"""Munsit Arabic live speech-to-text service implementation for Pipecat.

Streams audio to Munsit's live WebSocket API (``/api/v1/listen``) and turns the
typed JSON events into Pipecat frames:

    - ``Results`` (interim)              -> ``InterimTranscriptionFrame``
    - ``Results`` (``is_final: true``)   -> ``TranscriptionFrame``
    - ``Gender`` / ``Sentiment``         -> ``on_enrichment`` callback
    - ``Error``                          -> logged (recoverable) or pushed as an error

``speech_final`` is carried through as ``TranscriptionFrame.finalized`` — a final
with ``finalized=False`` is a forced split inside one long utterance (the server
splits turns at ~60s of unbroken speech), not the end of the user's turn.

Turn boundaries (endpointing) are decided server-side, configured with
``endpointing_ms``. Idle connections are kept open by sending protocol
``KeepAlive`` control frames — never silence, which would be billed as audio.
"""

import asyncio
import json
import os
from typing import AsyncGenerator, Awaitable, Callable, Optional, Sequence, Union
from urllib.parse import urlencode

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

DEFAULT_LISTEN_URL = "wss://api.munsit.com/api/v1/listen"

# Engine-accepted input rates; anything else is resampled to 16 kHz client-side.
_ENGINE_SAMPLE_RATES = (8000, 16000)

# Measured p95 speech-end -> final-transcript latency is ~1.51s at the default
# endpointing of 800ms (RTX 5090); 1.6 adds p99 margin.
MUNSIT_TTFS_P99 = 1.6

# Server closes a session after 12s without audio or KeepAlive.
_DEFAULT_KEEPALIVE_TIMEOUT = 8.0
_DEFAULT_KEEPALIVE_INTERVAL = 2.0

_MAX_HOTWORDS = 200
_MAX_HOTWORD_CHARS = 40

EnrichmentCallback = Callable[[dict], Union[None, Awaitable[None]]]


def _validate_hotwords(hotwords: Sequence[str]) -> list[str]:
    """Validate hotword entries against the API limits (<=200 entries, <=40 chars, no commas)."""
    if len(hotwords) > _MAX_HOTWORDS:
        raise ValueError(f"hotwords: at most {_MAX_HOTWORDS} entries allowed, got {len(hotwords)}")
    cleaned = []
    for w in hotwords:
        w = w.strip()
        if not w:
            continue
        if "," in w:
            raise ValueError(f"hotwords: entries must not contain commas: {w!r}")
        if len(w) > _MAX_HOTWORD_CHARS:
            raise ValueError(
                f"hotwords: entries must be <= {_MAX_HOTWORD_CHARS} characters: {w!r}"
            )
        cleaned.append(w)
    return cleaned


class MunsitSTTService(WebsocketSTTService):
    """Munsit Arabic live speech-to-text service.

    Streams audio to Munsit's ``/api/v1/listen`` WebSocket API and pushes
    interim and final transcription frames. Arabic only (v1).

    Example::

        stt = MunsitSTTService(
            api_key="your-api-key",
            endpointing_ms=800,
            hotwords=["فصيح", "منصت"],
        )

        # Per-turn speaker gender / sentiment events ({"type", "label", "score", ...})
        async def on_enrichment(event: dict):
            print(event["type"], event["label"], event["score"])

        stt = MunsitSTTService(api_key="...", on_enrichment=on_enrichment)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        url: str = DEFAULT_LISTEN_URL,
        sample_rate: Optional[int] = None,
        interim_results: bool = True,
        endpointing_ms: int = 800,
        smart_turn: bool = True,
        hotwords: Optional[Sequence[str]] = None,
        correlation_id: Optional[str] = None,
        on_enrichment: Optional[EnrichmentCallback] = None,
        stt_ttfb_timeout: float = 1.0,
        ttfs_p99_latency: Optional[float] = MUNSIT_TTFS_P99,
        keepalive_timeout: Optional[float] = _DEFAULT_KEEPALIVE_TIMEOUT,
        keepalive_interval: float = _DEFAULT_KEEPALIVE_INTERVAL,
        **kwargs,
    ):
        """Initialize the Munsit live STT service.

        Args:
            api_key: Munsit API key. Falls back to MUNSIT_API_KEY (or the legacy
                FASEEH_API_KEY) env var.
            url: Live STT WebSocket URL.
            sample_rate: Input audio sample rate in Hz. If None, taken from the
                pipeline StartFrame. Rates other than 8000/16000 are resampled
                to 16 kHz before sending.
            interim_results: Whether the server should send interim results.
            endpointing_ms: Server-side silence window (ms) that ends a turn.
                Valid range 100-5000. Defaults to 800.
            smart_turn: Gate end-of-turn on the server's semantic turn-completion
                model in addition to silence, reducing premature cuts when speakers
                pause mid-thought (hard ceiling of 2x ``endpointing_ms``). On by
                default, matching the server; pass False to end turns on silence
                alone.
            hotwords: Custom vocabulary to bias recognition toward. At most 200
                entries of up to 40 characters; entries must not contain commas.
            correlation_id: Client-chosen ID echoed on every event (<=128 chars).
            on_enrichment: Callback (sync or async) invoked with each per-turn
                ``Gender`` and ``Sentiment`` event dict.
            stt_ttfb_timeout: Fallback wait (s) after VAD stop before reporting the
                speech-end -> final TTFB metric when the final outraced the local
                VAD stop event. Munsit finals are single-shot, so 1.0s (down from
                the framework's 2.0s) keeps the metric inside typical user->bot
                latency windows.
            ttfs_p99_latency: P99 speech-end -> final-transcript latency hint
                broadcast to downstream turn strategies.
            keepalive_timeout: Seconds of no audio before a protocol KeepAlive is
                sent (the server closes idle sessions after 12s). None disables.
            keepalive_interval: Seconds between keepalive idle checks.
            **kwargs: Additional arguments passed to parent WebsocketSTTService.

        Raises:
            ValueError: If no API key is found, or a parameter is out of range.
        """
        super().__init__(
            sample_rate=sample_rate,
            settings=STTSettings(model="munsit-v2", language=Language.AR),
            stt_ttfb_timeout=stt_ttfb_timeout,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=keepalive_timeout,
            keepalive_interval=keepalive_interval,
            **kwargs,
        )

        self._api_key = (
            api_key or os.environ.get("MUNSIT_API_KEY") or os.environ.get("FASEEH_API_KEY", "")
        )
        if not self._api_key:
            raise ValueError(
                "Munsit API key is required. Pass api_key or set MUNSIT_API_KEY env var."
            )

        if not 100 <= endpointing_ms <= 5000:
            raise ValueError(f"endpointing_ms must be within 100-5000, got {endpointing_ms}")
        if correlation_id is not None and len(correlation_id) > 128:
            raise ValueError("correlation_id must be at most 128 characters")

        self._url = url
        self._interim_results = interim_results
        self._endpointing_ms = endpointing_ms
        self._smart_turn = smart_turn
        self._hotwords = _validate_hotwords(hotwords) if hotwords else []
        self._correlation_id = correlation_id
        self._on_enrichment = on_enrichment

        self._engine_rate = 16000
        self._resampler = None
        self._receive_task = None
        self._session_id: Optional[str] = None

    def can_generate_metrics(self) -> bool:
        """Munsit service supports metrics generation."""
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Map a Language enum to the service code. Arabic only (v1)."""
        if str(language.value).lower().startswith("ar"):
            return "ar"
        return None

    async def start(self, frame: StartFrame):
        """Start the service and open the websocket connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self.sample_rate in _ENGINE_SAMPLE_RATES:
            self._engine_rate = self.sample_rate
            self._resampler = None
        else:
            self._engine_rate = 16000
            self._resampler = create_stream_resampler()
            logger.debug(
                f"{self}: resampling {self.sample_rate} Hz -> {self._engine_rate} Hz "
                f"(server accepts {_ENGINE_SAMPLE_RATES})"
            )
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service gracefully, draining the server-side turn first.

        CloseStream forces the server to finalize buffered audio; the final
        Results, enrichment events, and billing Metadata arrive before the
        server closes with 1000. The receive task must stay alive (and must
        not auto-reconnect) until that drain completes.

        Args:
            frame: The end frame.
        """
        if self._websocket and self._websocket.state is State.OPEN:
            self._disconnecting = True  # the drain close is intentional: no reconnect
            await self._send_close_stream()
            try:
                await asyncio.wait_for(self._websocket.wait_closed(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"{self}: server did not close after CloseStream")
        await super().stop(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Optional[Frame], None]:
        """Send audio data to the live STT websocket.

        Args:
            audio: Raw 16-bit PCM audio bytes.

        Yields:
            Frame: None (transcription results come via the receive task).
        """
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                if self._resampler:
                    audio = await self._resampler.resample(
                        audio, self.sample_rate, self._engine_rate
                    )
                if audio:
                    await self._websocket.send(audio)
            except Exception as e:
                logger.warning(f"{self}: audio send failed: {e}")
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Munsit-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()

    async def _update_settings(self, delta):
        """Apply a settings delta; connection params are fixed for a live session.

        v1 of the API is single-model and Arabic-only, so changed fields are
        reported as unhandled instead of silently pretending to apply.
        """
        changed = await super()._update_settings(delta)
        self._warn_unhandled_updated_settings(changed)
        return changed

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    def _build_url(self) -> str:
        params: dict = {
            "encoding": "linear16",
            "sample_rate": self._engine_rate,
            "channels": 1,
            "language": "ar",
            "interim_results": "true" if self._interim_results else "false",
            "endpointing": self._endpointing_ms,
            # always explicit — the server default is true, so omitting the param
            # would make smart_turn=False silently ineffective
            "smart_turn": "true" if self._smart_turn else "false",
            "api_key": self._api_key,
        }
        if self._hotwords:
            params["hotwords"] = ",".join(self._hotwords)
        if self._correlation_id:
            params["correlation_id"] = self._correlation_id
        return f"{self._url}?{urlencode(params)}"

    async def _connect(self):
        """Open the websocket and start the receive and keepalive tasks."""
        await self._connect_websocket()

        await super()._connect()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Stop tasks and close the websocket."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the websocket connection."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"{self}: connecting to Munsit live STT")
            self._websocket = await websocket_connect(self._build_url())
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Munsit live STT: {e}", exception=e)

    async def _disconnect_websocket(self):
        """Close the websocket connection."""
        try:
            if self._websocket:
                logger.debug(f"{self}: disconnecting from Munsit live STT")
                await self._websocket.close()
        except Exception as e:
            logger.warning(f"{self}: error closing websocket: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _send_close_stream(self):
        """Ask the server to finalize buffered audio and close with billing metadata."""
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(json.dumps({"type": "CloseStream"}))
            except Exception as e:
                logger.warning(f"{self}: CloseStream send failed: {e}")

    async def _send_keepalive(self, silence: bytes):
        """Send a protocol KeepAlive instead of silence — silence is billed as audio.

        Args:
            silence: Ignored; the Munsit protocol has an explicit control frame.
        """
        if self._websocket and self._websocket.state is State.OPEN:
            await self._websocket.send(json.dumps({"type": "KeepAlive"}))

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and dispatch typed JSON events from the websocket."""
        async for message in self._get_websocket():
            if isinstance(message, (bytes, bytearray)):
                continue
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: non-JSON event ignored: {message[:120]}")
                continue
            await self._handle_event(event)

    async def _handle_event(self, event: dict):
        etype = event.get("type")

        if etype == "Results":
            await self._handle_results(event)
        elif etype == "Metadata":
            if "audio_seconds_billed" in event:
                logger.debug(
                    f"{self}: session {event.get('session_id')} closed — "
                    f"{event.get('audio_seconds_billed')}s billed, "
                    f"{event.get('turn_count')} turns"
                )
            else:
                self._session_id = event.get("session_id")
                dropped = event.get("dropped_hotwords") or []
                if dropped:
                    logger.warning(f"{self}: server dropped hotwords: {dropped}")
                logger.debug(f"{self}: session open {self._session_id}")
        elif etype in ("Gender", "Sentiment"):
            if self._on_enrichment:
                try:
                    result = self._on_enrichment(event)
                    if result is not None and hasattr(result, "__await__"):
                        await result
                except Exception as e:
                    logger.warning(f"{self}: on_enrichment callback failed: {e}")
        elif etype in ("SpeechStarted", "UtteranceEnd"):
            logger.trace(f"{self}: {etype} {event}")
        elif etype == "Error":
            if event.get("recoverable"):
                logger.warning(f"{self}: recoverable server error: {event.get('message')}")
            else:
                # auth / wallet / session-cap rejections: reconnecting would just
                # replay the same failure in a loop, so stop trying
                self._reconnect_on_error = False
                await self.push_error(
                    error_msg=f"Munsit live STT error: {event.get('message')} "
                    f"(code {event.get('code')})"
                )

    async def _handle_results(self, event: dict):
        transcript = event.get("transcript") or ""
        if not transcript:
            return
        if event.get("is_final"):
            await self.push_frame(
                TranscriptionFrame(
                    text=transcript,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                    language=Language.AR,
                    result=event,
                    # a final with speech_final=False is a forced split mid-utterance
                    finalized=bool(event.get("speech_final")),
                )
            )
            await self._handle_transcription(transcript, True, Language.AR)
            await self.stop_processing_metrics()
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text=transcript,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                    language=Language.AR,
                    result=event,
                )
            )
