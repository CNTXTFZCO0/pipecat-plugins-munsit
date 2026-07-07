"""Munsit Arabic text-to-speech service implementation for Pipecat.

This module provides TTS functionality using Munsit's API, specializing in
high-quality Arabic speech synthesis with streaming support.

Features:
    - HTTP streaming of PCM16 audio at 48kHz mono (24kHz also supported)
    - Configurable voice, model, stability, and speed parameters
    - Runtime settings updates via ``TTSUpdateSettingsFrame``
    - OpenTelemetry tracing via @traced_tts
"""

import os
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

MUNSIT_API_BASE_URL = "https://api.munsit.com/api/v1"


@dataclass
class MunsitTTSSettings(TTSSettings):
    """Runtime-updatable settings for Munsit TTS.

    Parameters:
        stability: Voice consistency (0.0 = more variable, 1.0 = more consistent).
        speed: Speech rate multiplier (0.7 = slower, 1.2 = faster).
    """

    stability: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class MunsitTTSService(TTSService):
    """Munsit Arabic text-to-speech service.

    Converts text to speech using Munsit's TTS API with support for multiple
    Arabic voices and models. Uses HTTP streaming to deliver low-latency
    PCM16 audio at 48kHz mono by default.

    Example::

        tts = MunsitTTSService(
            api_key="your-api-key",
            aiohttp_session=session,
            voice_id="ar-hijazi-female-2",
            model="faseeh-v1-preview",
        )

        # With custom parameters
        tts = MunsitTTSService(
            api_key="your-api-key",
            aiohttp_session=session,
            params=MunsitTTSService.InputParams(stability=0.7, speed=1.1),
        )
    """

    Settings = MunsitTTSSettings

    class InputParams(BaseModel):
        """Voice configuration parameters.

        Parameters:
            stability: Voice consistency (0.0 = more variable, 1.0 = more consistent).
            speed: Speech rate multiplier (0.7 = slower, 1.2 = faster).
        """

        stability: float = Field(default=0.5, ge=0.0, le=1.0)
        speed: float = Field(default=1.0, ge=0.7, le=1.2)

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str = "ar-hijazi-female-2",
        model: str = "faseeh-v1-preview",
        base_url: str = MUNSIT_API_BASE_URL,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        settings: Optional[MunsitTTSSettings] = None,
        **kwargs,
    ):
        """Initialize the Munsit TTS service.

        Args:
            api_key: Munsit API key. Falls back to MUNSIT_API_KEY (or the legacy
                FASEEH_API_KEY) env var.
            aiohttp_session: Shared aiohttp session for making requests.
            voice_id: Voice identifier. Defaults to "ar-hijazi-female-2".
            model: TTS model name. Defaults to "faseeh-v1-preview".
            base_url: Munsit API base URL.
            sample_rate: Output audio sample rate in Hz, requested from the API.
                Defaults to 48000; 24000 also supported.
            params: Voice configuration parameters.
            settings: Runtime-updatable settings delta; values here take
                precedence over ``voice_id``/``model``/``params``.
            **kwargs: Additional arguments passed to parent TTSService.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        default_settings = MunsitTTSSettings(
            model=model,
            voice=voice_id,
            language=None,
            stability=0.5,
            speed=1.0,
        )
        if params is not None:
            default_settings.stability = params.stability
            default_settings.speed = params.speed
        if settings is not None:
            default_settings.apply_update(settings)

        # The base pushes TTSStartedFrame/TTSStoppedFrame and manages the audio
        # context; run_tts must yield only audio (pipecat >= 1.5 contract).
        super().__init__(
            sample_rate=sample_rate or 48000,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = (
            api_key or os.environ.get("MUNSIT_API_KEY") or os.environ.get("FASEEH_API_KEY", "")
        )
        if not self._api_key:
            raise ValueError(
                "Munsit API key is required. Pass api_key or set MUNSIT_API_KEY env var."
            )

        self._session = aiohttp_session
        self._base_url = base_url

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Munsit service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Munsit's TTS API.

        Streams PCM16 audio chunks from Munsit's HTTP endpoint, yielding
        TTSAudioRawFrame objects as data arrives.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: TTSAudioRawFrame(s), or ErrorFrame on failure. Start/stop
            frames are pushed by the base service.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        model = self._settings.model
        url = f"{self._base_url}/text-to-speech/{model}"
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "voice_id": self._settings.voice,
            "text": text,
            "model": model,
            "stability": self._settings.stability,
            "speed": self._settings.speed,
            "streaming": True,
            # request the declared rate — the emitted frames are stamped with it
            "sample_rate": self.sample_rate,
        }

        try:
            await self.start_ttfb_metrics()
            request_start = time.perf_counter()
            first_chunk_received = False

            async with self._session.post(
                url,
                headers=headers,
                json=payload,
                # fail fast on unreachable hosts; no total limit — the body streams
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=self._parse_error(response.status, error_text))
                    return

                await self.start_tts_usage_metrics(text)

                audio_buffer = bytearray()
                async for chunk, _ in response.content.iter_chunks():
                    if chunk:
                        if not first_chunk_received:
                            first_chunk_received = True
                            ttfb = time.perf_counter() - request_start
                            logger.debug(f"{self} TTFB: {ttfb:.3f}s")
                        await self.stop_ttfb_metrics()
                        audio_buffer.extend(chunk)

                        # PCM16 = 2 bytes per sample. Yield only aligned data.
                        aligned_length = len(audio_buffer) & ~1
                        if aligned_length > 0:
                            yield TTSAudioRawFrame(
                                audio=bytes(audio_buffer[:aligned_length]),
                                sample_rate=self.sample_rate,
                                num_channels=1,
                                context_id=context_id,
                            )
                            audio_buffer = audio_buffer[aligned_length:]

                # Flush any remaining complete samples
                if len(audio_buffer) >= 2:
                    aligned_length = len(audio_buffer) & ~1
                    yield TTSAudioRawFrame(
                        audio=bytes(audio_buffer[:aligned_length]),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )

        except aiohttp.ClientError as e:
            yield ErrorFrame(error=f"Munsit connection error: {e}")
        except Exception as e:
            yield ErrorFrame(error=f"Munsit TTS error: {e}")

    def _parse_error(self, status: int, body: str) -> str:
        """Map Munsit HTTP errors to descriptive messages.

        Args:
            status: HTTP status code.
            body: Response body text.

        Returns:
            A human-readable error message.
        """
        error_messages = {
            400: f"Bad request: {body}",
            401: "Unauthorized - check your MUNSIT_API_KEY",
            402: f"Insufficient balance - add credits at app.munsit.com: {body}",
            403: f"Access denied - model/feature not enabled: {body}",
            404: "Voice or model not found - check voice_id and model parameters",
            429: "Rate limit exceeded - reduce request frequency",
        }
        return error_messages.get(status, f"Munsit API error (status {status}): {body}")
