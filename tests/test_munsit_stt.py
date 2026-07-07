"""Tests for MunsitSTTService against a local fake /listen server."""

import asyncio
import os
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest
from fake_munsit import FakeMunsitServer

from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.task import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_plugins_munsit.stt import MunsitSTTService


@pytest.fixture
async def fake():
    server = FakeMunsitServer()
    await server.start()
    yield server
    await server.stop()


def _service(fake, **kwargs):
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("url", fake.url)
    kwargs.setdefault("sample_rate", 16000)
    kwargs.setdefault("audio_passthrough", False)
    return MunsitSTTService(**kwargs)


def _audio(ms=100, rate=16000):
    return InputAudioRawFrame(audio=b"\x00" * (rate * 2 * ms // 1000), sample_rate=rate, num_channels=1)


def _results(transcript, is_final, speech_final=False):
    return {
        "type": "Results",
        "channel": 0,
        "turn_id": 1,
        "transcript": transcript,
        "words": (
            [{"word": transcript.split()[0], "start": 0.1, "end": 0.4, "confidence": 0.99}]
            if transcript
            else []
        ),
        "is_final": is_final,
        "speech_final": speech_final,
        "language": "ar",
        "confidence": 0.98 if is_final else None,
    }


# --- Constructor validation ---


def test_raises_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="API key is required"):
            MunsitSTTService()


def test_reads_env_vars():
    with patch.dict(os.environ, {"MUNSIT_API_KEY": "env-key"}, clear=True):
        assert MunsitSTTService()._api_key == "env-key"
    with patch.dict(os.environ, {"FASEEH_API_KEY": "legacy-key"}, clear=True):
        assert MunsitSTTService()._api_key == "legacy-key"


def test_endpointing_range():
    with pytest.raises(ValueError, match="endpointing_ms"):
        MunsitSTTService(api_key="k", endpointing_ms=50)
    with pytest.raises(ValueError, match="endpointing_ms"):
        MunsitSTTService(api_key="k", endpointing_ms=6000)


def test_hotword_validation():
    with pytest.raises(ValueError, match="commas"):
        MunsitSTTService(api_key="k", hotwords=["a,b"])
    with pytest.raises(ValueError, match="40"):
        MunsitSTTService(api_key="k", hotwords=["x" * 41])
    with pytest.raises(ValueError, match="200"):
        MunsitSTTService(api_key="k", hotwords=[f"w{i}" for i in range(201)])


def test_correlation_id_length():
    with pytest.raises(ValueError, match="correlation_id"):
        MunsitSTTService(api_key="k", correlation_id="x" * 129)


def test_url_building():
    svc = MunsitSTTService(
        api_key="k",
        endpointing_ms=300,
        smart_turn=True,
        hotwords=["فصيح", "منصت"],
        correlation_id="call-1",
    )
    q = parse_qs(urlparse(svc._build_url()).query)
    assert q["encoding"] == ["linear16"]
    assert q["sample_rate"] == ["16000"]
    assert q["language"] == ["ar"]
    assert q["endpointing"] == ["300"]
    assert q["smart_turn"] == ["true"]
    assert q["hotwords"] == ["فصيح,منصت"]
    assert q["correlation_id"] == ["call-1"]
    assert q["api_key"] == ["k"]


def test_smart_turn_always_explicit():
    # server default is true - the param must always be sent so False can take effect
    q_default = parse_qs(urlparse(MunsitSTTService(api_key="k")._build_url()).query)
    assert q_default["smart_turn"] == ["true"]
    q_off = parse_qs(urlparse(MunsitSTTService(api_key="k", smart_turn=False)._build_url()).query)
    assert q_off["smart_turn"] == ["false"]


# --- Pipeline integration against the fake server ---


async def test_interim_and_final_frames(fake):
    fake.state.on_audio_events = [
        (3200, _results("مرحبا", is_final=False)),
        (6400, _results("مرحبا بكم", is_final=True, speech_final=True)),
    ]
    stt = _service(fake)
    frames_to_send = [_audio(), SleepFrame(0.3), _audio(), SleepFrame(0.5)]
    down, _ = await run_test(stt, frames_to_send=frames_to_send)

    interims = [f for f in down if isinstance(f, InterimTranscriptionFrame)]
    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert [f.text for f in interims] == ["مرحبا"]
    assert [f.text for f in finals] == ["مرحبا بكم"]
    assert finals[0].finalized is True
    assert finals[0].result["confidence"] == 0.98
    assert str(finals[0].language).lower().count("ar")

    # query params reached the server
    assert fake.state.listen_received_query["language"] == "ar"
    assert fake.state.listen_received_query["sample_rate"] == "16000"
    # audio reached the server, and EndFrame drained with CloseStream
    assert fake.state.listen_binary_bytes == 6400
    assert {"type": "CloseStream"} in fake.state.listen_controls


async def test_forced_split_is_not_finalized(fake):
    fake.state.on_audio_events = [(3200, _results("كلام طويل", is_final=True, speech_final=False))]
    stt = _service(fake)
    down, _ = await run_test(stt, frames_to_send=[_audio(), SleepFrame(0.5)])
    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert len(finals) == 1
    assert finals[0].finalized is False


async def test_empty_transcript_skipped(fake):
    fake.state.on_audio_events = [(3200, _results("", is_final=False) | {"transcript": ""})]
    stt = _service(fake)
    down, _ = await run_test(stt, frames_to_send=[_audio(), SleepFrame(0.4)])
    assert not [f for f in down if isinstance(f, (InterimTranscriptionFrame, TranscriptionFrame))]


async def test_enrichment_callback(fake):
    fake.state.on_audio_events = [
        (3200, _results("مرحبا", is_final=True, speech_final=True)),
        (3200, {"type": "UtteranceEnd", "channel": 0, "last_word_end": 0.4}),
        (3200, {"type": "Gender", "channel": 0, "turn_id": 1, "gender": "male", "confidence": 0.99}),
        (3200, {"type": "Sentiment", "channel": 0, "turn_id": 1, "sentiment": "positive", "confidence": 0.9}),
    ]
    seen = []

    async def on_enrichment(event):
        seen.append(event["type"])

    stt = _service(fake, on_enrichment=on_enrichment)
    await run_test(stt, frames_to_send=[_audio(), SleepFrame(0.5)])
    assert seen == ["Gender", "Sentiment"]


async def test_nonrecoverable_error_pushed(fake):
    fake.state.on_audio_events = [
        (3200, {"type": "Error", "code": 1008, "message": "wallet balance exhausted", "recoverable": False}),
    ]
    stt = _service(fake)
    _, up = await run_test(stt, frames_to_send=[_audio(), SleepFrame(0.4)])
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert errors and "wallet balance exhausted" in errors[0].error


async def test_fatal_error_stops_reconnecting(fake):
    # non-recoverable Error then server close: the plugin must not reconnect-loop
    fake.state.on_audio_events = [
        (3200, {"type": "Error", "code": 1008, "message": "authentication required", "recoverable": False}),
    ]
    fake.state.close_after_bytes = 6400
    stt = _service(fake)
    _, up = await run_test(
        stt, frames_to_send=[_audio(), SleepFrame(0.3), _audio(), SleepFrame(1.0)]
    )
    assert [f for f in up if isinstance(f, ErrorFrame)]
    assert fake.state.listen_connect_count == 1, "must not reconnect after a fatal server error"


async def test_recoverable_error_not_pushed(fake):
    fake.state.on_audio_events = [
        (3200, {"type": "Error", "code": 4002, "message": "bad Configure", "recoverable": True}),
    ]
    stt = _service(fake)
    _, up = await run_test(stt, frames_to_send=[_audio(), SleepFrame(0.4)])
    assert not [f for f in up if isinstance(f, ErrorFrame)]


async def test_resamples_unsupported_rate(fake):
    stt = _service(fake, sample_rate=24000)
    # 100ms at 24kHz mono PCM16 = 4800 bytes -> 3200 bytes at 16kHz
    await run_test(stt, frames_to_send=[_audio(rate=24000), SleepFrame(0.4)])
    q = parse_qs(urlparse(stt._build_url()).query)
    assert q["sample_rate"] == ["16000"]
    assert fake.state.listen_binary_bytes > 0
    assert fake.state.listen_binary_bytes <= 3300  # resampled down, not passed through


async def test_keepalive_sends_control_not_audio(fake):
    stt = _service(fake, keepalive_timeout=0.2, keepalive_interval=0.1)
    await run_test(stt, frames_to_send=[SleepFrame(0.8)])
    keepalives = [c for c in fake.state.listen_controls if c.get("type") == "KeepAlive"]
    assert keepalives, "expected a protocol KeepAlive during idle"
    assert fake.state.listen_binary_bytes == 0, "keepalive must not send billable audio"


async def test_dropped_hotwords_logged(fake):
    from fake_munsit import default_open_metadata

    fake.state.on_open_events = [default_open_metadata(dropped_hotwords=["verylongword"])]
    stt = _service(fake, hotwords=["ok"])
    down, up = await run_test(stt, frames_to_send=[SleepFrame(0.3)])
    # session still usable, no errors
    assert not [f for f in up if isinstance(f, ErrorFrame)]


async def test_finalized_final_reports_stt_ttfb_immediately(fake):
    """speech_final finals must early-report TTFB (no 2s stt_ttfb_timeout wait),
    so the metric lands inside the user->bot latency window on fast turns."""
    fake.state.on_audio_events = [(3200, _results("مرحبا", is_final=True, speech_final=True))]
    stt = _service(fake)
    frames_to_send = [
        VADUserStartedSpeakingFrame(),
        _audio(),
        VADUserStoppedSpeakingFrame(stop_secs=0.2),
        SleepFrame(0.6),  # well under the 2.0s stt_ttfb_timeout
    ]
    down, _ = await run_test(
        stt,
        frames_to_send=frames_to_send,
        pipeline_params=PipelineParams(enable_metrics=True),
    )
    ttfb = [
        d
        for f in down
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, TTFBMetricsData) and "Munsit" in d.processor
    ]
    assert ttfb, "expected an immediate STT TTFB metric after a finalized final"


async def test_stt_ttfb_reported_when_final_outraces_vad_stop(fake):
    """If the server final lands before the local VAD stop event (fast endpointing,
    short utterance), the metric must still fire within the 1.0s fallback window."""
    fake.state.on_audio_events = [(3200, _results("نعم", is_final=True, speech_final=True))]
    stt = _service(fake)
    frames_to_send = [
        VADUserStartedSpeakingFrame(),
        _audio(),          # final fires on receipt — before any VAD stop
        SleepFrame(0.4),
        VADUserStoppedSpeakingFrame(stop_secs=0.2),
        SleepFrame(1.3),   # > 1.0s fallback, < framework default 2.0s
    ]
    down, _ = await run_test(
        stt,
        frames_to_send=frames_to_send,
        pipeline_params=PipelineParams(enable_metrics=True),
    )
    ttfb = [
        d
        for f in down
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, TTFBMetricsData) and "Munsit" in d.processor
    ]
    assert ttfb, "fallback window must report STT TTFB before typical bot speech"
