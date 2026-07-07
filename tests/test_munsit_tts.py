"""Tests for MunsitTTSService (ported from the faseeh plugin suite)."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame

from pipecat_plugins_munsit.tts import MunsitTTSService


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def api_key():
    return "test-api-key-123"


@pytest.fixture
def service(api_key, mock_session):
    return MunsitTTSService(api_key=api_key, aiohttp_session=mock_session)


# --- Constructor tests ---


def test_raises_without_api_key(mock_session):
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("MUNSIT_API_KEY", None)
        os.environ.pop("FASEEH_API_KEY", None)
        with pytest.raises(ValueError, match="API key is required"):
            MunsitTTSService(aiohttp_session=mock_session)


def test_accepts_api_key_param(mock_session):
    svc = MunsitTTSService(api_key="my-key", aiohttp_session=mock_session)
    assert svc._api_key == "my-key"


def test_reads_env_var(mock_session):
    with patch.dict(os.environ, {"MUNSIT_API_KEY": "env-key"}, clear=True):
        svc = MunsitTTSService(aiohttp_session=mock_session)
        assert svc._api_key == "env-key"


def test_reads_legacy_env_var(mock_session):
    with patch.dict(os.environ, {"FASEEH_API_KEY": "legacy-key"}, clear=True):
        svc = MunsitTTSService(aiohttp_session=mock_session)
        assert svc._api_key == "legacy-key"


def test_munsit_env_var_wins_over_legacy(mock_session):
    with patch.dict(
        os.environ, {"MUNSIT_API_KEY": "new-key", "FASEEH_API_KEY": "legacy-key"}, clear=True
    ):
        svc = MunsitTTSService(aiohttp_session=mock_session)
        assert svc._api_key == "new-key"


def test_default_model_and_voice(service):
    assert service._settings.model == "faseeh-v1-preview"
    assert service._settings.voice == "ar-hijazi-female-2"


def test_custom_model_and_voice(api_key, mock_session):
    svc = MunsitTTSService(
        api_key=api_key,
        aiohttp_session=mock_session,
        voice_id="ar-emirati-male-1",
        model="faseeh-v2",
    )
    assert svc._settings.model == "faseeh-v2"
    assert svc._settings.voice == "ar-emirati-male-1"


def test_custom_params(api_key, mock_session):
    params = MunsitTTSService.InputParams(stability=0.8, speed=1.1)
    svc = MunsitTTSService(api_key=api_key, aiohttp_session=mock_session, params=params)
    assert svc._settings.stability == 0.8
    assert svc._settings.speed == 1.1


def test_settings_delta_wins(api_key, mock_session):
    svc = MunsitTTSService(
        api_key=api_key,
        aiohttp_session=mock_session,
        voice_id="ar-hijazi-female-2",
        settings=MunsitTTSService.Settings(voice="ar-emirati-male-1", stability=0.3),
    )
    assert svc._settings.voice == "ar-emirati-male-1"
    assert svc._settings.stability == 0.3
    assert svc._settings.speed == 1.0


def test_default_sample_rate(service):
    # sample_rate is set during __init__ via _init_sample_rate, but actual
    # _sample_rate is set during start(). Check the init value.
    assert service._init_sample_rate == 48000


def test_can_generate_metrics(service):
    assert service.can_generate_metrics() is True


def test_params_validation():
    with pytest.raises(Exception):
        MunsitTTSService.InputParams(stability=2.0)

    with pytest.raises(Exception):
        MunsitTTSService.InputParams(speed=0.1)


# --- run_tts tests ---


async def _collect_frames(gen):
    """Collect all frames from an async generator."""
    frames = []
    async for frame in gen:
        frames.append(frame)
    return frames


def _make_mock_response(status=200, chunks=None):
    """Create a mock aiohttp response with streaming content."""
    response = AsyncMock()
    response.status = status
    response.text = AsyncMock(return_value="error body")

    if chunks is None:
        chunks = [b"\x00\x01" * 100, b"\x02\x03" * 100]

    async def iter_chunks():
        for chunk in chunks:
            yield chunk, True

    response.content = MagicMock()
    response.content.iter_chunks = iter_chunks

    return response


@pytest.mark.asyncio
async def test_run_tts_success(api_key):
    """Successful TTS should yield only audio frames."""
    audio_chunks = [b"\x00\x01" * 50, b"\x02\x03" * 50]
    mock_response = _make_mock_response(status=200, chunks=audio_chunks)

    session = MagicMock()
    session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock(return_value=False)))

    svc = MunsitTTSService(api_key=api_key, aiohttp_session=session)
    # Mock metric methods
    svc.start_ttfb_metrics = AsyncMock()
    svc.stop_ttfb_metrics = AsyncMock()
    svc.start_tts_usage_metrics = AsyncMock()
    svc._sample_rate = 24000

    frames = await _collect_frames(svc.run_tts("Hello", "ctx-1"))

    # Check frame sequence
    # run_tts yields only audio; start/stop frames come from the base service

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) >= 1
    for af in audio_frames:
        assert af.sample_rate == 24000
        assert af.num_channels == 1
        assert af.context_id == "ctx-1"
        assert len(af.audio) % 2 == 0  # PCM16 aligned


@pytest.mark.asyncio
async def test_run_tts_odd_byte_alignment(api_key):
    """Odd-length chunks should be buffered and aligned to 2-byte boundaries."""
    # 3 bytes: first 2 should be yielded, last 1 buffered
    # Then 3 more bytes: buffered byte + 3 = 4 bytes, yielded
    audio_chunks = [b"\x01\x02\x03", b"\x04\x05\x06"]
    mock_response = _make_mock_response(status=200, chunks=audio_chunks)

    session = MagicMock()
    session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock(return_value=False)))

    svc = MunsitTTSService(api_key=api_key, aiohttp_session=session)
    svc.start_ttfb_metrics = AsyncMock()
    svc.stop_ttfb_metrics = AsyncMock()
    svc.start_tts_usage_metrics = AsyncMock()
    svc._sample_rate = 24000

    frames = await _collect_frames(svc.run_tts("Test", "ctx-2"))

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    for af in audio_frames:
        assert len(af.audio) % 2 == 0


@pytest.mark.asyncio
async def test_run_tts_api_401(api_key):
    """401 error should yield ErrorFrame with auth message."""
    mock_response = _make_mock_response(status=401)

    session = MagicMock()
    session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock(return_value=False)))

    svc = MunsitTTSService(api_key=api_key, aiohttp_session=session)
    svc.start_ttfb_metrics = AsyncMock()
    svc.stop_ttfb_metrics = AsyncMock()
    svc._sample_rate = 24000

    frames = await _collect_frames(svc.run_tts("Test", "ctx-3"))

    error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
    assert len(error_frames) == 1
    assert "MUNSIT_API_KEY" in error_frames[0].error


@pytest.mark.asyncio
async def test_run_tts_api_429(api_key):
    """429 error should yield ErrorFrame with rate limit message."""
    mock_response = _make_mock_response(status=429)

    session = MagicMock()
    session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock(return_value=False)))

    svc = MunsitTTSService(api_key=api_key, aiohttp_session=session)
    svc.start_ttfb_metrics = AsyncMock()
    svc.stop_ttfb_metrics = AsyncMock()
    svc._sample_rate = 24000

    frames = await _collect_frames(svc.run_tts("Test", "ctx-4"))

    error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
    assert len(error_frames) == 1
    assert "Rate limit" in error_frames[0].error


@pytest.mark.asyncio
async def test_run_tts_connection_error(api_key):
    """aiohttp.ClientError should yield ErrorFrame."""
    import aiohttp as aiohttp_mod

    session = MagicMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(side_effect=aiohttp_mod.ClientError("Connection refused"))
    cm.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=cm)

    svc = MunsitTTSService(api_key=api_key, aiohttp_session=session)
    svc.start_ttfb_metrics = AsyncMock()
    svc.stop_ttfb_metrics = AsyncMock()
    svc._sample_rate = 24000

    frames = await _collect_frames(svc.run_tts("Test", "ctx-5"))

    error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
    assert len(error_frames) == 1
    assert "connection error" in error_frames[0].error.lower()


@pytest.mark.asyncio
async def test_run_tts_unexpected_error(api_key):
    """Unexpected exceptions should yield ErrorFrame."""
    session = MagicMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(side_effect=RuntimeError("Unexpected"))
    cm.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=cm)

    svc = MunsitTTSService(api_key=api_key, aiohttp_session=session)
    svc.start_ttfb_metrics = AsyncMock()
    svc.stop_ttfb_metrics = AsyncMock()
    svc._sample_rate = 24000

    frames = await _collect_frames(svc.run_tts("Test", "ctx-6"))

    error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
    assert len(error_frames) == 1
    assert "Munsit TTS error" in error_frames[0].error


# --- Settings update tests ---


@pytest.mark.asyncio
async def test_update_settings_voice(service):
    # voice_id is aliased to voice by the settings machinery
    delta = MunsitTTSService.Settings.from_mapping({"voice_id": "ar-emirati-male-1"})
    await service._update_settings(delta)
    assert service._settings.voice == "ar-emirati-male-1"


@pytest.mark.asyncio
async def test_update_settings_stability_speed(service):
    await service._update_settings(MunsitTTSService.Settings(stability=0.9, speed=0.8))
    assert service._settings.stability == 0.9
    assert service._settings.speed == 0.8


# --- Error parsing tests ---


def test_parse_error_messages(service):
    assert "MUNSIT_API_KEY" in service._parse_error(401, "")
    assert "balance" in service._parse_error(402, "low").lower()
    assert "Rate limit" in service._parse_error(429, "")
    assert "not found" in service._parse_error(404, "").lower()
    assert "status 500" in service._parse_error(500, "internal")
