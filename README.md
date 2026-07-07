# pipecat-plugins-munsit

[Pipecat](https://github.com/pipecat-ai/pipecat) services for [Munsit](https://app.munsit.com):
Arabic **speech-to-text** (live WebSocket streaming) and **text-to-speech** (HTTP streaming).

This package supersedes `pipecat-plugins-faseeh`, which is now a deprecation shim
that re-exports these services.

## Installation

```bash
pip install pipecat-plugins-munsit
```

Requires Python 3.10+ and `pipecat-ai >= 1.5.0`.

## Speech-to-Text (live streaming)

`MunsitSTTService` streams audio to Munsit's live WebSocket API
(`wss://api.munsit.com/api/v1/listen`) and pushes interim and final
transcription frames. Arabic only (v1).

```python
from pipecat_plugins_munsit import MunsitSTTService

stt = MunsitSTTService(
    api_key="your-api-key",       # or MUNSIT_API_KEY env var
    endpointing_ms=800,           # server-side silence window that ends a turn (100-5000)
    hotwords=["فصيح", "منصت"],    # custom vocabulary biasing (<=200 entries, <=40 chars)
)
```

### Per-turn enrichment (speaker gender, sentiment)

```python
async def on_enrichment(event: dict):
    if event["type"] == "Gender":
        print("speaker gender:", event["label"], event["score"])
    elif event["type"] == "Sentiment":
        print("turn sentiment:", event["label"], event["score"])

stt = MunsitSTTService(api_key="...", on_enrichment=on_enrichment)
```

### Notes

- Input at 8 kHz / 16 kHz is sent as-is; other pipeline rates are resampled
  to 16 kHz client-side.
- Idle connections are kept open with protocol `KeepAlive` control frames
  (never silence, which would be billed as audio).
- A final with `TranscriptionFrame.finalized == False` is a forced split
  inside one long utterance (the server splits turns at ~60s of unbroken
  speech), not the end of the user's turn.
- `smart_turn` (on by default) gates end-of-turn on the server's semantic
  turn-completion model, reducing premature cuts; pass `smart_turn=False`
  to end turns on silence alone.

## Text-to-Speech

```python
import aiohttp
from pipecat_plugins_munsit import MunsitTTSService

async with aiohttp.ClientSession() as session:
    tts = MunsitTTSService(
        api_key="your-api-key",   # or MUNSIT_API_KEY env var
        aiohttp_session=session,
        voice_id="ar-hijazi-female-2",
        model="faseeh-v1-preview",
    )
```

Streams PCM16 audio at 48 kHz mono by default (`sample_rate=24000` also supported). `stability` (0.0-1.0) and `speed`
(0.7-1.2) are configurable at construction and runtime-updatable via
`TTSUpdateSettingsFrame`.

## Full agent example

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat_plugins_munsit import MunsitSTTService, MunsitTTSService

pipeline = Pipeline([
    transport.input(),
    MunsitSTTService(api_key="..."),
    context_aggregator.user(),
    llm,
    MunsitTTSService(api_key="...", aiohttp_session=session),
    transport.output(),
    context_aggregator.assistant(),
])
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

Releases are published to PyPI by pushing a `v*` tag (see
`.github/workflows/publish.yml`; requires the `PYPI_API_TOKEN` repo secret).

## License

Apache-2.0
