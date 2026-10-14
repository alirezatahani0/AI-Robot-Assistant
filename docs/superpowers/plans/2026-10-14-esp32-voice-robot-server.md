# Robot Server (laptop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `uv run robot serve` runs a WebSocket server on the laptop that takes a button-started voice turn from a robot (or `robot client`), transcribes it with OpenAI, answers with an OpenAI model, speaks the answer with an Azure voice, and streams the audio back.

**Architecture:** A per-connection `TurnSession` state machine (idle → listening → thinking → speaking → idle) reuses the existing `UtteranceSegmenter` to end the turn on a pause, runs blocking cloud calls in threads with timeouts, and streams 16 kHz PCM back at ~1.1× real time. A thin `websockets` server checks the device token and feeds messages into the session. `robot client` is a laptop pretend-robot speaking the same protocol, so everything is testable before the ESP32 parts arrive. The ESP32 firmware is a separate plan.

**Tech Stack:** Python 3.12, uv, `websockets` (new asyncio API), `openai` (Responses + audio transcriptions), Azure Speech REST via `urllib`, numpy, sounddevice, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-esp32-voice-robot-design.md`

## Global Constraints

- Python ≥ 3.12; run everything with `uv run`; tests with `uv run pytest`.
- Audio on the wire, both directions: 16 kHz, 16-bit little-endian, mono, raw PCM. Mic frames are 20 ms (640 bytes).
- Endpoint: `ws://<host>:8765/robot`; handshake header `Authorization: Bearer <ROBOT_DEVICE_TOKEN>`. Wrong/missing token → HTTP 401 at the handshake; other paths → HTTP 404.
- Messages exactly as the spec: robot → `{"type":"hello","fw":…}`, `{"type":"start"}`, `{"type":"cancel"}`; server → `{"type":"state","state":"idle|listening|thinking|speaking|error"}` (optional `"reason":"no_speech"`), binary audio, `{"type":"audio_end"}`.
- Turn rules: end on 0.7 s pause after speech or 15 s of speech; `idle` + `no_speech` if no speech within 8 s or empty transcript; each cloud call times out after 10 s; reply audio paced at 1.1× real time after a 0.5 s burst.
- Speech-to-text: OpenAI `gpt-transcribe` (env `ROBOT_STT_MODEL`), no language pinned. Brain: OpenAI Responses API, model `gpt-6-luna` (env `ROBOT_CHAT_MODEL`), reasoning effort `low` (env `ROBOT_REASONING_EFFORT`, empty string = omit), `store=False`, last 10 turns of history per connection.
- Voice: Azure only, no fallback. Persian voice if the reply contains any Persian/Arabic-script letter, else English. Defaults `fa-IR-DilaraNeural` / `en-US-JennyNeural` (env `ROBOT_VOICE_FA` / `ROBOT_VOICE_EN`). Output format `raw-16khz-16bit-mono-pcm`.
- Error message «ببخشید، یه مشکلی پیش اومد.» generated once with Azure, cached at `~/.cache/robot/error_fa.wav`; if unavailable, show error state only.
- Secrets only from environment (`.env` via `uv run --env-file .env`), never logged or committed. `.env` is gitignored.
- Remove `piper-tts` and `edge-tts` dependencies.
- Match the codebase's comment density: comments only for constraints the code can't show.

## Review Focus

- Robot disconnects in the middle of a turn → server keeps running, the turn is cancelled, the agent is not called afterwards. (Task 6 test `test_disconnect_mid_turn_cancels_the_turn`.)
- Malformed JSON, unknown message types, or an odd-length audio frame → ignored, connection stays open. (Task 1 `test_trailing_odd_byte_is_dropped_instead_of_crashing`, Task 5 `test_garbage_and_unknown_messages_are_ignored`.)
- Reply text containing `<`, `&` or quotes → still valid SSML. (Task 2 `test_special_characters_are_escaped_in_ssml`.)
- Button pressed again while thinking/speaking → `start` ignored, `cancel` returns to idle with no more audio. (Task 5 `test_start_while_busy_is_ignored`, `test_cancel_while_thinking_stops_the_turn`.)
- A cloud call hangs → 10 s timeout → error state, robot returns to idle. (Task 5 `test_hung_service_times_out_into_the_error_path`.)

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/robot/pcm.py` | create | 16 kHz PCM ↔ `AudioChunk`, WAV bytes |
| `src/robot/server/__init__.py` | create | package marker |
| `src/robot/server/protocol.py` | create | message builders/parser, `State`, frame size |
| `src/robot/tts/synth.py` | create | `Synthesizer` interface (text → audio) |
| `src/robot/tts/voice.py` | create | Persian detection, voice pick, locale |
| `src/robot/tts/azure.py` | create | Azure REST synthesizer |
| `src/robot/stt/openai_stt.py` | create | OpenAI transcription |
| `src/robot/agent/openai_agent.py` | create | OpenAI Responses agent with history |
| `src/robot/agent/base.py` | modify | add `AgentError` |
| `src/robot/server/session.py` | create | per-connection turn state machine |
| `src/robot/server/app.py` | create | WebSocket server + token check |
| `src/robot/client.py` | create | laptop pretend-robot (WAV turn + interactive mic) |
| `src/robot/server/run.py` | create | env config, error-audio cache, `serve()` |
| `src/robot/main.py` | modify | `serve` and `client` subcommands |
| `src/robot/pipeline.py` | modify | `save_wav_mono` uses `encode_wav` |
| `tests/fakes.py` | create | shared fakes + server test helpers |
| `tests/test_protocol.py`, `test_azure_tts.py`, `test_openai_stt.py`, `test_openai_agent.py`, `test_session.py`, `test_server_app.py`, `test_client.py`, `test_serve_config.py` | create | tests |
| `.env.example`, `.gitignore`, `README.md`, spec | modify/create | config + docs |

---

### Task 1: PCM helpers and protocol messages

**Files:**
- Create: `src/robot/pcm.py`, `src/robot/server/__init__.py`, `src/robot/server/protocol.py`
- Modify: `pyproject.toml` / `uv.lock` (via `uv`)
- Test: `tests/test_protocol.py`

**Interfaces:**
- Produces: `robot.pcm.SAMPLE_RATE: int = 16000`; `pcm16_to_chunk(data: bytes, sample_rate: int = SAMPLE_RATE) -> AudioChunk`; `chunk_to_pcm16(chunk: AudioChunk) -> bytes` (raises `ValueError` if not 16 kHz).
- Produces: `robot.server.protocol.FRAME_BYTES = 640`; `State` (StrEnum: `IDLE, LISTENING, THINKING, SPEAKING, ERROR`); `ProtocolError(ValueError)`; `state_message(state: State, reason: str | None = None) -> str`; `audio_end_message() -> str`; `hello_message(fw: str) -> str`; `start_message() -> str`; `cancel_message() -> str`; `parse_message(text: str) -> dict`.

- [ ] **Step 1: Swap dependencies**

Run:
```bash
uv remove piper-tts edge-tts
uv add websockets openai
```
Expected: both commands succeed; `pyproject.toml` lists `websockets` and `openai`, no `piper-tts` / `edge-tts`.

- [ ] **Step 2: Write the failing tests**

`tests/test_protocol.py`:
```python
"""Robot ↔ server wire format."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot.device.base import AudioChunk
from robot.pcm import SAMPLE_RATE, chunk_to_pcm16, pcm16_to_chunk
from robot.server.protocol import (
    ProtocolError,
    State,
    audio_end_message,
    cancel_message,
    hello_message,
    parse_message,
    start_message,
    state_message,
)


def test_state_message_with_and_without_reason() -> None:
    assert json.loads(state_message(State.LISTENING)) == {"type": "state", "state": "listening"}
    assert json.loads(state_message(State.IDLE, "no_speech")) == {
        "type": "state",
        "state": "idle",
        "reason": "no_speech",
    }


def test_messages_round_trip_through_parse() -> None:
    assert parse_message(hello_message("0.1")) == {"type": "hello", "fw": "0.1"}
    assert parse_message(start_message()) == {"type": "start"}
    assert parse_message(cancel_message()) == {"type": "cancel"}
    assert parse_message(audio_end_message()) == {"type": "audio_end"}


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"state": "idle"}', '{"type": 5}'])
def test_malformed_messages_raise_protocol_error(raw: str) -> None:
    with pytest.raises(ProtocolError):
        parse_message(raw)


def test_float_audio_encodes_to_little_endian_int16() -> None:
    chunk = AudioChunk(samples=np.array([0.0, 0.5, -0.5], dtype=np.float32), sample_rate=SAMPLE_RATE)
    assert chunk_to_pcm16(chunk) == b"\x00\x00\xff\x3f\x01\xc0"


def test_out_of_range_samples_are_clipped() -> None:
    chunk = AudioChunk(samples=np.array([2.0, -2.0], dtype=np.float32), sample_rate=SAMPLE_RATE)
    assert chunk_to_pcm16(chunk) == b"\xff\x7f\x01\x80"


def test_pcm_decodes_to_float_samples() -> None:
    chunk = pcm16_to_chunk(b"\x00\x40\x00\xc0")
    assert chunk.sample_rate == SAMPLE_RATE
    np.testing.assert_allclose(chunk.samples, [0.5, -0.5])


def test_trailing_odd_byte_is_dropped_instead_of_crashing() -> None:
    chunk = pcm16_to_chunk(b"\x00\x40\x07")
    np.testing.assert_allclose(chunk.samples, [0.5])


def test_encoding_rejects_other_sample_rates() -> None:
    chunk = AudioChunk(samples=np.zeros(10, dtype=np.float32), sample_rate=24_000)
    with pytest.raises(ValueError):
        chunk_to_pcm16(chunk)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_protocol.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'robot.pcm'`.

- [ ] **Step 4: Implement**

`src/robot/pcm.py`:
```python
"""16 kHz 16-bit mono PCM — the audio format on the robot wire and from Azure."""

from __future__ import annotations

import numpy as np

from robot.device.base import AudioChunk

SAMPLE_RATE = 16_000


def pcm16_to_chunk(data: bytes, sample_rate: int = SAMPLE_RATE) -> AudioChunk:
    usable = len(data) - (len(data) % 2)
    samples = np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0
    return AudioChunk(samples=samples, sample_rate=sample_rate)


def chunk_to_pcm16(chunk: AudioChunk) -> bytes:
    if chunk.sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {chunk.sample_rate} Hz")
    pcm = np.clip(chunk.samples, -1.0, 1.0) * 32767.0
    return pcm.astype("<i2").tobytes()
```

`src/robot/server/__init__.py`:
```python
"""Robot server: the laptop (later a VPS) side of the ESP32 voice robot."""
```

`src/robot/server/protocol.py`:
```python
"""Robot ↔ server WebSocket messages (see docs/superpowers/specs/2026-10-14-esp32-voice-robot-design.md)."""

from __future__ import annotations

import json
from enum import StrEnum

from robot.pcm import SAMPLE_RATE

FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class ProtocolError(ValueError):
    """Malformed message from the other side."""


def state_message(state: State, reason: str | None = None) -> str:
    msg = {"type": "state", "state": state.value}
    if reason:
        msg["reason"] = reason
    return json.dumps(msg)


def audio_end_message() -> str:
    return json.dumps({"type": "audio_end"})


def hello_message(fw: str) -> str:
    return json.dumps({"type": "hello", "fw": fw})


def start_message() -> str:
    return json.dumps({"type": "start"})


def cancel_message() -> str:
    return json.dumps({"type": "cancel"})


def parse_message(text: str) -> dict:
    try:
        msg = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"not JSON: {text[:80]!r}") from exc
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        raise ProtocolError(f"missing 'type': {text[:80]!r}")
    return msg
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all tests pass (new 11 + existing 23).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/robot/pcm.py src/robot/server tests/test_protocol.py
git commit -m "Add robot wire protocol and PCM helpers; swap TTS deps for websockets/openai"
```

---

### Task 2: Azure synthesizer with voice selection

**Files:**
- Create: `src/robot/tts/synth.py`, `src/robot/tts/voice.py`, `src/robot/tts/azure.py`
- Test: `tests/test_azure_tts.py`

**Interfaces:**
- Consumes: `robot.pcm.pcm16_to_chunk`; `robot.tts.base.TtsError`.
- Produces: `robot.tts.synth.Synthesizer` (ABC) with `synthesize(self, text: str) -> AudioChunk` (16 kHz; raises `TtsError`).
- Produces: `robot.tts.voice.has_persian(text: str) -> bool`; `pick_voice(text: str, fa_voice: str, en_voice: str) -> str`; `voice_locale(voice: str) -> str`.
- Produces: `robot.tts.azure.AzureSynthesizer(endpoint: str, key: str, fa_voice: str = "fa-IR-DilaraNeural", en_voice: str = "en-US-JennyNeural", timeout: float = 10.0)`; `build_ssml(text: str, voice: str) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_azure_tts.py`:
```python
"""Azure text-to-speech: voice choice, SSML, HTTP handling (no network)."""

from __future__ import annotations

import io
import urllib.error

import numpy as np
import pytest

from robot.tts.azure import AzureSynthesizer, build_ssml
from robot.tts.base import TtsError
from robot.tts.voice import has_persian, pick_voice, voice_locale

FA = "fa-IR-DilaraNeural"
EN = "en-US-JennyNeural"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("سلام، خوبی؟", FA),
        ("Hello there", EN),
        ("فایل README رو باز کن", FA),
        ("Call me at 5 pm", EN),
        ("۱۲۳", FA),
    ],
)
def test_voice_is_persian_if_any_persian_letter_appears(text: str, expected: str) -> None:
    assert pick_voice(text, FA, EN) == expected


def test_has_persian_is_false_for_empty_text() -> None:
    assert has_persian("") is False


def test_voice_locale_comes_from_voice_name() -> None:
    assert voice_locale(FA) == "fa-IR"
    assert voice_locale(EN) == "en-US"
    with pytest.raises(ValueError):
        voice_locale("Dilara")


def test_special_characters_are_escaped_in_ssml() -> None:
    ssml = build_ssml('a < b & "c"', EN)
    assert ssml == (
        '<speak version="1.0" xml:lang="en-US">'
        '<voice name="en-US-JennyNeural">a &lt; b &amp; "c"</voice></speak>'
    )


class _Resp:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _capture_urlopen(monkeypatch, body: bytes) -> dict:
    captured: dict = {}

    def fake_urlopen(req, timeout):
        captured["req"] = req
        captured["timeout"] = timeout
        return _Resp(body)

    monkeypatch.setattr("robot.tts.azure.urllib.request.urlopen", fake_urlopen)
    return captured


def test_persian_reply_is_sent_with_persian_voice_and_returns_pcm(monkeypatch) -> None:
    captured = _capture_urlopen(monkeypatch, b"\x00\x40" * 4)
    synth = AzureSynthesizer("https://robot.cognitiveservices.azure.com/", "secret-key")

    audio = synth.synthesize("سلام")

    req = captured["req"]
    assert req.full_url == "https://robot.cognitiveservices.azure.com/tts/cognitiveservices/v1"
    assert req.get_method() == "POST"
    assert req.get_header("Ocp-apim-subscription-key") == "secret-key"
    assert req.get_header("X-microsoft-outputformat") == "raw-16khz-16bit-mono-pcm"
    assert req.get_header("Content-type") == "application/ssml+xml"
    assert 'xml:lang="fa-IR"' in req.data.decode("utf-8")
    assert f'name="{FA}"' in req.data.decode("utf-8")
    assert captured["timeout"] == 10.0
    assert audio.sample_rate == 16_000
    np.testing.assert_allclose(audio.samples, [0.5] * 4)


def test_english_reply_uses_english_voice(monkeypatch) -> None:
    captured = _capture_urlopen(monkeypatch, b"\x00\x00")
    AzureSynthesizer("https://robot.cognitiveservices.azure.com", "k").synthesize("Hi!")
    assert f'name="{EN}"' in captured["req"].data.decode("utf-8")


def test_http_error_becomes_tts_error(monkeypatch) -> None:
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr("robot.tts.azure.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(TtsError, match="401"):
        AzureSynthesizer("https://x.cognitiveservices.azure.com", "k").synthesize("Hi")


def test_empty_audio_response_is_an_error(monkeypatch) -> None:
    _capture_urlopen(monkeypatch, b"")
    with pytest.raises(TtsError):
        AzureSynthesizer("https://x.cognitiveservices.azure.com", "k").synthesize("Hi")


def test_empty_text_is_rejected_without_a_request(monkeypatch) -> None:
    captured = _capture_urlopen(monkeypatch, b"\x00\x00")
    with pytest.raises(TtsError):
        AzureSynthesizer("https://x.cognitiveservices.azure.com", "k").synthesize("   ")
    assert captured == {}


def test_missing_credentials_fail_fast() -> None:
    with pytest.raises(TtsError):
        AzureSynthesizer("", "k")
    with pytest.raises(TtsError):
        AzureSynthesizer("https://x.cognitiveservices.azure.com", "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_azure_tts.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'robot.tts.azure'`.

- [ ] **Step 3: Implement**

`src/robot/tts/synth.py`:
```python
"""Text → audio, for sending to the robot (TextToSpeech speaks on this machine instead)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from robot.device.base import AudioChunk


class Synthesizer(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> AudioChunk:
        """Return 16 kHz mono audio for `text`. Raises TtsError on failure."""
```

`src/robot/tts/voice.py`:
```python
"""One voice per reply: Persian if the reply contains any Persian letter, else English."""

from __future__ import annotations

import re

_PERSIAN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def has_persian(text: str) -> bool:
    return bool(_PERSIAN.search(text))


def pick_voice(text: str, fa_voice: str, en_voice: str) -> str:
    return fa_voice if has_persian(text) else en_voice


def voice_locale(voice: str) -> str:
    parts = voice.split("-")
    if len(parts) < 3:
        raise ValueError(f"unexpected Azure voice name: {voice!r}")
    return f"{parts[0]}-{parts[1]}"
```

`src/robot/tts/azure.py`:
```python
"""Azure Speech text-to-speech over REST, returning 16 kHz raw PCM."""

from __future__ import annotations

import urllib.error
import urllib.request
from xml.sax.saxutils import escape

from robot.device.base import AudioChunk
from robot.pcm import pcm16_to_chunk
from robot.tts.base import TtsError
from robot.tts.synth import Synthesizer
from robot.tts.voice import pick_voice, voice_locale

OUTPUT_FORMAT = "raw-16khz-16bit-mono-pcm"


def build_ssml(text: str, voice: str) -> str:
    return (
        f'<speak version="1.0" xml:lang="{voice_locale(voice)}">'
        f'<voice name="{voice}">{escape(text)}</voice></speak>'
    )


class AzureSynthesizer(Synthesizer):
    def __init__(
        self,
        endpoint: str,
        key: str,
        fa_voice: str = "fa-IR-DilaraNeural",
        en_voice: str = "en-US-JennyNeural",
        timeout: float = 10.0,
    ) -> None:
        if not endpoint or not key:
            raise TtsError("AZURE_SPEECH_ENDPOINT and AZURE_SPEECH_KEY must be set.")
        self.url = endpoint.rstrip("/") + "/tts/cognitiveservices/v1"
        self.key = key
        self.fa_voice = fa_voice
        self.en_voice = en_voice
        self.timeout = timeout

    def synthesize(self, text: str) -> AudioChunk:
        cleaned = (text or "").strip()
        if not cleaned:
            raise TtsError("Nothing to synthesize (empty text).")

        voice = pick_voice(cleaned, self.fa_voice, self.en_voice)
        req = urllib.request.Request(
            self.url,
            data=build_ssml(cleaned, voice).encode("utf-8"),
            method="POST",
            headers={
                "Ocp-Apim-Subscription-Key": self.key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
                "User-Agent": "robot-server",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                pcm = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise TtsError(f"Azure TTS failed (HTTP {exc.code}): {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise TtsError(f"Azure TTS request failed: {exc}") from exc

        if not pcm:
            raise TtsError("Azure TTS returned no audio.")
        return pcm16_to_chunk(pcm)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/robot/tts/synth.py src/robot/tts/voice.py src/robot/tts/azure.py tests/test_azure_tts.py
git commit -m "Add Azure synthesizer with one-voice-per-reply selection"
```

---

### Task 3: OpenAI speech-to-text

**Files:**
- Modify: `src/robot/pcm.py` (add `encode_wav`), `src/robot/pipeline.py` (`save_wav_mono` uses it)
- Create: `src/robot/stt/openai_stt.py`
- Test: `tests/test_openai_stt.py`

**Interfaces:**
- Produces: `robot.pcm.encode_wav(chunk: AudioChunk) -> bytes` (16-bit mono WAV at the chunk's rate).
- Produces: `robot.stt.openai_stt.DEFAULT_STT_MODEL = "gpt-transcribe"`; `OpenAISTT(client=None, model: str | None = None, timeout: float = 10.0)` implementing `SpeechToText.transcribe(chunk) -> Transcript`; raises `SttError`.

- [ ] **Step 1: Write the failing tests**

`tests/test_openai_stt.py`:
```python
"""OpenAI transcription (fake client, no network)."""

from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from robot.device.base import AudioChunk
from robot.pcm import encode_wav
from robot.stt.base import SttError
from robot.stt.openai_stt import OpenAISTT


def _speech() -> AudioChunk:
    return AudioChunk(samples=np.full(16_000, 0.2, dtype=np.float32), sample_rate=16_000)


def _client(create) -> SimpleNamespace:
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))


def test_encode_wav_produces_16bit_mono_wav() -> None:
    chunk = AudioChunk(samples=np.array([0.0, 0.5], dtype=np.float32), sample_rate=16_000)
    with wave.open(io.BytesIO(encode_wav(chunk))) as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes())
    assert params == (1, 2, 16_000, 2)


def test_audio_is_uploaded_as_wav_without_a_language_pin() -> None:
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text="  سلام ربات  ")

    result = OpenAISTT(client=_client(create), model="gpt-transcribe").transcribe(_speech())

    assert result.text == "سلام ربات"
    (kwargs,) = calls
    assert kwargs["model"] == "gpt-transcribe"
    name, data, content_type = kwargs["file"]
    assert (name, content_type) == ("speech.wav", "audio/wav")
    assert data.startswith(b"RIFF")
    assert "language" not in kwargs and "languages" not in kwargs


def test_model_defaults_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_STT_MODEL", "gpt-4o-transcribe")
    calls: list[dict] = []
    stt = OpenAISTT(client=_client(lambda **kw: calls.append(kw) or SimpleNamespace(text="x")))
    stt.transcribe(_speech())
    assert calls[0]["model"] == "gpt-4o-transcribe"


def test_api_failure_becomes_stt_error() -> None:
    def create(**kwargs):
        raise RuntimeError("rate limited")

    with pytest.raises(SttError, match="rate limited"):
        OpenAISTT(client=_client(create)).transcribe(_speech())


def test_empty_audio_is_not_uploaded() -> None:
    calls: list[dict] = []
    stt = OpenAISTT(client=_client(lambda **kw: calls.append(kw)))
    empty = AudioChunk(samples=np.zeros(0, dtype=np.float32), sample_rate=16_000)
    assert stt.transcribe(empty).text == ""
    assert calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_stt.py -q`
Expected: collection error `ImportError: cannot import name 'encode_wav'`.

- [ ] **Step 3: Implement**

Append to `src/robot/pcm.py` (add `import io` and `import wave` to its imports):
```python
def encode_wav(chunk: AudioChunk) -> bytes:
    pcm = (np.clip(chunk.samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(chunk.sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
```

Replace `save_wav_mono` in `src/robot/pipeline.py`:
```python
def save_wav_mono(chunk: AudioChunk, path: str) -> None:
    """Write mono float audio as 16-bit PCM WAV."""
    from pathlib import Path

    from robot.pcm import encode_wav

    Path(path).write_bytes(encode_wav(chunk))
```

`src/robot/stt/openai_stt.py`:
```python
"""Cloud STT via OpenAI's transcription API (Persian and English, no language pin)."""

from __future__ import annotations

import os

from robot.device.base import AudioChunk
from robot.pcm import encode_wav
from robot.stt.base import SpeechToText, SttError, Transcript

DEFAULT_STT_MODEL = "gpt-transcribe"


class OpenAISTT(SpeechToText):
    def __init__(self, client=None, model: str | None = None, timeout: float = 10.0) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout, max_retries=1)
        self.client = client
        self.model = model or os.environ.get("ROBOT_STT_MODEL", DEFAULT_STT_MODEL)

    def transcribe(self, chunk: AudioChunk) -> Transcript:
        if chunk.is_empty:
            return Transcript(text="")
        try:
            result = self.client.audio.transcriptions.create(
                model=self.model,
                file=("speech.wav", encode_wav(chunk), "audio/wav"),
            )
        except Exception as exc:  # noqa: BLE001
            raise SttError(f"OpenAI transcription failed: {exc}") from exc
        return Transcript(text=(result.text or "").strip())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass, including the existing `test_save_wav_round_trips_through_loader`.

- [ ] **Step 5: Commit**

```bash
git add src/robot/pcm.py src/robot/pipeline.py src/robot/stt/openai_stt.py tests/test_openai_stt.py
git commit -m "Add OpenAI speech-to-text"
```

---

### Task 4: OpenAI agent with conversation history

**Files:**
- Modify: `src/robot/agent/base.py`, `src/robot/agent/__init__.py`
- Create: `src/robot/agent/openai_agent.py`
- Test: `tests/test_openai_agent.py`

**Interfaces:**
- Produces: `robot.agent.base.AgentError(RuntimeError)` (exported from `robot.agent`).
- Produces: `robot.agent.openai_agent.DEFAULT_CHAT_MODEL = "gpt-6-luna"`; `SYSTEM_PROMPT: str`; `OpenAIAgent(client=None, model: str | None = None, max_turns: int = 10, reasoning_effort: str | None = None, timeout: float = 10.0)` implementing `Agent.reply(user_text) -> AgentReply`; raises `AgentError`.

- [ ] **Step 1: Write the failing tests**

`tests/test_openai_agent.py`:
```python
"""OpenAI brain: prompt, history, errors (fake client, no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robot.agent import AgentError
from robot.agent.openai_agent import SYSTEM_PROMPT, OpenAIAgent


class _Responses:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.answers.pop(0))


def _agent(answers: list[str], **kw) -> tuple[OpenAIAgent, _Responses]:
    responses = _Responses(answers)
    return OpenAIAgent(client=SimpleNamespace(responses=responses), **kw), responses


def test_first_reply_sends_instructions_and_the_user_text() -> None:
    agent, responses = _agent(["سلام! خوبم."], model="gpt-6-luna", reasoning_effort="low")

    assert agent.reply("سلام، خوبی؟").text == "سلام! خوبم."

    (call,) = responses.calls
    assert call["model"] == "gpt-6-luna"
    assert call["instructions"] == SYSTEM_PROMPT
    assert call["input"] == [{"role": "user", "content": "سلام، خوبی؟"}]
    assert call["reasoning"] == {"effort": "low"}
    assert call["store"] is False


def test_later_replies_include_earlier_turns() -> None:
    agent, responses = _agent(["Hi Ali!", "You said your name is Ali."])
    agent.reply("I'm Ali")
    agent.reply("What's my name?")
    assert responses.calls[1]["input"] == [
        {"role": "user", "content": "I'm Ali"},
        {"role": "assistant", "content": "Hi Ali!"},
        {"role": "user", "content": "What's my name?"},
    ]


def test_history_keeps_only_the_last_max_turns() -> None:
    agent, responses = _agent(["a1", "a2", "a3", "a4"], max_turns=2)
    for text in ("u1", "u2", "u3", "u4"):
        agent.reply(text)
    assert [m["content"] for m in responses.calls[3]["input"]] == ["u2", "a2", "u3", "a3", "u4"]


def test_empty_reasoning_effort_omits_the_reasoning_option(monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_REASONING_EFFORT", "")
    agent, responses = _agent(["ok"])
    agent.reply("hi")
    assert "reasoning" not in responses.calls[0]


def test_api_failure_becomes_agent_error_and_is_not_remembered() -> None:
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("timeout")
        return SimpleNamespace(output_text="ok")

    agent = OpenAIAgent(client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    with pytest.raises(AgentError, match="timeout"):
        agent.reply("first")
    agent.reply("second")
    assert calls[1]["input"] == [{"role": "user", "content": "second"}]


def test_empty_user_text_is_rejected() -> None:
    agent, responses = _agent(["unused"])
    with pytest.raises(AgentError):
        agent.reply("  ")
    assert responses.calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_agent.py -q`
Expected: collection error `ImportError: cannot import name 'AgentError'`.

- [ ] **Step 3: Implement**

Append to `src/robot/agent/base.py`:
```python


class AgentError(RuntimeError):
    """Agent backend failure with a user-facing message."""
```

`src/robot/agent/__init__.py`:
```python
from robot.agent.base import Agent, AgentError, AgentReply
from robot.agent.stub import StubAgent

__all__ = ["Agent", "AgentError", "AgentReply", "StubAgent"]
```

`src/robot/agent/openai_agent.py`:
```python
"""The robot's brain: OpenAI Responses API with a short per-connection memory."""

from __future__ import annotations

import os

from robot.agent.base import Agent, AgentError, AgentReply

DEFAULT_CHAT_MODEL = "gpt-6-luna"

SYSTEM_PROMPT = (
    "You are Robot, a small friendly desk robot that talks with people out loud. "
    "Always answer in the same language the user spoke: Persian (Farsi) or English. "
    "When answering in Persian, use natural, conversational Iranian Persian. "
    "Keep replies short: one to three spoken sentences. "
    "Never use markdown, lists, emoji, code, or URLs; a voice reads your words aloud."
)


class OpenAIAgent(Agent):
    def __init__(
        self,
        client=None,
        model: str | None = None,
        max_turns: int = 10,
        reasoning_effort: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout, max_retries=1)
        self.client = client
        self.model = model or os.environ.get("ROBOT_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.max_turns = max_turns
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else os.environ.get("ROBOT_REASONING_EFFORT", "low")
        )
        self._history: list[dict[str, str]] = []

    def reply(self, user_text: str) -> AgentReply:
        text = (user_text or "").strip()
        if not text:
            raise AgentError("Empty user text.")

        messages = [*self._history, {"role": "user", "content": text}]
        kwargs: dict = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": messages,
            "store": False,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        try:
            response = self.client.responses.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise AgentError(f"OpenAI reply failed: {exc}") from exc

        answer = (response.output_text or "").strip()
        history = [*messages, {"role": "assistant", "content": answer}]
        self._history = history[-2 * self.max_turns :]
        return AgentReply(text=answer)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/robot/agent tests/test_openai_agent.py
git commit -m "Add OpenAI agent with short conversation history"
```

---

### Task 5: Turn session state machine

**Files:**
- Create: `src/robot/server/session.py`, `tests/fakes.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `UtteranceSegmenter(calibrate_sec=…, pause_sec=…, max_utterance_sec=…)` with `.feed(chunk) -> list[Segment]` (`Segment.kind`, `.audio`) and `.in_speech` (existing `robot.dictation`); `SpeechToText.transcribe`; `Agent.reply -> AgentReply`; `Synthesizer.synthesize`; protocol + PCM helpers from Task 1.
- Produces: `robot.server.session.TurnSession(send_text, send_bytes, *, stt, agent, synth, error_audio: AudioChunk | None = None, no_speech_sec=8.0, call_timeout=10.0, realtime_factor: float | None = 1.1, prebuffer_sec=0.5, segmenter_factory=make_segmenter)` where `send_text: Callable[[str], Awaitable[None]]`, `send_bytes: Callable[[bytes], Awaitable[None]]`. Methods: `async handle_text(raw: str)`, `async handle_audio(data: bytes)`, `async close()`, `async wait_for_turn()`; attribute `state: State`. Constant `NO_SPEECH = "no_speech"`. Function `make_segmenter() -> UtteranceSegmenter`.
- Produces (tests): `tests/fakes.py` with `SPEECH`, `QUIET`, `frames(seconds, amplitude) -> list[bytes]`, `utterance_frames(speech_sec=0.6, pause_sec=0.8) -> list[bytes]`, `FakeSTT(text="سلام", delay=0.0, error=None)` (`.calls: list[AudioChunk]`), `FakeAgent(reply="سلام! خوبم.")` (`.calls: list[str]`), `FakeSynth(seconds=0.1)` (`.calls: list[str]`).

- [ ] **Step 1: Write the shared fakes**

`tests/fakes.py`:
```python
"""Fakes shared by server/session/client tests (no network, no models)."""

from __future__ import annotations

import time

import numpy as np

from robot.agent.base import Agent, AgentReply
from robot.device.base import AudioChunk
from robot.pcm import SAMPLE_RATE, chunk_to_pcm16
from robot.server.protocol import FRAME_BYTES
from robot.stt.base import SpeechToText, Transcript
from robot.tts.synth import Synthesizer

SPEECH = 0.2
QUIET = 0.0005
FRAME_SEC = FRAME_BYTES / 2 / SAMPLE_RATE


def frames(seconds: float, amplitude: float) -> list[bytes]:
    frame = chunk_to_pcm16(
        AudioChunk(
            samples=np.full(FRAME_BYTES // 2, amplitude, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
        )
    )
    return [frame] * int(round(seconds / FRAME_SEC))


def utterance_frames(speech_sec: float = 0.6, pause_sec: float = 0.8) -> list[bytes]:
    return frames(speech_sec, SPEECH) + frames(pause_sec, QUIET)


class FakeSTT(SpeechToText):
    def __init__(self, text: str = "سلام", delay: float = 0.0, error: Exception | None = None) -> None:
        self.text = text
        self.delay = delay
        self.error = error
        self.calls: list[AudioChunk] = []

    def transcribe(self, chunk: AudioChunk) -> Transcript:
        self.calls.append(chunk)
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return Transcript(text=self.text)


class FakeAgent(Agent):
    def __init__(self, reply: str = "سلام! خوبم.") -> None:
        self.reply_text = reply
        self.calls: list[str] = []

    def reply(self, user_text: str) -> AgentReply:
        self.calls.append(user_text)
        return AgentReply(text=self.reply_text)


class FakeSynth(Synthesizer):
    def __init__(self, seconds: float = 0.1) -> None:
        self.seconds = seconds
        self.calls: list[str] = []

    def synthesize(self, text: str) -> AudioChunk:
        self.calls.append(text)
        n = int(self.seconds * SAMPLE_RATE)
        return AudioChunk(samples=np.full(n, 0.1, dtype=np.float32), sample_rate=SAMPLE_RATE)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_session.py`:
```python
"""One robot connection's turn state machine (no network)."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np

from fakes import QUIET, SPEECH, FakeAgent, FakeSTT, FakeSynth, frames, utterance_frames
from robot.device.base import AudioChunk
from robot.pcm import SAMPLE_RATE
from robot.server.protocol import State, cancel_message, hello_message, start_message
from robot.server.session import TurnSession
from robot.stt.base import SttError


class Robot:
    """Records what the server sends, the way the ESP32 would see it."""

    def __init__(self) -> None:
        self.texts: list[dict] = []
        self.audio = bytearray()
        self.log: list[str] = []

    async def send_text(self, raw: str) -> None:
        msg = json.loads(raw)
        self.texts.append(msg)
        self.log.append(f"state:{msg['state']}" if msg["type"] == "state" else msg["type"])

    async def send_bytes(self, data: bytes) -> None:
        self.audio.extend(data)
        if not self.log or self.log[-1] != "audio":
            self.log.append("audio")

    @property
    def states(self) -> list[str]:
        return [m["state"] for m in self.texts if m["type"] == "state"]


def _session(robot: Robot, stt=None, agent=None, synth=None, **kw) -> TurnSession:
    kw.setdefault("realtime_factor", None)
    return TurnSession(
        robot.send_text,
        robot.send_bytes,
        stt=stt or FakeSTT(),
        agent=agent or FakeAgent(),
        synth=synth or FakeSynth(),
        **kw,
    )


async def _speak(session: TurnSession, speech_sec: float = 0.6, pause_sec: float = 0.8) -> None:
    for frame in utterance_frames(speech_sec, pause_sec):
        await session.handle_audio(frame)


def test_full_turn_listens_thinks_speaks_and_returns_to_idle() -> None:
    robot, stt, agent, synth = Robot(), FakeSTT("سلام ربات"), FakeAgent("سلام! چطوری؟"), FakeSynth(0.1)

    async def scenario() -> None:
        s = _session(robot, stt, agent, synth)
        await s.handle_text(hello_message("0.1"))
        await s.handle_text(start_message())
        await _speak(s)
        await s.wait_for_turn()

    asyncio.run(scenario())

    assert robot.log == [
        "state:idle",
        "state:listening",
        "state:thinking",
        "state:speaking",
        "audio",
        "audio_end",
        "state:idle",
    ]
    assert len(robot.audio) == int(0.1 * SAMPLE_RATE) * 2
    assert len(stt.calls) == 1 and stt.calls[0].duration_sec >= 0.6
    assert agent.calls == ["سلام ربات"]
    assert synth.calls == ["سلام! چطوری؟"]


def test_silence_for_8_seconds_returns_to_idle_with_no_speech() -> None:
    robot, stt = Robot(), FakeSTT()

    async def scenario() -> None:
        s = _session(robot, stt)
        await s.handle_text(start_message())
        for frame in frames(8.2, QUIET):
            await s.handle_audio(frame)

    asyncio.run(scenario())

    assert robot.states == ["listening", "idle"]
    assert robot.texts[-1] == {"type": "state", "state": "idle", "reason": "no_speech"}
    assert stt.calls == []


def test_empty_transcript_returns_to_idle_without_asking_the_agent() -> None:
    robot, agent = Robot(), FakeAgent()

    async def scenario() -> None:
        s = _session(robot, FakeSTT(text="  "), agent)
        await s.handle_text(start_message())
        await _speak(s)
        await s.wait_for_turn()

    asyncio.run(scenario())

    assert robot.texts[-1] == {"type": "state", "state": "idle", "reason": "no_speech"}
    assert agent.calls == []


def test_empty_reply_skips_speech() -> None:
    robot, synth = Robot(), FakeSynth()

    async def scenario() -> None:
        s = _session(robot, agent=FakeAgent(reply=""), synth=synth)
        await s.handle_text(start_message())
        await _speak(s)
        await s.wait_for_turn()

    asyncio.run(scenario())

    assert robot.states == ["listening", "thinking", "idle"]
    assert synth.calls == [] and not robot.audio


def test_cancel_while_listening_returns_to_idle_and_ignores_later_audio() -> None:
    robot, stt = Robot(), FakeSTT()

    async def scenario() -> None:
        s = _session(robot, stt)
        await s.handle_text(start_message())
        for frame in frames(0.3, SPEECH):
            await s.handle_audio(frame)
        await s.handle_text(cancel_message())
        await _speak(s)

    asyncio.run(scenario())

    assert robot.states == ["listening", "idle"]
    assert stt.calls == []


def test_cancel_while_thinking_stops_the_turn() -> None:
    robot, agent = Robot(), FakeAgent()

    async def scenario() -> None:
        s = _session(robot, FakeSTT(delay=0.3), agent)
        await s.handle_text(start_message())
        await _speak(s)
        await asyncio.sleep(0.05)
        await s.handle_text(cancel_message())
        await asyncio.sleep(0.4)

    asyncio.run(scenario())

    assert robot.states == ["listening", "thinking", "idle"]
    assert agent.calls == []
    assert not robot.audio


def test_start_while_busy_is_ignored() -> None:
    robot = Robot()

    async def scenario() -> None:
        s = _session(robot, FakeSTT(delay=0.2))
        await s.handle_text(start_message())
        await _speak(s)
        await s.handle_text(start_message())
        await s.wait_for_turn()

    asyncio.run(scenario())

    assert robot.states == ["listening", "thinking", "speaking", "idle"]


def test_service_failure_plays_error_audio_then_returns_to_idle() -> None:
    robot = Robot()
    error_audio = AudioChunk(samples=np.full(1600, 0.1, dtype=np.float32), sample_rate=SAMPLE_RATE)

    async def scenario() -> None:
        s = _session(robot, FakeSTT(error=SttError("boom")), error_audio=error_audio)
        await s.handle_text(start_message())
        await _speak(s)
        await s.wait_for_turn()

    asyncio.run(scenario())

    assert robot.log == [
        "state:listening",
        "state:thinking",
        "state:error",
        "audio",
        "audio_end",
        "state:idle",
    ]
    assert len(robot.audio) == 1600 * 2


def test_service_failure_without_cached_message_shows_error_only() -> None:
    robot = Robot()

    async def scenario() -> None:
        s = _session(robot, FakeSTT(error=SttError("boom")))
        await s.handle_text(start_message())
        await _speak(s)
        await s.wait_for_turn()

    asyncio.run(scenario())

    assert robot.log == ["state:listening", "state:thinking", "state:error", "state:idle"]


def test_hung_service_times_out_into_the_error_path() -> None:
    robot = Robot()

    async def scenario() -> float:
        s = _session(robot, FakeSTT(delay=1.0), call_timeout=0.2)
        await s.handle_text(start_message())
        await _speak(s)
        started = time.monotonic()
        await s.wait_for_turn()
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())

    assert robot.states == ["listening", "thinking", "error", "idle"]
    assert elapsed < 0.9


def test_garbage_and_unknown_messages_are_ignored() -> None:
    robot = Robot()

    async def scenario() -> TurnSession:
        s = _session(robot)
        await s.handle_text("not json")
        await s.handle_text('{"type": "dance"}')
        await s.handle_audio(b"\x01")
        return s

    session = asyncio.run(scenario())

    assert robot.texts == []
    assert session.state is State.IDLE


def test_reply_audio_is_paced_slightly_faster_than_real_time() -> None:
    robot = Robot()

    async def scenario() -> float:
        s = _session(robot, synth=FakeSynth(seconds=1.5), realtime_factor=1.1, prebuffer_sec=0.5)
        await s.handle_text(start_message())
        await _speak(s)
        started = time.monotonic()
        await s.wait_for_turn()
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())

    # 0.5 s sent at once, the remaining 1.0 s at 1.1× real time ≈ 0.9 s.
    assert 0.75 <= elapsed <= 1.3
    assert len(robot.audio) == int(1.5 * SAMPLE_RATE) * 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_session.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'robot.server.session'`.

- [ ] **Step 4: Implement**

`src/robot/server/session.py`:
```python
"""One robot connection: button → listen → transcribe → reply → speak → idle."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from robot.agent.base import Agent
from robot.device.base import AudioChunk
from robot.dictation import UtteranceSegmenter
from robot.pcm import SAMPLE_RATE, chunk_to_pcm16, pcm16_to_chunk
from robot.server.protocol import (
    FRAME_BYTES,
    ProtocolError,
    State,
    audio_end_message,
    parse_message,
    state_message,
)
from robot.stt.base import SpeechToText
from robot.tts.synth import Synthesizer

log = logging.getLogger(__name__)

SendText = Callable[[str], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]

NO_SPEECH = "no_speech"


def make_segmenter() -> UtteranceSegmenter:
    # The button press is the cue to talk, so there's no quiet lead-in to calibrate on.
    return UtteranceSegmenter(calibrate_sec=0.0, pause_sec=0.7, max_utterance_sec=15.0)


class TurnSession:
    def __init__(
        self,
        send_text: SendText,
        send_bytes: SendBytes,
        *,
        stt: SpeechToText,
        agent: Agent,
        synth: Synthesizer,
        error_audio: AudioChunk | None = None,
        no_speech_sec: float = 8.0,
        call_timeout: float = 10.0,
        realtime_factor: float | None = 1.1,
        prebuffer_sec: float = 0.5,
        segmenter_factory: Callable[[], UtteranceSegmenter] = make_segmenter,
    ) -> None:
        self.send_text = send_text
        self.send_bytes = send_bytes
        self.stt = stt
        self.agent = agent
        self.synth = synth
        self.error_audio = error_audio
        self.no_speech_sec = no_speech_sec
        self.call_timeout = call_timeout
        self.realtime_factor = realtime_factor
        self.prebuffer_sec = prebuffer_sec
        self.segmenter_factory = segmenter_factory

        self.state = State.IDLE
        self._segmenter: UtteranceSegmenter | None = None
        self._listened_sec = 0.0
        self._turn: asyncio.Task | None = None

    async def handle_text(self, raw: str) -> None:
        try:
            msg = parse_message(raw)
        except ProtocolError as exc:
            log.warning("ignoring message: %s", exc)
            return

        kind = msg["type"]
        if kind == "hello":
            log.info("robot says hello (fw=%s)", msg.get("fw"))
            await self._set_state(self.state)
        elif kind == "start":
            await self._on_start()
        elif kind == "cancel":
            await self._on_cancel()
        else:
            log.warning("ignoring unknown message type %r", kind)

    async def handle_audio(self, data: bytes) -> None:
        if self.state is not State.LISTENING or self._segmenter is None:
            return
        chunk = pcm16_to_chunk(data)
        if chunk.is_empty:
            return

        self._listened_sec += chunk.duration_sec
        finals = [seg for seg in self._segmenter.feed(chunk) if seg.kind == "final"]
        if finals:
            self._segmenter = None
            await self._set_state(State.THINKING)
            self._turn = asyncio.create_task(self._run_turn(finals[0].audio))
        elif not self._segmenter.in_speech and self._listened_sec >= self.no_speech_sec:
            self._segmenter = None
            await self._set_state(State.IDLE, NO_SPEECH)

    async def close(self) -> None:
        await self._cancel_turn()

    async def wait_for_turn(self) -> None:
        if self._turn is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn

    async def _on_start(self) -> None:
        if self.state is not State.IDLE:
            log.info("ignoring start while %s", self.state.value)
            return
        self._segmenter = self.segmenter_factory()
        self._listened_sec = 0.0
        await self._set_state(State.LISTENING)

    async def _on_cancel(self) -> None:
        if self.state is State.IDLE:
            return
        await self._cancel_turn()
        self._segmenter = None
        await self._set_state(State.IDLE)

    async def _cancel_turn(self) -> None:
        turn, self._turn = self._turn, None
        if turn is not None and not turn.done():
            turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await turn

    async def _run_turn(self, audio: AudioChunk) -> None:
        try:
            transcript = await self._call(self.stt.transcribe, audio)
            heard = transcript.text.strip()
            if not heard:
                await self._set_state(State.IDLE, NO_SPEECH)
                return
            log.info("heard: %s", heard)

            reply = (await self._call(self.agent.reply, heard)).text.strip()
            if not reply:
                await self._set_state(State.IDLE)
                return
            log.info("reply: %s", reply)

            speech = await self._call(self.synth.synthesize, reply)
            await self._set_state(State.SPEAKING)
            await self._stream(speech)
            await self._set_state(State.IDLE)
        except Exception as exc:  # noqa: BLE001
            log.error("turn failed: %s", exc)
            # The robot may already be gone; a failed error report must not escape the task.
            with contextlib.suppress(Exception):
                await self._set_state(State.ERROR)
                if self.error_audio is not None:
                    await self._stream(self.error_audio)
                await self._set_state(State.IDLE)

    async def _call(self, fn, *args):
        return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=self.call_timeout)

    async def _stream(self, audio: AudioChunk) -> None:
        pcm = chunk_to_pcm16(audio)
        loop = asyncio.get_running_loop()
        frame_sec = FRAME_BYTES / 2 / SAMPLE_RATE
        prebuffer_frames = int(self.prebuffer_sec / frame_sec)
        started = loop.time()
        for i, offset in enumerate(range(0, len(pcm), FRAME_BYTES)):
            if self.realtime_factor and i >= prebuffer_frames:
                due = started + (i - prebuffer_frames) * frame_sec / self.realtime_factor
                delay = due - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            await self.send_bytes(pcm[offset : offset + FRAME_BYTES])
        await self.send_text(audio_end_message())

    async def _set_state(self, state: State, reason: str | None = None) -> None:
        self.state = state
        await self.send_text(state_message(state, reason))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/robot/server/session.py tests/fakes.py tests/test_session.py
git commit -m "Add per-connection turn session state machine"
```

---

### Task 6: WebSocket server with token check

**Files:**
- Create: `src/robot/server/app.py`
- Modify: `tests/fakes.py` (server helpers)
- Test: `tests/test_server_app.py`

**Interfaces:**
- Consumes: `TurnSession` (Task 5).
- Produces: `robot.server.app.ROBOT_PATH = "/robot"`; `robot_server(token: str, make_session: Callable[[ServerConnection], TurnSession], host: str = "0.0.0.0", port: int = 8765)` returning the `websockets` `serve(...)` async context manager (yields a `Server` with `.sockets`).
- Produces (tests): in `tests/fakes.py`: `TOKEN = "test-token"`, `AUTH = {"Authorization": f"Bearer {TOKEN}"}`, `session_factory(stt=None, agent=None, synth=None)`, `async with_server(make_session, body) -> Any` where `body(base_url: str)` is awaited with `base_url = "ws://127.0.0.1:<port>"`.

- [ ] **Step 1: Add server helpers to the fakes**

Append to `tests/fakes.py`:
```python


TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def session_factory(stt=None, agent=None, synth=None):
    from robot.server.session import TurnSession

    def make(connection):
        return TurnSession(
            connection.send,
            connection.send,
            stt=stt or FakeSTT(),
            agent=agent or FakeAgent(),
            synth=synth or FakeSynth(),
            realtime_factor=None,
        )

    return make


async def with_server(make_session, body):
    from robot.server.app import robot_server

    async with robot_server(TOKEN, make_session, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        return await body(f"ws://127.0.0.1:{port}")
```

- [ ] **Step 2: Write the failing tests**

`tests/test_server_app.py`:
```python
"""WebSocket server: token check, message routing, disconnects."""

from __future__ import annotations

import asyncio
import json

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from fakes import AUTH, FakeAgent, FakeSTT, session_factory, utterance_frames, with_server
from robot.server.protocol import hello_message, start_message


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_connections_without_the_right_token_are_rejected(headers: dict) -> None:
    async def body(base: str) -> int:
        with pytest.raises(InvalidStatus) as exc:
            async with connect(base + "/robot", additional_headers=headers):
                pass
        return exc.value.response.status_code

    assert asyncio.run(with_server(session_factory(), body)) == 401


def test_unknown_path_is_not_found() -> None:
    async def body(base: str) -> int:
        with pytest.raises(InvalidStatus) as exc:
            async with connect(base + "/other", additional_headers=AUTH):
                pass
        return exc.value.response.status_code

    assert asyncio.run(with_server(session_factory(), body)) == 404


def test_robot_with_token_gets_state_updates() -> None:
    async def body(base: str) -> tuple[dict, dict]:
        async with connect(base + "/robot", additional_headers=AUTH) as ws:
            await ws.send(hello_message("0.1"))
            first = json.loads(await ws.recv())
            await ws.send(start_message())
            second = json.loads(await ws.recv())
            return first, second

    first, second = asyncio.run(with_server(session_factory(), body))
    assert first == {"type": "state", "state": "idle"}
    assert second == {"type": "state", "state": "listening"}


def test_disconnect_mid_turn_cancels_the_turn() -> None:
    agent = FakeAgent()

    async def body(base: str) -> list[str]:
        async with connect(base + "/robot", additional_headers=AUTH) as ws:
            await ws.send(start_message())
            for frame in utterance_frames():
                await ws.send(frame)
            while json.loads(await ws.recv())["state"] != "thinking":
                pass
        await asyncio.sleep(0.6)
        return agent.calls

    calls = asyncio.run(with_server(session_factory(stt=FakeSTT(delay=0.3), agent=agent), body))
    assert calls == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_server_app.py -q`
Expected: FAIL / error `ModuleNotFoundError: No module named 'robot.server.app'`.

- [ ] **Step 4: Implement**

`src/robot/server/app.py`:
```python
"""WebSocket endpoint the robot (or `robot client`) connects to."""

from __future__ import annotations

import hmac
import http
import logging
from collections.abc import Callable

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from robot.server.session import TurnSession

log = logging.getLogger(__name__)

ROBOT_PATH = "/robot"

SessionFactory = Callable[[ServerConnection], TurnSession]


def _check_request(token: str):
    expected = f"Bearer {token}".encode()

    def process_request(connection, request):
        if request.path != ROBOT_PATH:
            return connection.respond(http.HTTPStatus.NOT_FOUND, "Not found\n")
        given = request.headers.get("Authorization", "").encode()
        if not hmac.compare_digest(given, expected):
            log.warning("rejected %s: bad or missing token", connection.remote_address)
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Invalid token\n")
        return None

    return process_request


async def _handle_robot(connection: ServerConnection, make_session: SessionFactory) -> None:
    session = make_session(connection)
    log.info("robot connected from %s", connection.remote_address)
    try:
        async for message in connection:
            if isinstance(message, bytes):
                await session.handle_audio(message)
            else:
                await session.handle_text(message)
    except ConnectionClosed:
        pass
    finally:
        await session.close()
        log.info("robot disconnected")


def robot_server(
    token: str,
    make_session: SessionFactory,
    host: str = "0.0.0.0",
    port: int = 8765,
):
    return serve(
        lambda connection: _handle_robot(connection, make_session),
        host,
        port,
        process_request=_check_request(token),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/robot/server/app.py tests/fakes.py tests/test_server_app.py
git commit -m "Add robot WebSocket server with device token check"
```

---

### Task 7: Laptop pretend-robot client

**Files:**
- Create: `src/robot/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: protocol (Task 1), server helpers in `tests/fakes.py` (Task 6).
- Produces: `robot.client.DEFAULT_SERVER = "ws://localhost:8765/robot"`; `auth_headers(token: str) -> dict[str, str]`; `WavTurnResult(states: list[str], reason: str | None, reply: AudioChunk)`; `async run_wav_turn(uri: str, token: str, audio: AudioChunk, *, trailing_silence_sec: float = 1.0, realtime: bool = False, timeout: float = 60.0) -> WavTurnResult`; `PlaybackBuffer` with `write(data: bytes)`, `read(n: int) -> bytes`, `clear()`; `async run_mic_client(uri: str, token: str) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_client.py`:
```python
"""Laptop pretend-robot against the real server (fake cloud services)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fakes import TOKEN, FakeAgent, FakeSynth, session_factory, with_server
from robot.client import PlaybackBuffer, run_wav_turn
from robot.device.base import AudioChunk
from robot.pcm import SAMPLE_RATE


def test_playback_buffer_pads_underruns_with_silence() -> None:
    buf = PlaybackBuffer()
    buf.write(b"\x01\x02")
    assert buf.read(4) == b"\x01\x02\x00\x00"
    assert buf.read(2) == b"\x00\x00"


def test_playback_buffer_returns_audio_in_order_and_clear_drops_it() -> None:
    buf = PlaybackBuffer()
    buf.write(b"\x01\x02")
    buf.write(b"\x03\x04")
    assert buf.read(2) == b"\x01\x02"
    buf.clear()
    assert buf.read(2) == b"\x00\x00"


def test_wav_turn_round_trip_through_the_real_server() -> None:
    agent, synth = FakeAgent("Hello!"), FakeSynth(seconds=0.2)
    speech = AudioChunk(samples=np.full(int(0.6 * SAMPLE_RATE), 0.2, dtype=np.float32), sample_rate=SAMPLE_RATE)

    async def body(base: str):
        return await run_wav_turn(base + "/robot", TOKEN, speech)

    result = asyncio.run(with_server(session_factory(agent=agent, synth=synth), body))

    assert result.states == ["idle", "listening", "thinking", "speaking", "idle"]
    assert result.reply.duration_sec == pytest.approx(0.2)
    assert agent.calls == ["سلام"]


def test_wav_turn_with_only_silence_reports_no_speech() -> None:
    silence = AudioChunk(samples=np.zeros(1600, dtype=np.float32), sample_rate=SAMPLE_RATE)

    async def body(base: str):
        return await run_wav_turn(base + "/robot", TOKEN, silence, trailing_silence_sec=8.5)

    result = asyncio.run(with_server(session_factory(), body))

    assert result.states[-1] == "idle"
    assert result.reason == "no_speech"
    assert result.reply.is_empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_client.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'robot.client'`.

- [ ] **Step 3: Implement**

`src/robot/client.py`:
```python
"""Laptop stand-in for the ESP32: talks to `robot serve` over the robot protocol."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
from dataclasses import dataclass

import numpy as np
from websockets.asyncio.client import connect

from robot.device.base import AudioChunk
from robot.pcm import SAMPLE_RATE, chunk_to_pcm16, pcm16_to_chunk
from robot.server.protocol import (
    FRAME_BYTES,
    cancel_message,
    hello_message,
    parse_message,
    start_message,
)

CLIENT_FW = "laptop-0.1"
DEFAULT_SERVER = "ws://localhost:8765/robot"


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@dataclass
class WavTurnResult:
    states: list[str]
    reason: str | None
    reply: AudioChunk


async def run_wav_turn(
    uri: str,
    token: str,
    audio: AudioChunk,
    *,
    trailing_silence_sec: float = 1.0,
    realtime: bool = False,
    timeout: float = 60.0,
) -> WavTurnResult:
    """Press the button, send a recording as the mic, collect the spoken reply."""
    silence = np.zeros(int(trailing_silence_sec * SAMPLE_RATE), dtype=np.float32)
    pcm = chunk_to_pcm16(
        AudioChunk(samples=np.concatenate([audio.samples, silence]), sample_rate=audio.sample_rate)
    )
    frames = [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    frame_sec = FRAME_BYTES / 2 / SAMPLE_RATE

    states: list[str] = []
    reason: str | None = None
    reply = bytearray()

    async with asyncio.timeout(timeout):
        async with connect(uri, additional_headers=auth_headers(token)) as ws:

            async def send_audio() -> None:
                for frame in frames:
                    await ws.send(frame)
                    if realtime:
                        await asyncio.sleep(frame_sec)

            sender: asyncio.Task | None = None
            await ws.send(hello_message(CLIENT_FW))
            await ws.send(start_message())
            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        reply.extend(message)
                        continue
                    msg = parse_message(message)
                    if msg["type"] != "state":
                        continue
                    states.append(msg["state"])
                    reason = msg.get("reason")
                    if msg["state"] == "listening" and sender is None:
                        sender = asyncio.create_task(send_audio())
                    elif msg["state"] == "idle" and sender is not None:
                        break
            finally:
                if sender is not None:
                    sender.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sender

    return WavTurnResult(states=states, reason=reason, reply=pcm16_to_chunk(bytes(reply)))


class PlaybackBuffer:
    """Reply audio waiting for the speaker; underruns play as silence."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()

    def write(self, data: bytes) -> None:
        with self._lock:
            self._buf.extend(data)

    def read(self, n: int) -> bytes:
        with self._lock:
            out = bytes(self._buf[:n])
            del self._buf[:n]
        return out + b"\x00" * (n - len(out))

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


async def run_mic_client(uri: str, token: str) -> None:
    """Interactive pretend-robot: Enter is the button, laptop mic and speakers are the robot's."""
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    mic_frames: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    button_presses: asyncio.Queue[str] = asyncio.Queue()
    playback = PlaybackBuffer()
    state = {"value": "offline"}

    def offer_frame(data: bytes) -> None:
        if mic_frames.full():
            mic_frames.get_nowait()
        mic_frames.put_nowait(data)

    def on_mic(indata, frames, time_info, status) -> None:  # noqa: ARG001
        loop.call_soon_threadsafe(offer_frame, bytes(indata))

    def on_speaker(outdata, frames, time_info, status) -> None:  # noqa: ARG001
        outdata[:] = playback.read(len(outdata))

    def read_stdin() -> None:
        for line in sys.stdin:
            loop.call_soon_threadsafe(button_presses.put_nowait, line)

    threading.Thread(target=read_stdin, daemon=True).start()

    async with connect(uri, additional_headers=auth_headers(token)) as ws:
        await ws.send(hello_message(CLIENT_FW))

        async def receive() -> None:
            async for message in ws:
                if isinstance(message, bytes):
                    playback.write(message)
                    continue
                msg = parse_message(message)
                if msg["type"] != "state":
                    continue
                state["value"] = msg["state"]
                if msg["state"] == "listening":
                    while not mic_frames.empty():
                        mic_frames.get_nowait()
                note = f" ({msg['reason']})" if msg.get("reason") else ""
                print(f"[{msg['state']}{note}]  Enter = button", flush=True)

        async def send_mic() -> None:
            while True:
                data = await mic_frames.get()
                if state["value"] == "listening":
                    await ws.send(data)

        async def button() -> None:
            while True:
                await button_presses.get()
                if state["value"] == "idle":
                    await ws.send(start_message())
                else:
                    playback.clear()
                    await ws.send(cancel_message())

        stream_opts = dict(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_BYTES // 2)
        with sd.RawInputStream(callback=on_mic, **stream_opts), sd.RawOutputStream(
            callback=on_speaker, **stream_opts
        ):
            tasks = [asyncio.create_task(job()) for job in (receive, send_mic, button)]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/robot/client.py tests/test_client.py
git commit -m "Add laptop pretend-robot client"
```

---

### Task 8: `serve` / `client` commands, config, error message cache, docs

**Files:**
- Create: `src/robot/server/run.py`, `.env.example`
- Modify: `src/robot/main.py`, `.gitignore`, `README.md`, `docs/superpowers/specs/2026-10-14-esp32-voice-robot-design.md`
- Test: `tests/test_serve_config.py`

**Interfaces:**
- Consumes: everything above; `FasterWhisperSTT`, `DEFAULT_MODEL` (existing); `load_wav_mono`, `save_wav_mono` (existing).
- Produces: `robot.server.run.ConfigError(RuntimeError)`; `ServeConfig` dataclass (`token, host, port, stt_backend, whisper_model, azure_endpoint, azure_key, fa_voice, en_voice`); `config_from_env(*, stt_backend: str = "openai", host: str = "0.0.0.0", port: int = 8765, whisper_model: str = DEFAULT_MODEL) -> ServeConfig`; `ERROR_TEXT`; `load_error_audio(synth: Synthesizer, path: Path) -> AudioChunk | None`; `async serve(config: ServeConfig) -> None`.
- Produces: CLI `robot serve [--host] [--port] [--stt openai|whisper] [--whisper-model]` and `robot client [--server URL] [--file WAV]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_serve_config.py`:
```python
"""Server configuration from the environment and the cached error message."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from fakes import FakeSynth
from robot.pipeline import save_wav_mono
from robot.device.base import AudioChunk
from robot.server.run import ERROR_TEXT, ConfigError, config_from_env, load_error_audio
from robot.tts.base import TtsError

REQUIRED = {
    "ROBOT_DEVICE_TOKEN": "tok",
    "OPENAI_API_KEY": "sk-test",
    "AZURE_SPEECH_KEY": "az",
    "AZURE_SPEECH_ENDPOINT": "https://robot.cognitiveservices.azure.com",
}


def _set_env(monkeypatch, **overrides) -> None:
    for name in [*REQUIRED, "ROBOT_VOICE_FA", "ROBOT_VOICE_EN"]:
        monkeypatch.delenv(name, raising=False)
    for name, value in {**REQUIRED, **overrides}.items():
        if value is not None:
            monkeypatch.setenv(name, value)


def test_config_reads_required_values_and_voice_defaults(monkeypatch) -> None:
    _set_env(monkeypatch)
    config = config_from_env()
    assert config.token == "tok"
    assert config.azure_endpoint == "https://robot.cognitiveservices.azure.com"
    assert (config.fa_voice, config.en_voice) == ("fa-IR-DilaraNeural", "en-US-JennyNeural")
    assert (config.host, config.port, config.stt_backend) == ("0.0.0.0", 8765, "openai")


def test_missing_settings_are_all_named_in_one_error(monkeypatch) -> None:
    _set_env(monkeypatch, OPENAI_API_KEY=None, AZURE_SPEECH_KEY=None)
    with pytest.raises(ConfigError) as exc:
        config_from_env()
    assert "OPENAI_API_KEY" in str(exc.value)
    assert "AZURE_SPEECH_KEY" in str(exc.value)
    assert "ROBOT_DEVICE_TOKEN" not in str(exc.value)


def test_cached_error_message_is_reused(tmp_path) -> None:
    path = tmp_path / "error_fa.wav"
    save_wav_mono(AudioChunk(samples=np.full(800, 0.25, dtype=np.float32), sample_rate=16_000), str(path))
    synth = FakeSynth()

    audio = load_error_audio(synth, path)

    assert audio is not None and audio.duration_sec == pytest.approx(0.05)
    assert synth.calls == []


def test_error_message_is_generated_once_and_cached(tmp_path) -> None:
    path = tmp_path / "cache" / "error_fa.wav"
    synth = FakeSynth(seconds=0.1)

    audio = load_error_audio(synth, path)

    assert synth.calls == [ERROR_TEXT]
    assert audio is not None and path.exists()


def test_error_message_is_skipped_if_azure_fails(tmp_path) -> None:
    class Broken(FakeSynth):
        def synthesize(self, text):
            raise TtsError("down")

    path = tmp_path / "error_fa.wav"
    assert load_error_audio(Broken(), path) is None
    assert not path.exists()


def test_serve_command_reports_missing_settings(monkeypatch, capsys) -> None:
    _set_env(monkeypatch, ROBOT_DEVICE_TOKEN=None)
    cli = importlib.import_module("robot.main")
    code = cli.main_args(["serve"])
    assert code == 1
    assert "ROBOT_DEVICE_TOKEN" in capsys.readouterr().err


def test_client_command_needs_the_device_token(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ROBOT_DEVICE_TOKEN", raising=False)
    cli = importlib.import_module("robot.main")
    assert cli.main_args(["client"]) == 1
    assert "ROBOT_DEVICE_TOKEN" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_serve_config.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'robot.server.run'`.

- [ ] **Step 3: Implement `run.py`**

`src/robot/server/run.py`:
```python
"""`robot serve`: wire real cloud services into the robot server."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from dataclasses import dataclass
from pathlib import Path

from robot.device.base import AudioChunk
from robot.pipeline import load_wav_mono, save_wav_mono
from robot.stt.whisper_local import DEFAULT_MODEL
from robot.tts.base import TtsError
from robot.tts.synth import Synthesizer

log = logging.getLogger(__name__)

ERROR_TEXT = "ببخشید، یه مشکلی پیش اومد."
ERROR_AUDIO_PATH = Path.home() / ".cache" / "robot" / "error_fa.wav"
REQUIRED_ENV = ("ROBOT_DEVICE_TOKEN", "OPENAI_API_KEY", "AZURE_SPEECH_KEY", "AZURE_SPEECH_ENDPOINT")


class ConfigError(RuntimeError):
    """Missing or invalid server settings."""


@dataclass(frozen=True)
class ServeConfig:
    token: str
    host: str
    port: int
    stt_backend: str
    whisper_model: str
    azure_endpoint: str
    azure_key: str
    fa_voice: str
    en_voice: str


def config_from_env(
    *,
    stt_backend: str = "openai",
    host: str = "0.0.0.0",
    port: int = 8765,
    whisper_model: str = DEFAULT_MODEL,
) -> ServeConfig:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]
    if missing:
        raise ConfigError(
            "Missing settings: " + ", ".join(missing) + ". Put them in .env (see .env.example) "
            "and run with `uv run --env-file .env robot serve`."
        )
    return ServeConfig(
        token=os.environ["ROBOT_DEVICE_TOKEN"].strip(),
        host=host,
        port=port,
        stt_backend=stt_backend,
        whisper_model=whisper_model,
        azure_endpoint=os.environ["AZURE_SPEECH_ENDPOINT"].strip(),
        azure_key=os.environ["AZURE_SPEECH_KEY"].strip(),
        fa_voice=os.environ.get("ROBOT_VOICE_FA", "fa-IR-DilaraNeural"),
        en_voice=os.environ.get("ROBOT_VOICE_EN", "en-US-JennyNeural"),
    )


def load_error_audio(synth: Synthesizer, path: Path = ERROR_AUDIO_PATH) -> AudioChunk | None:
    if path.exists():
        return load_wav_mono(str(path))
    try:
        audio = synth.synthesize(ERROR_TEXT)
    except TtsError as exc:
        log.warning("could not prepare the spoken error message: %s", exc)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    save_wav_mono(audio, str(path))
    return audio


def _robot_url(port: int) -> str:
    host = socket.gethostname()
    if not host.endswith(".local"):
        host += ".local"
    return f"ws://{host}:{port}/robot"


async def serve(config: ServeConfig) -> None:
    from robot.agent.openai_agent import OpenAIAgent
    from robot.server.app import robot_server
    from robot.server.session import TurnSession
    from robot.tts.azure import AzureSynthesizer

    synth = AzureSynthesizer(
        config.azure_endpoint, config.azure_key, fa_voice=config.fa_voice, en_voice=config.en_voice
    )
    if config.stt_backend == "whisper":
        from robot.stt.whisper_local import FasterWhisperSTT

        stt = FasterWhisperSTT(model_size=config.whisper_model, vad_filter=False)
        print(f"Loading local Farsi model {config.whisper_model}…", flush=True)
        await asyncio.to_thread(stt.warmup)
    else:
        from robot.stt.openai_stt import OpenAISTT

        stt = OpenAISTT()

    error_audio = await asyncio.to_thread(load_error_audio, synth)

    def make_session(connection) -> TurnSession:
        return TurnSession(
            connection.send,
            connection.send,
            stt=stt,
            agent=OpenAIAgent(),
            synth=synth,
            error_audio=error_audio,
        )

    async with robot_server(config.token, make_session, config.host, config.port):
        print(f"Robot server ready: {_robot_url(config.port)}  (Ctrl+C to stop)", flush=True)
        await asyncio.Future()
```

- [ ] **Step 4: Wire the CLI**

In `src/robot/main.py`:

Add imports at the top with the others:
```python
import asyncio
import logging
```

Add these functions before `_add_backend_flags`:
```python
def cmd_serve(args: argparse.Namespace) -> int:
    from robot.server.run import ConfigError, config_from_env, serve

    try:
        config = config_from_env(
            stt_backend=args.stt, host=args.host, port=args.port, whisper_model=args.whisper_model
        )
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(serve(config))
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except OSError as exc:
        print(f"Error: could not start the server on port {args.port}: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_client(args: argparse.Namespace) -> int:
    from websockets.exceptions import WebSocketException

    from robot.client import run_mic_client, run_wav_turn

    token = os.environ.get("ROBOT_DEVICE_TOKEN", "").strip()
    if not token:
        print(
            "Error: ROBOT_DEVICE_TOKEN is not set. Run with `uv run --env-file .env robot client`.",
            file=sys.stderr,
        )
        return 1

    try:
        if args.file:
            result = asyncio.run(run_wav_turn(args.server, token, load_wav_mono(args.file)))
            note = f" ({result.reason})" if result.reason else ""
            print("States: " + " → ".join(result.states) + note)
            if not result.reply.is_empty:
                LaptopSpeaker().play(result.reply)
        else:
            print("Press Enter to talk; press Enter again to cancel. Ctrl+C to quit.")
            asyncio.run(run_mic_client(args.server, token))
    except KeyboardInterrupt:
        print("\nStopped.")
    except DeviceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (OSError, WebSocketException) as exc:
        print(f"Error: could not talk to the robot server at {args.server}: {exc}", file=sys.stderr)
        return 1
    return 0
```

In `build_parser()`, before `return parser`, add:
```python
    p_serve = sub.add_parser("serve", help="Run the robot server (laptop) for the ESP32 / robot client")
    p_serve.add_argument("--host", default="0.0.0.0", help="Interface to listen on")
    p_serve.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    p_serve.add_argument(
        "--stt",
        choices=("openai", "whisper"),
        default="openai",
        help="Speech-to-text: OpenAI (default) or the local Farsi Whisper model",
    )
    p_serve.add_argument("--whisper-model", default=DEFAULT_MODEL, help="Model for --stt whisper")
    p_serve.set_defaults(func=cmd_serve)

    p_client = sub.add_parser("client", help="Pretend to be the robot using the laptop mic and speakers")
    p_client.add_argument("--server", default="ws://localhost:8765/robot", help="Robot server URL")
    p_client.add_argument("--file", default=None, help="Send this WAV as one turn instead of using the mic")
    p_client.set_defaults(func=cmd_client)
```

Replace `main` with a testable `main_args` plus `main`:
```python
def main_args(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(main_args(argv))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Config files**

Append to `.gitignore`:
```
.env
```

Create `.env.example`:
```bash
# Copy to .env (never commit .env) and fill in, then:
#   uv run --env-file .env robot serve
#   uv run --env-file .env robot client

# Any long random string; the robot firmware uses the same value.
ROBOT_DEVICE_TOKEN=
OPENAI_API_KEY=
AZURE_SPEECH_KEY=
# From the Azure portal → your Speech resource → Keys and Endpoint
AZURE_SPEECH_ENDPOINT=https://<your-resource-name>.cognitiveservices.azure.com

# Optional overrides
# ROBOT_CHAT_MODEL=gpt-6-luna
# ROBOT_REASONING_EFFORT=low
# ROBOT_STT_MODEL=gpt-transcribe
# ROBOT_VOICE_FA=fa-IR-DilaraNeural
# ROBOT_VOICE_EN=en-US-JennyNeural
```

- [ ] **Step 7: Docs**

In `README.md`, add this section directly before `## What shipped`:
````markdown
## Robot server (laptop) for the ESP32

The ESP32 robot streams its mic to the laptop over Wi-Fi; the laptop transcribes
with OpenAI, answers with an OpenAI model, and speaks with an Azure voice
(Persian voice if the reply has any Persian, otherwise English).

1. Get keys: an [OpenAI API key](https://platform.openai.com/api-keys), and an
   Azure **Speech** resource (free F0 tier) — copy its key and endpoint from
   *Keys and Endpoint*.
2. `cp .env.example .env` and fill it in. For the token, any long random string:
   `python3 -c "import secrets; print(secrets.token_urlsafe(24))"`.
3. Start the server:

```bash
uv run --env-file .env robot serve
```

4. Without the robot, pretend to be it from a second terminal (Enter = button):

```bash
uv run --env-file .env robot client
uv run --env-file .env robot client --file recordings/test.wav   # one turn from a recording
```

`robot serve --stt whisper` uses the local Farsi model instead of OpenAI for
speech-to-text (free, ~4 s slower per sentence).
````

In the spec `docs/superpowers/specs/2026-10-14-esp32-voice-robot-design.md`, update to match what was built:
- Decisions table, Speech-to-text row: `OpenAI \`gpt-4o-transcribe\`` → `OpenAI \`gpt-transcribe\` (OpenAI's current recommendation)`.
- Configuration paragraph: `AZURE_SPEECH_REGION` → `AZURE_SPEECH_ENDPOINT`; add `ROBOT_REASONING_EFFORT`; note chat model default `gpt-6-luna`.
- Errors table, Wrong token row: `Close the connection with a policy-violation code` → `Reject the WebSocket handshake with HTTP 401`.

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/robot/server/run.py src/robot/main.py tests/test_serve_config.py .gitignore .env.example README.md docs/superpowers/specs/2026-10-14-esp32-voice-robot-design.md
git commit -m "Add robot serve/client commands, env config and cached error message"
```

- [ ] **Step 10: Manual end-to-end check (needs real keys)**

With `.env` filled in:
1. Terminal 1: `uv run --env-file .env robot serve` → prints `Robot server ready: ws://<name>.local:8765/robot`. macOS may ask to allow incoming connections — allow.
2. Terminal 2: `uv run --env-file .env robot client --file recordings/test.wav` → `States: idle → listening → thinking → speaking → idle`, and a Persian answer plays.
3. Terminal 2: `uv run --env-file .env robot client`, press Enter, ask in English → English voice answers; press Enter, ask in Persian → Persian voice answers.
4. Stop the server mid-answer (Ctrl+C in terminal 1) → client prints a connection error and exits; restart the server.
5. Put a wrong `ROBOT_DEVICE_TOKEN` in the client's environment → client reports HTTP 401.

If step 2 reports an error state, the server log names the failing service (OpenAI or Azure); fix the key / endpoint / model name in `.env` and retry.
