# Robot — phase 1 (STT + TTS)

Assistant-robot host for a future **ESP32 + LCD** build. Phase 1 runs on a **MacBook** (Linux works too).

Right now the useful path is **live Farsi dictation**: laptop mic → speech-to-text on screen so you can confirm the words. Agent + TTS are still stubbed and optional.

## Quick start (macOS / Linux)

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Check audio devices + backends
uv run robot status

# Live mic → Farsi text (no agent). Ctrl+C to stop and confirm.
uv run robot listen
```

Terminals can't lay out right-to-left Persian properly (words come out in the wrong order or letters don't join). To read the text correctly, add `--web` — it opens a local page that updates live as you talk:

```bash
uv run robot listen --web
```

First run downloads the Farsi model [`oi-uae/farsi-faster-whisper-large-v3`](https://huggingface.co/oi-uae/farsi-faster-whisper-large-v3) (~3 GB). Speak Persian and pause briefly (~0.7 s) at the end of a sentence; the sentence appears a few seconds later. Long sentences (3 s+) show a rough preview while you are still talking. Ctrl+C prints the final text for you to check.

Why not word-by-word? Whisper processes a fixed 30 s window per call, so each call costs ~4 s on an M4 Pro CPU no matter how short the clip. Live mode therefore transcribes each sentence once, on a background thread, so the mic never stalls. The model runs in `float32` (~6 GB RAM), which measured faster than `int8` on Apple Silicon; override with `ROBOT_STT_COMPUTE=int8` on machines with less memory.

To check accuracy on a fixed clip (no live chunking):

```bash
uv run robot record --seconds 8          # saves recordings/test.wav
uv run robot transcribe recordings/test.wav
```

`--language` defaults to `fa` (Iranian Farsi). Stock Whisper `tiny` / `small` / `medium` garble conversational Persian — on the same 8 s clip only the Farsi fine-tune got every word right (~7 s to transcribe on an M-series Mac).

### Optional agent / TTS loop (English-oriented stub)

```bash
# Text → stub agent → speak (no mic needed)
uv run robot once --text "hello"

# Interactive typed loop (headless / no mic)
uv run robot loop --text-loop

# Live mic turn then stub reply (needs mic permission on macOS)
uv run robot once --duration 4
```

Optional: transcribe a WAV file instead of the mic:

```bash
uv run robot once --file path/to/utterance.wav
```

### Headless / CI-friendly mocks

```bash
uv run robot once --text "hello" --tts mock
uv run robot once --file sample.wav --stt mock --tts mock
uv run robot listen --stt mock --mock-text "سلام"
```

## What shipped

| Package | Role |
|---------|------|
| `robot.device` | `Microphone` / `Speaker` interfaces; `Laptop*` now; `Esp32*` placeholders |
| `robot.stt` | Local **faster-whisper** STT (default **Farsi fine-tuned large-v3** + **`fa`**); `MockSTT` for demos |
| `robot.dictation` | Live rolling transcript, commit on pause |
| `robot.tts` | Local **macOS `say`** / **espeak-ng**; optional OpenAI TTS if `OPENAI_API_KEY` is set |
| `robot.agent` | `StubAgent` echo / canned replies — skipped by `robot listen` |
| `robot.pipeline` | Thin listen → STT → agent → TTS loop |
| `robot.main` | CLI: `status`, `listen`, `once`, `loop` |

## Architecture (laptop → ESP32 later)

```
┌─────────────┐     ┌─────────┐     ┌──────────────┐     ┌─────────┐
│ Microphone  │────▶│   STT   │────▶│ Agent (stub) │────▶│   TTS   │
│ (device I/O)│     │ Farsi   │     │  skip in     │     │ module  │
└─────────────┘     └────┬────┘     │  `listen`    │     └────┬────┘
      ▲                  │          └──────────────┘          │
      │                  ▼                                    ▼
      │            screen (confirm)                      Speaker
      │  LaptopMicrophone / LaptopSpeaker today
      │  Esp32Microphone / Esp32Speaker later
```

- Keep **device I/O behind adapters**. Firmware (C++/Arduino or MicroPython on ESP32) should implement the same capture/play contracts — or stream PCM/text to this host over serial/BLE/Wi‑Fi.
- Swap `StubAgent` for a real HTTP/gRPC client when the agent API exists.
- LCD eye animations are out of scope for phase 1.

## STT / TTS notes

**STT (local-first, Farsi):** [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Default language is **`fa`**. Default model is **`oi-uae/farsi-faster-whisper-large-v3`** because stock Whisper models garble Iranian Persian. Override with `--whisper-model medium` or `ROBOT_WHISPER_MODEL`. Override language with `--language auto` or `ROBOT_LANGUAGE`.

**TTS (local-first):** macOS `say` does **not** speak Persian well. `robot listen` skips TTS on purpose until a Farsi voice is wired in.

| Platform | Backend |
|----------|---------|
| macOS | `say` (built-in, not Farsi) |
| Linux | `espeak-ng` (`apt install espeak-ng` / `brew install espeak`) |

**Optional cloud TTS:** set `OPENAI_API_KEY` to use OpenAI speech as a fallback. Leave it unset for fully local operation.

Other knobs: `ROBOT_TTS_BACKEND`, `ROBOT_TTS_VOICE`, `ROBOT_OPENAI_TTS_VOICE`.

## Empty / error cases

- **No mic / permission denied** — clear message; use `--text`, `--text-loop`, or `--file`.
- **Silence** — turn is skipped with a “no speech detected” hint (`once` / `loop`); `listen` prints `(چیزی شنیده نشد)`.
- **Empty STT** — skipped with guidance to speak longer / check the mic.
- **No TTS binary** — install `espeak-ng` or use `--tts mock`.
- **Missing WAV** — file-not-found error from `--file`.

## ESP32-ready notes (not implemented yet)

- Prefer **I2S mic + I2S amp/DAC** on ESP32.
- Firmware options: **Arduino/C++** (ESP-IDF / Arduino-ESP32) or **MicroPython**.
- Phase-2 bridge ideas: stream 16 kHz mono PCM to this host, or run wake-word / tiny STT on-device and send text only.
- Keep the Python `Microphone` / `Speaker` interfaces stable so adapters can plug in without rewriting the pipeline.

## Project layout

```
src/robot/
  device/     # I/O adapters (laptop, esp32 stub)
  stt/        # speech-to-text (Farsi-pinned Whisper)
  dictation.py
  tts/        # text-to-speech
  agent/      # stub brain (API later)
  pipeline.py
  main.py     # CLI
```
