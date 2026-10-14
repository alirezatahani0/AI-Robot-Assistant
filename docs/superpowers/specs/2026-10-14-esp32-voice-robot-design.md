# ESP32 voice robot — design

Date: 2026-10-14
Status: draft for review

## Goal

Make the robot work without the laptop's mic and speakers: a classic ESP32 with
its own mic, speaker and LCD eyes. You press a button, speak Persian or English,
and the robot answers out loud in the same language while its eyes show what it
is doing.

For now the laptop acts as the server on the home network. Later the same
server moves to a VPS, and the ESP32 is replaced by an ESP32-S3 with an on-chip
wake word, without changing the protocol.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Robot board | Classic ESP32 (DevKit) | What you have. Too little RAM for TLS audio to cloud AI APIs directly, so it acts as a thin audio terminal. |
| Starting a turn | Press the BOOT button once, then talk; server ends the turn on a ~0.7 s pause. Press again to cancel. | No wake word possible on classic ESP32. Same feel as a future wake word. |
| Server | This Python project, running on the laptop (`robot serve`) | No hosting cost now; VPS later. |
| Speech-to-text | OpenAI `gpt-4o-transcribe`, no language pinned | Good Persian and English. Local Farsi Whisper stays available via `--stt whisper`. |
| Brain | OpenAI chat model | Same provider, one API key. |
| Voice | Azure Speech neural voices (free tier); `edge-tts` fallback | Natural Persian + English; Azure is the official route, edge-tts is free but unofficial. |
| Mixed sentences | Split by script; Persian voice for Persian runs, English voice for English runs | Your choice. |
| Firmware | Arduino C++ via PlatformIO | Mature I2S, WebSocket and LCD libraries for ESP32. |

## Out of scope (this phase)

- Wake word (comes with the ESP32-S3).
- Interrupting the robot while it speaks (barge-in).
- Streaming the reply sentence by sentence (a later latency improvement).
- VPS deployment, TLS (`wss://`).
- Motors, sensors, battery power.
- Piper TTS: no longer needed with cloud voices; remove the `piper-tts` dependency added earlier.

## Architecture

```
 ROBOT (ESP32, Arduino)                         SERVER (laptop: `uv run robot serve`)
 ┌──────────────────────────┐   home Wi-Fi     ┌──────────────────────────────────────┐
 │ BOOT button ─► start     │   WebSocket      │ turn session (per connection)        │
 │ INMP441 mic ─► 16k PCM   │ ─ audio up ────► │   sentence end detection (reuse      │
 │                          │                  │   UtteranceSegmenter)                │
 │ MAX98357A ◄── play       │ ◄─ audio down ── │   ▼ OpenAI speech-to-text            │
 │ ST7789 LCD ◄── eyes      │ ◄─ state msgs ── │   ▼ OpenAI chat (short history)      │
 └──────────────────────────┘                  │   ▼ split Persian/English → Azure    │
                                               │     voices (edge-tts fallback)       │
                                               └──────────────────────────────────────┘
```

The laptop also runs `uv run robot client`: a pretend robot that uses the laptop
mic and speakers over exactly the same protocol, so the server can be built and
tested before the hardware arrives.

## Protocol

One WebSocket per robot: `ws://<laptop-name>.local:8765/robot`, with the header
`Authorization: Bearer <device token>`. The server rejects connections without
the right token.

Audio in both directions: 16 kHz, 16-bit little-endian, mono, raw PCM.

Robot → server

| Message | When |
|---|---|
| `{"type":"hello","fw":"0.1"}` | After connecting |
| `{"type":"start"}` | Button pressed while idle |
| `{"type":"cancel"}` | Button pressed while listening, thinking or speaking |
| binary frames (20 ms, 640 bytes) | Mic audio, only while listening |

Server → robot

| Message | Meaning |
|---|---|
| `{"type":"state","state":"idle"\|"listening"\|"thinking"\|"speaking"\|"error"}` | Drives the eyes; robot sends mic audio only in `listening` |
| `{"type":"state","state":"idle","reason":"no_speech"}` | Nothing understood; eyes show a "didn't catch that" look |
| binary frames | Reply audio |
| `{"type":"audio_end"}` | Reply audio finished |

The server sends reply audio at about 1.1× real time so the ESP32's ~1 s playback
buffer never overflows.

## Server

### One turn

1. `start` → state `listening`. Feed mic audio into `UtteranceSegmenter`.
   End on a 0.7 s pause after speech, or 15 s of speech. If no speech within 8 s,
   go back to `idle` with reason `no_speech`.
2. State `thinking`. Transcribe with OpenAI. Empty text → `idle` / `no_speech`.
3. Send the text to the chat model with a system prompt: answer in the language
   the user spoke, short spoken-style replies, no markdown. Keep the last 10 turns
   per connection.
4. State `speaking`. Split the reply into Persian and English runs (digits and
   punctuation join the neighbouring run). Synthesize each run with the matching
   Azure voice as 16 kHz raw PCM (edge-tts output decoded with PyAV if Azure
   fails). Stream audio to the robot, then `audio_end`.
5. State `idle`.

`cancel` at any point stops the turn, discards pending work, and returns to `idle`.

### Code layout (new modules)

| Module | Responsibility |
|---|---|
| `robot/server/protocol.py` | Encode/decode JSON messages; message constants |
| `robot/server/session.py` | Per-connection turn state machine; depends only on STT / agent / synthesizer interfaces, so it's testable with fakes |
| `robot/server/app.py` | WebSocket server (`websockets` asyncio), token check, listens on `0.0.0.0:8765` (macOS already advertises `<laptop-name>.local`) |
| `robot/stt/openai_stt.py` | OpenAI transcription behind the existing `SpeechToText` interface |
| `robot/agent/openai_agent.py` | Chat with short history behind the existing `Agent` interface |
| `robot/tts/synth.py` | New `Synthesizer` interface: `synthesize(text) -> AudioChunk` (16 kHz) |
| `robot/tts/azure.py`, `robot/tts/edge.py` | Azure and edge-tts synthesizers |
| `robot/tts/mixed.py` | Script splitting + per-language synthesizer, concatenated with short gaps |
| `robot/client.py` | Laptop pretend-robot (`robot client`) |

Configuration via environment variables: `OPENAI_API_KEY`, `AZURE_SPEECH_KEY`,
`AZURE_SPEECH_REGION`, `ROBOT_DEVICE_TOKEN`, plus `ROBOT_CHAT_MODEL`,
`ROBOT_STT_MODEL`, `ROBOT_VOICE_FA` (default `fa-IR-DilaraNeural`),
`ROBOT_VOICE_EN` (default `en-US-JennyNeural`). Model names are checked against
current OpenAI docs during implementation, not hard-coded from memory.

### Errors

| Failure | Behaviour |
|---|---|
| OpenAI / Azure error or timeout (10 s) | State `error`, play a short pre-generated Persian message («ببخشید، یه مشکلی پیش اومد»), then `idle` |
| Azure fails | Retry the run with edge-tts before treating it as an error |
| Robot disconnects mid-turn | Drop the session and its chat history; a reconnect starts fresh |
| Wrong token | Close the connection with a policy-violation code |

## Robot

### Wiring (classic ESP32 DevKit)

| Part | Pins |
|---|---|
| INMP441 mic (I2S port 0, RX) | BCLK 26, WS 25, SD 33, L/R → GND, VDD 3.3 V |
| MAX98357A amp (I2S port 1, TX) | BCLK 27, LRC 14, DIN 22, VIN 5 V; 4 Ω 3 W speaker |
| ST7789 240×240 LCD (VSPI) | SCK 18, MOSI 23, CS 5, DC 2, RST 4, BL 32 |
| Push-to-talk | BOOT button, GPIO 0 (on the board) |

Power from USB for now.

### Firmware (`firmware/`, PlatformIO, Arduino framework)

| Module | Responsibility |
|---|---|
| `secrets.h` (gitignored; `secrets.example.h` committed) | Wi-Fi SSID/password, server host, device token |
| `net` | Wi-Fi connect, mDNS lookup of the server, WebSocket client with reconnect backoff (1 s → 30 s) |
| `mic` | I2S RX at 16 kHz, 32-bit INMP441 samples → 16-bit, 20 ms frames, sent only in `listening` |
| `speaker` | I2S TX at 16 kHz; ~32 KB (1 s) ring buffer fed by binary frames |
| `eyes` | Simple shapes per state + blinking; partial redraws only (no full framebuffer) |
| `button` | Debounced GPIO 0; `start` when idle, `cancel` otherwise |
| `main` | State machine mirroring the server's `state` messages; `offline` look while disconnected |

Memory budget: roughly 100 KB of ~300 KB free heap.

## Testing

- Unit tests (pytest, no network): protocol encode/decode; session state machine
  with fake STT / agent / synthesizer (start → listening → thinking → speaking →
  idle, cancel at each stage, no-speech timeout, error path); script splitting
  for mixed Persian/English text; token rejection.
- Integration test: run the WebSocket server in-process with fakes and drive it
  with the real `robot client` protocol code, feeding a WAV file instead of the mic.
- Manual: `robot serve` + `robot client` on the laptop with real OpenAI / Azure keys.
- Firmware: bring-up order Wi-Fi/WebSocket → speaker (server sends a test tone)
  → mic (server echoes audio back) → eyes → button → full turn.

## Later phases

- ESP32-S3 with on-chip wake word sending the same `start` message. Note:
  Espressif's stock wake words are English/Chinese; a Persian wake word may need
  their custom-word service or another detector.
- Move the server to a VPS with Caddy for `wss://`.
- Stream the reply sentence by sentence to roughly halve the delay.
- Barge-in.
