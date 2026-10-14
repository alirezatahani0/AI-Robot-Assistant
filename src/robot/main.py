"""CLI entrypoint: laptop mic → STT (Farsi) → optional stub agent → TTS."""

from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser

from robot.agent import StubAgent
from robot.device import (
    DeviceError,
    LaptopMicrophone,
    LaptopSpeaker,
    describe_host,
)
from robot.dictation import LiveDictation
from robot.live_view import LiveView
from robot.pipeline import VoicePipeline, load_wav_mono, save_wav_mono
from robot.stt import FasterWhisperSTT, MockSTT, SttError
from robot.stt.whisper_local import DEFAULT_MODEL
from robot.tts import MockTTS, SystemTTS, TtsError


def build_pipeline(args: argparse.Namespace) -> VoicePipeline:
    mic = LaptopMicrophone()
    speaker = LaptopSpeaker()

    if args.stt == "mock":
        stt = MockSTT(canned=args.mock_text)
    else:
        stt = FasterWhisperSTT(model_size=args.whisper_model, language=args.language)

    if args.tts == "mock":
        tts = MockTTS()
    else:
        tts = SystemTTS(voice=args.voice)

    return VoicePipeline(
        microphone=mic,
        speaker=speaker,
        stt=stt,
        tts=tts,
        agent=StubAgent(),
    )


def cmd_status(args: argparse.Namespace) -> int:
    print(f"Host: {describe_host()}")
    mic = LaptopMicrophone()
    speaker = LaptopSpeaker()
    try:
        inputs = mic.list_devices()
        outputs = speaker.list_devices()
    except DeviceError as exc:
        print(f"Audio error: {exc}", file=sys.stderr)
        return 1

    print(f"Input devices ({len(inputs)}):")
    if inputs:
        for name in inputs:
            print(f"  - {name}")
    else:
        print("  (none — use --text or --file)")

    print(f"Output devices ({len(outputs)}):")
    if outputs:
        for name in outputs:
            print(f"  - {name}")
    else:
        print("  (none — SystemTTS may still use `say` / espeak-ng)")

    hint = speaker.speak_hint()
    print(f"Preferred local TTS: {hint or 'none detected'}")
    print(
        f"STT backend: {args.stt} (whisper model={args.whisper_model}, language={args.language})"
    )
    print("Agent: StubAgent (skipped by `robot listen`)")
    return 0


def cmd_once(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args)

    if args.text is not None:
        user = args.text.strip()
        if not user:
            print("Empty text; nothing to do.", file=sys.stderr)
            return 1
        print(f"You: {user}")
        result = pipeline.run_text_turn(user)
    elif args.file:
        try:
            chunk = load_wav_mono(args.file)
        except DeviceError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Loaded {args.file} ({chunk.duration_sec:.2f}s @ {chunk.sample_rate} Hz)")
        result = pipeline.run_audio_turn(chunk)
    else:
        print(f"Listening for {args.duration:g}s …")
        try:
            chunk = pipeline.listen_once(duration_sec=args.duration)
        except DeviceError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        result = pipeline.run_audio_turn(chunk)

    if result.skipped:
        print(f"Skipped: {result.reason}", file=sys.stderr)
        return 2

    if result.user_text:
        print(f"Heard: {result.user_text}")
    print(f"Robot: {result.reply_text}")
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args)
    print("Robot voice loop. Ctrl+C to quit.")
    print("Modes: speak after the beep prompt, or type when using --text-loop.\n")

    try:
        while True:
            if args.text_loop:
                try:
                    user = input("You> ").strip()
                except EOFError:
                    print()
                    break
                if not user:
                    print("(empty input — type something or Ctrl+C to quit)")
                    continue
                if user.lower() in {"quit", "exit", "q"}:
                    pipeline.speak("Goodbye!")
                    break
                result = pipeline.run_text_turn(user)
            else:
                print(f"Listening for {args.duration:g}s …")
                try:
                    chunk = pipeline.listen_once(duration_sec=args.duration)
                except DeviceError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    print("Tip: retry, check mic permissions, or pass --text-loop.")
                    continue
                result = pipeline.run_audio_turn(chunk)

            if result.skipped:
                print(f"Skipped: {result.reason}")
                continue
            if result.user_text:
                print(f"Heard: {result.user_text}")
            print(f"Robot: {result.reply_text}")
            if result.reply_text.lower().rstrip("!") == "goodbye":
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    except (SttError, TtsError) as exc:
        print(f"Fatal: {exc}", file=sys.stderr)
        return 1
    return 0


def _listen_status(snap) -> str:
    if snap.partial:
        return snap.partial
    if snap.listening:
        return "🎙 …"
    if snap.pending:
        return "⏳ …"
    return "…"


def cmd_listen(args: argparse.Namespace) -> int:
    """Mic → live Farsi text on screen. No agent, no TTS."""
    if args.stt == "mock":
        stt = MockSTT(canned=args.mock_text)
    else:
        print(
            f"Loading Farsi STT (model={args.whisper_model}, language={args.language})…",
            flush=True,
        )
        stt = FasterWhisperSTT(
            model_size=args.whisper_model,
            language=args.language,
            vad_filter=False,
        )
        try:
            stt.warmup()
        except SttError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    mic = LaptopMicrophone()
    session = LiveDictation(stt)

    view: LiveView | None = None
    if args.web:
        try:
            view = LiveView(port=args.port)
        except OSError as exc:
            print(f"Error: could not start live page on port {args.port}: {exc}", file=sys.stderr)
            return 1
        view.start()
        print(f"Live Persian text: {view.url}", flush=True)
        webbrowser.open(view.url)

    print("گوش می‌دهم — فارسی حرف بزن. برای پایان Ctrl+C را بزن.\n", flush=True)

    last_committed = ""
    heard_signal = False
    zero_sec = 0.0
    warned_muted = False
    try:
        for chunk in mic.iter_blocks(block_sec=0.1):
            if not heard_signal:
                if chunk.samples.size and float(abs(chunk.samples).max()) > 0.0:
                    heard_signal = True
                else:
                    zero_sec += chunk.duration_sec
                    # macOS feeds exact zeros when the host app lacks mic permission.
                    if zero_sec >= 2.0 and not warned_muted:
                        warned_muted = True
                        print(
                            "\nMicrophone is sending pure silence (all zeros). "
                            "macOS is blocking mic access for this terminal app.\n"
                            "Fix: System Settings → Privacy & Security → Microphone → "
                            "enable Cursor (or Terminal/iTerm), then fully quit and reopen it.",
                            file=sys.stderr,
                            flush=True,
                        )
            snap = session.feed(chunk)
            if snap.committed != last_committed:
                added = snap.committed
                if last_committed and snap.committed.startswith(last_committed):
                    added = snap.committed[len(last_committed) :].strip()
                if added:
                    sys.stdout.write("\033[2K\r" + added + "\n")
                    sys.stdout.flush()
                last_committed = snap.committed
            status = _listen_status(snap)
            sys.stdout.write("\033[2K\r" + status)
            sys.stdout.flush()
            if view:
                view.update(snap.committed, snap.partial or status)
    except DeviceError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    except SttError as exc:
        print(f"\nFatal: {exc}", file=sys.stderr)
        return 1

    if session.snapshot().pending or session.segmenter.in_speech:
        sys.stdout.write("\033[2K\rFinishing the last sentence…")
        sys.stdout.flush()
    final = session.finalize()
    sys.stdout.write("\n")
    print("---")
    print("متن نهایی (برای تایید):")
    print(final or "(چیزی شنیده نشد)")
    if view:
        view.finish(final)
        # Keep serving briefly so the page picks up the final text before exit.
        time.sleep(1.0)
        view.stop()
    return 0 if final else 2


def cmd_record(args: argparse.Namespace) -> int:
    """Mic → WAV file, so the same clip can be compared across STT models."""
    print(f"Recording {args.seconds:g}s — speak now…", flush=True)
    try:
        chunk = LaptopMicrophone().record(duration_sec=args.seconds)
    except DeviceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if chunk.is_empty:
        print(
            "Microphone returned silence. Check System Settings → Privacy & Security → "
            "Microphone for this terminal app.",
            file=sys.stderr,
        )
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_wav_mono(chunk, args.out)
    print(f"Saved {args.out}")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    """WAV file → text only. No agent, no TTS."""
    try:
        chunk = load_wav_mono(args.file)
    except DeviceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.stt == "mock":
        stt = MockSTT(canned=args.mock_text)
    else:
        stt = FasterWhisperSTT(model_size=args.whisper_model, language=args.language)

    started = time.monotonic()
    try:
        transcript = stt.transcribe(chunk)
    except SttError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started

    print(f"[model={args.whisper_model} language={args.language} {elapsed:.1f}s]")
    print(transcript.text or "(empty)")
    return 0 if not transcript.is_empty else 2


def _add_backend_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stt",
        choices=("whisper", "mock"),
        default="whisper",
        help="STT backend (default: local faster-whisper)",
    )
    parser.add_argument(
        "--whisper-model",
        default=os.environ.get("ROBOT_WHISPER_MODEL", DEFAULT_MODEL),
        help="faster-whisper model (default: Farsi fine-tuned large-v3; stock small/medium garble Persian). Env: ROBOT_WHISPER_MODEL",
    )
    parser.add_argument(
        "--language",
        default=os.environ.get("ROBOT_LANGUAGE", "fa"),
        help="STT language (default: fa). Use auto to let Whisper guess. Env: ROBOT_LANGUAGE",
    )
    parser.add_argument(
        "--tts",
        choices=("system", "mock"),
        default="system",
        help="TTS backend (default: macOS say / espeak-ng)",
    )
    parser.add_argument("--voice", default=None, help="TTS voice name (backend-specific)")
    parser.add_argument(
        "--mock-text",
        default="hello robot",
        help="Canned transcript when --stt mock",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot",
        description="Phase-1 robot host: mic → Farsi STT (listen) or STT → stub agent → TTS",
    )
    _add_backend_flags(parser)

    sub = parser.add_subparsers(dest="command", required=True)

    # Backend flags also on subcommands so `robot once --tts mock` works.
    p_status = sub.add_parser("status", help="Show host + audio devices")
    _add_backend_flags(p_status)
    p_status.set_defaults(func=cmd_status)

    p_once = sub.add_parser("once", help="Single listen→reply→speak turn")
    _add_backend_flags(p_once)
    p_once.add_argument("--duration", type=float, default=4.0, help="Mic seconds")
    p_once.add_argument("--text", type=str, default=None, help="Skip mic; use this text")
    p_once.add_argument("--file", type=str, default=None, help="Transcribe this WAV instead of mic")
    p_once.set_defaults(func=cmd_once)

    p_loop = sub.add_parser("loop", help="Repeated turns until quit")
    _add_backend_flags(p_loop)
    p_loop.add_argument("--duration", type=float, default=4.0, help="Mic seconds per turn")
    p_loop.add_argument(
        "--text-loop",
        action="store_true",
        help="Type text instead of using the mic (good for headless)",
    )
    p_loop.set_defaults(func=cmd_loop)

    p_listen = sub.add_parser(
        "listen",
        help="Live mic dictation in Farsi — print text as you speak, no agent",
    )
    _add_backend_flags(p_listen)
    p_listen.add_argument(
        "--web",
        action="store_true",
        help="Also show the live text in a browser page (terminals can't render Persian RTL)",
    )
    p_listen.add_argument("--port", type=int, default=8765, help="Port for --web (default 8765)")
    p_listen.set_defaults(func=cmd_listen)

    p_record = sub.add_parser("record", help="Record the mic to a WAV file")
    p_record.add_argument("--seconds", type=float, default=6.0, help="Seconds to record")
    p_record.add_argument("--out", default="recordings/test.wav", help="Output WAV path")
    p_record.set_defaults(func=cmd_record)

    p_transcribe = sub.add_parser("transcribe", help="Transcribe a WAV file to text (no agent)")
    _add_backend_flags(p_transcribe)
    p_transcribe.add_argument("file", help="WAV file to transcribe")
    p_transcribe.set_defaults(func=cmd_transcribe)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
