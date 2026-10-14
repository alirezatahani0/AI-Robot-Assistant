"""Local TTS via OS tools — no API key.

Order of preference:
1. macOS `say`
2. `espeak-ng` / `espeak` (Linux / Homebrew)
3. Optional OpenAI TTS if OPENAI_API_KEY is set (documented fallback only)
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from robot.tts.base import TextToSpeech, TtsError


class SystemTTS(TextToSpeech):
    def __init__(self, voice: str | None = None) -> None:
        self.voice = voice or os.environ.get("ROBOT_TTS_VOICE")
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        forced = os.environ.get("ROBOT_TTS_BACKEND", "").strip().lower()
        if forced:
            return forced

        if platform.system() == "Darwin" and shutil.which("say"):
            return "say"
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if shutil.which("espeak"):
            return "espeak"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return "none"

    def speak(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            raise TtsError("Nothing to speak (empty text).")

        backend = self._backend
        if backend == "say":
            self._say(cleaned)
        elif backend in {"espeak-ng", "espeak"}:
            self._espeak(cleaned, backend)
        elif backend == "openai":
            self._openai(cleaned)
        else:
            raise TtsError(
                "No local TTS backend found. On macOS, `say` is built-in. "
                "On Linux, install espeak-ng (`brew install espeak` / "
                "`apt install espeak-ng`). Or set OPENAI_API_KEY for cloud TTS."
            )

    def _say(self, text: str) -> None:
        cmd = ["say"]
        if self.voice:
            cmd.extend(["-v", self.voice])
        cmd.append(text)
        self._run(cmd, "macOS say")

    def _espeak(self, text: str, binary: str) -> None:
        cmd = [binary]
        if self.voice:
            cmd.extend(["-v", self.voice])
        cmd.append(text)
        self._run(cmd, binary)

    def _openai(self, text: str) -> None:
        """Optional paid fallback — only used when OPENAI_API_KEY is set."""
        try:
            import urllib.request
            import json
        except ImportError as exc:  # pragma: no cover
            raise TtsError("urllib unavailable for OpenAI TTS") from exc

        api_key = os.environ["OPENAI_API_KEY"]
        body = json.dumps(
            {
                "model": os.environ.get("ROBOT_OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
                "voice": self.voice or os.environ.get("ROBOT_OPENAI_TTS_VOICE", "alloy"),
                "input": text,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                audio = resp.read()
        except Exception as exc:  # noqa: BLE001
            raise TtsError(
                f"OpenAI TTS failed: {exc}. Unset OPENAI_API_KEY to use local TTS."
            ) from exc

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio)
            path = Path(tmp.name)

        try:
            # Prefer afplay (macOS) then ffplay.
            if shutil.which("afplay"):
                self._run(["afplay", str(path)], "afplay")
            elif shutil.which("ffplay"):
                self._run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
                    "ffplay",
                )
            else:
                raise TtsError(
                    "OpenAI TTS returned audio but neither afplay nor ffplay "
                    "is available to play it."
                )
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _run(cmd: list[str], label: str) -> None:
        try:
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except OSError as exc:
            raise TtsError(f"{label} failed to start: {exc}") from exc
        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "").strip()
            raise TtsError(f"{label} failed (exit {completed.returncode}): {err}")
