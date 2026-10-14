"""Print-only TTS for headless CI / no-speaker environments."""

from __future__ import annotations

from robot.tts.base import TextToSpeech, TtsError


class MockTTS(TextToSpeech):
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            raise TtsError("Nothing to speak (empty text).")
        self.spoken.append(cleaned)
        print(f"[mock-tts] {cleaned}")
