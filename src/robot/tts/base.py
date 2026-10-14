"""Text-to-speech contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextToSpeech(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Speak `text` aloud. Raises TtsError on hard failures."""


class TtsError(RuntimeError):
    """TTS backend failure with a user-facing message."""
