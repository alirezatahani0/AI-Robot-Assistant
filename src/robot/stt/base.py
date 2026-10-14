"""Speech-to-text contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from robot.device.base import AudioChunk


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, chunk: AudioChunk) -> Transcript:
        """Convert audio to text. Raises SttError on hard failures."""


class SttError(RuntimeError):
    """STT backend failure with a user-facing message."""
