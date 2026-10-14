"""Device I/O contracts — swap Laptop* for Esp32* when hardware arrives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """Mono PCM float32 samples in [-1, 1] at `sample_rate` Hz."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError("AudioChunk.samples must be 1-D mono audio")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def duration_sec(self) -> float:
        return float(len(self.samples)) / float(self.sample_rate)

    @property
    def is_empty(self) -> bool:
        return len(self.samples) == 0


class Microphone(ABC):
    """Capture microphone audio for STT."""

    @abstractmethod
    def list_devices(self) -> list[str]:
        """Human-readable input device names (may be empty)."""

    @abstractmethod
    def record(self, duration_sec: float, sample_rate: int = 16_000) -> AudioChunk:
        """Record for `duration_sec` seconds. Raises DeviceError on failure."""


class Speaker(ABC):
    """Play audio or speak text through the device output path."""

    @abstractmethod
    def list_devices(self) -> list[str]:
        """Human-readable output device names (may be empty)."""

    @abstractmethod
    def play(self, chunk: AudioChunk) -> None:
        """Play PCM audio. Raises DeviceError on failure."""

    def speak_hint(self) -> Optional[str]:
        """Optional backend hint (e.g. 'say', 'espeak-ng') for TTS engines."""
        return None


class DeviceError(RuntimeError):
    """Microphone / speaker / host audio failure with a user-facing message."""
