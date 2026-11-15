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


class Display(ABC):
    """RGB framebuffer for the ST7789 (or a laptop window stand-in).

    Coordinates are pixels with origin at the top-left. Colors are RGB
    triples in 0–255 unless a method says RGB565.
    """

    @property
    @abstractmethod
    def width(self) -> int:
        """Panel width in pixels (typical ST7789 1.3\" = 240)."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Panel height in pixels (typical ST7789 1.3\" = 240)."""

    @abstractmethod
    def clear(self, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        """Fill the whole panel with `color`."""

    @abstractmethod
    def fill(self, color: tuple[int, int, int]) -> None:
        """Alias for clear — fill the whole panel."""

    @abstractmethod
    def draw_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        """Set one pixel. Out-of-bounds pixels are ignored."""

    @abstractmethod
    def draw_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color: tuple[int, int, int],
        *,
        filled: bool = True,
    ) -> None:
        """Draw an axis-aligned rectangle."""

    @abstractmethod
    def blit_rgb(
        self,
        x: int,
        y: int,
        image: np.ndarray,
    ) -> None:
        """Blit an HxWx3 uint8 RGB image with top-left at (x, y)."""

    @abstractmethod
    def blit_rgb565(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: bytes | bytearray | memoryview,
    ) -> None:
        """Blit packed little-endian RGB565 pixels (row-major)."""

    def draw_text(
        self,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Optional bitmap text. Default: no-op; laptop adapter draws via pygame."""

    @abstractmethod
    def set_backlight(self, on: bool) -> None:
        """Enable or disable the panel backlight (BLK pin on hardware)."""

    def show(self) -> None:
        """Flush pending draws to the visible surface (laptop window / SPI)."""

    def close(self) -> None:
        """Release window / SPI resources."""


class DeviceError(RuntimeError):
    """Microphone / speaker / display / host I/O failure with a user-facing message."""
