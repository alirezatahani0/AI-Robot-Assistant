"""ESP32 device adapters — placeholders until firmware lands.

Future directions (not implemented in phase 1):
- MicroPython or Arduino/C++ firmware on ESP32 capturing I2S mic frames
- Push PCM (or Opus) over USB serial / BLE / Wi‑Fi to this host, *or*
  run a tiny on-device STT later and only send text upstream
- Speaker: I2S DAC / amplifier driven from TTS audio streamed from the host
  (or on-device TTS if memory allows)
- Display: Adafruit ST7789 or TFT_eSPI on the pins in `robot.device.pins`

Keep the same Microphone / Speaker / Display contracts so the pipeline does not change.
"""

from __future__ import annotations

import numpy as np

from robot.device.base import AudioChunk, DeviceError, Display, Microphone, Speaker
from robot.device.pins import (
    DEFAULT_ST7789_PINS,
    ST7789_HEIGHT,
    ST7789_WIDTH,
    St7789Esp32Pins,
)


class Esp32Microphone(Microphone):
    def list_devices(self) -> list[str]:
        return []

    def record(self, duration_sec: float, sample_rate: int = 16_000) -> AudioChunk:
        raise DeviceError(
            "Esp32Microphone is not implemented yet. Use LaptopMicrophone "
            "for the MacBook stand-in, then plug in ESP32 I2S capture here."
        )


class Esp32Speaker(Speaker):
    def list_devices(self) -> list[str]:
        return []

    def play(self, chunk: AudioChunk) -> None:
        raise DeviceError(
            "Esp32Speaker is not implemented yet. Use LaptopSpeaker for now, "
            "then stream PCM to the ESP32 DAC/amp adapter."
        )


class Esp32Display(Display):
    """Stub for the physical ST7789. Documents the default SPI pin map.

    Firmware (Adafruit_ST7789 / TFT_eSPI) will own the real SPI transfers.
    This Python adapter exists so host code can target `Display` without
    branching, and so `robot status` / docs can print the wiring table.
    """

    def __init__(
        self,
        pins: St7789Esp32Pins | None = None,
        width: int = ST7789_WIDTH,
        height: int = ST7789_HEIGHT,
    ) -> None:
        self.pins = pins or DEFAULT_ST7789_PINS
        self._width = width
        self._height = height
        self._backlight = True

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _nyi(self, op: str) -> None:
        p = self.pins
        raise DeviceError(
            f"Esp32Display.{op} is not implemented in Python yet. "
            f"Wire ST7789 SPI: SCK={p.sck} (SCL), MOSI={p.mosi} (SDA), "
            f"CS={p.cs}, DC={p.dc}, RES={p.res}, BLK={p.blk}; "
            "drive it from firmware with Adafruit ST7789 or TFT_eSPI. "
            "Use LaptopDisplay (`uv run robot eyes`) on the Mac for now."
        )

    def clear(self, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        self._nyi("clear")

    def fill(self, color: tuple[int, int, int]) -> None:
        self._nyi("fill")

    def draw_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        self._nyi("draw_pixel")

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
        self._nyi("draw_rect")

    def blit_rgb(self, x: int, y: int, image: np.ndarray) -> None:
        self._nyi("blit_rgb")

    def blit_rgb565(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: bytes | bytearray | memoryview,
    ) -> None:
        self._nyi("blit_rgb565")

    def set_backlight(self, on: bool) -> None:
        # Document-only until firmware; remember intent for callers.
        self._backlight = bool(on)
        self._nyi("set_backlight")
