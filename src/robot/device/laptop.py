"""MacBook / Linux host audio via PortAudio (sounddevice) + pygame LCD stand-in."""

from __future__ import annotations

import platform
import queue
import shutil
import sys
from collections.abc import Iterator
from typing import Optional

import numpy as np
import sounddevice as sd

from robot.device.base import AudioChunk, DeviceError, Display, Microphone, Speaker
from robot.device.pins import ST7789_HEIGHT, ST7789_WIDTH


def _input_device_names() -> list[str]:
    names: list[str] = []
    try:
        for idx, info in enumerate(sd.query_devices()):
            if int(info.get("max_input_channels", 0)) > 0:
                names.append(f"{idx}: {info['name']}")
    except Exception as exc:  # noqa: BLE001 — surface host audio quirks
        raise DeviceError(f"Could not query audio input devices: {exc}") from exc
    return names


def _output_device_names() -> list[str]:
    names: list[str] = []
    try:
        for idx, info in enumerate(sd.query_devices()):
            if int(info.get("max_output_channels", 0)) > 0:
                names.append(f"{idx}: {info['name']}")
    except Exception as exc:  # noqa: BLE001
        raise DeviceError(f"Could not query audio output devices: {exc}") from exc
    return names


class LaptopMicrophone(Microphone):
    """Laptop / desktop mic stand-in for the future ESP32 mic."""

    def list_devices(self) -> list[str]:
        return _input_device_names()

    def record(self, duration_sec: float, sample_rate: int = 16_000) -> AudioChunk:
        if duration_sec <= 0:
            raise DeviceError("Record duration must be positive.")

        devices = self.list_devices()
        if not devices:
            raise DeviceError(
                "No microphone found. Plug in a mic, grant OS permission "
                "(System Settings → Privacy → Microphone on macOS), or run "
                "with --text / --file to skip live capture."
            )

        frames = int(duration_sec * sample_rate)
        try:
            # sounddevice returns float32 in [-1, 1]
            data = sd.rec(
                frames,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocking=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(
                f"Microphone recording failed: {exc}. "
                "Check permissions and that another app is not locking the device."
            ) from exc

        samples = np.asarray(data, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise DeviceError("Microphone returned empty audio.")

        # Treat near-silence as empty so the pipeline can prompt clearly.
        peak = float(np.max(np.abs(samples)))
        if peak < 1e-4:
            return AudioChunk(samples=np.zeros(0, dtype=np.float32), sample_rate=sample_rate)

        return AudioChunk(samples=samples, sample_rate=sample_rate)

    def iter_blocks(
        self,
        sample_rate: int = 16_000,
        block_sec: float = 0.4,
    ) -> Iterator[AudioChunk]:
        """Yield short mic blocks until the caller stops iterating."""
        devices = self.list_devices()
        if not devices:
            raise DeviceError(
                "No microphone found. Plug in a mic, grant OS permission "
                "(System Settings → Privacy → Microphone on macOS), or run "
                "with --stt mock to skip live capture."
            )

        blocksize = max(1, int(sample_rate * block_sec))
        pending: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(indata, frames, time, status) -> None:  # noqa: ARG001
            pending.put(np.array(indata, dtype=np.float32, copy=True))

        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=_callback,
            )
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(
                f"Microphone streaming failed: {exc}. "
                "Check permissions and that another app is not locking the device."
            ) from exc

        with stream:
            while True:
                data = pending.get()
                samples = np.asarray(data, dtype=np.float32).reshape(-1)
                yield AudioChunk(samples=samples, sample_rate=sample_rate)


class LaptopSpeaker(Speaker):
    """Laptop / desktop speakers stand-in for the future ESP32 amp/speaker."""

    def list_devices(self) -> list[str]:
        return _output_device_names()

    def play(self, chunk: AudioChunk) -> None:
        if chunk.is_empty:
            raise DeviceError("Nothing to play (empty audio).")

        devices = self.list_devices()
        if not devices:
            raise DeviceError(
                "No speaker/output device found. Connect speakers/headphones "
                "or use a TTS backend that does not need PortAudio playback "
                "(macOS `say` / Linux `espeak-ng`)."
            )

        try:
            sd.play(chunk.samples, samplerate=chunk.sample_rate, blocking=True)
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(f"Speaker playback failed: {exc}") from exc

    def speak_hint(self) -> Optional[str]:
        system = platform.system().lower()
        if system == "darwin" and shutil.which("say"):
            return "say"
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if shutil.which("espeak"):
            return "espeak"
        return None


def describe_host() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()}), Python {sys.version.split()[0]}"


def _rgb565_to_rgb(data: bytes | bytearray | memoryview, n: int) -> np.ndarray:
    """Unpack little-endian RGB565 → Nx3 uint8 RGB."""
    raw = np.frombuffer(memoryview(data)[: n * 2], dtype="<u2")
    r = ((raw >> 11) & 0x1F) * 255 // 31
    g = ((raw >> 5) & 0x3F) * 255 // 63
    b = (raw & 0x1F) * 255 // 31
    return np.stack([r, g, b], axis=1).astype(np.uint8)


class LaptopDisplay(Display):
    """pygame window sized like a 1.3\" ST7789 (240×240) until the ESP32 panel is wired."""

    def __init__(
        self,
        width: int = ST7789_WIDTH,
        height: int = ST7789_HEIGHT,
        *,
        scale: int = 2,
        title: str = "Robot eyes (ST7789 sim)",
        headless: bool = False,
    ) -> None:
        if width <= 0 or height <= 0:
            raise DeviceError("Display size must be positive.")
        if scale <= 0:
            raise DeviceError("Display scale must be positive.")

        self._width = width
        self._height = height
        self._scale = scale
        self._backlight = True
        self._headless = headless
        self._closed = False
        # Logical RGB framebuffer (matches the panel).
        self._fb = np.zeros((height, width, 3), dtype=np.uint8)

        self._pygame = None
        self._screen = None
        self._clock = None
        if not headless:
            try:
                import pygame
            except ImportError as exc:  # pragma: no cover
                raise DeviceError(
                    "pygame is required for the laptop LCD simulator. "
                    "Run `uv sync` then `uv run robot eyes`."
                ) from exc
            self._pygame = pygame
            pygame.init()
            pygame.display.set_caption(title)
            self._screen = pygame.display.set_mode((width * scale, height * scale))
            self._clock = pygame.time.Clock()
            self.show()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def clear(self, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        self._fb[:, :] = color

    def fill(self, color: tuple[int, int, int]) -> None:
        self.clear(color)

    def draw_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self._width and 0 <= y < self._height:
            self._fb[y, x] = color

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
        if w <= 0 or h <= 0:
            return
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self._width, x + w)
        y1 = min(self._height, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        if filled:
            self._fb[y0:y1, x0:x1] = color
            return
        # 1px outline
        self._fb[y0, x0:x1] = color
        self._fb[y1 - 1, x0:x1] = color
        self._fb[y0:y1, x0] = color
        self._fb[y0:y1, x1 - 1] = color

    def blit_rgb(self, x: int, y: int, image: np.ndarray) -> None:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise DeviceError("blit_rgb expects an HxWx3 array.")
        h, w, _ = arr.shape
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self._width, x + w)
        y1 = min(self._height, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        sx0 = x0 - x
        sy0 = y0 - y
        self._fb[y0:y1, x0:x1] = arr[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]

    def blit_rgb565(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: bytes | bytearray | memoryview,
    ) -> None:
        need = width * height
        if len(data) < need * 2:
            raise DeviceError(
                f"RGB565 blit needs {need * 2} bytes, got {len(data)}."
            )
        rgb = _rgb565_to_rgb(data, need).reshape(height, width, 3)
        self.blit_rgb(x, y, rgb)

    def draw_text(
        self,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        if self._pygame is None or not text:
            return
        font = self._pygame.font.SysFont("menlo", 14)
        surf = font.render(text, True, color)
        # Stamp into framebuffer via a temporary surface → array path.
        arr = self._pygame.surfarray.array3d(surf).transpose(1, 0, 2)
        self.blit_rgb(x, y, arr)

    def set_backlight(self, on: bool) -> None:
        self._backlight = bool(on)

    def show(self) -> None:
        if self._closed or self._pygame is None or self._screen is None:
            return
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.close()
                return
        visible = self._fb if self._backlight else np.zeros_like(self._fb)
        # pygame surfarray wants WxHx3
        surf = self._pygame.surfarray.make_surface(
            np.transpose(visible, (1, 0, 2))
        )
        if self._scale != 1:
            surf = self._pygame.transform.scale(
                surf, (self._width * self._scale, self._height * self._scale)
            )
        self._screen.blit(surf, (0, 0))
        self._pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(60)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pygame is not None:
            self._pygame.display.quit()
            self._pygame.quit()
        self._screen = None
