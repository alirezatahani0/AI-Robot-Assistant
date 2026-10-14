"""MacBook / Linux host audio via PortAudio (sounddevice)."""

from __future__ import annotations

import platform
import queue
import shutil
import sys
from collections.abc import Iterator
from typing import Optional

import numpy as np
import sounddevice as sd

from robot.device.base import AudioChunk, DeviceError, Microphone, Speaker


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
