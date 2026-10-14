"""ESP32 device adapters — placeholders until firmware lands.

Future directions (not implemented in phase 1):
- MicroPython or Arduino/C++ firmware on ESP32 capturing I2S mic frames
- Push PCM (or Opus) over USB serial / BLE / Wi‑Fi to this host, *or*
  run a tiny on-device STT later and only send text upstream
- Speaker: I2S DAC / amplifier driven from TTS audio streamed from the host
  (or on-device TTS if memory allows)

Keep the same Microphone / Speaker contracts so the pipeline does not change.
"""

from __future__ import annotations

from robot.device.base import AudioChunk, DeviceError, Microphone, Speaker


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
