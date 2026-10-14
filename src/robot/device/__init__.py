from robot.device.base import AudioChunk, DeviceError, Microphone, Speaker
from robot.device.esp32 import Esp32Microphone, Esp32Speaker
from robot.device.laptop import LaptopMicrophone, LaptopSpeaker, describe_host

__all__ = [
    "AudioChunk",
    "DeviceError",
    "Esp32Microphone",
    "Esp32Speaker",
    "LaptopMicrophone",
    "LaptopSpeaker",
    "Microphone",
    "Speaker",
    "describe_host",
]
