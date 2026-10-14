"""Deterministic STT for demos / CI without a real model download."""

from __future__ import annotations

from robot.device.base import AudioChunk
from robot.stt.base import SpeechToText, Transcript


class MockSTT(SpeechToText):
    def __init__(self, canned: str = "hello robot") -> None:
        self.canned = canned

    def transcribe(self, chunk: AudioChunk) -> Transcript:
        if chunk.is_empty:
            return Transcript(text="")
        return Transcript(text=self.canned, language="en")
