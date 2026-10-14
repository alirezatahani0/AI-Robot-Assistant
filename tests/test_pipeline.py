"""Lightweight smoke tests (no mic / no model download)."""

from __future__ import annotations

import numpy as np

from robot.agent import StubAgent
from robot.device.base import AudioChunk, Microphone, Speaker
from robot.pipeline import VoicePipeline
from robot.stt import MockSTT
from robot.tts import MockTTS


class _FakeMic(Microphone):
    def list_devices(self) -> list[str]:
        return ["fake"]

    def record(self, duration_sec: float, sample_rate: int = 16_000) -> AudioChunk:
        n = int(duration_sec * sample_rate)
        return AudioChunk(samples=np.ones(n, dtype=np.float32) * 0.1, sample_rate=sample_rate)


class _FakeSpeaker(Speaker):
    def list_devices(self) -> list[str]:
        return ["fake"]

    def play(self, chunk: AudioChunk) -> None:
        return None


def test_text_turn_hello() -> None:
    tts = MockTTS()
    pipe = VoicePipeline(_FakeMic(), _FakeSpeaker(), MockSTT(), tts, StubAgent())
    result = pipe.run_text_turn("hello")
    assert not result.skipped
    assert "robot" in result.reply_text.lower()
    assert tts.spoken[-1] == result.reply_text


def test_empty_audio_skipped() -> None:
    pipe = VoicePipeline(
        _FakeMic(), _FakeSpeaker(), MockSTT(), MockTTS(), StubAgent()
    )
    empty = AudioChunk(samples=np.zeros(0, dtype=np.float32), sample_rate=16_000)
    result = pipe.run_audio_turn(empty)
    assert result.skipped
    assert result.reason


def test_audio_turn_uses_mock_stt() -> None:
    tts = MockTTS()
    pipe = VoicePipeline(
        _FakeMic(), _FakeSpeaker(), MockSTT("ping"), tts, StubAgent()
    )
    chunk = AudioChunk(samples=np.ones(1600, dtype=np.float32) * 0.2, sample_rate=16_000)
    result = pipe.run_audio_turn(chunk)
    assert result.user_text == "ping"
    assert "ping" in result.reply_text
