"""Farsi STT: language pinning and live dictation (no agent)."""

from __future__ import annotations

import numpy as np

from robot.device.base import AudioChunk
from robot.main import build_parser
from robot.stt import FasterWhisperSTT


def _speech(seconds: float, sample_rate: int = 16_000) -> AudioChunk:
    n = max(1, int(seconds * sample_rate))
    return AudioChunk(samples=np.ones(n, dtype=np.float32) * 0.2, sample_rate=sample_rate)


def _silence(seconds: float, sample_rate: int = 16_000) -> AudioChunk:
    n = max(1, int(seconds * sample_rate))
    return AudioChunk(samples=np.zeros(n, dtype=np.float32), sample_rate=sample_rate)


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeInfo:
    language = "fa"


class _FakeWhisper:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return iter([_FakeSegment("سلام خوبی")]), _FakeInfo()


def test_whisper_transcribe_pins_farsi_language(monkeypatch) -> None:
    fake = _FakeWhisper()
    monkeypatch.setattr("robot.stt.whisper_local._load_model", lambda *a, **k: fake)

    stt = FasterWhisperSTT()
    result = stt.transcribe(_speech(0.2))

    assert result.text == "سلام خوبی"
    assert result.language == "fa"
    assert fake.kwargs is not None
    assert fake.kwargs["language"] == "fa"


def test_farsi_aliases_pin_whisper_language(monkeypatch) -> None:
    fake = _FakeWhisper()
    monkeypatch.setattr("robot.stt.whisper_local._load_model", lambda *a, **k: fake)

    stt = FasterWhisperSTT(language="Farsi")
    stt.transcribe(_speech(0.2))

    assert fake.kwargs is not None
    assert fake.kwargs["language"] == "fa"


def test_listen_warns_when_mic_delivers_digital_silence(monkeypatch, capsys) -> None:
    import importlib

    cli = importlib.import_module("robot.main")

    class _MutedMic:
        def iter_blocks(self, sample_rate: int = 16_000, block_sec: float = 0.4):
            for _ in range(10):
                yield _silence(0.4)

    monkeypatch.setattr(cli, "LaptopMicrophone", _MutedMic)
    args = cli.build_parser().parse_args(["listen", "--stt", "mock"])
    cli.cmd_listen(args)

    err = capsys.readouterr().err
    assert "Microphone" in err
    assert "Privacy" in err


def test_listen_command_defaults_to_farsi_and_skips_agent() -> None:
    parser = build_parser()
    args = parser.parse_args(["listen"])
    assert args.command == "listen"
    assert args.language == "fa"
