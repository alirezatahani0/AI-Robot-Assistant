"""Record-to-file and file-only transcription (no agent) for Farsi accuracy checks."""

from __future__ import annotations

import importlib

import numpy as np

from robot.device.base import AudioChunk
from robot.pipeline import load_wav_mono, save_wav_mono


def test_save_wav_round_trips_through_loader(tmp_path) -> None:
    samples = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    path = tmp_path / "clip.wav"

    save_wav_mono(AudioChunk(samples=samples, sample_rate=16_000), str(path))
    loaded = load_wav_mono(str(path))

    assert loaded.sample_rate == 16_000
    np.testing.assert_allclose(loaded.samples, samples, atol=1e-4)


def test_transcribe_command_prints_text_without_agent_reply(tmp_path, capsys) -> None:
    cli = importlib.import_module("robot.main")
    path = tmp_path / "clip.wav"
    save_wav_mono(
        AudioChunk(samples=np.ones(8000, dtype=np.float32) * 0.2, sample_rate=16_000),
        str(path),
    )

    args = cli.build_parser().parse_args(
        ["transcribe", str(path), "--stt", "mock", "--mock-text", "سلام"]
    )
    code = args.func(args)

    out = capsys.readouterr().out
    assert code == 0
    assert "سلام" in out
    assert "Robot:" not in out
