"""Live dictation: pause-based sentence splitting + background transcription."""

from __future__ import annotations

import threading
import time

import numpy as np

from robot.device.base import AudioChunk
from robot.dictation import LiveDictation, UtteranceSegmenter
from robot.stt import MockSTT
from robot.stt.base import SpeechToText, Transcript

RATE = 16_000
BLOCK = 0.1


def _blocks(seconds: float, amplitude: float) -> list[AudioChunk]:
    n_blocks = int(round(seconds / BLOCK))
    size = int(BLOCK * RATE)
    return [
        AudioChunk(samples=np.full(size, amplitude, dtype=np.float32), sample_rate=RATE)
        for _ in range(n_blocks)
    ]


def _feed(target, seconds: float, amplitude: float) -> list:
    out = []
    for block in _blocks(seconds, amplitude):
        result = target.feed(block)
        if isinstance(result, list):
            out.extend(result)
    return out


def _segmenter(**overrides) -> UtteranceSegmenter:
    opts = dict(
        sample_rate=RATE,
        calibrate_sec=0.3,
        pause_sec=0.5,
        min_speech_sec=0.3,
        max_utterance_sec=15.0,
        pre_roll_sec=0.2,
        partial_every_sec=0.5,
    )
    opts.update(overrides)
    return UtteranceSegmenter(**opts)


def test_pause_after_speech_emits_one_final_sentence() -> None:
    seg = _segmenter()
    events = _feed(seg, 0.5, 0.001)
    events += _feed(seg, 1.0, 0.2)
    events += _feed(seg, 0.6, 0.001)

    finals = [e for e in events if e.kind == "final"]
    assert len(finals) == 1
    # pre-roll (0.2) + speech (1.0) + trailing pause (0.5) ≈ 1.7s
    assert 1.5 <= finals[0].audio.duration_sec <= 1.9


def test_partial_previews_are_emitted_during_long_speech() -> None:
    seg = _segmenter(partial_every_sec=0.5)
    _feed(seg, 0.5, 0.001)
    events = _feed(seg, 1.6, 0.2)

    partials = [e for e in events if e.kind == "partial"]
    assert len(partials) >= 2
    assert all(e.kind != "final" for e in events)


def test_short_click_is_not_treated_as_a_sentence() -> None:
    seg = _segmenter(min_speech_sec=0.3)
    events = _feed(seg, 0.5, 0.001)
    events += _feed(seg, 0.1, 0.3)
    events += _feed(seg, 1.0, 0.001)

    assert [e for e in events if e.kind == "final"] == []


def test_nonstop_speech_is_cut_at_max_length() -> None:
    seg = _segmenter(max_utterance_sec=2.0)
    _feed(seg, 0.5, 0.001)
    events = _feed(seg, 3.0, 0.2)

    finals = [e for e in events if e.kind == "final"]
    assert len(finals) == 1
    assert finals[0].audio.duration_sec <= 2.05


def test_room_noise_level_is_not_mistaken_for_speech() -> None:
    seg = _segmenter()
    events = _feed(seg, 0.5, 0.01)  # noisy room, louder than a fixed 0.008 threshold
    events += _feed(seg, 1.0, 0.01)
    assert events == []

    events = _feed(seg, 1.0, 0.2)
    events += _feed(seg, 0.6, 0.01)
    assert len([e for e in events if e.kind == "final"]) == 1


def test_flush_returns_unfinished_sentence() -> None:
    seg = _segmenter()
    _feed(seg, 0.5, 0.001)
    _feed(seg, 0.8, 0.2)
    finals = [e for e in seg.flush() if e.kind == "final"]
    assert len(finals) == 1


class _SlowSTT(SpeechToText):
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0
        self._lock = threading.Lock()

    def transcribe(self, chunk: AudioChunk) -> Transcript:
        time.sleep(self.delay)
        with self._lock:
            self.calls += 1
            n = self.calls
        return Transcript(text=f"جمله {n}", language="fa")


def _dictation(stt: SpeechToText, **overrides) -> LiveDictation:
    return LiveDictation(stt, segmenter=_segmenter(**overrides))


def test_feeding_audio_never_waits_for_slow_transcription() -> None:
    session = _dictation(_SlowSTT(delay=0.5), partial_every_sec=0.2)
    _feed(session, 0.5, 0.001)

    started = time.monotonic()
    _feed(session, 1.0, 0.2)
    _feed(session, 0.6, 0.001)
    elapsed = time.monotonic() - started

    assert elapsed < 0.3
    session.finalize()


def test_finalize_waits_for_and_returns_committed_text() -> None:
    session = _dictation(MockSTT("سلام خوبی"))
    _feed(session, 0.5, 0.001)
    _feed(session, 1.0, 0.2)
    _feed(session, 0.6, 0.001)

    assert session.finalize() == "سلام خوبی"
    assert session.snapshot().committed == "سلام خوبی"


def test_sentences_are_committed_in_spoken_order() -> None:
    session = _dictation(_SlowSTT(delay=0.05), partial_every_sec=100.0)
    _feed(session, 0.5, 0.001)
    for _ in range(3):
        _feed(session, 0.6, 0.2)
        _feed(session, 0.6, 0.001)

    assert session.finalize() == "جمله 1 جمله 2 جمله 3"


def test_preview_is_shown_while_the_final_sentence_is_still_transcribing() -> None:
    stt = _SlowSTT(delay=0.4)
    session = _dictation(stt, partial_every_sec=0.3)
    _feed(session, 0.5, 0.001)
    _feed(session, 0.4, 0.2)  # preview queued after 0.3s of speech
    _feed(session, 0.6, 0.001)  # pause → final queued behind the preview

    saw_preview = False
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        snap = session.snapshot()
        if snap.committed:
            break
        if snap.partial == "جمله 1":
            saw_preview = True
        time.sleep(0.01)

    assert saw_preview
    assert session.finalize() == "جمله 2"


def test_committed_text_appears_while_still_listening() -> None:
    session = _dictation(MockSTT("الان فارسیه"))
    _feed(session, 0.5, 0.001)
    _feed(session, 1.0, 0.2)
    _feed(session, 0.6, 0.001)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not session.snapshot().committed:
        time.sleep(0.02)
    assert session.snapshot().committed == "الان فارسیه"
    session.finalize()
