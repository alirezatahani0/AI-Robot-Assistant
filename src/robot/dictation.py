"""Live mic → text (no agent).

Whisper costs roughly the same per call regardless of clip length (it always
encodes a 30 s window), so re-transcribing a rolling buffer every second can't
keep up with a large Farsi model on CPU. Instead:

- `UtteranceSegmenter` splits the mic stream into sentences at natural pauses,
  using a silence threshold that adapts to the room's noise level.
- `LiveDictation` transcribes each finished sentence once on a background
  thread, and previews the sentence in progress whenever the worker is idle.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from robot.device.base import AudioChunk
from robot.stt.base import SpeechToText, SttError


@dataclass(frozen=True)
class Segment:
    kind: str  # "partial" | "final"
    uid: int
    audio: AudioChunk


@dataclass(frozen=True)
class DictationSnapshot:
    committed: str
    partial: str
    pending: int = 0
    listening: bool = False

    @property
    def display(self) -> str:
        parts = [p for p in (self.committed.strip(), self.partial.strip()) if p]
        return " ".join(parts)


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class UtteranceSegmenter:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        calibrate_sec: float = 0.5,
        pause_sec: float = 0.7,
        min_speech_sec: float = 0.3,
        max_utterance_sec: float = 15.0,
        pre_roll_sec: float = 0.3,
        partial_every_sec: float = 3.0,
        min_threshold: float = 0.004,
        noise_ratio: float = 3.0,
    ) -> None:
        self.sample_rate = sample_rate
        self._calibrate_n = int(calibrate_sec * sample_rate)
        self._pause_n = int(pause_sec * sample_rate)
        self._min_speech_n = int(min_speech_sec * sample_rate)
        self._max_n = int(max_utterance_sec * sample_rate)
        self._pre_roll_n = int(pre_roll_sec * sample_rate)
        self._partial_n = int(partial_every_sec * sample_rate)
        self.min_threshold = min_threshold
        self.noise_ratio = noise_ratio

        self._calibrated_n = 0
        self._calib_rms: list[float] = []
        self._noise: float | None = None
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_len = 0

        self._uid = 0
        self._in_speech = False
        self._buf: list[np.ndarray] = []
        self._buf_n = 0
        self._speech_n = 0
        self._silence_n = 0
        self._since_partial_n = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def threshold(self) -> float:
        if self._noise is None:
            return self.min_threshold
        return max(self.min_threshold, self._noise * self.noise_ratio)

    def feed(self, chunk: AudioChunk) -> list[Segment]:
        block = chunk.samples.astype(np.float32, copy=False)
        n = block.size
        if n == 0:
            return []
        level = _rms(block)

        if self._calibrated_n < self._calibrate_n:
            self._calibrated_n += n
            self._calib_rms.append(level)
            if self._calibrated_n >= self._calibrate_n:
                self._noise = float(np.mean(self._calib_rms))
            self._push_pre_roll(block)
            return []

        voiced = level >= self.threshold

        if not self._in_speech:
            if not voiced:
                self._update_noise(level)
                self._push_pre_roll(block)
                return []
            self._start_utterance(block)
            return []

        self._buf.append(block)
        self._buf_n += n
        self._since_partial_n += n
        if voiced:
            self._speech_n += n
            self._silence_n = 0
        else:
            self._silence_n += n

        if self._silence_n >= self._pause_n or self._buf_n >= self._max_n:
            return self._end_utterance()
        if self._since_partial_n >= self._partial_n:
            self._since_partial_n = 0
            return [Segment("partial", self._uid, self._audio())]
        return []

    def flush(self) -> list[Segment]:
        if not self._in_speech:
            return []
        return self._end_utterance()

    def _start_utterance(self, block: np.ndarray) -> None:
        self._uid += 1
        self._in_speech = True
        self._buf = list(self._pre_roll) + [block]
        self._buf_n = self._pre_roll_len + block.size
        self._speech_n = block.size
        self._silence_n = 0
        self._since_partial_n = block.size
        self._pre_roll.clear()
        self._pre_roll_len = 0

    def _end_utterance(self) -> list[Segment]:
        out: list[Segment] = []
        if self._speech_n >= self._min_speech_n:
            out.append(Segment("final", self._uid, self._audio()))
        self._in_speech = False
        self._buf = []
        self._buf_n = 0
        self._speech_n = 0
        self._silence_n = 0
        self._since_partial_n = 0
        return out

    def _audio(self) -> AudioChunk:
        return AudioChunk(samples=np.concatenate(self._buf), sample_rate=self.sample_rate)

    def _push_pre_roll(self, block: np.ndarray) -> None:
        self._pre_roll.append(block)
        self._pre_roll_len += block.size
        while self._pre_roll and self._pre_roll_len - self._pre_roll[0].size >= self._pre_roll_n:
            self._pre_roll_len -= self._pre_roll.popleft().size

    def _update_noise(self, level: float) -> None:
        if self._noise is None:
            self._noise = level
        else:
            self._noise = 0.95 * self._noise + 0.05 * level


class LiveDictation:
    """Non-blocking dictation: `feed` returns immediately; STT runs on a worker thread."""

    def __init__(self, stt: SpeechToText, segmenter: UtteranceSegmenter | None = None) -> None:
        self.stt = stt
        self.segmenter = segmenter or UtteranceSegmenter()
        self.last_error: str | None = None

        self._cond = threading.Condition()
        self._finals: deque[Segment] = deque()
        self._partial: Segment | None = None
        self._busy = False
        self._stopping = False
        self._committed: list[str] = []
        self._last_final_uid = 0
        self._partial_text: tuple[int, str] | None = None

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def feed(self, chunk: AudioChunk) -> DictationSnapshot:
        segments = self.segmenter.feed(chunk)
        if segments:
            self._enqueue(segments)
        return self.snapshot()

    def snapshot(self) -> DictationSnapshot:
        with self._cond:
            partial = ""
            if self._partial_text and self._partial_text[0] > self._last_final_uid:
                partial = self._partial_text[1]
            pending = len(self._finals) + (1 if self._busy else 0)
            return DictationSnapshot(
                committed=" ".join(self._committed).strip(),
                partial=partial,
                pending=pending,
                listening=self.segmenter.in_speech,
            )

    def finalize(self, timeout: float | None = None) -> str:
        self._enqueue(self.segmenter.flush())
        with self._cond:
            self._stopping = True
            self._partial = None
            self._cond.notify_all()
        self._worker.join(timeout)
        with self._cond:
            return " ".join(self._committed).strip()

    def _enqueue(self, segments: list[Segment]) -> None:
        with self._cond:
            for seg in segments:
                if seg.kind == "final":
                    self._finals.append(seg)
                    if self._partial and self._partial.uid <= seg.uid:
                        self._partial = None
                elif not self._stopping:
                    self._partial = seg
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._finals and self._partial is None and not self._stopping:
                    self._cond.wait()
                if self._finals:
                    seg = self._finals.popleft()
                elif self._partial is not None:
                    seg = self._partial
                    self._partial = None
                else:
                    return
                self._busy = True

            text = self._transcribe(seg)

            with self._cond:
                self._busy = False
                if seg.kind == "final":
                    if text:
                        self._committed.append(text)
                    self._last_final_uid = max(self._last_final_uid, seg.uid)
                elif seg.uid > self._last_final_uid:
                    self._partial_text = (seg.uid, text)
                self._cond.notify_all()

    def _transcribe(self, seg: Segment) -> str:
        try:
            return self.stt.transcribe(seg.audio).text.strip()
        except SttError as exc:
            self.last_error = str(exc)
            return ""
