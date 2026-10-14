"""Local-first STT via faster-whisper (no API key).

Downloads a Whisper model on first use into the Hugging Face cache
(or set HF_HOME / XDG_CACHE_HOME). Default is a Farsi fine-tune of
large-v3 + language `fa`; stock `tiny`/`small`/`medium` garble Iranian Persian.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from robot.device.base import AudioChunk
from robot.stt.base import SpeechToText, SttError, Transcript

DEFAULT_MODEL = "oi-uae/farsi-faster-whisper-large-v3"

_FARSI_PROMPT = "این یک گفتگوی روزمره به زبان فارسی ایران است."

_LANG_ALIASES = {
    "fa": "fa",
    "farsi": "fa",
    "persian": "fa",
    "iranian": "fa",
    "fa-ir": "fa",
    "fa_ir": "fa",
}


def _normalize_language(value: str | None) -> str | None:
    raw = (value if value is not None else os.environ.get("ROBOT_LANGUAGE", "fa")).strip()
    if not raw or raw.lower() in {"auto", "none"}:
        return None
    key = raw.lower().replace("_", "-")
    if key in _LANG_ALIASES:
        return _LANG_ALIASES[key]
    if key.startswith("fa-") or key.startswith("fa_"):
        return "fa"
    return key


def _default_threads() -> int:
    env = os.environ.get("ROBOT_STT_THREADS")
    if env:
        return int(env)
    # 8 threads measured ~1s/call faster than CTranslate2's default on an M-series Mac.
    return min(8, os.cpu_count() or 4)


@lru_cache(maxsize=2)
def _load_model(model_size: str, device: str, compute_type: str, cpu_threads: int = 0):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise SttError(
            "faster-whisper is not installed. Run: uv sync"
        ) from exc

    try:
        return WhisperModel(
            model_size, device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )
    except Exception as exc:  # noqa: BLE001
        raise SttError(
            f"Could not load Whisper model '{model_size}': {exc}. "
            "Check network for first-time download, or set ROBOT_WHISPER_MODEL."
        ) from exc


class FasterWhisperSTT(SpeechToText):
    def __init__(
        self,
        model_size: str | None = None,
        device: str = "cpu",
        compute_type: str | None = None,
        language: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 1,
    ) -> None:
        self.model_size = model_size or os.environ.get("ROBOT_WHISPER_MODEL", DEFAULT_MODEL)
        self.device = device
        # float32 uses Apple's Accelerate path: ~4s vs ~5.6s per call for int8 on M4 Pro (needs ~6 GB RAM).
        self.compute_type = compute_type or os.environ.get("ROBOT_STT_COMPUTE", "float32")
        self.language = _normalize_language(language)
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.cpu_threads = _default_threads()

    def warmup(self) -> None:
        _load_model(self.model_size, self.device, self.compute_type, self.cpu_threads)

    def transcribe(self, chunk: AudioChunk) -> Transcript:
        if chunk.is_empty:
            return Transcript(text="", language=self.language)

        # Whisper expects 16 kHz float32 mono.
        audio = chunk.samples.astype(np.float32, copy=False)
        if chunk.sample_rate != 16_000:
            audio = _resample_mono(audio, chunk.sample_rate, 16_000)

        kwargs: dict = {
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
        }
        if self.language:
            kwargs["language"] = self.language
        if self.language == "fa":
            kwargs["initial_prompt"] = _FARSI_PROMPT

        try:
            model = _load_model(self.model_size, self.device, self.compute_type, self.cpu_threads)
            segments, info = model.transcribe(audio, **kwargs)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except SttError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SttError(f"Transcription failed: {exc}") from exc

        language = getattr(info, "language", None) or self.language
        return Transcript(text=text, language=language)


def _resample_mono(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)
