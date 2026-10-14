"""Thin listen → STT → agent → TTS pipeline with clear extension points."""

from __future__ import annotations

from dataclasses import dataclass

from robot.agent.base import Agent
from robot.device.base import AudioChunk, DeviceError, Microphone, Speaker
from robot.stt.base import SpeechToText, SttError
from robot.tts.base import TextToSpeech, TtsError


@dataclass
class TurnResult:
    user_text: str
    reply_text: str
    skipped: bool = False
    reason: str | None = None


class VoicePipeline:
    def __init__(
        self,
        microphone: Microphone,
        speaker: Speaker,
        stt: SpeechToText,
        tts: TextToSpeech,
        agent: Agent,
    ) -> None:
        self.microphone = microphone
        self.speaker = speaker
        self.stt = stt
        self.tts = tts
        self.agent = agent

    def listen_once(self, duration_sec: float = 4.0, sample_rate: int = 16_000) -> AudioChunk:
        return self.microphone.record(duration_sec=duration_sec, sample_rate=sample_rate)

    def run_audio_turn(self, chunk: AudioChunk) -> TurnResult:
        if chunk.is_empty:
            return TurnResult(
                user_text="",
                reply_text="",
                skipped=True,
                reason="No speech detected (silence or empty audio). Try speaking louder or closer to the mic.",
            )

        try:
            transcript = self.stt.transcribe(chunk)
        except SttError as exc:
            return TurnResult(user_text="", reply_text="", skipped=True, reason=str(exc))

        if transcript.is_empty:
            return TurnResult(
                user_text="",
                reply_text="",
                skipped=True,
                reason="STT returned empty text. Try a longer utterance or check the mic.",
            )

        return self.run_text_turn(transcript.text)

    def run_text_turn(self, user_text: str) -> TurnResult:
        reply = self.agent.reply(user_text)
        try:
            self.tts.speak(reply.text)
        except TtsError as exc:
            return TurnResult(
                user_text=user_text,
                reply_text=reply.text,
                skipped=True,
                reason=f"TTS failed: {exc}",
            )
        return TurnResult(user_text=user_text, reply_text=reply.text)

    def speak(self, text: str) -> None:
        self.tts.speak(text)


def save_wav_mono(chunk: AudioChunk, path: str) -> None:
    """Write mono float audio as 16-bit PCM WAV."""
    import wave

    import numpy as np

    pcm = (np.clip(chunk.samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(chunk.sample_rate)
        wf.writeframes(pcm.tobytes())


def load_wav_mono(path: str, target_rate: int = 16_000) -> AudioChunk:
    """Load a WAV file for `--file` demos without a live mic."""
    import wave

    import numpy as np

    from robot.device.base import AudioChunk as AC

    try:
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except FileNotFoundError as exc:
        raise DeviceError(f"Audio file not found: {path}") from exc
    except wave.Error as exc:
        raise DeviceError(f"Invalid WAV file '{path}': {exc}") from exc

    if sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise DeviceError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if rate != target_rate and data.size:
        duration = data.size / float(rate)
        dst_len = max(1, int(round(duration * target_rate)))
        x_old = np.linspace(0.0, 1.0, num=data.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
        rate = target_rate

    return AC(samples=data.astype(np.float32), sample_rate=rate)
