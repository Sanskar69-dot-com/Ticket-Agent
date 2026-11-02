from __future__ import annotations

import asyncio
import io
import re

from .contracts import STTService, TTSService


class WhisperSTTService(STTService):
    """Local Speech-to-Text adapter using faster-whisper.

    The model is loaded lazily and failures are reported with a clear message
    instead of crashing the API at startup or being turned into an empty
    transcript.
    """

    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self.model is not None:
            return
        async with self._lock:
            if self.model is not None:
                return
            loop = asyncio.get_running_loop()
            try:
                from faster_whisper import WhisperModel

                # Run model loading in an executor so the event loop stays free
                self.model = await loop.run_in_executor(
                    None,
                    lambda: WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Speech-to-text model could not be loaded: {exc}. "
                    "The first run downloads the Whisper model, so check your internet connection and try again."
                ) from exc

    @staticmethod
    def _decode_wav(audio_bytes: bytes):
        """Read a PCM WAV recording without PyAV. Returns 16 kHz mono float32."""
        import wave

        import numpy as np

        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        if width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif width == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported WAV sample width: {width}")

        if channels > 1:
            data = data[: len(data) // channels * channels].reshape(-1, channels).mean(axis=1)

        if rate != 16000 and data.size:
            if rate % 16000 == 0:
                factor = rate // 16000
                data = data[: data.size // factor * factor].reshape(-1, factor).mean(axis=1)
            else:
                n_out = int(round(data.size * 16000 / rate))
                data = np.interp(np.linspace(0, data.size - 1, n_out), np.arange(data.size), data)
        return data.astype(np.float32)

    @staticmethod
    def _prepare_audio(audio_bytes: bytes):
        """Decode a recording to 16 kHz mono float32 and normalise its volume."""
        import numpy as np

        if audio_bytes[:4] == b"RIFF":
            try:
                audio = WhisperSTTService._decode_wav(audio_bytes)
            except Exception:
                audio = None
        else:
            audio = None

        if audio is None:
            # Non-WAV input (webm/ogg/mp3...) needs PyAV: pip install -U av
            from faster_whisper.audio import decode_audio

            audio = decode_audio(io.BytesIO(audio_bytes), sampling_rate=16000)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return audio

        audio = audio - float(audio.mean())  # remove DC offset
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-4:
            # Quiet microphones are the usual reason Whisper "hears nothing".
            gain = min(0.9 / peak, 30.0)
            audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
        return audio

    def _run_model(self, audio, use_vad: bool) -> str:
        kwargs = dict(
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.9,   # default 0.6 drops real speech on noisy mics
            log_prob_threshold=-2.0,   # default -1.0 is too strict for short clips
            temperature=0.0,
        )
        if use_vad:
            kwargs["vad_filter"] = True
            kwargs["vad_parameters"] = dict(
                threshold=0.3,                # default 0.5
                min_speech_duration_ms=100,
                min_silence_duration_ms=700,
                speech_pad_ms=400,
            )
        else:
            kwargs["vad_filter"] = False

        segments, _info = self.model.transcribe(audio, **kwargs)
        parts = [segment.text.strip() for segment in segments]
        return " ".join(p for p in parts if p).strip()

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        if not audio_bytes:
            return ""

        if self.model is None:
            await self.initialize()

        loop = asyncio.get_running_loop()

        def _do_transcribe() -> str:
            audio = self._prepare_audio(audio_bytes)
            if audio.size < 1600:  # < 0.1 s of audio
                return ""
            text = self._run_model(audio, use_vad=True)
            if not text:
                # VAD may have cut everything out; retry on the full clip.
                text = self._run_model(audio, use_vad=False)
            return text

        try:
            return await loop.run_in_executor(None, _do_transcribe)
        except Exception as exc:
            # Do not hide real failures behind "no speech detected".
            raise RuntimeError(f"Could not read the recorded audio ({exc}). Please record again.") from exc

    async def cleanup(self) -> None:
        self.model = None


_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D]")


def clean_text_for_speech(text: str) -> str:
    """Remove markdown, links and emojis so the voice reads clean sentences."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*[-*\u2022]\s+", "", text, flags=re.MULTILINE)  # bullet markers
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"[*`#>~|]", "", text)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


class EdgeTTSService(TTSService):
    """Text-to-Speech adapter using Microsoft Edge TTS (natural neural voices)."""

    ATTEMPTS = 2
    TIMEOUT_SECONDS = 40

    def __init__(self, voice: str = "en-US-AriaNeural") -> None:
        self.voice = voice

    async def initialize(self) -> None:
        pass

    async def _collect_audio(self, text: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        if not text.strip():
            raise ValueError("Text input is empty")

        clean_text = clean_text_for_speech(text)
        if not clean_text:
            raise ValueError("Nothing readable to speak")

        last_error: Exception | None = None
        for _ in range(self.ATTEMPTS):
            try:
                audio_bytes = await asyncio.wait_for(self._collect_audio(clean_text), self.TIMEOUT_SECONDS)
                if audio_bytes:
                    return audio_bytes, "audio/mpeg"
                last_error = RuntimeError("Edge TTS returned zero audio bytes")
            except Exception as exc:  # network errors, timeouts, service errors
                last_error = exc

        raise RuntimeError(
            f"Edge TTS synthesis error: {last_error}. Edge TTS needs an internet connection."
        ) from last_error

    async def cleanup(self) -> None:
        pass


class FakeSTT(STTService):
    """Deterministic mock STT for unit tests."""

    def __init__(self, fixed_text: str = "My payment was charged twice", fail: bool = False) -> None:
        self.fixed_text = fixed_text
        self.fail = fail

    async def initialize(self) -> None:
        pass

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        if self.fail:
            raise RuntimeError("Fake STT internal error")
        if not audio_bytes:
            return ""
        return self.fixed_text

    async def cleanup(self) -> None:
        pass


class FakeTTS(TTSService):
    """Deterministic mock TTS for unit tests."""

    def __init__(self, fixed_audio: bytes = b"RIFFfake-wav-audio-payload", media_type: str = "audio/mpeg", fail: bool = False) -> None:
        self.fixed_audio = fixed_audio
        self.media_type = media_type
        self.fail = fail

    async def initialize(self) -> None:
        pass

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        if self.fail:
            raise RuntimeError("Fake TTS internal error")
        if not text.strip():
            raise ValueError("Text input is empty")
        return self.fixed_audio, self.media_type

    async def cleanup(self) -> None:
        pass