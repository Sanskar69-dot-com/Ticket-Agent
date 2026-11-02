import sys
import types

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import app
from src.voice.adapters import EdgeTTSService, WhisperSTTService, clean_text_for_speech
from src.voice.pipeline import VoicePipeline
from src.voice.adapters import FakeSTT, FakeTTS


def test_clean_text_for_speech_removes_markdown_links_and_emojis() -> None:
    text = "✅ **Refund approved** for order_id 123!\n- See [the portal](https://example.com/x) now\n# Done 🎉"
    cleaned = clean_text_for_speech(text)
    assert "*" not in cleaned and "#" not in cleaned and "http" not in cleaned
    assert "✅" not in cleaned and "🎉" not in cleaned
    assert "the portal" in cleaned and "order id 123" in cleaned


@pytest.mark.asyncio
async def test_whisper_model_load_failure_gives_clear_error(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise OSError("no internet")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=boom))
    stt = WhisperSTTService()
    with pytest.raises(RuntimeError, match="could not be loaded"):
        await stt.transcribe(b"audio", "audio/wav")


@pytest.mark.asyncio
async def test_whisper_returns_joined_segments() -> None:
    class FakeModel:
        def transcribe(self, buf, **kwargs):
            seg = types.SimpleNamespace
            return [seg(text=" My payment "), seg(text="was charged twice ")], None

    stt = WhisperSTTService()
    stt.model = FakeModel()
    assert await stt.transcribe(b"audio", "audio/wav") == "My payment was charged twice"


@pytest.mark.asyncio
async def test_whisper_decode_error_is_not_reported_as_silence() -> None:
    class BrokenModel:
        def transcribe(self, buf, **kwargs):
            raise ValueError("invalid audio data")

    stt = WhisperSTTService()
    stt.model = BrokenModel()
    with pytest.raises(RuntimeError, match="Could not read the recorded audio"):
        await stt.transcribe(b"audio", "audio/wav")


@pytest.mark.asyncio
async def test_edge_tts_retries_once_then_succeeds(monkeypatch) -> None:
    tts = EdgeTTSService()
    calls = {"n": 0}

    async def flaky(text: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("temporary network error")
        return b"mp3-bytes"

    monkeypatch.setattr(tts, "_collect_audio", flaky)
    audio, media_type = await tts.synthesize("Your ticket has been created.")
    assert (audio, media_type, calls["n"]) == (b"mp3-bytes", "audio/mpeg", 2)


@pytest.mark.asyncio
async def test_edge_tts_failure_and_unreadable_text(monkeypatch) -> None:
    tts = EdgeTTSService()

    async def always_fail(text: str) -> bytes:
        raise ConnectionError("offline")

    monkeypatch.setattr(tts, "_collect_audio", always_fail)
    with pytest.raises(RuntimeError, match="Edge TTS synthesis error"):
        await tts.synthesize("Hello there")
    with pytest.raises(ValueError):
        await tts.synthesize("🎉 ** **")


@pytest.mark.asyncio
async def test_api_returns_real_error_text_for_voice_failures() -> None:
    original = app.state.voice_pipeline
    app.state.voice_pipeline = VoicePipeline(FakeSTT(fail=True), FakeTTS(fail=True))
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            stt = await client.post("/voice/transcribe", files={"file": ("r.wav", b"abc", "audio/wav")})
            tts = await client.post("/voice/synthesize", json={"message_id": "m1", "text": "hello"})
        assert stt.status_code == 500 and "Fake STT internal error" in stt.json()["error"]
        assert tts.status_code == 500 and "Fake TTS internal error" in tts.json()["error"]
    finally:
        app.state.voice_pipeline = original
