import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import app
from src.voice.adapters import FakeSTT, FakeTTS
from src.voice.pipeline import VoicePipeline


@pytest.mark.asyncio
async def test_fake_stt_successful_transcription() -> None:
    """Requirement: Successful transcription using a fake STT adapter."""
    fake_stt = FakeSTT(fixed_text="My payment was charged twice")
    fake_tts = FakeTTS()
    pipeline = VoicePipeline(fake_stt, fake_tts)

    transcript, timing = await pipeline.transcribe(b"test-audio-bytes", "audio/wav")
    assert transcript == "My payment was charged twice"
    assert timing >= 0


@pytest.mark.asyncio
async def test_empty_audio_validation() -> None:
    """Requirement: Empty-audio validation."""
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    with pytest.raises(ValueError, match="Audio input is empty"):
        await pipeline.transcribe(b"", "audio/wav")


@pytest.mark.asyncio
async def test_stt_failure_handling() -> None:
    """Requirement: STT failure handling."""
    failing_stt = FakeSTT(fail=True)
    pipeline = VoicePipeline(failing_stt, FakeTTS())
    with pytest.raises(RuntimeError, match="Fake STT internal error"):
        await pipeline.transcribe(b"valid-bytes", "audio/wav")


@pytest.mark.asyncio
async def test_fake_tts_successful_synthesis() -> None:
    """Requirement: Successful synthesis using a fake TTS adapter."""
    fake_tts = FakeTTS(fixed_audio=b"custom-mp3-bytes", media_type="audio/mpeg")
    pipeline = VoicePipeline(FakeSTT(), fake_tts)

    audio, media_type, timing = await pipeline.synthesize("Ticket CST-2026-0001 has been created.")
    assert audio == b"custom-mp3-bytes"
    assert media_type == "audio/mpeg"
    assert timing >= 0


@pytest.mark.asyncio
async def test_tts_failure_handling() -> None:
    """Requirement: TTS failure handling."""
    failing_tts = FakeTTS(fail=True)
    pipeline = VoicePipeline(FakeSTT(), failing_tts)
    with pytest.raises(RuntimeError, match="Fake TTS internal error"):
        await pipeline.synthesize("Some agent text")


@pytest.mark.asyncio
async def test_voice_pipeline_timing_and_media_type() -> None:
    """Requirement: Voice-pipeline timing and media type validation."""
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    _, timing_stt = await pipeline.transcribe(b"dummy", "audio/wav")
    audio, media_type, timing_tts = await pipeline.synthesize("Valid response text")

    assert isinstance(timing_stt, int)
    assert isinstance(timing_tts, int)
    assert media_type == "audio/mpeg"
    assert len(audio) > 0


@pytest.mark.asyncio
async def test_typed_chat_regression_after_voice_integration(monkeypatch) -> None:
    """Requirement: Typed-chat regression after voice integration."""
    from src.api.server import pipeline
    from src.models import ChatResponse
    pipeline.ready = True

    async def mock_process(session_id: str, message: str) -> ChatResponse:
        return ChatResponse(
            success=True,
            session_id=session_id,
            response="Standard delivery usually takes 3 to 5 business days.",
            sources=["shipping.md"],
            ticket_id=None,
        )

    monkeypatch.setattr(pipeline, "process", mock_process)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/chat",
            json={"session_id": "regression-sess-1", "message": "How long does standard delivery take?"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "shipping.md" in data["sources"]


@pytest.mark.asyncio
async def test_correct_association_between_response_and_playback_audio() -> None:
    """Requirement: Correct association between a response and its playback audio."""
    mock_pipeline = VoicePipeline(
        FakeSTT("Dictated test message"),
        FakeTTS(b"test-audio-for-msg-101", media_type="audio/mpeg"),
    )
    app.state.voice_pipeline = mock_pipeline

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test synthesis endpoint with specific message_id
        msg_id = "msg-101"
        res = await client.post(
            "/voice/synthesize",
            json={"message_id": msg_id, "text": "Here is your response audio"},
        )
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/mpeg"
        assert res.headers["x-message-id"] == msg_id
        assert res.content == b"test-audio-for-msg-101"


@pytest.mark.asyncio
async def test_transcribe_endpoint_validation_errors() -> None:
    """Test API /voice/transcribe empty audio validation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/voice/transcribe", content=b"")
        assert res.status_code == 400
        assert res.json()["success"] is False
