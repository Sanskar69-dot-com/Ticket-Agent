import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import load_settings
from src.models import ChatRequest, ChatResponse, Ticket
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError
from src.voice.adapters import EdgeTTSService, WhisperSTTService
from src.voice.models import SynthesisRequest, TranscriptionResponse
from src.voice.pipeline import VoicePipeline

settings = load_settings()
pipeline = SupportPipeline(settings, Path(__file__).resolve().parents[2] / "knowledge_base")
voice_pipeline = VoicePipeline(WhisperSTTService(), EdgeTTSService())
logger = logging.getLogger("support_agent.voice")


async def _warm_up_voice() -> None:
    """Load the Whisper model in the background.

    A failed or slow model download must never stop the chat API from starting;
    the model is loaded again on the first transcription request if needed.
    """
    try:
        await voice_pipeline.initialize()
    except Exception as exc:
        logger.warning("Voice warm-up failed (will retry on first use): %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup owns expensive shared initialization.
    await pipeline.initialize()
    warm_up = asyncio.create_task(_warm_up_voice())
    yield
    warm_up.cancel()
    pipeline.ready = False
    await voice_pipeline.cleanup()


app = FastAPI(title="Customer Support Ticket Agent", lifespan=lifespan)
app.state.voice_pipeline = voice_pipeline

# Allow CORS for all origins for seamless client/Streamlit access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    if not pipeline.ready:
        raise HTTPException(status_code=503, detail="Support pipeline is not ready")
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Keep this transport boundary thin: Pydantic validates the public request,
    # the pipeline owns orchestration, and known service errors are translated
    # to stable HTTP responses here.
    try:
        return await pipeline.process(request.session_id, request.message)
    except ComponentNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc


@app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: str) -> Ticket:
    ticket = pipeline.tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


# ==============================================================================
# MID-SESSION VOICE ENDPOINTS
# ==============================================================================
@app.post("/voice/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    request: Request,
    file: UploadFile | None = File(None),
):
    """Transcribe recorded audio file to text transcript."""
    active_pipeline: VoicePipeline = getattr(app.state, "voice_pipeline", voice_pipeline)

    audio_bytes = b""
    media_type = "audio/wav"

    if file is not None:
        audio_bytes = await file.read()
        media_type = file.content_type or "audio/wav"
    else:
        audio_bytes = await request.body()
        media_type = request.headers.get("content-type", "audio/wav")

    if not audio_bytes:
        return JSONResponse(status_code=400, content={"success": False, "error": "Audio input is empty"})

    try:
        transcript, elapsed_ms = await active_pipeline.transcribe(audio_bytes, media_type)
        return TranscriptionResponse(
            success=True,
            transcript=transcript,
            processing_time_ms=elapsed_ms,
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"success": False, "error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Transcription error: {exc}"})


@app.post("/voice/synthesize")
async def synthesize_speech(req: SynthesisRequest):
    """Synthesize text response to playable audio bytes."""
    active_pipeline: VoicePipeline = getattr(app.state, "voice_pipeline", voice_pipeline)

    if not req.text.strip():
        return JSONResponse(status_code=400, content={"success": False, "error": "Text input is empty"})

    try:
        audio_bytes, media_type, elapsed_ms = await active_pipeline.synthesize(req.text)
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-Processing-Time-Ms": str(elapsed_ms),
                "X-Message-ID": req.message_id,
            },
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"success": False, "error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Synthesis error: {exc}"})
