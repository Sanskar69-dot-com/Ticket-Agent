from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import SupportWorkflowState, build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize shared components once during FastAPI startup."""
        await self.retriever.initialize()
        self.workflow = build_support_workflow(
            self.model,
            retriever=self.retriever,
            sessions=self.sessions,
            tickets=self.tickets,
        )
        self.ready = True

    async def process(self, session_id: str, message: str) -> ChatResponse:
        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError("Support pipeline is not ready")

        clean_session_id = session_id.strip() if session_id else ""
        clean_message = message.strip() if message else ""

        if not clean_session_id:
            raise AgentProcessingError("session_id must not be blank")
        if not clean_message:
            raise AgentProcessingError("message must not be blank")

        session = self.sessions.get_or_create(clean_session_id)
        session.history.append({"role": "user", "content": clean_message})

        initial_state: SupportWorkflowState = {
            "session_id": clean_session_id,
            "customer_message": clean_message,
            "messages": [HumanMessage(content=clean_message)],
        }

        config = {
            "configurable": {
                "retriever": self.retriever,
                "sessions": self.sessions,
                "tickets": self.tickets,
            }
        }

        try:
            result = await self.workflow.ainvoke(initial_state, config=config)
        except AgentProcessingError:
            raise
        except Exception as exc:
            raise AgentProcessingError(f"Error executing support workflow: {exc}") from exc

        response_text = (result.get("response_text") or "").strip()
        if not response_text:
            raise AgentProcessingError("Workflow produced an empty response")

        sources = result.get("sources") or []
        raw_ticket_id = result.get("ticket_id")

        # Validate that the ticket actually exists in repository
        verified_ticket_id = (
            raw_ticket_id if raw_ticket_id and self.tickets.get(raw_ticket_id) is not None else None
        )

        session.history.append({"role": "assistant", "content": response_text})

        return ChatResponse(
            success=True,
            session_id=clean_session_id,
            response=response_text,
            sources=sources,
            ticket_id=verified_ticket_id,
        )
