from __future__ import annotations

import json
import re
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.models import TicketCategory, TicketCreate
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import ConversationState, SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError


class SupportWorkflowState(TypedDict, total=False):
    """Typed state shared by the supplied LangGraph node skeletons."""

    session_id: str
    customer_message: str
    messages: Annotated[list, add_messages]
    retrieved_chunks: list[dict[str, str]]
    route: str
    extracted_fields: dict[str, str]
    response_text: str
    sources: list[str]
    ticket_id: str | None


class DecisionSchema(BaseModel):
    intent: Literal["answer", "ticket"] = Field(
        description="Choose 'answer' for general policy/troubleshooting questions, or 'ticket' if customer wants to report an unresolved issue, file a complaint, or provide details for a support ticket."
    )
    customer_name: Optional[str] = Field(
        default=None, description="Explicit customer name mentioned in this turn, or null."
    )
    customer_email: Optional[str] = Field(
        default=None, description="Explicit customer email address mentioned in this turn, or null."
    )
    issue_description: Optional[str] = Field(
        default=None, description="Explicit description of the unresolved problem or issue, or null."
    )
    category: Optional[TicketCategory] = Field(
        default=None,
        description="Category if stated or implied: order, payment, account, technical, or other.",
    )


DECIDE_SYSTEM_PROMPT = """You are an intent classification and information extraction assistant for a customer support platform.
Your task is to analyze the customer's message in the context of recent conversation history.

Rules:
1. Intent:
   - Select "ticket" if:
     a) The customer wants to report an unresolved issue, file a complaint, dispute a charge, or open/raise a support ticket.
     b) A ticket collection is already in progress and the user is responding with details (like name, email, or problem details).
   - Select "answer" if the customer is asking a general policy, shipping, return, payment rule, or account information question.

2. Information Extraction:
   - Extract customer_name ONLY if the customer explicitly provides their name in this turn.
   - Extract customer_email ONLY if the customer explicitly provides a valid email address in this turn.
   - Extract issue_description ONLY if the customer describes an issue or problem in this turn.
   - Extract category (must be one of: order, payment, account, technical, other) if discernible.
   - NEVER invent or hallucinate customer details. If not mentioned in this turn, leave as null.

You must respond with a JSON object adhering to this schema:
{{
  "intent": "answer" | "ticket",
  "customer_name": string | null,
  "customer_email": string | null,
  "issue_description": string | null,
  "category": "order" | "payment" | "account" | "technical" | "other" | null
}}
"""

ANSWER_SYSTEM_PROMPT = """You are a helpful and professional customer-support agent.

Strict Instructions:
1. Use the supplied knowledge context to answer policy, shipping, return, account, and payment questions.
2. Apply reasonable policy reasoning (for example, if returns are allowed within 30 days of delivery, a return requested at 20 days is within the 30-day window and therefore eligible).
3. If the knowledge context truly has no relation to the question (e.g. asking for unrelated recipes, weather, coding help, or general trivia), state clearly: "I am sorry, but our support documentation does not contain information to answer that question." Do NOT fabricate non-existent policies.
4. NEVER request or accept passwords, one-time codes (OTPs), or full payment-card numbers.
5. Do NOT claim a ticket exists unless explicitly told it was created.
6. Keep your tone polite, concise, and helpful.
"""


def _clean_json_text(text: str) -> str:
    """Extract JSON object from text that may contain markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def build_support_workflow(
    model: BaseChatModel,
    retriever: KnowledgeRetriever | None = None,
    sessions: SessionStore | None = None,
    tickets: TicketRepository | None = None,
):
    """Build the agent graph with robust nodes for RAG, intent routing, and ticket management."""

    async def retrieve(state: SupportWorkflowState) -> SupportWorkflowState:
        active_retriever = retriever
        customer_message = state.get("customer_message", "").strip()

        if not active_retriever or not customer_message:
            return {"retrieved_chunks": []}

        try:
            chunks = await active_retriever.search(customer_message)
        except Exception:
            chunks = []

        # Deduplicate identical chunk contents while preserving sources
        unique_chunks: list[dict[str, str]] = []
        seen_contents: set[str] = set()
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if content and content not in seen_contents:
                seen_contents.add(content)
                unique_chunks.append({
                    "content": content,
                    "source": chunk.get("source", "support.md"),
                })

        return {"retrieved_chunks": unique_chunks}

    async def decide(state: SupportWorkflowState) -> SupportWorkflowState:
        active_sessions = sessions
        session_id = state.get("session_id", "default")
        customer_message = state.get("customer_message", "")

        session: ConversationState | None = None
        if active_sessions:
            session = active_sessions.get_or_create(session_id)

        # Contextual clue: Is ticket creation already in progress?
        ticket_in_progress = False
        if session:
            missing = session.missing_ticket_fields()
            # If some fields have already been collected and ticket_id is not set
            if 0 < len(missing) < 4 and not session.ticket_id:
                ticket_in_progress = True

        prompt = (
            f"Context: Ticket collection currently in progress: {ticket_in_progress}\n"
            f"Customer message: {customer_message}\n"
            "Return JSON only."
        )

        try:
            messages = [
                SystemMessage(content=DECIDE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = await model.ainvoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
            cleaned_json = _clean_json_text(str(raw_text))
            parsed = json.loads(cleaned_json)
            decision = DecisionSchema(**parsed)
        except Exception:
            # Fallback heuristic if model json parsing fails
            msg_lower = customer_message.lower()
            if ticket_in_progress or any(w in msg_lower for w in ["ticket", "charged twice", "refund", "issue", "problem", "broken", "stolen", "report"]):
                intent = "ticket"
            else:
                intent = "answer"
            decision = DecisionSchema(intent=intent)

        # Ensure route is either "answer" or "ticket"
        route = decision.intent if decision.intent in ("answer", "ticket") else "answer"

        extracted: dict[str, str] = {}
        if decision.customer_name:
            extracted["customer_name"] = decision.customer_name.strip()
        if decision.customer_email:
            extracted["customer_email"] = decision.customer_email.strip()
        if decision.issue_description:
            extracted["issue_description"] = decision.issue_description.strip()
        if decision.category:
            extracted["category"] = decision.category

        # Regex fallback for email extraction if customer typed an email
        if "customer_email" not in extracted:
            email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", customer_message)
            if email_match:
                extracted["customer_email"] = email_match.group(0)

        return {
            "route": route,
            "extracted_fields": extracted,
        }

    async def answer(state: SupportWorkflowState) -> SupportWorkflowState:
        retrieved_chunks = state.get("retrieved_chunks", [])
        customer_message = state.get("customer_message", "")

        if not retrieved_chunks:
            return {
                "response_text": (
                    "I am sorry, but our support documentation does not contain information to answer that question."
                ),
                "sources": [],
                "ticket_id": None,
            }

        context_texts = [f"[{c['source']}]:\n{c['content']}" for c in retrieved_chunks]
        context_block = "\n\n".join(context_texts)

        prompt = (
            f"Knowledge Context:\n{context_block}\n\n"
            f"Customer Question: {customer_message}\n\n"
            "Provide a clear, grounded answer using only the knowledge context provided above."
        )

        try:
            messages = [
                SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = await model.ainvoke(messages)
            response_text = str(response.content if hasattr(response, "content") else response).strip()
        except Exception as exc:
            raise AgentProcessingError(f"Model generation failed: {exc}") from exc

        # Collect unique sources
        sources = list(dict.fromkeys(c["source"] for c in retrieved_chunks))

        return {
            "response_text": response_text,
            "sources": sources,
            "ticket_id": None,
        }

    async def collect_or_create(state: SupportWorkflowState) -> SupportWorkflowState:
        active_sessions = sessions
        active_tickets = tickets
        session_id = state.get("session_id", "default")
        customer_message = state.get("customer_message", "").strip()
        extracted = state.get("extracted_fields", {})

        if not active_sessions or not active_tickets:
            raise AgentProcessingError("Session store or ticket repository not available")

        session = active_sessions.get_or_create(session_id)

        # Attribute sources if relevant documents were retrieved during this turn
        sources = list(dict.fromkeys(c["source"] for c in state.get("retrieved_chunks", [])))

        # Idempotency check: if session already completed a ticket
        if session.ticket_id:
            existing_ticket = active_tickets.get(session.ticket_id)
            if existing_ticket:
                return {
                    "response_text": (
                        f"Your ticket has already been submitted with ID {session.ticket_id}. "
                        f"Status: {existing_ticket.status}."
                    ),
                    "sources": sources,
                    "ticket_id": session.ticket_id,
                }

        # Update session with newly extracted fields (never overwrite with empty)
        if extracted.get("customer_name") and not session.customer_name:
            session.customer_name = extracted["customer_name"]
        if extracted.get("customer_email") and not session.customer_email:
            session.customer_email = extracted["customer_email"]
        if extracted.get("issue_description") and not session.issue_description:
            session.issue_description = extracted["issue_description"]
        if extracted.get("category") and not session.category:
            session.category = extracted["category"]

        # If issue description is still missing but message is substantive, use customer_message
        if not session.issue_description and len(customer_message) >= 5:
            if "@" not in customer_message and len(customer_message.split()) > 2:
                session.issue_description = customer_message

        # Default category to 'other' if problem is described but category wasn't specific
        if not session.category and session.issue_description:
            desc_lower = session.issue_description.lower()
            if "pay" in desc_lower or "charg" in desc_lower or "card" in desc_lower or "bill" in desc_lower:
                session.category = "payment"
            elif "ship" in desc_lower or "deliver" in desc_lower or "track" in desc_lower:
                session.category = "order"
            elif "password" in desc_lower or "login" in desc_lower or "account" in desc_lower:
                session.category = "account"
            elif "bug" in desc_lower or "error" in desc_lower or "broken" in desc_lower or "crash" in desc_lower:
                session.category = "technical"
            else:
                session.category = "other"

        missing_fields = session.missing_ticket_fields()

        # If fields are missing, ask one targeted follow-up question
        if missing_fields:
            next_missing = missing_fields[0]
            if next_missing == "customer_name":
                question = "I can certainly help you create a ticket. Could you please share your full name?"
            elif next_missing == "customer_email":
                name_prefix = f"Thank you, {session.customer_name}. " if session.customer_name else "I can help you create a ticket. "
                question = f"{name_prefix}What email address should we use?"
            elif next_missing == "issue_description":
                question = "Please provide a brief description of the issue you are experiencing."
            elif next_missing == "category":
                question = (
                    "Which category best fits your issue? (order, payment, account, technical, or other)"
                )
            else:
                question = f"Could you please provide your {next_missing.replace('_', ' ')}?"

            return {
                "response_text": question,
                "sources": sources,
                "ticket_id": None,
            }

        # All required fields are available -> create the ticket
        summary = (
            session.issue_description[:150]
            if len(session.issue_description) <= 150
            else session.issue_description[:147] + "..."
        )
        if len(summary) < 5:
            summary = f"Support request: {session.issue_description}"

        try:
            ticket_data = TicketCreate(
                customer_name=session.customer_name,  # type: ignore[arg-type]
                customer_email=session.customer_email,  # type: ignore[arg-type]
                issue_description=session.issue_description,  # type: ignore[arg-type]
                category=session.category,  # type: ignore[arg-type]
                summary=summary,
            )
            created_ticket = active_tickets.create(session_id, ticket_data)
            session.ticket_id = created_ticket.ticket_id
        except Exception as exc:
            raise AgentProcessingError(f"Ticket validation or creation failed: {exc}") from exc

        confirmation_text = (
            f"Your support ticket has been created successfully with ID {created_ticket.ticket_id}. "
            f"A confirmation will be sent to {created_ticket.customer_email}. "
            f"Our team will follow up on your {created_ticket.category} issue shortly."
        )

        return {
            "response_text": confirmation_text,
            "sources": sources,
            "ticket_id": created_ticket.ticket_id,
        }

    def select_route(state: SupportWorkflowState) -> str:
        route = state.get("route")
        if route in ("answer", "ticket"):
            return route
        raise AgentProcessingError(f"Invalid workflow route selected: {route}")

    graph = StateGraph(SupportWorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", decide)
    graph.add_node("answer", answer)
    graph.add_node("ticket", collect_or_create)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide",
        select_route,
        {"answer": "answer", "ticket": "ticket"},
    )
    graph.add_edge("answer", END)
    graph.add_edge("ticket", END)

    return graph.compile()
