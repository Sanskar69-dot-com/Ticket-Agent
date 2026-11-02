import pytest
from src.sessions.store import ConversationState, SessionStore


def test_session_reports_only_missing_ticket_fields() -> None:
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")
    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_session_empty_and_complete_states() -> None:
    empty_state = ConversationState()
    assert empty_state.missing_ticket_fields() == [
        "customer_name",
        "customer_email",
        "issue_description",
        "category",
    ]

    complete_state = ConversationState(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="Login authentication failed repeatedly",
        category="account",
    )
    assert complete_state.missing_ticket_fields() == []


def test_session_store_isolation_between_sessions() -> None:
    store = SessionStore()
    s1 = store.get_or_create("session-1")
    s2 = store.get_or_create("session-2")

    s1.customer_name = "User One"
    s1.customer_email = "user1@example.com"

    # s2 must not inherit s1's values
    assert s2.customer_name is None
    assert s2.customer_email is None


def test_session_store_rejects_blank_id() -> None:
    store = SessionStore()
    with pytest.raises(ValueError):
        store.get_or_create("")

    with pytest.raises(ValueError):
        store.get_or_create("   ")


def test_session_history_preserves_order() -> None:
    state = ConversationState()
    state.history.append({"role": "user", "content": "Hello"})
    state.history.append({"role": "assistant", "content": "How can I help you?"})
    state.history.append({"role": "user", "content": "My card was charged twice"})

    assert len(state.history) == 3
    assert state.history[0]["role"] == "user"
    assert state.history[1]["role"] == "assistant"
    assert state.history[2]["content"] == "My card was charged twice"


def test_session_retains_completed_ticket_id() -> None:
    state = ConversationState(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="Card charged twice",
        category="payment",
        ticket_id="CST-2026-0042",
    )
    assert state.ticket_id == "CST-2026-0042"
    assert state.missing_ticket_fields() == []
