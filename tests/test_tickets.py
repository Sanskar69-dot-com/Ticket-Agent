import pytest
from pydantic import ValidationError

from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository, create_ticket_tool


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )

    first = repository.create("session-1", request)
    second = repository.create("session-1", request)

    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


def test_repository_unique_ids_across_different_sessions() -> None:
    repository = TicketRepository()
    req1 = TicketCreate(
        customer_name="Alice",
        customer_email="alice@example.com",
        issue_description="Package did not arrive on time",
        category="order",
        summary="Delayed shipping arrival",
    )
    req2 = TicketCreate(
        customer_name="Bob",
        customer_email="bob@example.com",
        issue_description="Cannot reset account password",
        category="account",
        summary="Password reset issue",
    )

    t1 = repository.create("session-alice", req1)
    t2 = repository.create("session-bob", req2)

    assert t1.ticket_id != t2.ticket_id
    assert t1.customer_name == "Alice"
    assert t2.customer_name == "Bob"
    assert len(list(repository.all())) == 2


def test_repository_get_existing_and_missing() -> None:
    repository = TicketRepository()
    req = TicketCreate(
        customer_name="Charlie",
        customer_email="charlie@example.com",
        issue_description="Refund has not appeared in bank account",
        category="payment",
        summary="Pending refund inquiry",
    )
    ticket = repository.create("session-charlie", req)

    found = repository.get(ticket.ticket_id)
    assert found is not None
    assert found.customer_email == "charlie@example.com"
    assert found.status == "open"

    missing = repository.get("CST-9999-9999")
    assert missing is None


def test_ticket_create_validation_rules() -> None:
    # Invalid email must raise ValidationError
    with pytest.raises(ValidationError):
        TicketCreate(
            customer_name="David",
            customer_email="not-an-email",
            issue_description="Valid issue description",
            category="payment",
            summary="Payment issue",
        )

    # Short description must raise ValidationError
    with pytest.raises(ValidationError):
        TicketCreate(
            customer_name="David",
            customer_email="david@example.com",
            issue_description="tiny",  # min_length is 5
            category="payment",
            summary="Valid summary",
        )

    # Invalid category must raise ValidationError
    with pytest.raises(ValidationError):
        TicketCreate(
            customer_name="David",
            customer_email="david@example.com",
            issue_description="Valid issue description",
            category="invalid_category",  # type: ignore[arg-type]
            summary="Valid summary",
        )


def test_structured_ticket_tool_invocation() -> None:
    repository = TicketRepository()
    tool = create_ticket_tool(repository, "session-tool-test")

    ticket_id = tool.invoke({
        "customer_name": "Eva",
        "customer_email": "eva@example.com",
        "issue_description": "Damaged goods received in parcel",
        "category": "order",
        "summary": "Damaged items received",
    })

    assert ticket_id.startswith("CST-")
    saved = repository.get(ticket_id)
    assert saved is not None
    assert saved.customer_name == "Eva"
