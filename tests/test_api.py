from fastapi.testclient import TestClient
from src.api.server import app, pipeline
from src.models import TicketCreate


client = TestClient(app, raise_server_exceptions=False)


def test_health_endpoint() -> None:
    pipeline.ready = True
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

    pipeline.ready = False
    response = client.get("/health")
    assert response.status_code == 503
    pipeline.ready = True


def test_get_ticket_not_found() -> None:
    response = client.get("/tickets/CST-9999-9999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Ticket not found"


def test_get_ticket_success() -> None:
    ticket = pipeline.tickets.create(
        "api-test-session",
        TicketCreate(
            customer_name="API User",
            customer_email="apiuser@example.com",
            issue_description="API test ticket issue description",
            category="technical",
            summary="API test ticket summary",
        ),
    )
    response = client.get(f"/tickets/{ticket.ticket_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["ticket_id"] == ticket.ticket_id
    assert data["customer_name"] == "API User"
    assert data["category"] == "technical"


def test_chat_validation_error() -> None:
    # Blank session_id or message should return 422
    response = client.post("/chat", json={"session_id": "", "message": "hello"})
    assert response.status_code == 422

    response = client.post("/chat", json={"session_id": "s1", "message": ""})
    assert response.status_code == 422
