from fastapi.testclient import TestClient

from app.main import app


def test_ingest_and_ask_without_external_api_key():
    client = TestClient(app)
    response = client.post(
        "/v1/ingest",
        files={"files": ("policy.txt", b"Employees request access through the security portal.", "text/plain")},
    )
    assert response.status_code == 200
    answer = client.post("/v1/ask", json={"question": "How do employees request access?"})
    assert answer.status_code == 200
    assert "security portal" in answer.json()["answer"]
    assert answer.json()["retrieval_trace"]["original_query"] == "How do employees request access?"
