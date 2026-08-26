from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from llm_service.api import app
from llm_service.models import LLMResponse


client = TestClient(app)


def test_generate_endpoint():
    fake_response = LLMResponse(
        message="Hello there, friend!",
        input_tokens=16,
        output_tokens=8,
        estimated_cost_usd=0.000168,
    )

    with patch("llm_service.api.call_llm", new=AsyncMock(return_value=fake_response)):
        response = client.post("/generate", json={"prompt": "Say hello in exactly 3 words."})

    assert response.status_code == 200
    assert response.json()["message"] == "Hello there, friend!"

def test_generate_endpoint_with_system_prompt():
    fake_response = LLMResponse(
        message="Bonjour!",
        input_tokens=20,
        output_tokens=5,
        estimated_cost_usd=0.0002,
    )

    with patch("llm_service.api.call_llm", new=AsyncMock(return_value=fake_response)) as mock_call:
        response = client.post(
            "/generate",
            json={"prompt": "Say hello", "system": "Respond in French"},
        )

    assert response.status_code == 200
    sent_request = mock_call.call_args.args[0]
    assert sent_request.system == "Respond in French"