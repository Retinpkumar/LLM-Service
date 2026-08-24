import pytest
from unittest.mock import AsyncMock, MagicMock
from llm_service.client import call_llm, call_llm_batch


@pytest.mark.asyncio
async def test_call_llm_success(monkeypatch):
    # Mock the response from the client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello world!")]
    mock_response.usage.input_tokens = 5
    mock_response.usage.output_tokens = 5
    mock_response.usage.estimated_cost_usd = 0.01

    # Mock the AsyncAnthropic client
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    
    # Patch the client in the module
    monkeypatch.setattr("llm_service.client.get_client", MagicMock(return_value=mock_client))
    monkeypatch.setattr("llm_service.client.estimate_cost_usd", MagicMock(return_value=0.01))

    # Create a sample request
    request = MagicMock()
    request.model = "test-model"
    request.messages = [{"role": "user", "content": "Hello"}]
    request.max_tokens = 10

    # Call the function
    response = await call_llm(request)

    # Assertions
    assert response.message == "Hello world!"
    assert response.input_tokens == 5
    assert response.output_tokens == 5
    assert response.estimated_cost_usd == 0.01

@pytest.mark.asyncio
async def test_call_llm_batch_success(monkeypatch):
    # Mock the response from the client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello world!")]
    mock_response.usage.input_tokens = 5
    mock_response.usage.output_tokens = 5
    mock_response.usage.estimated_cost_usd = 0.01

    # Mock the AsyncAnthropic client
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    
    # Patch the client in the module
    monkeypatch.setattr("llm_service.client.get_client", MagicMock(return_value=mock_client))
    monkeypatch.setattr("llm_service.client.estimate_cost_usd", MagicMock(return_value=0.01))

    # Create sample requests
    request1 = MagicMock()
    request1.model = "test-model"
    request1.messages = [{"role": "user", "content": "Hello"}]
    request1.max_tokens = 10

    request2 = MagicMock()
    request2.model = "test-model"
    request2.messages = [{"role": "user", "content": "Hi"}]
    request2.max_tokens = 10

    requests = [request1, request2]

    # Call the function
    responses = await call_llm_batch(requests)

    # Assertions
    assert len(responses) == 2
    for response in responses:
        assert response.message == "Hello world!"
        assert response.input_tokens == 5
        assert response.output_tokens == 5
        assert response.estimated_cost_usd == 0.01
