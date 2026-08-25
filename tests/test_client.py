import pytest
import httpx
from anthropic import RateLimitError
from unittest.mock import AsyncMock, MagicMock
from llm_service.client import call_llm, call_llm_batch



@pytest.mark.asyncio
async def test_call_llm_retries_on_rate_limit(monkeypatch):
    # Build a mock successful response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello world!")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    # Build tow fake RateLimitError instances to simulate two failed attempts
    rate_limit_error1 = RateLimitError(
        message="rate limited",
        response=httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.anthropic.com")),
        body=None,
    )
    rate_limit_error2 = RateLimitError(
        message="rate limited",
        response=httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.anthropic.com")),
        body=None,
    )

    # Mock the AsyncAnthropic client
    mock_client = MagicMock()

    # Scripted sequence: fail, fail, then succeed
    mock_client.messages.create = AsyncMock(side_effect=[rate_limit_error1, rate_limit_error2, mock_response])

    # Patch the client in the module
    monkeypatch.setattr("llm_service.client.get_client", MagicMock(return_value=mock_client))
    monkeypatch.setattr("llm_service.client.estimate_cost_usd", MagicMock(return_value=0.01))
    monkeypatch.setattr("llm_service.client.asyncio.sleep", AsyncMock())  # Mock sleep to avoid actual delay

    # Create a sample request
    request = MagicMock()
    request.model = "test-model"
    request.messages = [{"role": "user", "content": "Hello"}]
    request.max_tokens = 10
    request.system = None

    # Call the function
    response = await call_llm(request)

    # Assertions
    assert response.message == "Hello world!"
    assert mock_client.messages.create.call_count == 3  # Ensure it retried twice before succeeding


@pytest.mark.asyncio
async def test_call_llm_raises_after_max_retries(monkeypatch):
    rate_limit_error1 = RateLimitError(
        message="rate limited",
        response=httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.anthropic.com")),
        body=None,
    )
    rate_limit_error2 = RateLimitError(
        message="rate limited",
        response=httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.anthropic.com")),
        body=None,
    )
    rate_limit_error3 = RateLimitError(
        message="rate limited",
        response=httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.anthropic.com")),
        body=None,
    )

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=[rate_limit_error1, rate_limit_error2, rate_limit_error3]
    )

    monkeypatch.setattr("llm_service.client.get_client", MagicMock(return_value=mock_client))
    monkeypatch.setattr("llm_service.client.asyncio.sleep", AsyncMock())

    request = MagicMock()
    request.model = "test-model"
    request.messages = [{"role": "user", "content": "Hello"}]
    request.max_tokens = 10
    request.system = None

    with pytest.raises(RateLimitError):
        await call_llm(request)

    assert mock_client.messages.create.call_count == 3


@pytest.mark.asyncio
async def test_call_llm_passes_system_prompt(monkeypatch):
    # Mock the response from the client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello world!")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    # Mock the AsyncAnthropic client
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    # Patch the client in the module
    monkeypatch.setattr("llm_service.client.get_client", MagicMock(return_value=mock_client))
    monkeypatch.setattr("llm_service.client.estimate_cost_usd", MagicMock(return_value=0.01))

    # Create a sample request with a system prompt
    request = MagicMock()
    request.model = "test-model"
    request.messages = [{"role": "user", "content": "Hello"}]
    request.max_tokens = 10
    request.system = "You are a helpful assistant."

    # Call the function
    response = await call_llm(request)

    # Assertions
    call_kwargs = mock_client.messages.create.call_args.kwargs 
    assert call_kwargs["system"] == "You are a helpful assistant."

@pytest.mark.asyncio
async def test_call_llm_omits_system_when_none(monkeypatch):
    # Mock the response from the client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello world!")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    # Mock the AsyncAnthropic client
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    # Patch the client in the module
    monkeypatch.setattr("llm_service.client.get_client", MagicMock(return_value=mock_client))
    monkeypatch.setattr("llm_service.client.estimate_cost_usd", MagicMock(return_value=0.01))

    # Create a sample request without a system prompt
    request = MagicMock()
    request.model = "test-model"
    request.messages = [{"role": "user", "content": "Hello"}]
    request.max_tokens = 10
    request.system = None

    # Call the function
    response = await call_llm(request)

    # Assertions
    call_kwargs = mock_client.messages.create.call_args.kwargs 
    assert "system" not in call_kwargs


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
