from llm_service.models import LLMRequest, LLMResponse

def test_llm_request():
    request = LLMRequest(prompt="Hello, world!", max_tokens=50, n=2)
    assert request.prompt == "Hello, world!"
    assert request.max_tokens == 50
    assert request.n == 2

def test_llm_response():
    response = LLMResponse(message="Hello, world!", token_count=10)
    assert response.message == "Hello, world!"
    assert response.token_count == 10