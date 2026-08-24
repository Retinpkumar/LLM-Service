from httpx import request

from llm_service.models import LLMRequest, LLMResponse


def test_llm_request():
    request = LLMRequest(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "Hello, world!"}], max_tokens=50)
    assert request.model == "claude-sonnet-4-6"
    assert request.messages == [{"role": "user", "content": "Hello, world!"}]
    assert request.max_tokens == 50
    

def test_llm_response():
    response = LLMResponse(message="Hello, world!", input_tokens=5, output_tokens=5, estimated_cost_usd=0.01)
    assert response.message == "Hello, world!"
    assert response.input_tokens == 5
    assert response.output_tokens == 5
    assert response.estimated_cost_usd == 0.01
