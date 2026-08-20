import asyncio
from anthropic import AsyncAnthropic
from llm_service.models import LLMRequest, LLMResponse


def get_client() -> AsyncAnthropic:
    return AsyncAnthropic()

async def call_llm(request: LLMRequest):
    client = get_client()
    response = await client.messages.create(
        model=request.model,
        messages=request.messages,
        max_tokens=request.max_tokens,
    )
    return LLMResponse(message=response.content[0].text, token_count=response.usage.total_tokens)

async def call_llm_batch(requests: list[LLMRequest]):
    tasks = [call_llm(request) for request in requests]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for result in raw_results:
        if isinstance(result, Exception):
            results.append(LLMResponse(message=str(result), token_count=0))
        else:
            results.append(result)
    return results

