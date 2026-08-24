import asyncio
from anthropic import AsyncAnthropic
from llm_service.models import LLMRequest, LLMResponse
from llm_service.pricing import estimate_cost_usd
from pydantic import ValidationError


def get_client() -> AsyncAnthropic:
    return AsyncAnthropic()

async def call_llm(request: LLMRequest):
    client = get_client()
    response = await client.messages.create(
        model=request.model,
        messages=request.messages,
        max_tokens=request.max_tokens,
    )
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = estimate_cost_usd(request.model, input_tokens, output_tokens)
    try:
        message_text = response.content[0].text
        return LLMResponse(message=message_text, input_tokens=input_tokens, output_tokens=output_tokens, estimated_cost_usd=cost)
    except (IndexError, AttributeError, ValidationError) as e:
        return LLMResponse(message=f"Error processing response: {str(e)}", input_tokens=input_tokens, output_tokens=output_tokens, estimated_cost_usd=cost)

async def call_llm_batch(requests: list[LLMRequest]):
    tasks = [call_llm(request) for request in requests]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for result in raw_results:
        if isinstance(result, Exception):
            results.append(LLMResponse(message=str(result), input_tokens=0, output_tokens=0, estimated_cost_usd =0))
        else:
            results.append(result)
    return results

