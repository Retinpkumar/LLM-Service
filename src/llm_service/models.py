from pydantic import BaseModel

class LLMRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 100
    

class LLMResponse(BaseModel):
    message: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float





