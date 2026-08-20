from pydantic import BaseModel

class LLMRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 100
    

class LLMResponse(BaseModel):
    message: str
    token_count: int





