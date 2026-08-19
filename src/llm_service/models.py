from pydantic import BaseModel

class LLMRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    n: int = 1

class LLMResponse(BaseModel):
    message: str
    token_count: int





