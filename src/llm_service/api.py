from fastapi import FastAPI
from pydantic import BaseModel

from llm_service.client import call_llm
from llm_service.models import LLMRequest, LLMResponse

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate", response_model=LLMResponse)
async def generate(request: GenerateRequest) -> LLMResponse:
    llm_request = LLMRequest(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": request.prompt}],
    )
    return await call_llm(llm_request)