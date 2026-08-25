import json
from typing import List
from pydantic import BaseModel
from llm_service.models import LLMRequest
from llm_service.client import call_llm


class ExtractedEntities(BaseModel):
    names: List[str]
    organizations: List[str]
    locations: List[str]
    contact_numbers: List[str]


async def extract_entities(text: str) -> ExtractedEntities:
    system_prompt = """You are an entity extraction service. Extract all names, organizations, locations, and contact numbers mentioned in the user's text.

Respond with ONLY a valid JSON object matching this exact structure, and nothing else — no markdown code fences, no explanation, no preamble:

{
  "names": ["..."],
  "organizations": ["..."],
  "locations": ["..."],
  "contact_numbers": ["..."]
}

If no entities of a given type are found, use an empty list for that field. Do not invent entities that are not present in the text."""

    request = LLMRequest(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": text}],
        system=system_prompt,
    )
    response = await call_llm(request)

    data = json.loads(response.message)
    
    return ExtractedEntities(**data)