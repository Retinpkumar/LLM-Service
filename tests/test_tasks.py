import json
import pytest
from pydantic import ValidationError
from unittest.mock import AsyncMock, MagicMock
from llm_service.tasks import extract_entities, ExtractedEntities


@pytest.mark.asyncio
async def test_extract_entities_success(monkeypatch):
    mock_response = MagicMock()
    mock_response.message = '{"names": ["Alice"], "organizations": ["Acme Corp"], "locations": ["Bangalore"], "contact_numbers": ["+91-9876543210"]}'

    mock_call_llm = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("llm_service.tasks.call_llm", mock_call_llm)

    result = await extract_entities("Alice from Acme Corp is based in Bangalore. Call +91-9876543210.")

    assert isinstance(result, ExtractedEntities)
    assert result.names == ["Alice"]
    assert result.organizations == ["Acme Corp"]
    assert result.locations == ["Bangalore"]
    assert result.contact_numbers == ["+91-9876543210"]

@pytest.mark.asyncio
async def test_extract_entities_invalid_json(monkeypatch):
    mock_response = MagicMock()
    mock_response.message = 'This is an invalid JSON'

    mock_call_llm = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("llm_service.tasks.call_llm", mock_call_llm)

    with pytest.raises(json.JSONDecodeError):
        await extract_entities("This will return an invalid JSON.")

@pytest.mark.asyncio
async def test_extract_entities_validation_error(monkeypatch):
    mock_response = MagicMock()
    mock_response.message = '{"names": "Alice", "organizations": ["Acme Corp"], "locations": ["Bangalore"], "contact_numbers": ["+91-9876543210"]}'

    mock_call_llm = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("llm_service.tasks.call_llm", mock_call_llm)

    with pytest.raises(ValidationError):
        await extract_entities("This will return a validation error.")