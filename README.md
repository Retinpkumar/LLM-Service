### What is this?

This is an LLM Service skeleton project. This project is the foundation for a larger curriculum project — eventually a FastAPI service wrapping LLM calls, extended with RAG/agents/evals in later stages.

### How do I run it?

To run the project, install dependencies with:
`uv sync`

Then start the FastAPI server:
`uv run uvicorn llm_service.api:app --reload`

The server will be available at `http://127.0.0.1:8000`.
Interactive API docs are available at `http://127.0.0.1:8000/docs`.

### API Endpoints

**`GET /health`**
Returns `{"status": "ok"}` if the server is running. Useful for checking the service is up.

**`POST /generate`**
Accepts a JSON body with a `prompt` (required) and an optional `system` prompt, calls the LLM, and returns a structured response containing the generated message, token usage, and estimated cost.

Example request:

```json
{
  "prompt": "Say hello in exactly 3 words.",
  "system": "Respond in French"
}
```

### How do I run these tests?

To run the test cases, use: `uv run pytest`

### Why is it structured this way?

The src/ layout forces proper installation rather than accidental path-based imports. Without src/ file structure, the script might run fine locally but will fail the moment it is installed somewhere else, as it can import any locally available files next to the one running the main script.

pyproject.toml declares what dependencies the project needs, and uv.lock pins the exact versions actually resolved and tested against. These 2 files help in exact reproducibility of the working environment anywhere.

The API layer (`api.py`) uses its own request models (e.g. `GenerateRequest`) separate from the internal models (`LLMRequest`, `LLMResponse`) — this keeps what external callers are allowed to send decoupled from the shape the internal LLM wrapper actually needs.

### Running with Docker

Build the image:
`docker build -t llm-service .`

Run the container, mapping port 8000 so it's reachable from your host machine:
`docker run -p 8000:8000 llm-service`

Once running, the API is available at `http://127.0.0.1:8000`, same as running it locally.
