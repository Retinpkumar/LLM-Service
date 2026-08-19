### What is this?
This is an LLM Service skeleton project. This project is the foundation for a larger curriculum project — eventually a FastAPI service wrapping LLM calls, extended with RAG/agents/evals in later stages.  

### How do I run it?
To run the project, you need to create a virtual environment and then install the required dependencies.  
This can be done by running the command:  <code>uv sync</code>  
Once that is successfully completed, then use the command <code>uv run llm-service</code> to run the project.  

### How do I run these tests?
To run the test cases, use: <code>uv run pytest</code>  

### Why is it structured this way?
The src/ layout forces proper installation rather than accidental path-based imports. Without src/ file structure, the script might run fine locally but will fail the moment it is installed somewhere else as it can import any locally available files next to the one running the main script.  

pyproject.toml declares what dependencies the project needs, and uv.lock pins the exact versions actually resolved and tested against.
These 2 files help in exact reproducibility of the working environment anywhere.