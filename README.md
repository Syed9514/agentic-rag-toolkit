# Agentic RAG Toolkit

A production-friendly, agent-agnostic retrieval-augmented generation tool designed to plug into Claude, Gemini, Codex, or any custom agentic workflow.

## Features

- Document ingestion from local folders
- Chunking, indexing, and vector retrieval
- Semantic search over your knowledge base
- Grounded answer generation with source citations
- REST API for direct integration with agents or tools
- Optional OpenAI-backed answer generation
- Local embedding fallback without requiring API keys

## Quick start

1. Create a virtual environment

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies

   pip install -r requirements.txt

3. Copy environment variables

   cp .env.example .env

4. Add your docs to `docs/`

5. Start the API server

   uvicorn app.main:app --reload

6. Ingest your docs

   curl -X POST http://localhost:8000/api/ingest \
     -H "Content-Type: application/json" \
     -d '{"paths":["docs"]}'

7. Ask a question

   curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question":"What is this project about?","top_k":5}'

## Agent tool interface

The app exposes a standard tool-like REST endpoint suitable for agent orchestration.

Example schema for agent wrappers:

```json
{
  "name": "rag_query",
  "description": "Search the indexed knowledge base and answer grounded in retrieved documents.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "question": { "type": "string" },
      "top_k": { "type": "integer", "default": 5 }
    },
    "required": ["question"]
  }
}
```

You can wire this into Claude, Gemini, or custom tooling by calling:

- `POST /api/query`
- `POST /api/ingest`
- `GET /health`

## API endpoints

- `GET /health` - server health check
- `POST /api/ingest` - index local documents
- `POST /api/query` - answer a question from the knowledge base

## Example request bodies

### Ingest

```json
{
  "paths": ["docs"],
  "clear_existing": true
}
```

### Query

```json
{
  "question": "Explain the deployment steps for this product.",
  "top_k": 4
}
```

## Environment variables

See `.env.example` for examples.

Required for OpenAI-backed generation:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Optional:

- `LLM_PROVIDER`
- `EMBEDDING_PROVIDER`
- `VECTORSTORE_PATH`
- `DOCS_PATH`

## Architecture

The system is intentionally simple and portable:

- Embeddings: local Hugging Face embedding model by default
- Vector database: FAISS
- Document splitting: LangChain recursive splitter
- Generation: OpenAI if configured; otherwise grounded fallback answer
- Interface: FastAPI REST API for agent compatibility

## Notes

This toolkit is designed as a base that can be adapted to:

- MCP servers
- LangChain agents
- custom tool gateway wrappers
- browser-based or desktop assistants
- local-only private RAG deployments

## License

MIT
