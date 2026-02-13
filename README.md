# CERN Magazine LLMOps

LLMOps framework for ingesting CERN Courier PDFs, chunking and indexing content,
and serving a retrieval-augmented QA API.

## Quickstart

1. **Install**

   ```bash
   python3 -m pip install -e ".[dev]"
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   # then edit .env and set OPENAI_API_KEY
   ```

3. **Run pipelines**

   ```bash
   # Download PDFs
   cern-mag-llmops ingest

   # Process PDFs and build Chroma vector DB
   cern-mag-llmops process

   # Start API server
   cern-mag-llmops serve
   ```

## API

- `GET /health` - service and model availability
- `POST /query` - answer a question with `rag`, `fine_tuned`, or `compare`
- `POST /documents` - retrieve relevant chunks for a query

Optional API key authentication is supported via:

- `API_AUTH_ENABLED=true`
- `API_AUTH_KEY=...`
- Request header: `X-API-Key`

## Developer Commands

```bash
ruff check .
mypy cern_mag_llmops
pytest
```

## Project Layout

- `cern_mag_llmops/config` - settings and environment handling
- `cern_mag_llmops/data_ingestion` - CERN Courier crawler
- `cern_mag_llmops/data_processing` - PDF extraction/chunking/vector DB builder
- `cern_mag_llmops/model` - RAG, fine-tuning, and model comparison
- `cern_mag_llmops/api` - FastAPI app and request/response models
- `cern_mag_llmops/pipelines` - ingestion/processing/fine-tuning pipelines
- `tests` - unit and API regression tests

## Notes

- OpenAI credentials are validated at runtime only for features that need them.
- CORS is environment-driven and defaults to local development origins.
- Fine-tuning monitoring includes timeout controls to avoid infinite polling.