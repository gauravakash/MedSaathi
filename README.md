# MedSaathi

MedSaathi is a multilingual (Hindi/English/Hinglish) medical RAG assistant designed for accessible rural-health guidance.

## Features

- FastAPI backend with full RAG pipeline:
  - query rewriting
  - hybrid retrieval (semantic + BM25)
  - cross-encoder reranking
  - grounded answer generation
  - safety guardrails
  - session memory
- Frontend chat UI optimized for Hindi/English users
- Offline evaluator with RAGAS metrics:
  - faithfulness
  - answer relevancy
  - context recall
  - context precision

## Project Structure

- `backend/` - API, RAG pipeline, evaluator
- `frontend/` - static chat UI (`index.html`, `style.css`, `chat.js`)
- `data/docs/` - source PDFs for ingestion
- `vector_store/` - local Chroma persistence (ignored in git)

## Setup

1. Install dependencies:
   - `uv sync`
2. Create `.env` file:
   - `GEMINI_API_KEY=your_key_here`
3. Add/replace medical PDFs in `data/docs/`
4. Run ingestion:
   - `uv run backend/ingest.py`

## Run Backend

- `uv run backend/main.py`
- API docs: `http://localhost:8000/docs`

## Run Frontend

Serve the `frontend/` directory with any static server and open `frontend/index.html`.

Example (Python):

- `python -m http.server 3000 --directory frontend`

Then open:

- `http://localhost:3000`

## Run Evaluation

- `uv run backend/evaluator.py`

This runs offline RAG evaluation and saves timestamped results under `eval_results/`.

## Notes

- This project provides informational support only and is not a replacement for professional medical advice.
- Validate `EVAL_DATASET` ground truths with qualified medical reviewers before production benchmarking.
