# MedSaathi

**Multilingual AI health assistant for rural India — Hindi, English, and Hinglish.**

MedSaathi is a production-grade RAG (Retrieval-Augmented Generation) application built with FastAPI and Gemini Flash. It answers health and medical queries grounded in verified PDF documents (WHO guidelines, AYUSH manuals), with safety guardrails, hybrid retrieval, cross-encoder reranking, and session memory.

> This project is informational only and is not a substitute for professional medical advice.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Evaluation](#evaluation)
- [Safety Design](#safety-design)
- [API Reference](#api-reference)
- [Roadmap](#roadmap)

---

## Architecture

Every user message goes through an 8-step pipeline:

```
User Query
    |
    v
[1] Session       -- create or restore conversation session
    |
    v
[2] Guardrails    -- pre-LLM safety check (crisis / unsafe / sensitive / OOS)
    |                 if blocked -> return hardcoded safe response immediately
    v
[3] Query Rewrite -- optimize raw query for retrieval (Gemini Flash, temp=0)
    |
    v
[4] Hybrid Search -- semantic (ChromaDB embeddings) + keyword (BM25) retrieval
    |                 fused via Reciprocal Rank Fusion (RRF, k=60)
    |
    v
[5] Reranking     -- cross-encoder (sentence-transformers) reranks top-10 to top-5
    |
    v
[6] History       -- fetch last N conversation turns from in-memory store
    |
    v
[7] Generation    -- grounded answer via Gemini Flash (temp=0.3), context-only prompt
    |
    v
[8] Memory Save   -- persist this turn for future context
    |
    v
Response (answer + sources + metadata)
```

---

## Features

**RAG Pipeline**
- Hybrid retrieval: semantic (all-MiniLM-L6-v2 embeddings + ChromaDB) combined with BM25 keyword search via Reciprocal Rank Fusion
- Cross-encoder reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2` for precision
- Query rewriting with Gemini Flash to optimize retrieval from informal Hindi/Hinglish queries
- Grounded generation: LLM can only answer from retrieved context — no hallucination from training data
- Source citations returned with every answer (PDF filename + page number)

**Safety System**
- Pre-LLM guardrails with five categories: `safe`, `sensitive`, `unsafe`, `crisis`, `out_of_scope`
- Crisis queries (self-harm, suicidal language) return hardcoded responses with verified Indian helpline numbers — the LLM never handles these
- Sensitive topics (mental health, reproductive health, substance abuse) receive topic-specific professional referral disclaimers
- Zero ML overhead on guardrails: pure rule-based keyword matching for speed

**API & Serving**
- FastAPI backend with both standard and streaming (`/chat/stream`) endpoints
- Server-Sent Events (SSE) streaming for typewriter effect — user sees response within ~200ms
- Session management: create, retrieve history, and clear sessions
- Frontend served from the same FastAPI process via `StaticFiles` — one command to run everything
- Health check endpoint (`/health`) reports vector store status and chunk count

**Evaluation**
- Offline RAGAS evaluation with four metrics: faithfulness, answer relevancy, context recall, context precision
- Timestamped results saved to `eval_results/` for tracking improvements over time

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 1.5 Flash (via LangChain Google GenAI) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Vector Store | ChromaDB (local persistence) |
| Keyword Search | BM25Okapi (rank-bm25) |
| Backend Framework | FastAPI + Uvicorn |
| Evaluation | RAGAS |
| PDF Ingestion | PyPDF |
| Package Manager | uv (Python >= 3.12) |

---

## Project Structure

```
MedSaathi/
|-- backend/
|   |-- main.py              # FastAPI app, pipeline orchestration
|   |-- ingest.py            # PDF ingestion: chunk, embed, store in ChromaDB
|   |-- embeddings.py        # Embedding model singleton + ChromaDB collection
|   |-- retriever.py         # Hybrid search: semantic + BM25 + RRF fusion
|   |-- reranker.py          # Cross-encoder reranking
|   |-- query_rewriter.py    # Query optimization via Gemini Flash
|   |-- generator.py         # Grounded response generation + streaming
|   |-- guardrails.py        # Pre-LLM safety checks
|   |-- memory.py            # In-memory session and conversation history
|   |-- evaluator.py         # RAGAS offline evaluation
|-- frontend/
|   |-- index.html
|   |-- style.css
|   |-- chat.js
|-- data/
|   |-- docs/                # Place your source medical PDFs here
|-- vector_store/            # ChromaDB persistence (auto-created, git-ignored)
|-- eval_results/            # RAGAS evaluation output (auto-created)
|-- pyproject.toml
|-- .env                     # GEMINI_API_KEY (not committed)
```

---

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), a free [Gemini API key](https://aistudio.google.com/apikey)

**1. Clone the repository**

```bash
git clone https://github.com/gauravakash/MedSaathi.git
cd MedSaathi
```

**2. Install dependencies**

```bash
uv sync
```

**3. Configure environment**

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

**4. Add medical PDFs**

Place your source PDFs in `data/docs/`. The project is designed to work with WHO guidelines, AYUSH manuals, or any structured medical reference documents.

**5. Run ingestion**

```bash
uv run backend/ingest.py
```

This chunks the PDFs, generates embeddings, and persists them to `vector_store/`. Run this once, or re-run whenever you add new documents.

---

## Running the App

**Start the backend (serves frontend too)**

```bash
uv run backend/main.py
```

The FastAPI server starts on `http://localhost:8000` and automatically serves the frontend at the same address.

- Chat UI: `http://localhost:8000`
- API docs (Swagger): `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

**Test with curl**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "bukhar mein kya lena chahiye"}'
```

---

## Evaluation

```bash
uv run backend/evaluator.py
```

Runs the RAGAS evaluation suite against a built-in test dataset and saves timestamped results to `eval_results/`. Metrics evaluated:

| Metric | What it measures |
|---|---|
| Faithfulness | Does the answer stay within the retrieved context? |
| Answer Relevancy | Is the answer relevant to the question asked? |
| Context Recall | Did retrieval find the right information? |
| Context Precision | How much of the retrieved context was actually useful? |

> Ground truth answers in `EVAL_DATASET` should be reviewed by a qualified medical professional before using evaluation results for production benchmarking.

---

## Safety Design

MedSaathi serves users in rural India who may have limited access to healthcare professionals. The safety system is designed around this reality.

**Query categories and handling:**

| Category | Example | Action |
|---|---|---|
| `safe` | "bukhar mein kya lena chahiye" | Full RAG pipeline |
| `sensitive` | "depression ke liye kya karein" | RAG pipeline + topic-specific disclaimer |
| `unsafe` | "overdose kaise karein" | Blocked — hardcoded warning + ambulance number |
| `crisis` | "mujhe jeena nahi hai" | Blocked — verified helpline numbers (Vandrevala, iCall, NIMHANS) |
| `out_of_scope` | "aaj cricket match kaun jeeta" | Blocked — polite redirect |

**Key design decisions:**
- Guardrails run before the LLM on every request. Blocked queries never reach Gemini.
- Crisis responses are hardcoded, never LLM-generated. Helpline numbers (1860-2662-345, 9152987821) are manually verified.
- Sensitive topics are not blocked — users deserve medical information — but receive professional referral messages appended to the answer.

---

## API Reference

### POST /chat

Run the full RAG pipeline.

**Request body:**
```json
{
  "query": "string",
  "session_id": "string | null",
  "stream": false
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [{"index": 1, "file": "who_guidelines.pdf", "page": 12}],
  "session_id": "string",
  "language": "hindi | english | hinglish",
  "category": "safe | sensitive | unsafe | crisis | out_of_scope",
  "chunks_used": 5,
  "rewritten_query": "string"
}
```

### POST /chat/stream

Identical to `/chat` but returns Server-Sent Events (SSE). Each event is prefixed with `data:` and the stream ends with `data: [DONE]`.

### GET /health

Returns vector store status, chunk count, and model name.

### POST /session/new

Create a new conversation session. Returns `session_id`.

### DELETE /session/{session_id}

Clear conversation history for a session.

### GET /session/{session_id}/history

Retrieve full conversation history for a session.

---

## Roadmap

- [ ] Replace in-memory session store with Redis for multi-instance support
- [ ] Add ML-based query classifier to replace keyword guardrails (fine-tuned BERT on Indian medical safety data)
- [ ] Containerize with Docker + Docker Compose
- [ ] CI/CD with GitHub Actions (lint, test, evaluate on PR)
- [ ] Deploy to AWS (ECS + ALB + S3 for PDFs)
- [ ] Add support for voice input (Whisper ASR for Hindi)
- [ ] Expand PDF corpus with state-specific ASHA worker guidelines
- [ ] Add user feedback loop to improve retrieval quality over time

---

## Notes

- This project is for informational and educational purposes only. It is not a replacement for professional medical advice, diagnosis, or treatment.
- Always validate evaluation ground truths with qualified medical reviewers before using results to make production decisions.
- In production, replace `allow_origins=["*"]` in CORS settings with your actual frontend domain.
