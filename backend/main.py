"""
MedSaathi — FastAPI Backend Server (The Conductor)
====================================================

This file wires ALL backend modules together into a single working API.
It doesn't do any heavy lifting itself — it just calls the right files
in the right order for each user request.

Full pipeline for every chat message:
──────────────────────────────────────
  Frontend → POST /chat → main.py orchestrates:

  ┌─────────────────────────────────────────────────────────────┐
  │  Step 1: Session       → memory.py (create/get session)     │
  │  Step 2: Guardrails    → guardrails.py (safety check)       │
  │  Step 3: Query Rewrite → query_rewriter.py (optimize query) │
  │  Step 4: Retrieval     → retriever.py (hybrid search)       │
  │  Step 5: Reranking     → reranker.py (cross-encoder)        │
  │  Step 6: History       → memory.py (get past turns)         │
  │  Step 7: Generation    → generator.py (Gemini Flash)        │
  │  Step 8: Save Memory   → memory.py (store this turn)        │
  └─────────────────────────────────────────────────────────────┘

  If guardrails block the query at Step 2, steps 3-7 are SKIPPED entirely.
  The user gets an immediate pre-written safe response.

What is CORS?
─────────────
  Cross-Origin Resource Sharing. Without it, a browser running the frontend
  on http://localhost:3000 would refuse to talk to the API on http://localhost:8000
  because they're different "origins." CORS headers tell the browser:
  "It's OK, this API allows requests from any origin." We use allow_origins=["*"]
  during development; in production, restrict to your actual frontend domain.

What are Pydantic models?
─────────────────────────
  Pydantic validates incoming request data automatically. If someone sends
  {"query": 123} instead of {"query": "some text"}, FastAPI returns a 422
  error with a helpful message — before our code even runs. This prevents
  crashes from bad input and documents the API shape via /docs.

Why preload models on startup?
──────────────────────────────
  The embedding model (~120MB) and reranker (~80MB) take 3-5 seconds each
  to load. If we loaded them on the first request, that user would wait
  10+ seconds for a response. By loading at startup, the server is ready
  before any user connects.

Test with:
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "bukhar mein kya lena chahiye"}'
"""

import time
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

# ─── Load environment variables FIRST (before any module that reads them) ─────
load_dotenv()

# ─── Backend module imports ───────────────────────────────────────────────────
from backend.embeddings import get_embedding_model, get_collection  # noqa: E402
from backend.retriever import hybrid_search  # noqa: E402
from backend.reranker import rerank, get_reranker_model  # noqa: E402
from backend.query_rewriter import process_query  # noqa: E402
from backend.generator import (  # noqa: E402
    generate,
    generate_stream,
    format_sources,
    FALLBACK_RESPONSE_HINDI,
)
from backend.memory import (  # noqa: E402
    create_session,
    add_message,
    get_history,
    format_history_for_llm,
    clear_session,
    ConversationTurn,
)
from backend.guardrails import (  # noqa: E402
    run_all_checks,
    get_sensitive_disclaimer,
    QueryCategory,
)

# ─── Configuration ────────────────────────────────────────────────────────────

HOST = "0.0.0.0"
PORT = 8000
RETRIEVAL_N_RESULTS = 10   # Fetch top 10 from hybrid search (wide net)
RERANK_TOP_N = 5           # Rerank down to top 5 (precise selection)

console = Console(force_terminal=True)


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedSaathi API",
    description="AI Health Assistant for Rural India",
    version="1.0.0",
)

# CORS: Allow frontend on any origin/port during development.
# In production, replace ["*"] with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Request/Response Models ────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None   # None = create a new session automatically
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    language: str
    category: str                   # "safe", "sensitive", "crisis", etc.
    chunks_used: int
    rewritten_query: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    total_chunks: int
    model: str


# ─── Pipeline Logging Helper ─────────────────────────────────────────────────

def log_pipeline_step(step: int, total: int, name: str, detail: str, time_ms: float) -> None:
    """Print a colored pipeline step for server-side debugging."""
    console.print(
        f"  [dim][Step {step}/{total}][/dim] {name} "
        f"[green]OK[/green] [dim]({time_ms:.0f}ms)[/dim] — {detail}"
    )


# ─── Timing Middleware ────────────────────────────────────────────────────────

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """
    Log the duration of every request with color-coded timing.
    Green = fast (<1s), Yellow = moderate (1-3s), Red = slow (>3s).
    """
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    # Color by speed
    if duration_ms < 1000:
        color = "green"
    elif duration_ms < 3000:
        color = "yellow"
    else:
        color = "red"

    console.print(
        f"[dim]{request.method}[/dim] {request.url.path} "
        f"— [{color}]{duration_ms:.0f}ms[/{color}]"
    )
    return response


# ─── Startup Event ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """
    Preload ML models and verify ChromaDB connection on server start.
    This ensures the first user request is fast — no model loading delay.
    """
    console.print("\n[bold cyan]Starting MedSaathi API...[/bold cyan]\n")

    # Load embedding model (~120MB, ~3-5s)
    embedding_status = "Loaded"
    try:
        get_embedding_model()
    except Exception as e:
        embedding_status = f"FAILED: {e}"

    # Load reranker model (~80MB, ~2-3s)
    reranker_status = "Loaded"
    try:
        get_reranker_model()
    except Exception as e:
        reranker_status = f"FAILED: {e}"

    # Check ChromaDB vector store
    chunk_count = 0
    vector_status = "Empty — run ingest.py first"
    try:
        collection = get_collection()
        chunk_count = collection.count()
        vector_status = f"Ready ({chunk_count} chunks)"
    except SystemExit:
        # get_collection() calls sys.exit if collection not found
        vector_status = "Not found — run ingest.py first"
    except Exception as e:
        vector_status = f"Error: {e}"

    # Print startup summary
    console.print(Panel(
        f"[bold green]MedSaathi API Started[/bold green]\n\n"
        f"  Vector Store:    {vector_status}\n"
        f"  Embedding Model: {embedding_status}\n"
        f"  Reranker:        {reranker_status}\n"
        f"  Gemini Flash:    Initialized on first query\n"
        f"  Server:          http://localhost:{PORT}\n"
        f"  API Docs:        http://localhost:{PORT}/docs",
        title="MedSaathi",
        border_style="green",
    ))


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint — used by frontend to verify API is running.
    Also reports vector store status so the UI can warn if ingestion needed.
    """
    chunk_count = 0
    ready = False
    try:
        collection = get_collection()
        chunk_count = collection.count()
        ready = True
    except (SystemExit, Exception):
        pass

    return HealthResponse(
        status="healthy",
        vector_store_ready=ready,
        total_chunks=chunk_count,
        model="gemini-1.5-flash",
    )


# ─── Session Endpoints ───────────────────────────────────────────────────────

@app.post("/session/new", response_model=SessionResponse)
async def new_session():
    """
    Create a new chat session. Frontend calls this on page load or "New Chat."
    """
    session_id = create_session()
    console.print(f"[dim]New session created:[/dim] {session_id}")

    return SessionResponse(
        session_id=session_id,
        message=(
            "Namaste! Main MedSaathi hoon. Aap apni health se judi "
            "koi bhi baat pooch sakte hain. Main Hindi, English, "
            "aur Hinglish mein jawab de sakta hoon."
        ),
    )


@app.delete("/session/{session_id}", response_model=SessionResponse)
async def delete_session(session_id: str):
    """Clear conversation history for a session."""
    clear_session(session_id)
    console.print(f"[dim]Session cleared:[/dim] {session_id}")

    return SessionResponse(
        session_id=session_id,
        message="Session cleared successfully.",
    )


# ─── Get Session History ──────────────────────────────────────────────────────

@app.get("/session/{session_id}/history")
async def session_history(session_id: str):
    """
    Retrieve conversation history for a session.
    Frontend uses this to restore the chat on page refresh.
    """
    turns = get_history(session_id, last_n=50)  # Get more turns for display

    return {
        "session_id": session_id,
        "turns": [
            {
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp,
                "language": turn.language,
            }
            for turn in turns
        ],
    }


# ─── Helper: Convert Memory Turns to Generator History Format ─────────────────

def _build_history_pairs(turns: list[ConversationTurn]) -> list[dict]:
    """
    Convert memory's ConversationTurn list into the {user, assistant} pair
    format that generator.py's build_prompt() expects.

    Memory stores turns sequentially: [user, assistant, user, assistant, ...]
    Generator expects paired dicts: [{"user": "...", "assistant": "..."}, ...]
    """
    pairs = []
    i = 0
    while i < len(turns) - 1:
        if turns[i].role == "user" and turns[i + 1].role == "assistant":
            pairs.append({
                "user": turns[i].content,
                "assistant": turns[i + 1].content,
            })
            i += 2
        else:
            i += 1
    return pairs


# ─── Main Chat Endpoint ──────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    The main chat endpoint — runs the FULL RAG pipeline.

    Pipeline: session → guardrails → rewrite → retrieve → rerank → history → generate → save
    """
    pipeline_start = time.time()
    total_steps = 8

    try:
        # ── Step 1: Session handling ──────────────────────────────────
        t = time.time()
        session_id = request.session_id or create_session()
        log_pipeline_step(1, total_steps, "Session", f"id={session_id[:8]}...", (time.time() - t) * 1000)

        # ── Step 2: Guardrails ────────────────────────────────────────
        t = time.time()
        guard_result = run_all_checks(request.query)
        log_pipeline_step(2, total_steps, "Guardrails", f"category={guard_result.category.value}", (time.time() - t) * 1000)

        if guard_result.is_blocked:
            # Query is blocked — return safe response without touching the LLM
            safe_response = guard_result.safe_response or FALLBACK_RESPONSE_HINDI

            # Save to memory so chat history shows this exchange
            add_message(session_id, "user", request.query)
            add_message(session_id, "assistant", safe_response)

            console.print(f"  [yellow]Query blocked:[/yellow] {guard_result.category.value}")

            total_ms = (time.time() - pipeline_start) * 1000
            console.print(f"  [dim]Pipeline complete: {total_ms:.0f}ms (blocked at guardrails)[/dim]")

            return ChatResponse(
                answer=safe_response,
                sources=[],
                session_id=session_id,
                language="hindi",
                category=guard_result.category.value,
                chunks_used=0,
                rewritten_query=request.query,
            )

        # ── Step 3: Query rewriting ───────────────────────────────────
        t = time.time()
        query_info = process_query(request.query)
        rewritten = query_info["best"]
        language = query_info["language"]
        log_pipeline_step(3, total_steps, "Rewrite", f"'{request.query[:30]}' -> '{rewritten[:30]}'", (time.time() - t) * 1000)

        # ── Step 4: Retrieval ─────────────────────────────────────────
        t = time.time()
        retrieved_chunks = hybrid_search(rewritten, n_results=RETRIEVAL_N_RESULTS)
        log_pipeline_step(4, total_steps, "Retrieval", f"{len(retrieved_chunks)} chunks", (time.time() - t) * 1000)

        # ── Step 5: Reranking ─────────────────────────────────────────
        t = time.time()
        reranked_chunks = rerank(rewritten, retrieved_chunks, top_n=RERANK_TOP_N)
        log_pipeline_step(5, total_steps, "Reranking", f"top {len(reranked_chunks)} chunks", (time.time() - t) * 1000)

        # ── Step 6: Get conversation history ──────────────────────────
        t = time.time()
        history_turns = get_history(session_id)
        history_pairs = _build_history_pairs(history_turns)
        log_pipeline_step(6, total_steps, "History", f"{len(history_pairs)} past exchanges", (time.time() - t) * 1000)

        # ── Step 7: Generate response ─────────────────────────────────
        t = time.time()
        result = generate(
            query=request.query,
            chunks=reranked_chunks,
            language=language,
            history=history_pairs,
        )
        answer = result["answer"]

        # Append sensitive disclaimer if applicable
        if guard_result.category == QueryCategory.SENSITIVE:
            disclaimer = get_sensitive_disclaimer(request.query)
            answer += disclaimer

        log_pipeline_step(7, total_steps, "Generation", f"{len(answer)} chars", (time.time() - t) * 1000)

        # ── Step 8: Save to memory ────────────────────────────────────
        t = time.time()
        add_message(session_id, "user", request.query, language)
        add_message(session_id, "assistant", answer, language)
        log_pipeline_step(8, total_steps, "Memory", "saved", (time.time() - t) * 1000)

        # Pipeline summary
        total_ms = (time.time() - pipeline_start) * 1000
        console.print(f"  [bold green]Pipeline complete: {total_ms:.0f}ms[/bold green]")

        return ChatResponse(
            answer=answer,
            sources=result["sources"],
            session_id=session_id,
            language=language,
            category=guard_result.category.value,
            chunks_used=result["chunks_used"],
            rewritten_query=rewritten,
        )

    except Exception as e:
        console.print(f"[red]Pipeline error: {type(e).__name__}: {e}[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")

        # Always return a safe fallback — never crash on the user
        session_id = request.session_id or "error"
        return ChatResponse(
            answer=FALLBACK_RESPONSE_HINDI,
            sources=[],
            session_id=session_id,
            language="hindi",
            category="error",
            chunks_used=0,
            rewritten_query=request.query,
        )


# ─── Streaming Chat Endpoint ─────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming version of /chat — returns response as Server-Sent Events (SSE).

    What is SSE (Server-Sent Events)?
    ─────────────────────────────────
      A protocol for the server to push data to the client as it becomes
      available. Instead of waiting for the full response, the frontend
      receives small text chunks every ~50-100ms:

        data: Bukhar
        data:  mein
        data:  paracetamol
        data:  500mg
        data: [DONE]

      The frontend appends each chunk to the chat bubble, creating a
      typewriter effect. Users start reading within ~200ms instead of
      waiting 2-3 seconds for the full response.

      Format: each chunk is prefixed with "data: " and followed by two
      newlines. The final "[DONE]" event signals the stream is complete.
    """
    try:
        # ── Steps 1-6: Same as /chat (non-streaming) ──
        session_id = request.session_id or create_session()

        guard_result = run_all_checks(request.query)
        if guard_result.is_blocked:
            safe_response = guard_result.safe_response or FALLBACK_RESPONSE_HINDI
            add_message(session_id, "user", request.query)
            add_message(session_id, "assistant", safe_response)

            async def blocked_stream():
                yield f"data: {safe_response}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(blocked_stream(), media_type="text/event-stream")

        query_info = process_query(request.query)
        rewritten = query_info["best"]
        language = query_info["language"]

        retrieved_chunks = hybrid_search(rewritten, n_results=RETRIEVAL_N_RESULTS)
        reranked_chunks = rerank(rewritten, retrieved_chunks, top_n=RERANK_TOP_N)

        history_turns = get_history(session_id)
        history_pairs = _build_history_pairs(history_turns)

        # ── Step 7: Streaming generation ──
        # Collect full response for memory while streaming to client
        async def response_stream():
            full_response = []

            for text_chunk in generate_stream(
                query=request.query,
                chunks=reranked_chunks,
                language=language,
                history=history_pairs,
            ):
                full_response.append(text_chunk)
                yield f"data: {text_chunk}\n\n"

            # Append sensitive disclaimer if needed
            if guard_result.category == QueryCategory.SENSITIVE:
                disclaimer = get_sensitive_disclaimer(request.query)
                full_response.append(disclaimer)
                yield f"data: {disclaimer}\n\n"

            yield "data: [DONE]\n\n"

            # Save complete response to memory after streaming finishes
            complete_answer = "".join(full_response)
            add_message(session_id, "user", request.query, language)
            add_message(session_id, "assistant", complete_answer, language)

        return StreamingResponse(response_stream(), media_type="text/event-stream")

    except Exception as e:
        console.print(f"[red]Streaming error: {type(e).__name__}: {e}[/red]")

        async def error_stream():
            yield f"data: {FALLBACK_RESPONSE_HINDI}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")


# ─── Main Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    console.print(Panel(
        "[bold green]MedSaathi Backend[/bold green]\n"
        "Rural AI Health Assistant\n\n"
        f"Starting server on http://localhost:{PORT}\n"
        f"API docs at http://localhost:{PORT}/docs",
        title="MedSaathi",
        border_style="cyan",
    ))

    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=True,   # Auto-restart on file changes during development
    )
