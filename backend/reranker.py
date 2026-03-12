"""
MedSaathi — Cross-Encoder Reranker (Step 2.5 of RAG)
======================================================

This file sits between retrieval and generation — it takes the rough candidates
from hybrid search and reranks them with much higher precision.

  1. INGEST     (ingest.py)     — PDF → chunks → embeddings → ChromaDB
  2. RETRIEVE   (retriever.py)  — user query → top 10 candidates (fast, rough)
  2.5 RERANK   (this file)     — top 10 → reorder → top 5 (slow, precise)
  3. GENERATE   (generator.py)  — top 5 chunks + query → LLM → answer

Two-stage retrieval pattern:
────────────────────────────
  Stage 1 (retriever.py): Cast a WIDE net — get top 10 candidates quickly
  Stage 2 (this file):    Examine each candidate CAREFULLY — keep the best 5

  Why two stages instead of one?
    The cross-encoder is ~100x slower than bi-encoder retrieval. Scoring all
    10,000+ chunks with a cross-encoder would take minutes per query.
    But scoring just 10 candidates takes ~0.1 seconds. So we use the fast
    method to narrow down, then the accurate method to finalize.

Bi-encoder vs Cross-encoder — the dating analogy:
──────────────────────────────────────────────────
  Bi-encoder (retriever.py) is like speed-dating:
    You form a quick impression of each person independently. You describe
    yourself in a few words (query embedding), they describe themselves
    (document embedding), and you compare notes. Fast — you can evaluate
    100 people in minutes. But you might miss a great match because the
    descriptions are too compressed.

  Cross-encoder (this file) is like a real date:
    You sit down with the person and have a full conversation. The model
    reads your query AND the document TOGETHER, word by word, with full
    attention between them. Much slower — you can only do a few per evening.
    But the judgment is far more accurate because it sees the interaction
    between query and document, not just their separate summaries.

  In technical terms:
    - Bi-encoder: encode(query) separately, encode(doc) separately → compare
    - Cross-encoder: encode(query + doc) together → single relevance score
"""

from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.table import Table

# ─── Configuration ────────────────────────────────────────────────────────────

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_N = 5
MIN_RERANK_SCORE = -5.0
# Cross-encoder score ranges (for ms-marco-MiniLM-L-6-v2):
#   +10 to +5  → Highly relevant — the chunk directly answers the query
#    +5 to  0  → Somewhat relevant — related topic, may contain useful info
#     0 to -5  → Weakly relevant — tangentially related at best
#    -5 to -10 → Irrelevant — the chunk has nothing to do with the query
# We filter out anything below -5.0 to avoid sending noise to the LLM.

console = Console()


# ─── Singleton: Cross-Encoder Model ──────────────────────────────────────────
#
# Same singleton pattern as embeddings.py — load once, reuse on every query.
# The cross-encoder is ~80MB and takes ~2-3 seconds to load. Without the
# singleton, every user query would pay this cost.

_reranker_model = None


def get_reranker_model() -> CrossEncoder:
    """
    Return the shared cross-encoder model instance, loading it on first call.

    The model takes a (query, document) pair and outputs a single relevance
    score. Unlike the embedding model which produces vectors, this model
    directly outputs "how relevant is this document to this query?"
    """
    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    console.print("[dim]Loading reranker model (first time only)...[/dim]")

    _reranker_model = CrossEncoder(
        model_name=RERANKER_MODEL,
        # ms-marco-MiniLM-L-6-v2: trained on the MS MARCO passage ranking dataset
        # (millions of real search queries + relevant/irrelevant passages).
        # 6 layers, ~80MB — the smallest accurate cross-encoder available.
        # Runs comfortably on CPU in <100ms for 10 pairs.
    )

    console.print(f"[green]✓[/green] Reranker model loaded: [dim]{RERANKER_MODEL}[/dim]")
    return _reranker_model


# ─── Rerank Function ─────────────────────────────────────────────────────────

def rerank(query: str, chunks: list[dict], top_n: int = DEFAULT_TOP_N) -> list[dict]:
    """
    Rerank retrieved chunks using cross-encoder for higher precision.

    Takes the rough candidates from hybrid search and reorders them based on
    true query-document relevance. Often dramatically changes the ranking —
    a chunk at position #7 from hybrid search might jump to #1 after reranking
    because the cross-encoder understands the nuanced relationship between
    the query and that specific chunk.

    Args:
        query: The user's question
        chunks: List of chunk dicts from retriever.py (each has text, source, page, score)
        top_n: Number of top results to return after reranking

    Returns:
        Top-n chunks sorted by rerank_score, with both original and new scores
    """
    if not chunks:
        return []

    model = get_reranker_model()

    # Create (query, document) pairs for the cross-encoder.
    # The model needs to see BOTH texts together to judge relevance — this is
    # what makes it more accurate than bi-encoders which encode them separately.
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs in one batch — much faster than scoring one at a time
    # because the model can process multiple pairs in a single forward pass.
    scores = model.predict(pairs)

    # Attach rerank_score to each chunk, keeping the original hybrid score too.
    # This lets us compare before/after to see how reranking changed things.
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        enriched = chunk.copy()
        enriched["rerank_score"] = float(score)
        scored_chunks.append(enriched)

    # Sort by rerank_score descending — highest relevance first
    scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

    # ── Filter low-confidence results ──
    # Chunks scoring below MIN_RERANK_SCORE are almost certainly irrelevant.
    # Sending irrelevant chunks to the LLM causes hallucination — the model
    # tries to "use" the context even when it doesn't match the question.
    filtered = [c for c in scored_chunks if c["rerank_score"] >= MIN_RERANK_SCORE]

    # Safety net: if ALL chunks were filtered out, keep the top 2 anyway.
    # It's better to give the LLM some context (even weak) than none at all —
    # with no context, the LLM has nothing to ground its answer on and will
    # either refuse to answer or hallucinate entirely.
    if not filtered:
        filtered = scored_chunks[:2]
        console.print("[yellow]⚠ All chunks scored below threshold — keeping top 2 as fallback[/yellow]")

    return filtered[:top_n]


# ─── Pretty Print Helper ─────────────────────────────────────────────────────

def print_reranked(results: list[dict]) -> None:
    """
    Display reranked results showing both original and new scores.
    The side-by-side comparison makes it easy to see how reranking
    reshuffled the results.
    """
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    table = Table(title="Reranked Results", show_lines=True)
    table.add_column("Rank", style="bold cyan", width=5)
    table.add_column("Rerank Score", style="bold magenta", width=13)
    table.add_column("Hybrid Score", style="dim", width=13)
    table.add_column("Source", style="green", width=25)
    table.add_column("Page", style="yellow", width=5)
    table.add_column("Text Preview", style="white")

    for i, result in enumerate(results, start=1):
        text = result["text"]
        preview = text[:150] + "..." if len(text) > 150 else text
        preview = " ".join(preview.split())

        # Show both scores: rerank_score (new) and score (original from hybrid search)
        rerank_str = f"{result['rerank_score']:.4f}"
        hybrid_str = f"{result.get('score', 0):.4f}"

        table.add_row(
            str(i),
            rerank_str,
            hybrid_str,
            str(result["source"]),
            str(result["page"]),
            preview,
        )

    console.print(table)


# ─── Test Block ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from backend.retriever import hybrid_search, print_results

    console.print("[bold magenta]═══ MedSaathi Reranker Test ═══[/bold magenta]\n")

    # Hindi query: "what medicine should I take for fever?"
    test_query = "bukhar mein kya dawa leni chahiye"
    console.print(f"Test query: [bold]{test_query}[/bold]\n")

    # Stage 1: Get top 10 from hybrid search (fast, rough ranking)
    console.print("[bold cyan]── Before Reranking (Hybrid Search, top 10) ──[/bold cyan]")
    candidates = hybrid_search(test_query, n_results=10)
    print_results(candidates)

    # Stage 2: Rerank to top 5 (slow, precise ranking)
    console.print("\n[bold cyan]── After Reranking (Cross-Encoder, top 5) ──[/bold cyan]")
    reranked = rerank(test_query, candidates, top_n=5)
    print_reranked(reranked)

    console.print("\n[green]✓[/green] Reranker test complete.")
    console.print("[dim]Compare the two tables — notice how the order changed![/dim]")
