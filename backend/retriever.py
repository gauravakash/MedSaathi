"""
MedSaathi — Hybrid Retriever (Step 2 of RAG)
==============================================

This file is the SECOND step in the RAG pipeline. Given a user's question,
it finds the most relevant document chunks from the vector store.

  1. INGEST   (ingest.py)     — PDF → chunks → embeddings → ChromaDB
  2. RETRIEVE (this file)     — user query → find best chunks
  3. GENERATE (generator.py)  — best chunks + query → LLM → answer

Why HYBRID search? Two search strategies, one problem:
─────────────────────────────────────────────────────────
  Semantic search (embeddings):
    ✓ Great for meaning — "fever reducer" matches "antipyretic medication"
    ✗ Bad for exact terms — "Paracetamol 500mg" might match "Ibuprofen 400mg"
      because they're semantically similar (both are painkillers)

  Keyword search (BM25):
    ✓ Great for exact matches — "Paracetamol 500mg" finds exactly that
    ✗ Bad for meaning — "sir dard ki dawa" won't match "headache medicine"

  Hybrid search combines both: if a chunk scores high on BOTH semantic AND
  keyword search, it's almost certainly the right result. This is especially
  important in medical contexts where drug names, dosages, and technical
  terms must match exactly, but symptoms are described in natural language.

This file imports from:
  - embeddings.py — for embed_query() and get_collection()
  - rank_bm25 — for BM25Okapi keyword search
  - rich — for pretty console output
"""

from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

from backend.embeddings import embed_query, get_collection

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_N_RESULTS = 5

console = Console()


# ─── Semantic Search (Embedding-based) ────────────────────────────────────────

def semantic_search(query: str, n_results: int = DEFAULT_N_RESULTS) -> list[dict]:
    """
    Find chunks whose MEANING is closest to the query using vector similarity.

    How it works:
      1. Convert the query text into a 384-float vector (embedding)
      2. Ask ChromaDB to find stored chunks with the closest vectors
      3. "Closest" = smallest cosine distance

    What is "embeddings distance"?
      Think of each text as a point in 384-dimensional space. Distance is how
      far apart two points are. Cosine distance specifically measures the angle
      between two vectors:
        - 0.0 = identical meaning (vectors point in same direction)
        - 1.0 = completely unrelated (vectors are perpendicular)
        - 2.0 = opposite meaning (vectors point opposite ways)
      So LOWER distance = MORE similar. This is the opposite of a "score"
      where higher is better — we handle this in the result formatting.
    """
    collection = get_collection()

    # Embed the query into the same vector space as the stored documents
    query_embedding = embed_query(query)

    # ChromaDB returns results sorted by distance (lowest = most similar)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Reshape ChromaDB's nested list format into clean dicts
    # ChromaDB returns lists-of-lists because it supports batched queries,
    # but we only query one at a time, so we unwrap the outer list [0].
    formatted = []
    for i in range(len(results["documents"][0])):
        metadata = results["metadatas"][0][i]
        formatted.append({
            "text": results["documents"][0][i],
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", -1),
            "score": results["distances"][0][i],  # Lower = more similar
        })

    return formatted


# ─── BM25 Keyword Search ─────────────────────────────────────────────────────

def bm25_search(query: str, n_results: int = DEFAULT_N_RESULTS) -> list[dict]:
    """
    Find chunks that contain the same KEYWORDS as the query using BM25.

    Why BM25 alongside semantic search?
      Drug names are the #1 reason. Consider:
        Query: "Paracetamol 500mg side effects"
        Semantic search might return chunks about Ibuprofen or Aspirin because
        they're all painkillers — semantically similar but medically WRONG.
        BM25 scores based on exact word overlap, so "Paracetamol" in the query
        will strongly match chunks containing "Paracetamol" — not "Ibuprofen".

      Same applies to: dosages ("500mg" vs "200mg"), lab values ("HbA1c"),
      Hindi drug names, and medical abbreviations.

    BM25Okapi is a proven algorithm from information retrieval (used by
    Elasticsearch under the hood). It scores based on:
      - Term frequency: how often the query word appears in a chunk
      - Inverse document frequency: rare words matter more than common ones
      - Document length normalization: long chunks don't get unfair advantage
    """
    collection = get_collection()

    # Load ALL documents from ChromaDB for BM25 indexing
    # BM25 needs the full corpus to compute term frequencies — unlike semantic
    # search which only needs the query vector. This is a tradeoff:
    # more memory, but accurate keyword matching.
    all_docs = collection.get(include=["documents", "metadatas"])
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    if not documents:
        return []

    # Tokenize: split on whitespace and lowercase for case-insensitive matching
    # This is intentionally simple — BM25 handles the statistical heavy lifting.
    # For Hindi text, whitespace tokenization works well since Hindi words are
    # space-separated (unlike Chinese/Japanese which need special tokenizers).
    tokenized_corpus = [doc.lower().split() for doc in documents]
    tokenized_query = query.lower().split()

    # Build BM25 index over the full corpus
    bm25 = BM25Okapi(tokenized_corpus)

    # Get BM25 scores for every document against this query
    scores = bm25.get_scores(tokenized_query)

    # Get indices of top n_results, sorted by score (highest first)
    # argsort returns ascending order, so we reverse with [::-1]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    formatted = []
    for idx in top_indices:
        metadata = metadatas[idx]
        formatted.append({
            "text": documents[idx],
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", -1),
            "score": float(scores[idx]),  # Higher = more relevant (opposite of semantic!)
        })

    return formatted


# ─── Reciprocal Rank Fusion (RRF) ────────────────────────────────────────────

def _reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge two ranked result lists into one using Reciprocal Rank Fusion.

    RRF in simple terms:
      Each result gets a score based on its RANK (position), not its raw score.
      Formula: rrf_score = 1 / (rank + k)

      Example with k=60:
        Rank 1 → 1/61 = 0.0164
        Rank 2 → 1/62 = 0.0161
        Rank 5 → 1/65 = 0.0154

      If the same chunk appears in BOTH semantic and BM25 results, its
      RRF scores are ADDED together. This is the key insight:
        - A chunk ranked #1 in semantic + #3 in BM25 = 1/61 + 1/63 = 0.0323
        - A chunk ranked #1 in semantic only = 1/61 = 0.0164
      The chunk found by BOTH methods gets roughly 2x the score.

    Why k=60?
      The constant k controls how much rank position matters. With k=60:
        - The difference between rank 1 and rank 2 is small (0.0164 vs 0.0161)
        - This prevents a single high-ranking result from dominating
        - k=60 is the value from the original RRF paper (Cormack et al., 2009)
          and has become the standard default
      Lower k (e.g., 1) would heavily favor top-ranked results.
      Higher k (e.g., 1000) would make all ranks nearly equal.

    Args:
        semantic_results: Ranked results from embedding-based search
        bm25_results: Ranked results from keyword-based search
        k: RRF constant (default 60, from the original paper)

    Returns:
        Merged and deduplicated results, sorted by combined RRF score
    """
    # Use chunk text as the deduplication key.
    # We store the full result dict alongside the accumulated score.
    fused_scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}

    # Score semantic results by rank
    for rank, result in enumerate(semantic_results, start=1):
        text = result["text"]
        rrf_score = 1.0 / (rank + k)
        fused_scores[text] = fused_scores.get(text, 0.0) + rrf_score
        result_map[text] = result  # Store the result dict for later

    # Score BM25 results by rank — if a chunk already appeared in semantic
    # results, its score gets ADDED (this is where fusion happens)
    for rank, result in enumerate(bm25_results, start=1):
        text = result["text"]
        rrf_score = 1.0 / (rank + k)
        fused_scores[text] = fused_scores.get(text, 0.0) + rrf_score
        if text not in result_map:
            result_map[text] = result

    # Sort by combined RRF score (highest first) and attach the score
    sorted_texts = sorted(fused_scores.keys(), key=lambda t: fused_scores[t], reverse=True)

    fused_results = []
    for text in sorted_texts:
        result = result_map[text].copy()
        result["score"] = fused_scores[text]  # Replace original score with RRF score
        fused_results.append(result)

    return fused_results


# ─── Hybrid Search (Main Entry Point) ────────────────────────────────────────

def hybrid_search(query: str, n_results: int = DEFAULT_N_RESULTS) -> list[dict]:
    """
    The main retrieval function — combines semantic + keyword search via RRF.

    This is what the rest of the pipeline should call. It runs both search
    strategies, fuses the results, and returns the top chunks.

    Why not just pick the better search method?
      Because we don't know in advance which method will work better for a
      given query. A symptom description benefits from semantic search;
      a drug name benefits from BM25. Hybrid search handles both cases
      without needing to classify the query type first.
    """
    console.print(f"[dim]Searching for:[/dim] {query}")

    # Run both search strategies with the same n_results
    # We fetch n_results from each, then RRF picks the best n_results overall
    semantic_results = semantic_search(query, n_results=n_results)
    bm25_results = bm25_search(query, n_results=n_results)

    # Fuse results using Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(semantic_results, bm25_results)

    # Return only the top n_results after fusion
    return fused[:n_results]


# ─── Pretty Print Helper (for debugging) ─────────────────────────────────────

def print_results(results: list[dict]) -> None:
    """Display retrieval results in a clean table for debugging."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="Retrieval Results", show_lines=True)
    table.add_column("Rank", style="bold cyan", width=5)
    table.add_column("Source", style="green", width=25)
    table.add_column("Page", style="yellow", width=5)
    table.add_column("Score", style="magenta", width=10)
    table.add_column("Text Preview", style="white")

    for i, result in enumerate(results, start=1):
        # Truncate text to 150 chars for display, preserving word boundaries
        text = result["text"]
        preview = text[:150] + "..." if len(text) > 150 else text
        # Clean up whitespace for display
        preview = " ".join(preview.split())

        table.add_row(
            str(i),
            str(result["source"]),
            str(result["page"]),
            f"{result['score']:.4f}",
            preview,
        )

    console.print(table)


# ─── Test Block ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold magenta]═══ MedSaathi Retriever Test ═══[/bold magenta]\n")

    test_query = "paracetamol dosage for fever"
    console.print(f"Test query: [bold]{test_query}[/bold]\n")

    # Test semantic search alone
    console.print("[bold cyan]── Semantic Search Results ──[/bold cyan]")
    sem_results = semantic_search(test_query)
    print_results(sem_results)

    # Test BM25 search alone
    console.print("\n[bold cyan]── BM25 Keyword Search Results ──[/bold cyan]")
    bm25_results = bm25_search(test_query)
    print_results(bm25_results)

    # Test hybrid search (the main function)
    console.print("\n[bold cyan]── Hybrid Search Results (RRF Fusion) ──[/bold cyan]")
    hybrid_results = hybrid_search(test_query)
    print_results(hybrid_results)

    console.print("\n[green]✓[/green] Retriever test complete.")
