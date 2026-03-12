"""
MedSaathi — Embedding Model & Vector Store Access (Shared Utility)
===================================================================

This file provides the embedding model and ChromaDB collection as shared
utilities for the rest of the application. It sits between two stages:

  ingest.py  (runs ONCE)  →  embeddings.py  ←  retriever.py (runs EVERY query)
                                ↑
                          You are here

Why is this separate from ingest.py?
  ingest.py is a one-time batch job — it processes all PDFs and exits.
  But retriever.py needs the embedding model on EVERY user query to convert
  the question into a vector for similarity search. If we loaded the model
  fresh on every query, each request would take ~3-5 extra seconds just for
  model loading. Instead, this file uses a singleton pattern to load once
  and reuse forever.

Why is this separate from retriever.py?
  Multiple files need embeddings — retriever.py for search, potentially
  reranker.py for scoring, evaluator.py for metrics. Centralizing here
  avoids duplicate model instances eating RAM.
"""

import sys

from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from rich.console import Console

# ─── Configuration ────────────────────────────────────────────────────────────

VECTOR_STORE_PATH = "./vector_store"
COLLECTION_NAME = "medsaathi_docs"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

console = Console()

# ─── Singleton: Embedding Model ──────────────────────────────────────────────
#
# Singleton pattern in plain English:
#   "Create the expensive thing once, stash it in a global variable,
#    and hand out the same instance to everyone who asks."
#
# _embedding_model starts as None. The first call to get_embedding_model()
# loads the model (~3-5 sec) and stores it here. Every subsequent call
# sees it's not None and returns instantly. The model stays in memory
# for the lifetime of the server process.

_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return the shared embedding model instance, loading it on first call.

    The model converts text into a 384-dimensional vector of floats.
    Same model is used for both queries and documents — this is what makes
    similarity search work (both live in the same vector space).
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    console.print("[dim]Loading embedding model (first time only)...[/dim]")

    _embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # CPU keeps deployment simple — no CUDA needed
        encode_kwargs={
            "normalize_embeddings": True,
            # normalize_embeddings = True:
            #   Scales every vector to unit length (magnitude = 1.0).
            #   This makes cosine similarity equivalent to a simple dot product,
            #   which is faster to compute. It also ensures that longer documents
            #   don't get artificially higher similarity scores just because
            #   their vectors have larger magnitudes.
        },
    )

    console.print(f"[green]✓[/green] Embedding model loaded: [dim]{EMBEDDING_MODEL}[/dim]")
    return _embedding_model


# ─── Embed a Single Query ────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    """
    Embed a single user query into a vector.

    Query vs Document embedding — what's the difference?
      Technically, the same model and same math. But semantically they serve
      different purposes:
        - embed_query():  used at SEARCH TIME on the user's question.
                          Input is short (a sentence or two).
        - embed_texts():  used at INGESTION TIME on document chunks.
                          Input is longer (paragraphs of medical text).

      Some advanced models (e.g., E5) actually use different prefixes for
      queries vs documents. Our MiniLM model treats them the same, but keeping
      separate functions makes the code's intent clear and future-proof.

    Returns:
        A list of 384 floats — the query's position in semantic space.
    """
    model = get_embedding_model()
    return model.embed_query(text)


# ─── Embed Multiple Texts (Batch) ────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of text strings into vectors.

    Used during ingestion (ingest.py calls its own embedding, but this function
    is available if any other module needs batch embedding — e.g., evaluator.py
    computing similarity scores for RAGAS metrics).

    Batching is more efficient than embedding one-by-one because the model
    can process multiple texts in a single forward pass through the neural network.

    Returns:
        A list of vectors, one per input text. Each vector is 384 floats.
    """
    model = get_embedding_model()
    return model.embed_documents(texts)


# ─── ChromaDB Collection Access ──────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    """
    Connect to ChromaDB and return the medsaathi_docs collection.

    This does NOT load embeddings into memory — ChromaDB uses memory-mapped
    files, so it only reads what's needed for each query. Safe to call
    repeatedly; ChromaDB handles connection pooling internally.
    """
    client = chromadb.PersistentClient(
        path=VECTOR_STORE_PATH,
        settings=Settings(
            anonymized_telemetry=False,  # Don't send usage data to ChromaDB cloud
        ),
    )

    # Check if the collection exists before trying to access it
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing_collections:
        console.print(f"[red]✗ Collection '{COLLECTION_NAME}' not found in {VECTOR_STORE_PATH}[/red]")
        console.print("  [yellow]Run ingest.py first before starting the server.[/yellow]")
        console.print("  [dim]  uv run backend/ingest.py[/dim]")
        sys.exit(1)

    collection = client.get_collection(name=COLLECTION_NAME)
    return collection
