"""
MedSaathi — Document Ingestion Pipeline (Step 1 of RAG)
========================================================

This file is the FIRST step in the RAG (Retrieval-Augmented Generation) pipeline.
Its job: take raw medical PDFs → split them into small chunks → convert each chunk
into a numerical vector (embedding) → store everything in ChromaDB for fast retrieval.

RAG Pipeline Overview:
  1. INGEST (this file)  — PDF → chunks → embeddings → vector DB
  2. RETRIEVE             — user query → find relevant chunks from vector DB
  3. GENERATE             — feed retrieved chunks + query to LLM → answer

Why this matters for MedSaathi:
  Medical documents can be in Hindi, English, or mixed. We use a multilingual
  embedding model so that a Hindi query can match English content and vice versa.
  ChromaDB stores these embeddings locally — no cloud dependency, no API costs
  for the retrieval step.
"""

import os
import sys

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from rich.console import Console
from rich.progress import track

# ─── Configuration ────────────────────────────────────────────────────────────

DOCS_PATH = "./data/docs"
VECTOR_STORE_PATH = "./vector_store"
COLLECTION_NAME = "medsaathi_docs"
CHUNK_SIZE = 800           # ~200 words per chunk — small enough for precise retrieval,
                           # large enough to preserve medical context (symptoms, dosages).
CHUNK_OVERLAP = 100        # Overlap prevents losing info that falls on chunk boundaries.
                           # E.g., "Take 500mg paracetamol" shouldn't get split across chunks.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                           # Multilingual model — supports Hindi + English in one vector space.
                           # MiniLM-L12 is lightweight (120MB) and fast on CPU while still accurate.
BATCH_SIZE = 50            # Process 50 chunks at a time to avoid OOM on low-RAM machines.

console = Console()


# ─── Step 1: Load PDFs ───────────────────────────────────────────────────────

def load_documents():
    """
    Load all PDFs from the data/docs folder.
    Returns a list of Document objects, each containing page text + metadata
    (source filename, page number) so we can cite sources later.
    """
    if not os.path.exists(DOCS_PATH):
        console.print(f"[red]✗ Folder not found:[/red] {DOCS_PATH}")
        console.print("  Create the folder and add your medical PDFs there.")
        sys.exit(1)

    # Check if there are any PDF files in the directory
    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.lower().endswith(".pdf")]
    if not pdf_files:
        console.print(f"[yellow]⚠ No PDF files found in {DOCS_PATH}[/yellow]")
        console.print("  Add your medical PDF documents to this folder and re-run.")
        sys.exit(0)

    console.print(f"\n[bold cyan]Step 1:[/bold cyan] Loading PDFs from [green]{DOCS_PATH}[/green]")

    # DirectoryLoader walks the folder recursively and uses PyPDFLoader for each .pdf
    # PyPDFLoader extracts text page-by-page, keeping page numbers in metadata
    loader = DirectoryLoader(
        path=DOCS_PATH,
        glob="**/*.pdf",            # Recursive — picks up PDFs in subdirectories too
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,    # Faster when there are many PDFs
    )
    documents = loader.load()

    console.print(f"  [green]✓[/green] Loaded [bold]{len(documents)}[/bold] pages from [bold]{len(pdf_files)}[/bold] PDF(s)")
    return documents


# ─── Step 2: Split into Chunks ──────────────────────────────────────────────

def split_documents(documents):
    """
    Split long pages into smaller chunks for better retrieval precision.

    Why chunk at all?
      A full PDF page (~2000-3000 chars) is too broad — if a user asks about
      "paracetamol dosage", we want to retrieve just the relevant paragraph,
      not the entire page about multiple drugs.

    Why RecursiveCharacterTextSplitter?
      It tries splitting at paragraph breaks first (\n\n), then sentences,
      then words — preserving natural text boundaries instead of cutting
      mid-sentence.
    """
    console.print(f"\n[bold cyan]Step 2:[/bold cyan] Splitting documents into chunks")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",   # Paragraph breaks (strongest boundary)
            "\n",      # Line breaks
            "।",       # Hindi full stop (devanagari purna viram) — critical for Hindi PDFs
            ".",        # English full stop
            " ",        # Word boundary (last resort before character split)
            "",         # Character-level split (fallback)
        ],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Print stats so the user can sanity-check the chunking
    avg_size = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
    console.print(f"  [green]✓[/green] Created [bold]{len(chunks)}[/bold] chunks (avg size: [bold]{avg_size:.0f}[/bold] chars)")

    return chunks


# ─── Step 3: Embed and Store in ChromaDB ─────────────────────────────────────

def embed_and_store(chunks):
    """
    Convert each text chunk into an embedding vector and store in ChromaDB.

    What is an embedding?
      An embedding is a list of numbers (e.g., 384 floats) that captures the
      MEANING of a text. Similar texts get similar numbers, so "headache medicine"
      and "सिरदर्द की दवा" end up close together in this number space — even though
      they're in different languages. This is how we find relevant chunks later:
      convert the user's question to numbers, then find stored chunks with the
      closest numbers (cosine similarity).

    Why cosine similarity?
      It measures the angle between two vectors, ignoring magnitude. This means
      "headache" and "HEADACHE" (same direction, different length) score as identical.
      More robust than euclidean distance for text embeddings.
    """
    console.print(f"\n[bold cyan]Step 3:[/bold cyan] Embedding chunks and storing in ChromaDB")

    # ── Initialize the embedding model ──
    # This runs locally on CPU — no API key needed, no data leaves the machine.
    # normalize_embeddings=True makes cosine similarity work correctly.
    console.print(f"  Loading embedding model: [dim]{EMBEDDING_MODEL}[/dim]")
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},         # Use CPU — works everywhere, GPU optional
        encode_kwargs={"normalize_embeddings": True},  # Required for cosine similarity
    )

    # ── Initialize ChromaDB with persistent storage ──
    # PersistentClient saves to disk so embeddings survive restarts.
    # No need to re-embed every time we start the server.
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    # Delete existing collection if present — ensures a clean, fresh ingestion.
    # This avoids duplicate chunks if the user re-runs ingestion after adding new PDFs.
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        console.print(f"  [yellow]↻[/yellow] Deleted existing collection '{COLLECTION_NAME}' for fresh ingestion")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Use cosine distance for similarity search
    )

    # ── Process in batches to avoid memory issues ──
    # Embedding 1000+ chunks at once can OOM on 8GB RAM machines.
    # Batching keeps peak memory usage low.
    console.print(f"  Embedding and storing {len(chunks)} chunks in batches of {BATCH_SIZE}...")

    for i in track(range(0, len(chunks), BATCH_SIZE), description="  Ingesting..."):
        batch = chunks[i : i + BATCH_SIZE]

        # Extract text and metadata from langchain Document objects
        texts = [chunk.page_content for chunk in batch]
        metadatas = [chunk.metadata for chunk in batch]

        # Generate unique IDs for each chunk (required by ChromaDB)
        ids = [f"chunk_{i + j}" for j, _ in enumerate(batch)]

        # Embed the batch — this is the CPU-intensive step
        embeddings = embedding_function.embed_documents(texts)

        # Store in ChromaDB: text + embedding + metadata together
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    console.print(f"  [green]✓[/green] Stored [bold]{len(chunks)}[/bold] chunks in ChromaDB at [green]{VECTOR_STORE_PATH}[/green]")
    console.print(f"  Collection: [bold]{COLLECTION_NAME}[/bold] | Similarity: cosine")


# ─── Main ────────────────────────────────────────────────────────────────────

def run_ingestion():
    """Run the full ingestion pipeline: load → split → embed → store."""
    console.print("[bold magenta]═══ MedSaathi Document Ingestion ═══[/bold magenta]")

    documents = load_documents()
    chunks = split_documents(documents)
    embed_and_store(chunks)

    console.print("\n[bold green]✓ Ingestion complete![/bold green] Run main.py to start the server.")


if __name__ == "__main__":
    run_ingestion()
