"""
MedSaathi — Query Rewriter (Step 1.5 of RAG)
==============================================

This file sits BEFORE the retriever — it transforms the user's raw query
into a better search query before it hits the vector store.

  User types: "mere bachche ko raat mein khansi aa rahi hai"
  Retriever sees: "pediatric nocturnal cough treatment children"

  1. INGEST     (ingest.py)         — PDF → chunks → ChromaDB
  1.5 REWRITE  (this file)         — raw query → optimized search query
  2. RETRIEVE   (retriever.py)      — search query → top chunks
  2.5 RERANK   (reranker.py)       — top chunks → reorder by precision
  3. GENERATE   (generator.py)      — best chunks + query → LLM → answer

Why does query rewriting dramatically improve RAG quality?
──────────────────────────────────────────────────────────
  The problem: medical PDFs are typically in clinical English, but Indian
  users ask questions in Hindi, Hinglish, or casual English. There's a
  vocabulary gap between how people ASK and how documents are WRITTEN.

  Example:
    User says: "pet mein dard hai aur ulti aa rahi hai"
    PDF contains: "abdominal pain accompanied by nausea and vomiting..."

  Even our multilingual embedding model struggles with this gap because
  casual Hindi phrasing and clinical English terminology are far apart in
  semantic space. Query rewriting bridges this gap by converting the query
  into terminology that matches what's actually in the documents.

  In benchmarks, query rewriting typically improves retrieval recall by
  20-40% — meaning we find the RIGHT chunks much more often.

Two strategies in this file:
────────────────────────────
  1. Rewriting = one BETTER version of the same query
     "pet dard" → "abdominal pain causes treatment"
     Same intent, better vocabulary for retrieval.

  2. Expansion = multiple ANGLES on the same query
     "abdominal pain" → ["stomach ache causes", "abdominal cramps diagnosis",
                          "gastric pain treatment options"]
     Different phrasings catch chunks that one query might miss.
     (Expansion is optional — not all pipelines use it, but it helps.)
"""

import ast
import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel

# ─── Configuration ────────────────────────────────────────────────────────────

REWRITER_MODEL = "gemini-1.5-flash"
REWRITER_TEMPERATURE = 0   # Deterministic — we want consistent, predictable rewrites.
                            # Temperature=0 means the model always picks the most likely
                            # token. For rewriting, creativity is harmful — we want the
                            # same input to always produce the same optimized query.
REWRITER_MAX_TOKENS = 200   # Rewrites should be short (under 15 words). 200 tokens
                            # gives plenty of room without allowing rambling.

# Load environment variables from .env (GEMINI_API_KEY)
load_dotenv()

console = Console()

# ─── System Prompts ───────────────────────────────────────────────────────────

REWRITE_SYSTEM_PROMPT = (
    "You are a medical query rewriter for an Indian health assistant.\n"
    "Your job is to convert conversational queries (in Hindi, English,\n"
    "or Hinglish) into precise medical search queries in English.\n"
    "\n"
    "Rules:\n"
    "- Output ONLY the rewritten query, nothing else\n"
    "- Keep it under 15 words\n"
    "- Use medical terminology where appropriate\n"
    "- Preserve specific drug names, dosages, or symptoms exactly\n"
    "- If query is already clear English, improve it slightly\n"
    "\n"
    "Examples:\n"
    "Input: 'pet mein dard hai aur ulti aa rahi hai'\n"
    "Output: 'abdominal pain nausea vomiting causes treatment'\n"
    "\n"
    "Input: 'sugar ki bimari mein kya khana chahiye'\n"
    "Output: 'diabetes diet recommendations food to eat avoid'\n"
    "\n"
    "Input: 'paracetamol 500mg dose for child'\n"
    "Output: 'paracetamol acetaminophen pediatric dosage 500mg'"
)

EXPANSION_SYSTEM_PROMPT = (
    "Generate 3 alternative search queries for this medical question.\n"
    "Each should approach the topic differently.\n"
    "Output ONLY a Python list of 3 strings, nothing else.\n"
    "Example output: ['query one', 'query two', 'query three']"
)


# ─── Singleton: Gemini Flash LLM ─────────────────────────────────────────────
#
# Same singleton pattern as embeddings.py and reranker.py.
# Gemini Flash is lightweight and fast, but the langchain wrapper still has
# setup overhead (API key validation, client init). Loading once avoids
# repeating this on every query.

_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return the shared Gemini Flash instance, initializing on first call.

    We use Gemini Flash (not Pro) because:
      - Query rewriting is a simple task — doesn't need the smartest model
      - Flash is ~5x faster and ~10x cheaper than Pro
      - Latency matters here — this runs on EVERY user query before retrieval
    """
    global _llm

    if _llm is not None:
        return _llm

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        console.print("[red]✗ GEMINI_API_KEY not set in .env file[/red]")
        console.print("  Get your free key at: [dim]https://aistudio.google.com/apikey[/dim]")
        console.print("  Then add it to .env: [dim]GEMINI_API_KEY=your_actual_key[/dim]")
        # Don't sys.exit — query rewriting is optional, the pipeline can work without it
        raise ValueError("GEMINI_API_KEY not configured")

    console.print("[dim]Initializing Gemini Flash for query rewriting...[/dim]")

    _llm = ChatGoogleGenerativeAI(
        model=REWRITER_MODEL,
        google_api_key=api_key,
        temperature=REWRITER_TEMPERATURE,
        max_output_tokens=REWRITER_MAX_TOKENS,
    )

    console.print(f"[green]✓[/green] Query rewriter ready: [dim]{REWRITER_MODEL}[/dim]")
    return _llm


# ─── Language Detection ───────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect whether input text is Hindi, English, or Hinglish.

    Uses a simple Unicode heuristic — no ML model needed:
      - Hindi characters live in Unicode range U+0900 to U+097F (Devanagari block)
      - If we find Devanagari characters, Hindi is present
      - If we also find ASCII letters (a-z), it's Hinglish (mixed)

    This is used later by generator.py to respond in the user's language.
    A user who asks in Hindi expects an answer in Hindi.

    Returns:
        "hindi", "english", or "hinglish"
    """
    # Check for Devanagari characters (Hindi, Marathi, Sanskrit, etc.)
    # Unicode range: \u0900-\u097F covers the entire Devanagari block
    has_hindi = bool(re.search(r"[\u0900-\u097F]", text))

    # Check for ASCII Latin letters (English)
    has_english = bool(re.search(r"[a-zA-Z]", text))

    if has_hindi and has_english:
        return "hinglish"  # Mixed — e.g., "mujhe diabetes hai, what should I eat?"
    elif has_hindi:
        return "hindi"     # Pure Hindi — e.g., "बुखार में क्या दवा लें"
    else:
        return "english"   # Default — includes pure English and romanized Hindi
                           # Note: romanized Hindi ("bukhar mein dawa") is detected as
                           # "english" since it uses Latin letters. This is fine because
                           # our rewriter handles romanized Hindi in its prompt.


# ─── Query Rewriting ─────────────────────────────────────────────────────────

def rewrite_query(query: str) -> str:
    """
    Rewrite a raw user query into an optimized medical search query.

    This is the core function — takes casual/Hindi input and produces
    clinical English output that matches PDF terminology.

    If Gemini fails for ANY reason (API down, rate limit, bad key), we
    silently return the original query. Query rewriting is an enhancement,
    not a requirement — the pipeline must never crash because of it.
    """
    try:
        llm = get_llm()

        response = llm.invoke([
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])

        # Extract the text and clean it up
        rewritten = response.content.strip()

        # Sanity check — if the LLM returned something empty or way too long,
        # fall back to the original query
        if not rewritten or len(rewritten) > 500:
            return query

        return rewritten

    except Exception as e:
        # Query rewriting failure should NEVER crash the app.
        # Log it and return the original query — retrieval will still work,
        # just slightly less optimized.
        console.print(f"[yellow]⚠ Query rewrite failed ({type(e).__name__}), using original query[/yellow]")
        return query


# ─── Query Expansion ──────────────────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    """
    Generate 3 alternative phrasings of the query for multi-angle retrieval.

    Why expand?
      A single query might miss chunks that use different terminology.
      "pediatric cough" might not match a chunk about "children's respiratory
      infection." By searching with 3 variations, we cast a wider net.

    The expansions are generated from the REWRITTEN query (not the raw query)
    so they're already in clinical English.
    """
    try:
        llm = get_llm()

        response = llm.invoke([
            SystemMessage(content=EXPANSION_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])

        raw = response.content.strip()

        # Parse the LLM's output as a Python list
        # ast.literal_eval safely parses string representations of Python literals
        # without executing arbitrary code (unlike eval()).
        expansions = ast.literal_eval(raw)

        # Validate: must be a list of strings
        if isinstance(expansions, list) and all(isinstance(q, str) for q in expansions):
            return expansions

        return [query]

    except Exception as e:
        # Expansion failure is even less critical than rewriting failure.
        # Just return the original query in a list — pipeline continues normally.
        console.print(f"[yellow]⚠ Query expansion failed ({type(e).__name__}), using original[/yellow]")
        return [query]


# ─── Combined: Process Query ─────────────────────────────────────────────────

def process_query(raw_query: str) -> dict:
    """
    Full query processing pipeline: detect language → rewrite → expand.

    Returns a dict with all versions of the query so downstream components
    can pick what they need:
      - "original":   what the user typed (kept for chat display)
      - "rewritten":  optimized English version (used by retriever)
      - "expansions": alternative phrasings (optional, for multi-query retrieval)
      - "best":       the single best query for retrieval (= rewritten)
      - "language":   detected language (used by generator for response language)
    """
    language = detect_language(raw_query)

    console.print(f"\n[bold cyan]Query Processing:[/bold cyan]")
    console.print(f"  Original  → [dim]{raw_query}[/dim]")
    console.print(f"  Language  → [dim]{language}[/dim]")

    rewritten = rewrite_query(raw_query)
    console.print(f"  Rewritten → [green]{rewritten}[/green]")

    expansions = expand_query(rewritten)
    console.print(f"  Expansions → [dim]{expansions}[/dim]")

    return {
        "original": raw_query,
        "rewritten": rewritten,
        "expansions": expansions,
        "best": rewritten,       # The retriever will use this for search
        "language": language,
    }


# ─── Test Block ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold magenta]═══ MedSaathi Query Rewriter Test ═══[/bold magenta]\n")

    test_queries = [
        "bukhar aur sardard hai",                                         # Romanized Hindi
        "mujhe diabetes hai, what should I eat?",                         # Hinglish
        "what is the dosage of amoxicillin for throat infection",          # English
    ]

    for query in test_queries:
        console.print(Panel(f"[bold]Input:[/bold] {query}", expand=False))

        lang = detect_language(query)
        console.print(f"  Detected language: [cyan]{lang}[/cyan]")

        result = process_query(query)
        console.print(f"  Best query for retriever: [bold green]{result['best']}[/bold green]")
        console.print()

    console.print("[green]✓[/green] Query rewriter test complete.")
