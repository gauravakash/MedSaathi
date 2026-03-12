"""
MedSaathi — Grounded Response Generator (Step 3 of RAG)
=========================================================

This is where the RAG pipeline COMPLETES. All previous steps feed into this:

  1. INGEST     (ingest.py)         — PDF → chunks → ChromaDB
  1.5 REWRITE  (query_rewriter.py) — raw query → optimized search query
  2. RETRIEVE   (retriever.py)      — search query → top 10 candidates
  2.5 RERANK   (reranker.py)       — top 10 → best 5 chunks
  3. GENERATE  (this file)         — best chunks + query → LLM → answer
                                      ↑ You are here

What is "grounded generation"?
──────────────────────────────
  Regular LLM generation: "Hey Gemini, what's the dosage for paracetamol?"
    → LLM answers from its training data. Might be correct, might be outdated,
      might be completely made up. You can't verify, you can't cite sources.
    → This is dangerous for medical information.

  Grounded generation (what we do):
    "Here are 5 specific paragraphs from WHO guidelines and AYUSH manuals.
     Answer this question ONLY using these paragraphs."
    → LLM can only use what we give it. If the answer isn't in the chunks,
      it says "I don't have this information." Every claim can be traced
      back to a specific PDF page. This is verifiable and safe.

What is hallucination?
──────────────────────
  When an LLM confidently generates information that isn't true. In medical
  contexts, this could mean inventing drug dosages, contraindications, or
  treatments that don't exist. Grounding prevents this by constraining the
  LLM to ONLY reference the provided context. If it tries to go beyond the
  context, the system prompt explicitly tells it to say "I don't know."

Why temperature=0.3 here but 0 in query_rewriter.py?
─────────────────────────────────────────────────────
  Query rewriting needs CONSISTENCY — same input should always produce
  the same optimized query (temperature=0, fully deterministic).

  Response generation needs NATURALNESS — we want warm, human-sounding
  responses in Hindi/English. A tiny bit of temperature (0.3) allows the
  LLM to vary word choice and sentence structure without becoming unreliable.
  Higher values (0.7+) would risk creative "additions" to medical facts.

Why we always add a medical disclaimer:
───────────────────────────────────────
  MedSaathi serves rural patients who may not have easy access to doctors.
  Even with grounded generation, we are NOT a replacement for medical
  professionals. The disclaimer "Yeh sirf jaankari hai, doctor ki salah
  zaroor lein" ensures users know to seek real medical help.
"""

import os
from collections.abc import Generator as GenType

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console

# ─── Configuration ────────────────────────────────────────────────────────────

GENERATOR_MODEL = "gemini-1.5-flash"
GENERATOR_TEMPERATURE = 0.3   # Slight creativity for natural-sounding responses,
                               # low enough to stay factually grounded.
GENERATOR_MAX_TOKENS = 1000    # Medical answers can be detailed — allow up to ~250 words.

FALLBACK_RESPONSE_HINDI = (
    "Maafi chahta hoon, abhi jawab dene mein dikkat aa rahi hai. "
    "Kripya thodi der baad try karein ya doctor se milein."
)

DISCLAIMER = "Yeh sirf jaankari hai, doctor ki salah zaroor lein."

load_dotenv()

console = Console()


# ─── Singleton: Generator LLM ────────────────────────────────────────────────
#
# We do NOT reuse get_llm() from query_rewriter.py because that instance
# is configured for rewriting (temperature=0, max_tokens=200). Generation
# needs different settings (temperature=0.3, max_tokens=1000).
# Two singletons, two configurations, same underlying Gemini Flash model.

_generator_llm = None


def get_generator_llm() -> ChatGoogleGenerativeAI:
    """
    Return the shared Gemini Flash instance configured for generation.

    Separate from the rewriter LLM because generation needs:
      - Higher temperature (0.3 vs 0) for natural responses
      - More tokens (1000 vs 200) for detailed medical answers
    """
    global _generator_llm

    if _generator_llm is not None:
        return _generator_llm

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        console.print("[red]✗ GEMINI_API_KEY not set in .env file[/red]")
        console.print("  Get your free key at: [dim]https://aistudio.google.com/apikey[/dim]")
        raise ValueError("GEMINI_API_KEY not configured")

    console.print("[dim]Initializing Gemini Flash for response generation...[/dim]")

    _generator_llm = ChatGoogleGenerativeAI(
        model=GENERATOR_MODEL,
        google_api_key=api_key,
        temperature=GENERATOR_TEMPERATURE,
        max_output_tokens=GENERATOR_MAX_TOKENS,
    )

    console.print(f"[green]✓[/green] Generator ready: [dim]{GENERATOR_MODEL}[/dim]")
    return _generator_llm


# ─── Build Context from Chunks ───────────────────────────────────────────────

def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Format reranked chunks into a numbered, citable context string.

    Input: list of chunk dicts from reranker.py
    Output: (context_string, sources_list)

    The context string looks like:
      [Source 1 — WHO_guidelines.pdf, Page 12]
      Paracetamol is used for mild to moderate pain and fever...

      [Source 2 — AYUSH_manual.pdf, Page 45]
      For children under 12, dosage should not exceed...

    The sources list tracks which files/pages were used so we can
    show citations below the answer in the UI.
    """
    if not chunks:
        return "No medical context available.", []

    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks, start=1):
        source_file = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk["text"]

        # Format each chunk with a numbered source header
        context_parts.append(
            f"[Source {i} — {source_file}, Page {page}]\n{text}"
        )

        sources.append({
            "index": i,
            "file": source_file,
            "page": page,
        })

    context_string = "\n\n".join(context_parts)
    return context_string, sources


# ─── Build the Full Prompt ────────────────────────────────────────────────────

def build_prompt(
    query: str,
    context: str,
    language: str,
    history: list | None = None,
) -> list:
    """
    Construct the full message list sent to Gemini.

    Returns a list of langchain Message objects (SystemMessage + HumanMessage)
    that together form the complete grounded generation prompt.

    The system prompt enforces:
      - Only use provided context (grounding)
      - Admit when answer isn't available (honesty)
      - Respond in user's language (accessibility)
      - Always cite sources (verifiability)
      - Always add disclaimer (safety)
    """
    if history is None:
        history = []

    # Format conversation history as text
    if history:
        history_text = "\n".join(
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in history
        )
    else:
        history_text = "No previous conversation"

    system_prompt = (
        "You are MedSaathi, a compassionate Indian health assistant\n"
        "helping rural patients who lack access to doctors.\n"
        "\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using the provided medical context below\n"
        "2. If the answer is not in the context, say:\n"
        "   'Yeh jaankari mere paas nahi hai. Kripya doctor se milein.'\n"
        "3. Always recommend seeing a doctor for serious symptoms\n"
        "4. Never prescribe specific dosages without context support\n"
        "5. Respond in {language} — same language as the user's question\n"
        "6. Be warm, simple, and clear — your users are not medical professionals\n"
        "7. Always cite which source your answer comes from (Source 1, Source 2 etc.)\n"
        "8. End every response with: '{disclaimer}'\n"
        "\n"
        "CONVERSATION HISTORY:\n"
        "{history}\n"
        "\n"
        "MEDICAL CONTEXT:\n"
        "{context}\n"
    ).format(
        language=language,
        disclaimer=DISCLAIMER,
        history=history_text,
        context=context,
    )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]


# ─── Generate Response (Main Function) ───────────────────────────────────────

def generate(
    query: str,
    chunks: list[dict],
    language: str = "hindi",
    history: list | None = None,
) -> dict:
    """
    The main generation function — produces a grounded, cited medical response.

    This is what the API endpoint calls. It orchestrates:
      1. Format chunks into a context string with source labels
      2. Build the full prompt with system instructions + context + query
      3. Call Gemini Flash to generate the answer
      4. Return everything the frontend needs (answer, sources, metadata)

    If Gemini fails for any reason, returns a safe Hindi fallback message.
    Medical AI should NEVER crash — a graceful failure is always better.

    Args:
        query: The user's original question (displayed in chat)
        chunks: Reranked chunks from reranker.py
        language: "hindi", "english", or "hinglish" (from detect_language)
        history: Past conversation turns [{user: "...", assistant: "..."}]

    Returns:
        Dict with answer, sources, chunks_used count, and language
    """
    if history is None:
        history = []

    try:
        llm = get_generator_llm()

        # Step 1: Format chunks into context + extract source citations
        context, sources = build_context(chunks)

        # Step 2: Build the complete prompt
        messages = build_prompt(query, context, language, history)

        # Step 3: Call Gemini Flash
        response = llm.invoke(messages)
        answer = response.content.strip()

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "language": language,
        }

    except Exception as e:
        console.print(f"[red]✗ Generation failed: {type(e).__name__}: {e}[/red]")
        return {
            "answer": FALLBACK_RESPONSE_HINDI,
            "sources": [],
            "chunks_used": 0,
            "language": "hindi",
        }


# ─── Streaming Generation ────────────────────────────────────────────────────

def generate_stream(
    query: str,
    chunks: list[dict],
    language: str = "hindi",
    history: list | None = None,
) -> GenType[str, None, None]:
    """
    Streaming version of generate() — yields text chunks as they arrive.

    Regular generation vs streaming:
    ─────────────────────────────────
      generate():        Waits for the ENTIRE response to be generated,
                         then returns it all at once. User sees nothing
                         for 2-3 seconds, then the full answer appears.

      generate_stream(): Yields each piece of text as Gemini produces it,
                         typically every ~50-100ms. Frontend displays each
                         piece immediately, creating a "typewriter effect."
                         User starts reading within 200ms — feels instant.

    Streaming is especially important for MedSaathi because:
      - Anxious patients want to see the answer forming immediately
      - Long medical answers (dosage tables, multi-step instructions)
        can take 3-5 seconds to generate fully
      - Typewriter effect feels conversational and reassuring

    Yields:
        String chunks of the response as they arrive from Gemini
    """
    if history is None:
        history = []

    try:
        llm = get_generator_llm()

        context, _ = build_context(chunks)
        messages = build_prompt(query, context, language, history)

        # .stream() returns an iterator of AIMessageChunk objects.
        # Each chunk has a .content attribute with the partial text.
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        console.print(f"[red]✗ Streaming generation failed: {type(e).__name__}: {e}[/red]")
        yield FALLBACK_RESPONSE_HINDI


# ─── Format Sources for Display ──────────────────────────────────────────────

def format_sources(sources: list[dict]) -> str:
    """
    Convert sources list into a human-readable citation string.

    Input:  [{"index": 1, "file": "WHO_guidelines.pdf", "page": 12}, ...]
    Output: "Sources: WHO_guidelines.pdf (p.12), AYUSH_manual.pdf (p.45)"

    Used by the frontend to show citations below the answer so users
    can verify the information themselves.
    """
    if not sources:
        return ""

    parts = [
        f"{s['file']} (p.{s['page']})"
        for s in sources
    ]

    return "Sources: " + ", ".join(parts)


# ─── Test Block ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.pretty import Pretty

    console.print("[bold magenta]═══ MedSaathi Generator Test ═══[/bold magenta]\n")

    # Create fake chunks — we test generation in isolation without calling
    # retriever or reranker. This lets us verify the prompt and LLM logic
    # independently from the rest of the pipeline.
    fake_chunks = [
        {
            "text": (
                "Paracetamol (Acetaminophen) 500mg tablet is commonly used "
                "for relief of mild to moderate pain and fever. For adults, "
                "the recommended dose is 1-2 tablets every 4-6 hours, not "
                "exceeding 8 tablets (4000mg) in 24 hours. It should be taken "
                "with water and can be taken with or without food."
            ),
            "source": "test_pharma.pdf",
            "page": 1,
            "rerank_score": 2.1,
        },
        {
            "text": (
                "For children aged 6-12 years, paracetamol dosage should be "
                "calculated based on body weight: 10-15 mg per kg body weight, "
                "every 4-6 hours. Maximum 4 doses in 24 hours. Children's "
                "paracetamol suspension (120mg/5ml) is preferred over tablets. "
                "Consult a pediatrician if fever persists beyond 3 days."
            ),
            "source": "test_pharma.pdf",
            "page": 2,
            "rerank_score": 1.8,
        },
    ]

    test_query = "bukhar mein kya lena chahiye"

    console.print(f"Query: [bold]{test_query}[/bold]")
    console.print(f"Chunks: [dim]{len(fake_chunks)} fake chunks[/dim]\n")

    # Test generate()
    console.print("[bold cyan]── Full Generation ──[/bold cyan]")
    result = generate(test_query, fake_chunks, language="hindi")
    console.print(Pretty(result))

    # Test format_sources()
    console.print(f"\n[bold cyan]── Formatted Sources ──[/bold cyan]")
    console.print(format_sources(result["sources"]))

    # Test streaming
    console.print(f"\n[bold cyan]── Streaming Generation ──[/bold cyan]")
    for text_chunk in generate_stream(test_query, fake_chunks, language="hindi"):
        console.print(text_chunk, end="")
    console.print()  # Final newline

    console.print("\n[green]✓[/green] Generator test complete.")
