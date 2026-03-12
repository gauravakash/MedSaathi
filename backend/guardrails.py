"""
MedSaathi — Safety Guardrails (Pre-LLM Filter)
================================================

⚠️  This file can save lives — handle with care.  ⚠️

This is the most ethically important file in the entire project.
MedSaathi serves rural Indian users who may have no access to doctors,
mental health professionals, or emergency services nearby. A careless
response to a suicidal user or a dangerous self-medication query could
have real-world consequences.

Why guardrails run BEFORE the LLM, not after:
──────────────────────────────────────────────
  If we let the LLM process a dangerous query first and then filter its
  response, several things can go wrong:
    1. The LLM might generate a harmful answer before we can stop it
    2. In streaming mode, harmful tokens may reach the user before filtering
    3. We waste API calls and latency on queries we'll block anyway
    4. The LLM might find creative ways to bypass post-generation filters

  By checking BEFORE the LLM, dangerous queries never reach the model.
  The user gets an immediate, pre-written safe response. No ambiguity.

Why pre-written responses, not LLM responses, for crisis situations:
────────────────────────────────────────────────────────────────────
  LLMs can hallucinate — they might invent a helpline number that doesn't
  exist, or give a suicide hotline number that's actually a pizza place.
  For crisis situations, EVERY word and EVERY number must be verified and
  hardcoded. Our crisis response has real, working Indian helpline numbers
  that we've verified. The LLM never gets a chance to improvise here.

Why sensitive queries are NOT blocked:
──────────────────────────────────────
  Topics like depression, pregnancy, and substance abuse are legitimate
  medical concerns. Blocking them would deny users the help they need.
  Instead, we let the LLM answer but append extra disclaimers and
  helpline numbers. The goal: provide information while strongly
  encouraging professional help.

This file is pure rule-based — zero ML, zero API calls.
Speed is critical: this runs on EVERY single query before anything else.
"""

from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.table import Table

# ─── Configuration ────────────────────────────────────────────────────────────

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 1000
HELPLINE_VANDREVALA = "1860-2662-345"
HELPLINE_ICALL = "9152987821"
AMBULANCE = "108"


# ─── Query Categories ────────────────────────────────────────────────────────

class QueryCategory(Enum):
    """
    Classification of user queries by safety level.

    SAFE:         Normal medical question — proceed through full RAG pipeline.
                  Example: "bukhar mein kya lena chahiye" (what to take for fever)

    UNSAFE:       Dangerous query — block completely, show warning.
                  Example: "overdose kaise karein" (how to overdose)

    SENSITIVE:    Legitimate but delicate topic — answer with extra disclaimers.
                  Example: "depression ke liye kya karein" (what to do for depression)

    CRISIS:       Mental health emergency — show helpline numbers IMMEDIATELY.
                  Example: "mujhe jeena nahi hai" (I don't want to live)

    OUT_OF_SCOPE: Non-medical question — politely redirect to medical topics.
                  Example: "aaj cricket match kaun jeeta" (who won the cricket match)
    """
    SAFE = "safe"
    UNSAFE = "unsafe"
    SENSITIVE = "sensitive"
    CRISIS = "crisis"
    OUT_OF_SCOPE = "out_of_scope"


# ─── Guardrail Result ────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """
    The output of a guardrail check — tells the pipeline what to do next.

    is_blocked=True means: do NOT send to LLM, return safe_response directly.
    is_blocked=False means: proceed to LLM, but maybe attach a warning/disclaimer.
    """
    category: QueryCategory
    is_blocked: bool
    safe_response: str | None = None    # Pre-written response if blocked
    warning: str | None = None          # Warning to log (even if not blocked)
    confidence: float = 1.0             # 0.0-1.0, how sure we are about this category


# ─── Trigger Pattern Lists ───────────────────────────────────────────────────
#
# These are starter patterns for keyword-based detection.
# In production, replace with an ML classifier trained on medical safety data
# (e.g., a fine-tuned BERT model on Indian medical safety datasets).
# Keyword matching is fast and transparent but has limitations:
#   - Can miss paraphrased harmful queries
#   - Can false-positive on innocent queries containing trigger words
# For a learning project, this provides a solid safety baseline.

CRISIS_PATTERNS = [
    "marna chahta", "marna chahti", "jeena nahi", "suicide",
    "khud ko hurt", "zindagi khatam", "mar jaana chahta",
    "nahi rehna chahta", "self harm", "khud ko nuksan",
    "khudkhushi", "aatmhatya", "kill myself", "want to die",
    "end my life", "maut chahiye",
]

UNSAFE_PATTERNS = [
    "overdose kaise", "zyada dawa khaoon", "kitni dawa se mar",
    "ghar par surgery", "ghar par abortion", "illegal dawa",
    "bina doctor ke operation", "nasha kaise", "drug kaise banaye",
    "poison", "zeher", "acid attack", "how to harm",
]

SENSITIVE_PATTERNS = [
    "abortion", "pregnancy rokna", "garbhpat", "baby girana",
    "nasha", "alcohol", "sharab", "cigarette band karna",
    "depression", "anxiety", "mental health", "pagalpan",
    "sexual health", "std", "hiv", "periods",
    "addiction", "nashe ki lat",
]

OUT_OF_SCOPE_PATTERNS = [
    "cricket", "movie", "politics", "recipe", "khana banana",
    "weather", "mausam", "news", "share market", "paisa",
    "boyfriend", "girlfriend", "pyaar", "love", "exam",
    "job", "naukri", "game", "song", "gaana",
]


# ─── Pre-Written Safe Responses ──────────────────────────────────────────────
#
# These responses are HARDCODED — every word and number is verified.
# Never let an LLM generate crisis responses. One wrong helpline number
# in a suicide prevention message could cost a life.

CRISIS_RESPONSE = (
    "Aapki baat sun kar dil bhaari ho gaya.\n"
    "Aap akele nahi hain — madad maujood hai.\n"
    "\n"
    "HELPLINE NUMBERS (24/7, free):\n"
    f"  Vandrevala Foundation: {HELPLINE_VANDREVALA}\n"
    f"  iCall: {HELPLINE_ICALL}\n"
    "  NIMHANS: 080-46110007\n"
    "\n"
    "Kripya abhi inhe call karein. Aapki zindagi kimti hai."
)

UNSAFE_RESPONSE = (
    "Maafi chahta hoon, yeh jaankari main nahi de sakta.\n"
    "Yeh aapke liye khatarnaak ho sakta hai.\n"
    f"Kripya kisi doctor ya health worker se milein.\n"
    f"Agar koi emergency hai: {AMBULANCE} (Ambulance) par call karein."
)

OUT_OF_SCOPE_RESPONSE = (
    "Main sirf health aur medical sawaalon mein madad kar sakta hoon.\n"
    "Aap mujhse sehat se judi koi bhi baat pooch sakte hain!\n"
    "\n"
    "Udaharan: 'bukhar mein kya lena chahiye?', 'sugar ki dawa batao'"
)

TOO_SHORT_RESPONSE = (
    "Kripya apna sawaal thoda detail mein poochein.\n"
    "Udaharan: 'bukhar mein kya dawa leni chahiye?'"
)

TOO_LONG_RESPONSE = (
    "Aapka sawaal bahut lamba hai. Kripya chhota karke poochein.\n"
    "Ek ya do lines mein apni problem bataiye."
)


console = Console(force_terminal=True)


# ─── Pattern Matching Helpers ─────────────────────────────────────────────────
#
# Each check function lowercases the query and scans for pattern matches.
# Order of checks in check_query() matters critically:
#   1. Crisis   — highest priority (a suicidal query might also match "unsafe")
#   2. Unsafe   — block before it can reach the LLM
#   3. Sensitive — don't block, but flag for extra disclaimers
#   4. Out of scope — lowest priority redirect

def _check_crisis(query: str) -> bool:
    """Check if query indicates a mental health crisis or self-harm intent."""
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in CRISIS_PATTERNS)


def _check_unsafe(query: str) -> bool:
    """Check if query seeks dangerously harmful information."""
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in UNSAFE_PATTERNS)


def _check_sensitive(query: str) -> bool:
    """Check if query touches a sensitive but legitimate medical topic."""
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in SENSITIVE_PATTERNS)


def _check_out_of_scope(query: str) -> bool:
    """Check if query is clearly not related to health or medicine."""
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in OUT_OF_SCOPE_PATTERNS)


# ─── Sensitive Topic Disclaimers ──────────────────────────────────────────────

def get_sensitive_disclaimer(topic: str) -> str:
    """
    Return an extra disclaimer to append to LLM responses for sensitive topics.

    These disclaimers are topic-specific because different sensitive topics
    need different professional referrals:
      - Mental health → counseling helpline
      - Pregnancy/reproductive → registered doctor or ANM (auxiliary nurse midwife)
      - Substance abuse → de-addiction center
    """
    topic_lower = topic.lower()

    if any(kw in topic_lower for kw in ["depression", "anxiety", "mental health", "pagalpan"]):
        return (
            "\n\nYaad rakhein: mental health ke liye professional help bahut zaroori hai.\n"
            f"iCall helpline: {HELPLINE_ICALL} (Mon-Sat, 8am-10pm)"
        )

    if any(kw in topic_lower for kw in ["abortion", "pregnancy", "garbhpat", "baby girana", "periods"]):
        return (
            "\n\nKripya kisi registered doctor ya ANM (Auxiliary Nurse Midwife) se zaroor milein.\n"
            "Apne najdiki PHC (Primary Health Centre) mein free jaanch karwa sakte hain."
        )

    if any(kw in topic_lower for kw in ["nasha", "alcohol", "sharab", "addiction", "nashe ki lat"]):
        return (
            "\n\nNashe se chutkara paane ke liye professional madad lein.\n"
            "NIMHANS De-addiction Centre: 080-46110007\n"
            "Apne najdiki government hospital mein de-addiction clinic hota hai."
        )

    if any(kw in topic_lower for kw in ["sexual health", "std", "hiv"]):
        return (
            "\n\nSexual health ke baare mein kisi doctor se khulkar baat karein.\n"
            "Government hospitals mein free aur confidential testing milti hai."
        )

    # Generic sensitive disclaimer
    return (
        "\n\nYeh ek sensitive topic hai. Kripya kisi doctor se zaroor milein."
    )


# ─── Query Length Check ───────────────────────────────────────────────────────

def check_query_length(query: str) -> GuardrailResult:
    """
    Validate query length before processing.

    Too short: likely accidental or meaningless ("hi", "ok")
    Too long: possible prompt injection attempt or copy-pasted text
    """
    stripped = query.strip()

    if len(stripped) < MIN_QUERY_LENGTH:
        return GuardrailResult(
            category=QueryCategory.SAFE,  # Not a safety issue, just guidance
            is_blocked=True,
            safe_response=TOO_SHORT_RESPONSE,
            warning="Query too short",
            confidence=1.0,
        )

    if len(stripped) > MAX_QUERY_LENGTH:
        return GuardrailResult(
            category=QueryCategory.SAFE,
            is_blocked=True,
            safe_response=TOO_LONG_RESPONSE,
            warning="Query too long",
            confidence=1.0,
        )

    return GuardrailResult(category=QueryCategory.SAFE, is_blocked=False)


# ─── Main Content Check ──────────────────────────────────────────────────────

def check_query(query: str) -> GuardrailResult:
    """
    Classify a query into a safety category based on keyword patterns.

    Check order is critical — most dangerous categories first:
      1. CRISIS    — suicidal/self-harm → helpline numbers (highest priority)
      2. UNSAFE    — harmful intent → block completely
      3. SENSITIVE — delicate topic → allow with disclaimers
      4. OUT_OF_SCOPE — non-medical → polite redirect
      5. SAFE      — normal medical question → proceed

    Why this order? A query like "jeena nahi hai, overdose kaise karein"
    matches BOTH crisis AND unsafe patterns. We want the crisis response
    (with helpline numbers) not the generic unsafe block — because the
    helpline numbers could save a life.
    """
    # 1. CRISIS — immediate danger, highest priority
    if _check_crisis(query):
        return GuardrailResult(
            category=QueryCategory.CRISIS,
            is_blocked=True,
            safe_response=CRISIS_RESPONSE,
            warning=f"CRISIS DETECTED: '{query[:50]}...'",
            confidence=0.9,
        )

    # 2. UNSAFE — harmful intent, block completely
    if _check_unsafe(query):
        return GuardrailResult(
            category=QueryCategory.UNSAFE,
            is_blocked=True,
            safe_response=UNSAFE_RESPONSE,
            warning=f"UNSAFE query blocked: '{query[:50]}...'",
            confidence=0.8,
        )

    # 3. SENSITIVE — legitimate but needs careful handling
    # NOT blocked — user deserves information, but with extra disclaimers
    if _check_sensitive(query):
        return GuardrailResult(
            category=QueryCategory.SENSITIVE,
            is_blocked=False,
            safe_response=None,  # LLM will answer, disclaimer appended after
            warning=f"Sensitive topic detected: '{query[:50]}...'",
            confidence=0.7,
        )

    # 4. OUT OF SCOPE — non-medical, redirect politely
    if _check_out_of_scope(query):
        return GuardrailResult(
            category=QueryCategory.OUT_OF_SCOPE,
            is_blocked=True,
            safe_response=OUT_OF_SCOPE_RESPONSE,
            warning=f"Out of scope query: '{query[:50]}...'",
            confidence=0.7,
        )

    # 5. SAFE — normal medical question, full pipeline proceeds
    return GuardrailResult(
        category=QueryCategory.SAFE,
        is_blocked=False,
        confidence=1.0,
    )


# ─── Combined Check (Main Entry Point) ───────────────────────────────────────

def run_all_checks(query: str) -> GuardrailResult:
    """
    Run ALL guardrail checks in sequence and return the first non-SAFE result.

    This is the ONLY function that main.py should call. It combines:
      1. Query length validation
      2. Content safety classification

    Returns the most critical result found. If all checks pass, returns SAFE.
    """
    # Length check first — reject empty/huge queries before content analysis
    length_result = check_query_length(query)
    if length_result.is_blocked:
        return length_result

    # Content safety check
    content_result = check_query(query)
    return content_result


# ─── Test Block ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold magenta]=== MedSaathi Guardrails Test ===[/bold magenta]\n")

    test_cases = [
        ("bukhar mein kya lena chahiye", "Normal medical query"),
        ("mujhe jeena nahi hai", "Crisis / self-harm"),
        ("overdose kaise karein", "Unsafe / harmful"),
        ("depression ke liye kya karein", "Sensitive topic"),
        ("aaj cricket match kaun jeeta", "Out of scope"),
        ("hi", "Too short"),
    ]

    table = Table(title="Guardrail Test Results", show_lines=True)
    table.add_column("Query", style="white", width=35)
    table.add_column("Expected", style="dim", width=18)
    table.add_column("Category", width=14)
    table.add_column("Blocked?", width=9)
    table.add_column("Response Preview", width=40)

    for query, expected in test_cases:
        result = run_all_checks(query)

        # Color based on category
        if result.is_blocked and result.category == QueryCategory.CRISIS:
            cat_style = "[bold red]"
        elif result.is_blocked:
            cat_style = "[red]"
        elif result.category == QueryCategory.SENSITIVE:
            cat_style = "[yellow]"
        else:
            cat_style = "[green]"

        blocked_str = "[red]YES[/red]" if result.is_blocked else "[green]NO[/green]"
        response_preview = ""
        if result.safe_response:
            # First non-empty line, truncated
            first_line = next(
                (line for line in result.safe_response.strip().splitlines() if line.strip()),
                "",
            )
            response_preview = first_line[:40] + "..." if len(first_line) > 40 else first_line

        table.add_row(
            query,
            expected,
            f"{cat_style}{result.category.value}[/]",
            blocked_str,
            response_preview,
        )

    console.print(table)

    # Test sensitive disclaimer
    console.print("\n[bold cyan]Sensitive Disclaimers:[/bold cyan]")
    for topic in ["depression", "pregnancy", "alcohol", "sexual health"]:
        disclaimer = get_sensitive_disclaimer(topic)
        first_line = disclaimer.strip().splitlines()[0]
        console.print(f"  [{topic}] {first_line}")

    console.print("\n[green]OK[/green] Guardrails test complete.")
