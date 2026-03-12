"""
MedSaathi — Conversation Memory (Session Management)
======================================================

This file manages conversation history so MedSaathi can have multi-turn
conversations — remembering what was discussed earlier in the same chat.

  Without memory (stateless):
    User: "mujhe diabetes hai"
    Bot:  "Diabetes ke baare mein jaankari..."
    User: "iske liye kya khana chahiye?"     ← "for this" — but what is "this"?
    Bot:  "Kiske liye? Kripya bataiye."      ← Bot forgot diabetes was mentioned!

  With memory (stateful):
    Bot remembers the diabetes context → correctly answers about diabetic diet.

Stateless vs Stateful:
──────────────────────
  Stateless: Every request is independent. Server forgets everything between
             requests. Like talking to a goldfish.
  Stateful:  Server remembers past interactions within a session. Like talking
             to a person who was paying attention.

Why medical conversations SPECIFICALLY need memory:
────────────────────────────────────────────────────
  Medical discussions are inherently sequential:
    1. Patient describes symptoms over multiple messages
    2. Follow-up questions refine the diagnosis ("does it hurt when you press?")
    3. Treatment discussions reference earlier symptoms
    4. Dosage questions refer to a previously mentioned drug

  Without memory, the bot would ask "which medicine?" every time the user
  says "how much should I take?" — frustrating and medically dangerous if
  the user assumes the bot remembers their condition.

Storage: In-memory (RAM) — simple and fast
──────────────────────────────────────────
  All sessions are stored in a Python dict. This means:
    ✓ Zero setup — no database, no Redis, no config
    ✓ Fast — dict lookup is O(1)
    ✗ Data lost on server restart — sessions disappear
    ✗ Single server only — can't share sessions across instances

  For production, replace _sessions dict with Redis or PostgreSQL.
  For this learning project, in-memory is perfectly fine.

This file is a pure utility — zero ML, zero API calls, just data management.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_HISTORY_LENGTH = 6   # Last 6 turns = 3 user + 3 assistant messages
MAX_SESSION_AGE_MINUTES = 60  # Auto-cleanup sessions inactive for 1 hour


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """
    A single message in a conversation.

    We store language PER TURN (not per session) because users often
    switch languages mid-conversation:
      Turn 1 (Hindi):   "mujhe bukhar hai"
      Turn 2 (English): "what medicine should I take?"
      Turn 3 (Hinglish): "aur koi side effects toh nahi hai?"
    Tracking per-turn language helps generator.py respond in the
    user's most recent language preference.
    """
    role: str             # "user" or "assistant"
    content: str          # The message text
    timestamp: str = ""   # ISO format datetime — when the message was sent
    language: str = "hindi"  # Detected language of this turn

    def __post_init__(self):
        """Auto-fill timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ─── Session Store ────────────────────────────────────────────────────────────
#
# In-memory storage for all active conversations.
# Keys: session_id strings (e.g., "a3f8c2d1-7e4b-4a9f-b6c3-1d2e3f4a5b6c")
# Values: ordered lists of ConversationTurn objects
#
# Why a plain dict?
#   - Zero dependencies — no Redis, no database setup
#   - Fast — O(1) lookup by session_id
#   - Good enough for a single-server learning project
#
# For production, swap this dict with:
#   - Redis (fast, supports TTL for auto-expiry, multi-server)
#   - PostgreSQL (persistent, queryable, good for analytics)

_sessions: dict[str, list[ConversationTurn]] = {}

console = Console(force_terminal=True)


# ─── Create Session ──────────────────────────────────────────────────────────

def create_session() -> str:
    """
    Start a new conversation session.

    Called when a user opens the chat for the first time or clicks "New Chat."
    Generates a unique ID using UUID4 (random, collision-proof) and
    initializes an empty conversation history.

    Returns:
        A unique session_id string
    """
    session_id = str(uuid.uuid4())
    _sessions[session_id] = []
    return session_id


# ─── Add Message ──────────────────────────────────────────────────────────────

def add_message(
    session_id: str,
    role: str,
    content: str,
    language: str = "hindi",
) -> None:
    """
    Append a message to a conversation session.

    Called twice per user interaction:
      1. After user sends a message (role="user")
      2. After bot generates a response (role="assistant")

    If the session_id doesn't exist yet, it's created automatically.
    This prevents crashes if the frontend skips the create_session() call.
    """
    # Validate role — only "user" and "assistant" are valid
    if role not in ("user", "assistant"):
        raise ValueError(
            f"Invalid role '{role}'. Must be 'user' or 'assistant'."
        )

    # Auto-create session if it doesn't exist
    if session_id not in _sessions:
        _sessions[session_id] = []

    turn = ConversationTurn(
        role=role,
        content=content,
        language=language,
    )

    _sessions[session_id].append(turn)


# ─── Get History ──────────────────────────────────────────────────────────────

def get_history(
    session_id: str,
    last_n: int = DEFAULT_HISTORY_LENGTH,
) -> list[ConversationTurn]:
    """
    Retrieve the last N turns of a conversation.

    Why limit history instead of sending everything?
    ─────────────────────────────────────────────────
      Token cost tradeoff:
        - Every turn sent to the LLM costs tokens (= money + latency)
        - A 20-turn conversation adds ~2000 tokens to the prompt
        - That's ~2000 fewer tokens available for context chunks
        - It also slows response time by ~0.5-1 second

      Relevance decay:
        - Recent turns are almost always more relevant than older ones
        - "What medicine should I take?" refers to the symptom mentioned
          1-2 turns ago, not something from 15 turns back
        - Last 6 turns (3 exchanges) covers the immediate conversational
          context, which is enough for most medical follow-ups

      Default last_n=6: captures the last 3 user-assistant exchanges.
      For longer context needs, caller can increase this.
    """
    if session_id not in _sessions:
        return []

    history = _sessions[session_id]
    return history[-last_n:] if last_n else history


# ─── Format History for LLM ──────────────────────────────────────────────────

def format_history_for_llm(
    session_id: str,
    last_n: int = DEFAULT_HISTORY_LENGTH,
) -> str:
    """
    Format conversation history as a clean string for the LLM prompt.

    Output format:
      User: mujhe diabetes hai
      Assistant: Diabetes ek chronic condition hai...
      User: iske liye kya khana chahiye
      Assistant: Diabetes mein aapko ye khana chahiye...

    This output goes directly into generator.py's build_prompt() as the
    CONVERSATION HISTORY section. The LLM reads this to understand what
    was discussed before the current question.

    Returns empty string "" if no history exists — generator.py handles
    this by displaying "No previous conversation" in the prompt.
    """
    turns = get_history(session_id, last_n)

    if not turns:
        return ""

    lines = []
    for turn in turns:
        # Capitalize role for readability in the prompt
        role_label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{role_label}: {turn.content}")

    return "\n".join(lines)


# ─── Clear Session ────────────────────────────────────────────────────────────

def clear_session(session_id: str) -> None:
    """
    Delete a conversation session completely.

    Called when:
      - User clicks "New Chat" in the frontend
      - Auto-cleanup removes stale sessions
      - Testing/debugging

    Silently does nothing if session doesn't exist — no error raised.
    This makes it safe to call without checking session existence first.
    """
    _sessions.pop(session_id, None)


# ─── Session Summary ─────────────────────────────────────────────────────────

def get_session_summary(session_id: str) -> dict | None:
    """
    Get metadata about a conversation session.

    Useful for:
      - Debugging ("how many turns has this session had?")
      - Analytics ("which languages are users speaking?")
      - Admin dashboard (future feature)

    Returns None if session doesn't exist.
    """
    if session_id not in _sessions:
        return None

    turns = _sessions[session_id]
    if not turns:
        return {
            "session_id": session_id,
            "total_turns": 0,
            "languages_used": [],
            "started_at": None,
            "last_active": None,
        }

    languages_used = list({turn.language for turn in turns})

    return {
        "session_id": session_id,
        "total_turns": len(turns),
        "languages_used": languages_used,
        "started_at": turns[0].timestamp,
        "last_active": turns[-1].timestamp,
    }


# ─── Auto Cleanup ────────────────────────────────────────────────────────────

def cleanup_old_sessions(max_age_minutes: int = MAX_SESSION_AGE_MINUTES) -> int:
    """
    Delete sessions that have been inactive for too long.

    Without cleanup, _sessions dict grows forever as new users chat.
    Each session holds text data (small) but over hours/days with many
    users, this adds up and causes a memory leak.

    In production, this would be a background scheduled task (e.g., APScheduler
    or a cron job running every 15 minutes). For this project, we call it
    manually or from the server startup/periodic health check.

    Returns:
        Number of sessions deleted
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=max_age_minutes)
    sessions_to_delete = []

    for session_id, turns in _sessions.items():
        if not turns:
            # Empty session — mark for cleanup
            sessions_to_delete.append(session_id)
            continue

        # Check timestamp of the most recent message
        last_turn = turns[-1]
        try:
            last_active = datetime.fromisoformat(last_turn.timestamp)
            if last_active < cutoff:
                sessions_to_delete.append(session_id)
        except (ValueError, TypeError):
            # If timestamp is malformed, delete the session to be safe
            sessions_to_delete.append(session_id)

    for session_id in sessions_to_delete:
        del _sessions[session_id]

    return len(sessions_to_delete)


# ─── Test Block ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold magenta]═══ MedSaathi Memory Test ═══[/bold magenta]\n")

    # Step 1: Create a new session
    console.print(Panel("[bold]Step 1: Create a new session[/bold]", expand=False))
    sid = create_session()
    console.print(f"  Session ID: [cyan]{sid}[/cyan]\n")

    # Step 2: Simulate a multi-turn diabetes conversation
    console.print(Panel("[bold]Step 2: Simulate a 4-turn conversation[/bold]", expand=False))

    add_message(sid, "user", "mujhe diabetes hai", language="hindi")
    add_message(
        sid, "assistant",
        "Diabetes ek chronic condition hai jisme blood sugar level badh jaata hai. "
        "Kya aap Type 1 ya Type 2 diabetes ke baare mein jaanna chahte hain?",
        language="hindi",
    )
    add_message(sid, "user", "Type 2 hai, iske liye kya khana chahiye?", language="hinglish")
    add_message(
        sid, "assistant",
        "Type 2 diabetes mein aapko yeh khana chahiye: whole grains, green vegetables, "
        "dal, sprouts. Sugar, maida, aur processed food se bachein. "
        "Yeh sirf jaankari hai, doctor ki salah zaroor lein.",
        language="hindi",
    )

    console.print("  Added 4 turns (2 user + 2 assistant)\n")

    # Step 3: Display history in a table
    console.print(Panel("[bold]Step 3: Get conversation history[/bold]", expand=False))
    history = get_history(sid)

    table = Table(title="Conversation History", show_lines=True)
    table.add_column("Turn", style="bold cyan", width=5)
    table.add_column("Role", style="yellow", width=10)
    table.add_column("Language", style="green", width=10)
    table.add_column("Content", style="white")

    for i, turn in enumerate(history, start=1):
        content = turn.content[:80] + "..." if len(turn.content) > 80 else turn.content
        table.add_row(str(i), turn.role, turn.language, content)

    console.print(table)

    # Step 4: Show formatted history for LLM
    console.print(Panel("[bold]Step 4: Format history for LLM prompt[/bold]", expand=False))
    formatted = format_history_for_llm(sid)
    console.print(f"[dim]{formatted}[/dim]\n")

    # Step 5: Show session summary
    console.print(Panel("[bold]Step 5: Session summary[/bold]", expand=False))
    summary = get_session_summary(sid)
    console.print(Pretty(summary))
    console.print()

    # Step 6: Clear session and verify
    console.print(Panel("[bold]Step 6: Clear session[/bold]", expand=False))
    clear_session(sid)
    remaining = get_history(sid)
    console.print(f"  History after clear: [cyan]{remaining}[/cyan] (should be empty)")
    console.print(f"  Summary after clear: [cyan]{get_session_summary(sid)}[/cyan] (should be None)\n")

    console.print("[green]✓[/green] Memory test complete.")
