"""wyzer.policy.tool_relevance_gate

GLOBAL Tool-Relevance Gate.

Determines whether a user query is "tool-relevant" — i.e., it asks about
desktop state, actions, results, recent events, windows, progress, errors,
installs, file operations, etc.

If a query IS tool-relevant but NO tool was executed and NO deterministic
WorldState fact exists for the requested information, the gate BLOCKS the
LLM and returns a deterministic refusal.

This gate works for ALL tools, not just screen/perception tools.

HARD INVARIANT:
    If is_tool_relevant_query(text) and not executed_any_tool
       and not deterministic_world_fact_exists(text):
        -> BLOCK LLM, return refusal.
"""

from __future__ import annotations

import re
from typing import Optional

from wyzer.core.logger import get_logger

_logger = get_logger()

# ============================================================================
# TOOL-RELEVANT QUERY PATTERNS
# ============================================================================
# A query matches if it asks about state/actions/results that Wyzer can ONLY
# know through tool execution or deterministic tracking.

_TOOL_RELEVANT_PATTERNS = [
    # Screen / Vision
    r"\b(?:see|seeing|saw|look|looking|visible|display(?:ed|ing)?|shown?|showing|screen|monitor)\b",
    r"\b(?:what(?:'s|s|\s+is)\s+on\s+(?:my\s+)?(?:screen|display|monitor))",
    r"\b(?:what\s+do\s+you\s+see|what\s+can\s+you\s+see|describe\s+(?:the\s+)?(?:screen|what\s+you\s+see))",
    r"\b(?:tell\s+me\s+what\s+you\s+see|what\s+(?:is|are)\s+(?:on\s+)?(?:the\s+)?screen)",

    # Window / Focus / Active app
    r"\b(?:focused|active\s+window|foreground|(?:which|what)\s+(?:window|app)\s+(?:is|am))",
    r"\b(?:what(?:'s|s|\s+is)\s+(?:the\s+)?(?:focused|active|current)\s+(?:window|app|application))",

    # Open / Close / Recent events
    r"\b(?:what\s+did\s+(?:i|you)\s+(?:just\s+)?(?:open|close|launch|start|run|do|click|type|press))",
    r"\b(?:what\s+(?:was|were)\s+(?:just\s+)?(?:opened|closed|launched|started|clicked|typed|pressed))",
    r"\b(?:recently?\s+(?:opened|closed|launched|started))",
    r"\b(?:what\s+(?:just\s+)?happened|what\s+changed)",
    r"\b(?:what\s+(?:is|was)\s+(?:the\s+)?last\s+(?:thing|action|command|app|window))",

    # Click / Type / Hotkey / Scroll / Input actions
    r"\b(?:did\s+(?:you|it)\s+click|did\s+(?:you|it)\s+type|did\s+(?:you|it)\s+press)",
    r"\b(?:click|right[- ]?click|double[- ]?click)\s+(?:on\s+)?(?:the\s+)?",
    r"\b(?:type|enter|input)\s+(?:in(?:to)?\s+)?(?:the\s+)?",
    r"\b(?:press|hit|hotkey|shortcut)\s+",
    r"\b(?:scroll)\s+(?:up|down|left|right)",

    # Install / Download / Progress / Error / Dialog
    r"\b(?:install(?:ed|ing|ation)?|download(?:ed|ing)?|progress|complet(?:ed|e|ion))\b",
    r"\b(?:error|fail(?:ed|ure)?|crash(?:ed)?|dialog|popup|prompt|notification)\b",
    r"\b(?:did\s+(?:it|that)\s+(?:work|succeed|finish|complete|fail|crash))",
    r"\b(?:is\s+(?:it|that)\s+(?:done|finished|complete|installed|downloaded|working))",

    # File / Folder / Create / Delete / Move / Rename / Save
    r"\b(?:file|folder|directory|path)\s+(?:is|was|are|were|exist|created|deleted|moved|renamed|saved)",
    r"\b(?:create|delete|move|rename|save|copy|paste)\s+(?:the\s+)?(?:file|folder|directory)",
    r"\b(?:did\s+(?:you|it)\s+(?:create|delete|move|rename|save|copy))",

    # "Did it succeed" / "Did that work" / aftermath questions
    r"\b(?:did\s+(?:it|that)\s+(?:work|succeed|go\s+through|happen))\b",
    r"\b(?:was\s+(?:it|that)\s+successful|(?:is|was)\s+(?:it|that)\s+done)\b",
    r"\b(?:what(?:'s|s|\s+is)\s+the\s+(?:result|outcome|status))\b",
    r"\b(?:what\s+(?:did\s+(?:you|it)\s+(?:find|get|return)))\b",

    # Tool name / action references
    r"\b(?:weather|forecast|temperature|humidity)\b.*\b(?:right\s+now|currently|today)\b",
    r"\b(?:timer|alarm|countdown)\s+(?:status|remaining|left|done)\b",
]

# Compile into a single pattern for speed
_TOOL_RELEVANT_RE = re.compile(
    "|".join(f"(?:{p})" for p in _TOOL_RELEVANT_PATTERNS),
    re.IGNORECASE,
)

# ============================================================================
# NEGATIVE PATTERNS (override: these are informational even if they match above)
# ============================================================================
# E.g. "what is an error message" is a knowledge question, not a tool query.
_INFORMATIONAL_OVERRIDE_RE = re.compile(
    r"^(?:"
    r"what\s+(?:is|are)\s+(?:a|an|the\s+concept\s+of|the\s+meaning\s+of)|"
    r"(?:explain|define|describe)\s+(?:what\s+)?(?:a|an)\s|"
    r"how\s+(?:do(?:es)?|does\s+a)\s+.*\s+work|"
    r"tell\s+me\s+about\s+"
    r")",
    re.IGNORECASE,
)

# Knowledge qualifiers: if the query has these after "what is", it's informational
_KNOWLEDGE_QUALIFIER_RE = re.compile(
    r"\b(?:concept|definition|meaning|example|difference\s+between|"
    r"type\s+of|kind\s+of|category)\b",
    re.IGNORECASE,
)


def is_tool_relevant_query(text: str) -> bool:
    """
    Determine if a user query is "tool-relevant".

    A tool-relevant query asks about desktop state, actions, results,
    recent events, windows, progress, errors, installs, file operations,
    or any other information that Wyzer can ONLY provide by executing a
    tool or reading deterministic WorldState.

    Returns True if the query should ONLY be answered with tool evidence.
    """
    if not text or not text.strip():
        return False

    stripped = text.strip()

    # ---- negative / informational override ----
    if _INFORMATIONAL_OVERRIDE_RE.match(stripped):
        return False
    if _KNOWLEDGE_QUALIFIER_RE.search(stripped):
        return False

    # ---- positive match ----
    return bool(_TOOL_RELEVANT_RE.search(stripped))


# ============================================================================
# GATE DECISION
# ============================================================================

# Default deterministic refusal message
_DEFAULT_REFUSAL = (
    "I can't verify that right now. No tool ran to check. "
    "Try asking me to perform the action first, or ask a specific question "
    "I can look up with a tool."
)


def gate_decision(
    user_text: str,
    executed_any_tool: bool,
    world_state=None,
) -> Optional[str]:
    """
    GLOBAL gate: decide whether to block LLM on a tool-relevant query.

    Args:
        user_text:         The user's input text.
        executed_any_tool: Whether any tool was executed this turn.
        world_state:       Optional WorldState instance.

    Returns:
        None   if LLM is ALLOWED to answer (either not tool-relevant, or
               tool evidence exists).
        str    deterministic refusal message if LLM must be BLOCKED.
    """
    if not is_tool_relevant_query(user_text):
        return None  # Not tool-relevant → LLM is free to answer

    # Tool ran → allow LLM (it will be constrained by evidence envelope)
    if executed_any_tool:
        _logger.info(
            "[GATE] llm_narration=true tools_executed=1+ "
            f"query={user_text[:60]!r}"
        )
        return None

    # Check if deterministic WorldState has a relevant fact
    if world_state is not None and _world_state_has_relevant_fact(user_text, world_state):
        _logger.info(
            "[GATE] llm_narration=true tools_executed=0 world_state_fact=true "
            f"query={user_text[:60]!r}"
        )
        return None

    # BLOCK: tool-relevant query with no evidence
    _logger.warning(
        "[GATE] blocked_llm=true tool_relevant=true reason=\"no_tool_evidence\" "
        f"query={user_text[:60]!r}"
    )
    return _DEFAULT_REFUSAL


def _world_state_has_relevant_fact(text: str, ws) -> bool:
    """
    Check if WorldState contains a deterministic fact relevant to the query.

    This is intentionally conservative — it only returns True when we are
    confident the WorldState field directly answers the question.
    """
    lower = text.lower()

    # "What did I open / what's active / which app"
    if any(kw in lower for kw in ("active", "focused", "foreground", "current app", "current window")):
        if getattr(ws, "active_app", None) or getattr(ws, "active_window_title", None):
            return True

    # "What did I just open/close"
    if any(kw in lower for kw in ("last", "recent", "just open", "just close")):
        if getattr(ws, "last_tool", None) and getattr(ws, "last_target", None):
            return True

    # "Did it work / did that succeed"
    if any(kw in lower for kw in ("did it work", "did that work", "succeed", "successful")):
        if getattr(ws, "last_tool", None):
            return True

    return False
