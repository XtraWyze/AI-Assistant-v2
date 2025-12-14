"""wyzer.core.hybrid_router

Hybrid deterministic router for obvious commands.

It returns either:
- a deterministic tool plan (one or more tool calls), OR
- a decision to use the LLM.

This module is intentionally conservative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal


@dataclass
class HybridDecision:
    mode: Literal["tool_plan", "llm"]
    intents: Optional[List[Dict[str, Any]]] = None
    reply: str = ""
    confidence: float = 0.0


MULTI_INTENT_MARKERS = [
    " and ",
    " then ",
    " after ",
    " before ",
    " while ",
    " also ",
    " plus ",
    " & ",
    ";",
    ",",
]


def looks_multi_intent(text: str) -> bool:
    tl = (text or "").strip().lower()
    if not tl:
        return False

    # Normalize whitespace to make marker checks more reliable.
    tl = re.sub(r"\s+", " ", tl)
    tl = f" {tl} "

    return any(marker in tl for marker in MULTI_INTENT_MARKERS)


# Anchored time patterns: only match whole-utterance variants.
_TIME_RE = re.compile(
    r"^(what\s+time\s+is\s+it\??|time\s+is\s+it\??|current\s+time\??)$",
    re.IGNORECASE,
)

# Anchored open/launch/start.
_OPEN_RE = re.compile(r"^(open|launch|start)\s+(.+)$", re.IGNORECASE)

# Conservative URL/domain detection: if it looks like a URL/domain, we force LLM.
_URL_SCHEME_RE = re.compile(r"\bhttps?://", re.IGNORECASE)
_WWW_RE = re.compile(r"\bwww\.", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"\b[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.(?:[a-z]{2,})(?:/[^\s]*)?\b",
    re.IGNORECASE,
)


def _looks_like_url_or_domain(text: str) -> bool:
    tl = (text or "").strip().lower()
    if not tl:
        return False

    if _URL_SCHEME_RE.search(tl) or _WWW_RE.search(tl):
        return True

    # Any obvious domain-like token (foo.com, foo.co.uk, etc.).
    if _DOMAIN_RE.search(tl):
        return True

    return False


def _decide_single_clause(text: str) -> HybridDecision:
    clause = (text or "").strip()
    if not clause:
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.0)

    # Safety: URLs/domains go to LLM.
    if _looks_like_url_or_domain(clause):
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.8)

    # Time queries.
    if _TIME_RE.match(clause.strip()):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_time", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # Open/launch/start X (non-URL) -> open_target.
    m = _OPEN_RE.match(clause)
    if m:
        target = (m.group(2) or "").strip().strip('"').strip("'")
        # If the target is missing or too ambiguous, defer to LLM.
        if not target or target.lower() in {"it", "this", "that", "something", "anything"}:
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.4)

        # Double-check: the extracted target itself may look like a URL.
        if _looks_like_url_or_domain(target):
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.8)

        # Extra defense: if target includes other action verbs, defer to LLM.
        target_l = re.sub(r"\s+", " ", target.lower()).strip()
        if any(v in target_l.split() for v in ["play", "pause", "resume", "then", "and", "also", "plus"]):
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.3)

        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "open_target",
                    "args": {"query": target},
                    "continue_on_error": False,
                }
            ],
            reply=f"Opening {target}.",
            confidence=0.9,
        )

    # Minimal media/volume controls (only if tools exist; existence checked upstream).
    tl = clause.lower()

    if re.search(r"\b(?:mute|unmute)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "volume_mute_toggle", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.9,
        )

    if re.search(r"\b(?:volume\s+up|turn\s+up|louder)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "volume_up", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.85,
        )

    if re.search(r"\b(?:volume\s+down|turn\s+down|quieter)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "volume_down", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.85,
        )

    if re.search(r"\b(?:play\s*pause|play/pause|pause|play|resume)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "media_play_pause", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.8,
        )

    return HybridDecision(mode="llm", intents=None, reply="", confidence=0.3)


def decide(text: str) -> HybridDecision:
    """Decide whether to run tools deterministically or use the LLM.

    Args:
        text: Raw user text

    Returns:
        HybridDecision with mode tool_plan or llm.
    """
    raw = (text or "").strip()
    if not raw:
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.0)

    # Multi-intent guard: never short-circuit multi-action phrases.
    if looks_multi_intent(raw):
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.2)

    return _decide_single_clause(raw)
