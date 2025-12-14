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


def _strip_trailing_punct(text: str) -> str:
    return (text or "").strip().rstrip(".?!,;:\"'")


def _extract_volume_percent(text: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,3})\s*%?\b", text or "")
    if not m:
        return None
    try:
        v = int(m.group(1))
    except Exception:
        return None
    if 0 <= v <= 100:
        return v
    return None


def _parse_volume_delta_hint(text_lower: str) -> int:
    tl = text_lower or ""
    if any(k in tl for k in ["a little", "a bit", "slightly", "tiny bit", "small"]):
        return 5
    if any(k in tl for k in ["a lot", "much", "way", "significantly"]):
        return 20
    return 10


def _parse_volume_scope_and_process(clause: str) -> tuple[str, str]:
    """Return (scope, process) where scope is 'master' or 'app'.

    Keep this conservative: only treat a token as an app/process hint when it looks
    like one of the common "<proc> volume" / "volume ... for <proc>" patterns.
    """
    c = (clause or "").strip()
    cl = c.lower()

    # Strip common query prefixes so we don't treat them as app names.
    cl = re.sub(r"^(?:what\s+is|what's|whats|get|check|show|tell\s+me|current)\s+", "", cl)
    cl = re.sub(r"^the\s+", "", cl)

    # "set <proc> volume ..." should treat <proc> as app.
    m = re.match(r"^set\s+(?P<proc>.+?)\s+(?:volume|sound|audio)\b", cl)
    if m:
        proc = _strip_trailing_punct(m.group("proc")).strip()
        if proc and proc not in {"the", "my", "this", "that", "volume", "sound", "audio"}:
            return "app", proc

    # "volume 30 for spotify" / "mute discord" style: trailing "for/in/on <proc>".
    m = re.search(r"\b(?:for|in|on)\s+(?P<proc>[a-z0-9][a-z0-9 _\-\.]{1,60})$", cl)
    if m:
        proc = _strip_trailing_punct(m.group("proc")).strip()
        if proc:
            return "app", proc

    # "spotify volume ..." / "chrome sound ..." (but NOT "set volume ...")
    m = re.match(r"^(?P<first>[a-z0-9][a-z0-9 _\-\.]{1,60})\s+(?:volume|sound|audio)\b", cl)
    if m:
        first = _strip_trailing_punct(m.group("first")).strip()
        if first and first not in {"the", "my", "this", "that", "set", "volume", "sound", "audio"}:
            return "app", first

    # "mute spotify" / "turn down chrome" etc.
    m = re.match(r"^(?:turn\s+(?:up|down)|mute|unmute|quieter|louder)\s+(?P<proc>.+)$", cl)
    if m:
        proc = _strip_trailing_punct(m.group("proc")).strip()
        proc_l = proc.lower()
        if proc_l in {"it", "it up", "it down", "the volume", "volume", "sound", "audio"}:
            return "master", ""
        if proc and proc not in {"it", "volume", "sound", "audio"}:
            return "app", proc

    return "master", ""


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

    # --- True volume control (pycaw) ---
    # If the command looks like volume/mute and the tool exists, prefer volume_control.
    # We keep this conservative and only match obvious phrases.
    if re.search(r"\b(?:mute|unmute|volume|sound|audio|louder|quieter|turn\s+it\s+(?:up|down))\b", tl):
        scope, proc = _parse_volume_scope_and_process(clause)

        # Get volume / what is the volume
        is_query = bool(re.search(r"\b(?:get|check|show|tell\s+me|what\s+is|what's|whats|current)\b", tl))
        is_volume_worded = any(k in tl for k in ["volume", "sound", "audio"])
        if is_query and is_volume_worded and _extract_volume_percent(tl) is None and not re.search(
            r"\b(?:up|down|louder|quieter|increase|decrease|raise|lower)\b", tl
        ):
            args: Dict[str, Any] = {"scope": scope, "action": "get"}
            if scope == "app":
                args["process"] = proc
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.92,
            )

        # Mute/unmute
        if re.search(r"\bunmute\b", tl):
            args = {"scope": scope, "action": "unmute"}
            if scope == "app":
                args["process"] = proc
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.93,
            )

        if re.search(r"\bmute\b", tl) and not re.search(r"\bunmute\b", tl):
            args = {"scope": scope, "action": "mute"}
            if scope == "app":
                args["process"] = proc
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.93,
            )

        # Absolute set: "volume 35" / "set volume to 35" / "spotify volume 35"
        if is_volume_worded:
            percent = _extract_volume_percent(tl)
            # Avoid interpreting "volume down 10" as set-to.
            has_direction = bool(re.search(r"\b(?:up|down|increase|decrease|raise|lower|louder|quieter)\b", tl))
            if percent is not None and not has_direction:
                args = {"scope": scope, "action": "set", "level": int(percent)}
                if scope == "app":
                    args["process"] = proc
                return HybridDecision(
                    mode="tool_plan",
                    intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                    reply="",
                    confidence=0.9,
                )

        # Relative change: up/down/louder/quieter, optional numeric delta.
        if re.search(r"\b(?:volume\s+up|turn\s+up|louder|raise|increase|sound\s+up)\b", tl) or re.search(
            r"\bturn\s+it\s+up\b", tl
        ):
            pct = _extract_volume_percent(tl)
            delta = int(pct) if pct is not None else _parse_volume_delta_hint(tl)
            args = {"scope": scope, "action": "change", "delta": int(delta)}
            if scope == "app":
                args["process"] = proc
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.88,
            )

        if re.search(r"\b(?:volume\s+down|turn\s+down|quieter|lower|decrease|sound\s+down)\b", tl) or re.search(
            r"\bturn\s+it\s+down\b", tl
        ):
            pct = _extract_volume_percent(tl)
            delta = int(pct) if pct is not None else _parse_volume_delta_hint(tl)
            args = {"scope": scope, "action": "change", "delta": -int(delta)}
            if scope == "app":
                args["process"] = proc
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.88,
            )

    if re.search(r"\b(?:mute|unmute)\b", tl):
        # Fallback for older setups without volume_control.
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
