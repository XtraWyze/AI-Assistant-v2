"""wyzer.agent_core.router

Regex+LLM hybrid router.

Routing pipeline:
  1. Run existing ``hybrid_router.decide()`` (regex fast-path).
  2. If regex confidence >= THRESH and slots ok  → route="regex".
  3. Else if last tool failed with slot/arg error → route="repair".
  4. Else                                        → route="plan".
  5. After execution, optionally                 → route="speak".

Pronoun resolution is attempted deterministically *before* falling back
to the LLM planner.

Public API:
  router_route(user_text, world_state, tool_registry, compact_mode,
               regex_router, last_tool_error) -> RouterDecision
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds / knobs
# ---------------------------------------------------------------------------
REGEX_CONFIDENCE_THRESH = 0.90   # above this → trust the regex route
REPAIR_ERROR_TYPES = {
    "window_not_found",
    "missing_argument",
    "invalid_args",
    "not_found",
}

# Verbs that should be stripped when the user's follow-up is a
# bare command like "Close Notepad" → "Notepad" (for slot extraction)
_VERB_PREFIX_RE = re.compile(
    r"^(?:close|open|focus|minimize|maximize|switch\s+to|launch|start|run)\s+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RouterDecision:
    """Result of router_route()."""

    route: Literal["regex", "plan", "repair", "speak"]
    intents: Optional[List[Dict[str, Any]]] = None
    prompt_profile: Optional[str] = None      # "plan" | "repair" | "speak" (if LLM used)
    reply: str = ""

    # Debug / observability
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    regex_mode: Optional[str] = None          # "tool_plan" | "llm" from hybrid
    needs_perception: bool = False


# ---------------------------------------------------------------------------
# Pronoun resolution helpers
# ---------------------------------------------------------------------------
_PRONOUN_TOKENS = {"it", "this", "that", "the app", "the window"}

WYZER_SELF_TITLES = {"wyzer", "ai assistant", "ai-assistant"}


def _has_pronoun(text: str) -> bool:
    tl = text.lower()
    return any(p in tl for p in _PRONOUN_TOKENS)


def _resolve_pronoun_from_world(ws: Any) -> Optional[str]:
    """Attempt deterministic pronoun resolution from world state.

    Priority:
      1. Foreground window (if not Wyzer itself)
      2. last opened window title (last_target)
      3. Most-recent non-system window from open_windows
    """
    # 1. Foreground
    fg = getattr(ws, "active_window_title", None) or ""
    if fg and fg.lower() not in WYZER_SELF_TITLES:
        return fg

    # 2. last_target
    lt = getattr(ws, "last_target", None)
    if lt:
        return lt

    # 3. Most-recent non-system open window
    open_wins = getattr(ws, "open_windows", None) or []
    for w in open_wins:
        title = w.get("title", w.get("name", ""))
        proc = (w.get("process", "") or "").lower()
        if proc in {"explorer", "shellexperiencehost", "searchhost", "textinputhost"}:
            continue
        if title and title.lower() not in WYZER_SELF_TITLES:
            return title

    return None


def strip_verb_prefix(text: str) -> str:
    """Strip leading action verb so 'Close Notepad' → 'Notepad'."""
    return _VERB_PREFIX_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Slot checking
# ---------------------------------------------------------------------------
def _all_slots_filled(intents: Optional[List[Dict[str, Any]]], ws: Any) -> bool:
    """Check that no intent has an obvious pronoun / empty required slot."""
    if not intents:
        return True
    for intent in intents:
        args = intent.get("args") or {}
        tool = intent.get("tool", "")
        # Window tools require a title/process
        if tool in {"close_window", "focus_window", "minimize_window", "maximize_window"}:
            title = args.get("title", "")
            process = args.get("process", "")
            if not title and not process:
                return False
            if title and title.lower() in _PRONOUN_TOKENS:
                return False
        # open_target requires a target
        if tool == "open_target":
            if not args.get("target"):
                return False
    return True


# ---------------------------------------------------------------------------
# Main routing function
# ---------------------------------------------------------------------------
def router_route(
    user_text: str,
    world_state: Any,
    tool_registry: Any,
    compact_mode: bool = False,
    regex_router: Any = None,
    last_tool_error: Optional[Dict[str, Any]] = None,
) -> RouterDecision:
    """Route a user utterance through the regex→LLM pipeline.

    Args:
        user_text:       Raw transcript.
        world_state:     WorldState singleton.
        tool_registry:   ToolRegistry instance.
        compact_mode:    Whether prompt budgeting should be aggressive.
        regex_router:    The existing ``hybrid_router`` module (needs ``.decide()``).
        last_tool_error: If a tool just failed, dict with
                         ``{tool_name, args, error_type, error_message}``.

    Returns:
        RouterDecision describing the chosen path.
    """
    reasons: List[str] = []

    # -----------------------------------------------------------------
    # Step 0: If a tool just failed with a slot/arg error → REPAIR
    # -----------------------------------------------------------------
    if last_tool_error:
        etype = (last_tool_error.get("error_type") or "").lower()
        if etype in REPAIR_ERROR_TYPES:
            reasons.append(f"tool_error={etype}")
            logger.info(
                f"[ROUTE] route=repair reason=tool_error "
                f"tool={last_tool_error.get('tool_name')} error_type={etype}"
            )
            return RouterDecision(
                route="repair",
                intents=None,
                prompt_profile="repair",
                confidence=0.0,
                reasons=reasons,
            )

    # -----------------------------------------------------------------
    # Step 1: Pronoun resolution (deterministic, before regex)
    # -----------------------------------------------------------------
    resolved_text = user_text
    if _has_pronoun(user_text):
        target = _resolve_pronoun_from_world(world_state)
        if target:
            # Replace pronoun with resolved target
            resolved_text = _substitute_pronoun(user_text, target)
            reasons.append(f"pronoun_resolved={target}")
            logger.info(f'[ROUTE] pronoun resolved: "{user_text}" → "{resolved_text}"')

    # -----------------------------------------------------------------
    # Step 2: Run regex router
    # -----------------------------------------------------------------
    regex_decision = None
    if regex_router is not None:
        decide_fn = getattr(regex_router, "decide", None)
        if decide_fn:
            regex_decision = decide_fn(resolved_text)

    if regex_decision is None:
        reasons.append("no_regex_router")
        logger.info("[ROUTE] route=plan reason=no_regex_router confidence=0")
        return RouterDecision(
            route="plan",
            prompt_profile="plan",
            confidence=0.0,
            reasons=reasons,
        )

    conf = getattr(regex_decision, "confidence", 0.0)
    mode = getattr(regex_decision, "mode", "llm")
    intents = getattr(regex_decision, "intents", None)
    needs_perc = getattr(regex_decision, "_needs_perception", False)

    reasons.append(f"regex_confidence={conf:.2f}")
    reasons.append(f"regex_mode={mode}")

    # -----------------------------------------------------------------
    # Step 3: High-confidence regex → "regex" route
    # -----------------------------------------------------------------
    if mode == "tool_plan" and conf >= REGEX_CONFIDENCE_THRESH:
        # Verify all slots are filled
        if _all_slots_filled(intents, world_state):
            logger.info(
                f"[ROUTE] route=regex confidence={conf:.2f} "
                f"intents={[i.get('tool') for i in (intents or [])]}"
            )
            return RouterDecision(
                route="regex",
                intents=intents,
                confidence=conf,
                reasons=reasons,
                regex_mode=mode,
                reply=getattr(regex_decision, "reply", ""),
                needs_perception=needs_perc,
            )
        else:
            reasons.append("slots_incomplete")

    # -----------------------------------------------------------------
    # Step 4: Low-confidence / missing slots → "plan" route
    # -----------------------------------------------------------------
    logger.info(
        f"[ROUTE] route=plan confidence={conf:.2f} "
        f"reasons={reasons}"
    )
    return RouterDecision(
        route="plan",
        intents=intents if mode == "tool_plan" else None,
        prompt_profile="plan",
        confidence=conf,
        reasons=reasons,
        regex_mode=mode,
        needs_perception=needs_perc,
    )


# ---------------------------------------------------------------------------
# Speak decision helper
# ---------------------------------------------------------------------------
def needs_speak(execution_summary: Any, user_text: str) -> bool:
    """Decide whether we should call SPEAK profile after tool execution.

    Simple confirmations ("Opened Chrome", "Paused music") can be generated
    deterministically.  SPEAK is only needed when the result is complex or
    the user asked a question that tools answered with data.
    """
    if execution_summary is None:
        return False
    ran = getattr(execution_summary, "ran", None) or []
    if not ran:
        return False

    # If any tool returned a data payload (not a simple ok/error), we need SPEAK
    for r in ran:
        result = getattr(r, "result", None) or r.get("result", None) if isinstance(r, dict) else getattr(r, "result", None)
        if isinstance(result, dict) and len(result) > 2:
            return True
        if isinstance(result, str) and len(result) > 100:
            return True

    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _substitute_pronoun(text: str, replacement: str) -> str:
    """Replace the first pronoun in *text* with *replacement*."""
    tl = text.lower()
    for p in sorted(_PRONOUN_TOKENS, key=len, reverse=True):
        idx = tl.find(p)
        if idx != -1:
            return text[:idx] + replacement + text[idx + len(p):]
    return text
