"""wyzer.agent_core.prompt_builder_profiles

Three-profile prompt builder with priority-based token budgeting.

Profiles:
  PLAN   – full tool schemas + world state; used when regex is unsure.
  REPAIR – tiny; used after a tool fails with bad/missing args.
  SPEAK  – TTS-friendly summary; used after successful execution.

Public API:
  build_prompt(profile, compact, ctx) -> BuiltPrompt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config constants (importable, overridable via env or tests)
# ---------------------------------------------------------------------------
PLAN_TOKEN_BUDGET = 1600       # target tokens for PLAN profile
REPAIR_TOKEN_BUDGET = 600      # target tokens for REPAIR profile
SPEAK_TOKEN_BUDGET = 400       # target tokens for SPEAK profile

MAX_EVENTS_NORMAL = 10         # recent events kept in normal mode
MAX_EVENTS_COMPACT = 5         # recent events kept in compact mode
MAX_WINDOWS_COMPACT = 6        # open windows kept in compact mode
MAX_TOOL_SCHEMAS_COMPACT = 10  # tool schemas kept when budget is tight

# Priority order (1 = highest = drop last)
# 1 = output-format rules   (MUST KEEP)
# 2 = ground-truth rule     (MUST KEEP)
# 3 = world snapshot min    (MUST KEEP)
# 4 = recent events
# 5 = tool schemas
# 6 = examples/extra

PRIORITY_FORMAT = 1
PRIORITY_GROUND_TRUTH = 2
PRIORITY_WORLD_MIN = 3
PRIORITY_EVENTS = 4
PRIORITY_TOOLS = 5
PRIORITY_EXTRAS = 6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PromptContext:
    """All inputs the prompt builder needs."""

    user_text: str = ""

    # Tool info (list of dicts with at least "name", "description"; optionally "args_schema")
    tool_schemas: List[Dict[str, Any]] = field(default_factory=list)

    # World-state snapshot
    foreground_window: str = ""
    last_action: str = ""
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    open_windows: List[Dict[str, Any]] = field(default_factory=list)

    # Regex result (from hybrid_router.decide)
    regex_result: Optional[Dict[str, Any]] = None   # {confidence, intents, leftover_text, debug_matches}

    # Last tool error (for REPAIR)
    last_tool_error: Optional[Dict[str, Any]] = None  # {tool_name, args, error_type, error_message}

    # Extra context blocks (memories, session history, etc.)
    extra_context: str = ""


@dataclass
class BuiltPrompt:
    """Result of build_prompt()."""
    system: str
    user: str
    profile: str          # "plan" | "repair" | "speak"
    compact: bool
    tokens_est: int       # rough token estimate
    sections_kept: List[str]      # names of kept sections (for logging)
    sections_dropped: List[str]   # names of dropped sections


# ---------------------------------------------------------------------------
# Token estimation (re-uses existing tokeniser if available)
# ---------------------------------------------------------------------------
_tokenizer = None
_tok_loaded = False


def _estimate_tokens(text: str) -> int:
    """Rough token count. Uses tiktoken if available, else len/4."""
    global _tokenizer, _tok_loaded
    if not text:
        return 0
    if not _tok_loaded:
        _tok_loaded = True
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tokenizer = None
    if _tokenizer is not None:
        try:
            return len(_tokenizer.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------
OUTPUT_FORMAT_RULES = """\
OUTPUT FORMAT RULES (MUST FOLLOW):
- Respond with VALID JSON only; no markdown fences, no commentary outside JSON.
- For tool calls: {"intents": [{"tool": "<name>", "args": {}}], "reply": "short msg"}
- For reply only: {"reply": "your answer"}
- Never invent tool names or args not listed below.\
"""

GROUND_TRUTH_RULE = """\
GROUND TRUTH RULE (NON-NEGOTIABLE):
- NEVER claim anything about the screen, UI, windows, buttons, dialogs,
  text on screen, progress, or installation status UNLESS the fact appears
  in WORLD STATE or a tool result provided in this prompt.
- If asked about the screen and no perception data is present, you MUST
  call a perception tool (describe_screen, list_open_windows) first.
- NEVER say "I see a button" or any UI-state claim without evidence.\
"""

REPAIR_OUTPUT_RULES = """\
OUTPUT FORMAT: Return ONLY a single corrected tool call as JSON.
{"tool": "<name>", "args": {<corrected_args>}}
No prose. No extra keys. No explanation.\
"""

SPEAK_OUTPUT_RULES = """\
OUTPUT FORMAT: Return a short, spoken-friendly sentence summarising the outcome.
No JSON. No tool calls. 1-2 sentences max.\
"""


# ---------------------------------------------------------------------------
# Section builders (each returns (text, priority, name))
# ---------------------------------------------------------------------------
def _section_format_rules(profile: str) -> tuple:
    if profile == "repair":
        return REPAIR_OUTPUT_RULES, PRIORITY_FORMAT, "format_rules"
    if profile == "speak":
        return SPEAK_OUTPUT_RULES, PRIORITY_FORMAT, "format_rules"
    return OUTPUT_FORMAT_RULES, PRIORITY_FORMAT, "format_rules"


def _section_ground_truth() -> tuple:
    return GROUND_TRUTH_RULE, PRIORITY_GROUND_TRUTH, "ground_truth"


def _section_world_min(ctx: PromptContext) -> tuple:
    parts = []
    if ctx.foreground_window:
        parts.append(f"Foreground: {ctx.foreground_window}")
    if ctx.last_action:
        parts.append(f"Last action: {ctx.last_action}")
    text = "[WORLD STATE]\n" + "\n".join(parts) if parts else ""
    return text, PRIORITY_WORLD_MIN, "world_min"


def _section_events(ctx: PromptContext, compact: bool) -> tuple:
    limit = MAX_EVENTS_COMPACT if compact else MAX_EVENTS_NORMAL
    events = ctx.recent_events[-limit:] if ctx.recent_events else []
    if not events:
        return "", PRIORITY_EVENTS, "events"
    lines = ["[RECENT EVENTS]"]
    for ev in events:
        etype = ev.get("event", ev.get("type", "?"))
        detail = ev.get("detail", ev.get("tool", ""))
        ts = ev.get("ts", "")
        lines.append(f"  - {etype}: {detail}" + (f" (ts={ts})" if ts else ""))
    return "\n".join(lines), PRIORITY_EVENTS, "events"


def _section_tool_schemas(ctx: PromptContext, profile: str, compact: bool) -> tuple:
    """Build tool-schema section.

    PLAN  -> full schemas (compressed in compact mode).
    REPAIR -> only the failing tool.
    SPEAK  -> none.
    """
    if profile == "speak":
        return "", PRIORITY_TOOLS, "tool_schemas"

    schemas = ctx.tool_schemas or []

    # REPAIR: only include the failing tool schema
    if profile == "repair" and ctx.last_tool_error:
        fail_name = ctx.last_tool_error.get("tool_name", "")
        schemas = [s for s in schemas if s.get("name") == fail_name]
        if not schemas:
            # Include just the name so the LLM knows what it's fixing
            return f"[TOOL] {fail_name} (schema unavailable)", PRIORITY_TOOLS, "tool_schemas"

    # Compact: limit count and compress descriptions
    if compact and len(schemas) > MAX_TOOL_SCHEMAS_COMPACT:
        schemas = schemas[:MAX_TOOL_SCHEMAS_COMPACT]

    lines = ["[AVAILABLE TOOLS]"]
    for s in schemas:
        name = s.get("name", "?")
        desc = s.get("description", "")
        args = s.get("args_schema") or s.get("parameters") or {}
        if compact:
            desc = desc[:80] + "..." if len(desc) > 80 else desc
        lines.append(f"  - {name}: {desc}")
        if args and profile == "plan":
            # Include arg names for plan profile
            if isinstance(args, dict):
                arg_names = list(args.get("properties", args).keys())
                if arg_names:
                    lines.append(f"    args: {', '.join(arg_names)}")
    return "\n".join(lines), PRIORITY_TOOLS, "tool_schemas"


def _section_open_windows(ctx: PromptContext, compact: bool) -> tuple:
    windows = ctx.open_windows or []
    if not windows:
        return "", PRIORITY_EVENTS, "open_windows"
    if compact:
        windows = windows[:MAX_WINDOWS_COMPACT]
    lines = ["[OPEN WINDOWS]"]
    for w in windows:
        title = w.get("title", w.get("name", "?"))
        proc = w.get("process", w.get("app", ""))
        line = f"  - {title}"
        if proc:
            line += f" ({proc})"
        lines.append(line)
    return "\n".join(lines), PRIORITY_EVENTS, "open_windows"


def _section_repair_error(ctx: PromptContext) -> tuple:
    err = ctx.last_tool_error
    if not err:
        return "", PRIORITY_WORLD_MIN, "repair_error"
    lines = [
        "[FAILED TOOL CALL]",
        f"  tool: {err.get('tool_name', '?')}",
        f"  args: {err.get('args', {})}",
        f"  error_type: {err.get('error_type', '?')}",
        f"  error_message: {err.get('error_message', '?')}",
    ]
    return "\n".join(lines), PRIORITY_WORLD_MIN, "repair_error"


def _section_extras(ctx: PromptContext, compact: bool) -> tuple:
    text = ctx.extra_context or ""
    if compact:
        text = text[:200] + "..." if len(text) > 200 else text
    return text, PRIORITY_EXTRAS, "extras"


# ---------------------------------------------------------------------------
# Budget enforcer – drops lowest-priority sections first
# ---------------------------------------------------------------------------
def _enforce_budget(sections: List[tuple], budget: int) -> tuple:
    """Return (kept_text, kept_names, dropped_names).

    Sections are (text, priority, name) tuples. Lower priority number = higher
    importance.  We drop from highest number first until we fit the budget.
    """
    # Sort by descending priority number (drop-first = highest number)
    ordered = sorted(sections, key=lambda s: -s[1])

    total = sum(_estimate_tokens(s[0]) for s in ordered)
    dropped: list[str] = []

    # Drop sections from lowest importance until budget met
    while total > budget and ordered:
        candidate = ordered[0]  # highest number = lowest importance
        if candidate[1] <= PRIORITY_WORLD_MIN:
            break  # never drop must-keep sections
        total -= _estimate_tokens(candidate[0])
        dropped.append(candidate[2])
        ordered.pop(0)

    # Re-sort by priority ascending (output order)
    ordered.sort(key=lambda s: s[1])

    kept_text = "\n\n".join(s[0] for s in ordered if s[0])
    kept_names = [s[2] for s in ordered if s[0]]
    return kept_text, kept_names, dropped


# ---------------------------------------------------------------------------
# Profile-specific system preambles
# ---------------------------------------------------------------------------
PLAN_PREAMBLE = """\
You are Wyzer, a local voice assistant.
Your job: produce a tool-call plan OR a reply-only answer.
Do NOT hallucinate tool names or UI state.\
"""

REPAIR_PREAMBLE = """\
You are Wyzer's REPAIR module.
A tool call just failed.  Fix the args using the data below and return the corrected call.\
"""

SPEAK_PREAMBLE = """\
You are Wyzer's speech module.
Summarise the tool outcomes below into a short spoken reply (1-2 sentences).
Be concise, conversational, and TTS-friendly. No JSON.\
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_prompt(
    profile: Literal["plan", "repair", "speak"],
    compact: bool,
    ctx: PromptContext,
) -> BuiltPrompt:
    """Build a prompt for the given profile with token budgeting.

    Args:
        profile: "plan", "repair", or "speak"
        compact: If True, aggressively compress context.
        ctx: PromptContext with all inputs.

    Returns:
        BuiltPrompt with system/user text and metadata.
    """
    budget = {
        "plan": PLAN_TOKEN_BUDGET,
        "repair": REPAIR_TOKEN_BUDGET,
        "speak": SPEAK_TOKEN_BUDGET,
    }.get(profile, PLAN_TOKEN_BUDGET)

    # -- Gather sections --
    sections: List[tuple] = []

    # Preamble (always kept, priority 1 alongside format)
    preamble = {"plan": PLAN_PREAMBLE, "repair": REPAIR_PREAMBLE, "speak": SPEAK_PREAMBLE}[profile]
    sections.append((preamble, PRIORITY_FORMAT, "preamble"))

    # Format rules
    sections.append(_section_format_rules(profile))

    # Ground truth (skip for speak – it only narrates, doesn't claim state)
    if profile != "speak":
        sections.append(_section_ground_truth())

    # World snapshot minimal
    sections.append(_section_world_min(ctx))

    # Recent events
    sections.append(_section_events(ctx, compact))

    # Tool schemas
    sections.append(_section_tool_schemas(ctx, profile, compact))

    # Open windows (useful for PLAN and REPAIR)
    if profile in ("plan", "repair"):
        sections.append(_section_open_windows(ctx, compact))

    # Repair error block
    if profile == "repair":
        sections.append(_section_repair_error(ctx))

    # Regex debug info (for PLAN, so LLM knows what regex saw)
    if profile == "plan" and ctx.regex_result:
        rr = ctx.regex_result
        debug = (
            f"[REGEX ROUTER RESULT]\n"
            f"  confidence: {rr.get('confidence', '?')}\n"
            f"  intents: {rr.get('intents', [])}\n"
            f"  leftover: {rr.get('leftover_text', '')}"
        )
        sections.append((debug, PRIORITY_EVENTS, "regex_debug"))

    # Extras (examples, memories, etc.)
    sections.append(_section_extras(ctx, compact))

    # -- Enforce budget --
    system_body, kept, dropped = _enforce_budget(sections, budget)

    # -- Build user message --
    user_msg = f"User: {ctx.user_text}" if ctx.user_text else ""

    # -- Token estimate --
    total_est = _estimate_tokens(system_body) + _estimate_tokens(user_msg)

    logger.info(
        f"[PROMPT] profile={profile} compact={compact} "
        f"tokens_est={total_est} kept={kept} dropped={dropped}"
    )

    return BuiltPrompt(
        system=system_body,
        user=user_msg,
        profile=profile,
        compact=compact,
        tokens_est=total_est,
        sections_kept=kept,
        sections_dropped=dropped,
    )
