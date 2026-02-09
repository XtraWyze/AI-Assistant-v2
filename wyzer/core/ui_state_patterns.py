"""wyzer.core.ui_state_patterns

Shared deterministic regexes for UI / window / screen-state queries.

Both the hybrid_router and orchestrator import from here so the same
patterns are used for routing AND for the defense-in-depth guard.
"""

from __future__ import annotations

import re
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# 1.  OPEN-WINDOWS QUERY  →  list_open_windows
# ═══════════════════════════════════════════════════════════════════════════
OPEN_WINDOWS_QUERY_RE = re.compile(
    r"^(?:"
    r"what\s+windows\s+are\s+open(?:\s+(?:right\s+)?now)?|"
    r"what(?:'?s|\s+is)\s+open(?:\s+(?:right\s+)?now)?|"
    r"what\s+do\s+i\s+have\s+open|"
    r"what\s+windows\s+do\s+i\s+have\s+open|"
    r"what\s+apps?\s+(?:are|do\s+i\s+have)\s+open(?:\s+(?:right\s+)?now)?|"
    r"list\s+(?:the\s+)?(?:open\s+)?windows|"
    r"show\s+(?:me\s+)?(?:the\s+)?(?:open\s+)?windows|"
    r"which\s+windows\s+are\s+open|"
    r"open\s+windows"
    r")\??$",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# 2.  RECENT-EVENTS / "WHAT JUST HAPPENED" QUERY  →  get_recent_events
# ═══════════════════════════════════════════════════════════════════════════
RECENT_EVENTS_QUERY_RE = re.compile(
    r"^(?:"
    # what did I open recently / what did I just open / what did I open most recently
    r"what\s+did\s+(?:i|you)\s+(?:just\s+)?open(?:\s+(?:most\s+)?recently)?|"
    r"what\s+(?:was|were)\s+(?:just\s+)?opened|"
    r"recently?\s+opened(?:\s+windows?)?|"
    # last / most recent window
    r"(?:the\s+)?last\s+window|"
    r"(?:the\s+)?most\s+recent\s+window|"
    r"(?:my\s+)?recent\s+windows?|"
    # what's the most recent / last thing/app/window I opened / I'll open
    r"what(?:'s|\s+is|\s+was)\s+the\s+(?:most\s+)?(?:recent|last)\s+(?:thing|app|application|window|program)\s+(?:i(?:'ve|'ll|\s+have|\s+will|\s+did)?\s+)?open(?:ed)?|"
    r"(?:the\s+)?(?:most\s+)?(?:recent|last)\s+(?:thing|app|application|window|program)\s+(?:i(?:'ve|'ll|\s+have|\s+will|\s+did)?\s+)?open(?:ed)?|"
    # what's the most recent / last thing done / that happened
    r"what(?:'s|\s+is|\s+was)\s+the\s+(?:most\s+)?(?:recent|last)\s+(?:thing|action|event)\s+(?:done|that\s+happened|i(?:'ve|\s+have)?\s+done)|"
    r"(?:the\s+)?(?:most\s+)?(?:recent|last)\s+(?:thing|action|event)\s+(?:done|that\s+happened|i(?:'ve|\s+have)?\s+done)|"
    # what just happened / recent events
    r"what\s+(?:just\s+)?happened|"
    r"recent\s+events|"
    r"what\s+changed|"
    r"what(?:'?s|\s+is|\s+has)\s+changed|"
    r"what\s+did\s+(?:i|you)\s+(?:just\s+)?do|"
    r"what\s+(?:was|were)\s+(?:the\s+)?(?:last|recent)\s+(?:action|event)s?|"
    # what's/what are the last N actions you performed/did
    # Allow optional qualifier words like 'tool' before the noun: 'last five tool actions'
    r"what(?:'s|\s+is|\s+are|\s+were)\s+(?:the\s+)?(?:last|recent)\s+(?:\d+\s+|(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+)?(?:\w+\s+)?(?:thing|action|event|command|step|tool)s?\s+(?:you(?:'ve|\s+have)?\s+)?(?:perform(?:ed)?|did|done|taken|execut(?:ed)?|made|complet(?:ed)?|us(?:ed)?|r[au]n|call(?:ed)?)|"
    # what actions/things/tools have you performed/done/used
    r"what\s+(?:\w+\s+)?(?:action|thing|event|command|step|tool)s?\s+(?:have\s+you|did\s+you|you)\s+(?:perform(?:ed)?|done?|taken|execut(?:ed)?|made|complet(?:ed)?|us(?:ed)?|r[au]n|call(?:ed)?)|"
    # list/show recent actions/tools
    r"(?:list|show|tell\s+me)\s+(?:the\s+)?(?:last|recent)\s+(?:\d+\s+|(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+)?(?:\w+\s+)?(?:action|event|command|thing|step|tool)s?"
    r")\??$",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# 3.  MONITOR / SCREEN WINDOW QUERY  →  list_open_windows
# ═══════════════════════════════════════════════════════════════════════════
MONITOR_WINDOWS_QUERY_RE = re.compile(
    r"^(?:"
    r"what(?:'?s|\s+is)\s+on\s+(?:my\s+)?(?:screen|monitor)s?|"
    r"what\s+(?:windows|apps?)\s+(?:are\s+)?on\s+(?:my\s+)?(?:screen|monitor)s?|"
    r"show\s+(?:me\s+)?what(?:'?s|\s+is)\s+on\s+(?:my\s+)?(?:screen|monitor)s?"
    r")\??$",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  PER-MONITOR WINDOWS QUERY  →  list_open_windows
#     "what windows are on each/every/per monitor", "windows opened on each monitor"
# ═══════════════════════════════════════════════════════════════════════════
PER_MONITOR_WINDOWS_QUERY_RE = re.compile(
    r"^(?:"
    # "what windows are opened/open on each/every/per monitor/screen"
    r"what\s+(?:windows|apps?)\s+(?:are\s+)?(?:open(?:ed)?|running|showing)\s+(?:on\s+)?(?:each|every|per|all(?:\s+my)?)\s+(?:monitor|screen|display)s?|"
    # "what apps are on every monitor" (no action verb, straight to 'on')
    r"what\s+(?:windows|apps?)\s+are\s+on\s+(?:each|every|per|all(?:\s+my)?)\s+(?:monitor|screen|display)s?|"
    # "what's open on each/every monitor"
    r"what(?:'s|\s+is)\s+(?:open(?:ed)?|running|showing)\s+(?:on\s+)?(?:each|every|per|all(?:\s+my)?)\s+(?:monitor|screen|display)s?|"
    # "show me windows on each/every/all monitors"
    r"(?:show|list|display)\s+(?:me\s+)?(?:the\s+)?(?:windows|apps?)\s+(?:on\s+)?(?:each|every|per|all(?:\s+my)?)\s+(?:monitor|screen|display)s?|"
    # "windows on each/every monitor" / "windows per monitor"
    r"(?:the\s+)?windows\s+(?:on|per)\s+(?:each|every|all(?:\s+my)?)\s+(?:monitor|screen|display)s?|"
    # "what's on each/every monitor"
    r"what(?:'s|\s+is)\s+on\s+(?:each|every|per|all(?:\s+my)?)\s+(?:monitor|screen|display)s?"
    r")\??$",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# 5.  UI-CONTENT QUERY  →  perception tool (NOT get_window_context)
#     Queries about buttons, text, progress, elements visible on screen.
# ═══════════════════════════════════════════════════════════════════════════
_UI_CONTENT_QUERY_RE = re.compile(
    r"(?:"
    # "is there a button/link/text that says ..." / "do you see a button"
    r"(?:is\s+there|do\s+you\s+see|can\s+you\s+(?:see|find))\s+(?:a\s+)?(?:button|link|text|element|control|checkbox|field|input|label|tab|option|menu\s+item)|"
    # "what buttons/text/elements are on the screen/window"
    r"what\s+(?:buttons?|text|elements?|controls?|options?|links?|tabs?)\s+(?:are|do\s+i\s+see)|"
    # "read the screen/window" / "what does the dialog/popup say"
    r"(?:read|check|verify|look\s+at)\s+(?:the\s+|this\s+|that\s+)?(?:screen|window|dialog|popup)|"
    # "what does the error/message/dialog say/show"
    r"what\s+does\s+(?:the\s+|this\s+|that\s+)?(?:dialog|popup|prompt|notification|alert|window|error|warning|message)(?:\s+(?:message|text|box))?\s+(?:say|show)|"
    # "is there an error" / "is there a progress bar"
    r"(?:is\s+there|do\s+you\s+see|can\s+you\s+see)\s+(?:a\s+|an\s+)?(?:error|warning|progress|loading|dialog|popup|notification)"
    r")",
    re.IGNORECASE,
)


def is_ui_content_query(text: str) -> bool:
    """Return True if the query asks about on-screen UI *content* (buttons, text, progress).

    These must NOT be answered by get_window_context (metadata only) — they need
    a perception tool like perceive_uia_focused_window or ui_find_text.
    """
    s = (text or "").strip()
    if not s:
        return False
    return bool(_UI_CONTENT_QUERY_RE.search(s))


def is_ui_state_tool_query(text: str) -> Optional[str]:
    """Return the deterministic tool name if *text* is a UI/state query, else None.

    This is the single source of truth shared by hybrid_router and orchestrator.
    It does NOT cover focused-window queries (those have their own dedicated
    handler in both modules).

    Returns:
        "list_open_windows"  – for open-windows queries
        "get_recent_events"  – for recent-events / what-just-happened queries
        "list_open_windows"  – for monitor/screen window queries
        None                 – not a UI/state query
    """
    s = (text or "").strip()
    if not s:
        return None

    if OPEN_WINDOWS_QUERY_RE.match(s):
        return "list_open_windows"

    if RECENT_EVENTS_QUERY_RE.match(s):
        return "get_recent_events"

    if MONITOR_WINDOWS_QUERY_RE.match(s):
        return "list_open_windows"

    if PER_MONITOR_WINDOWS_QUERY_RE.match(s):
        return "list_open_windows"

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 6.  AGENT-GRADE: Broad perception-first patterns  (Phase 17)
#     If the user asks anything that requires seeing the screen, these
#     patterns force the orchestrator to run perception BEFORE LLM.
# ═══════════════════════════════════════════════════════════════════════════

_AGENT_UI_QUERY_RE = re.compile(
    r"(?:"
    # what's on screen / what windows are open
    r"what(?:'?s|\s+is)\s+on\s+(?:my\s+)?screen|"
    r"what\s+windows\s+are\s+open|"
    # is it still downloading / did it finish
    r"is\s+it\s+still\s+\w+ing|"
    r"did\s+it\s+finish|"
    r"did\s+(?:the\s+)?\w+\s+finish|"
    r"is\s+it\s+(?:done|finished|complete|ready)|"
    r"is\s+it\s+still\s+there|"
    # Tab-specific UI actions (need perception to identify tabs)
    r"open\s+(?:the\s+)?\w+\s+tab|"
    r"chat\s+history\s+tab"
    r")",
    re.IGNORECASE,
)


def needs_perception_first(text: str) -> bool:
    """Return True if *text* is a query/command that requires running
    perception tools BEFORE letting the LLM answer or plan.

    This covers:
    - UI-state questions ("what's on screen", "is it still downloading")
    - Click/select/tab commands that need to see the UI first
    - Follow-up queries about visible state ("is it still there", "did it finish")
    - UI-content queries from the existing ``is_ui_content_query``
    """
    s = (text or "").strip()
    if not s:
        return False
    if _AGENT_UI_QUERY_RE.search(s):
        return True
    if is_ui_content_query(s):
        return True
    if is_ui_state_tool_query(s) is not None:
        return True
    return False
