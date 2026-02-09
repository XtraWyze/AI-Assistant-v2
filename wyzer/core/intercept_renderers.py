"""wyzer.core.intercept_renderers

Deterministic reply renderers for forced-tool intercept paths.

These functions convert raw tool-result JSON into spoken replies
without any LLM involvement.  They are called from the HARD OVERRIDE
blocks in orchestrator.py.

Public API
----------
format_ui_content_reply(user_text, result)
    Renders a perceive_uia_focused_window result into a spoken answer
    (e.g. "Yes, I found a button called Install." or
    "No, I didn't find a button matching 'install'.")

format_recent_events_reply(result)
    Renders a get_recent_events result into a short spoken summary
    (e.g. "Here's what happened recently: Perceived focused window …").
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# 1.  UI-CONTENT REPLY  (perceive_uia_focused_window)
# ═══════════════════════════════════════════════════════════════

# Control types that count as "clickable / interactive"
_INTERACTIVE_TYPES = frozenset({"Button", "Hyperlink", "MenuItem", "TabItem",
                                 "CheckBox", "RadioButton", "ComboBox"})

# Regex that pulls the search term from queries like
#   "is there a button that says Install"
#   "do you see a link called Sign In"
#   "can you find a button named OK"
_SEARCH_TERM_RE = re.compile(
    r"(?:(?:that|which)\s+says?|"
    r"(?:called|named|labeled|labelled|with\s+(?:the\s+)?(?:text|label|name)))"
    r"\s+['\"]?(.+?)['\"]?\s*\??$",
    re.IGNORECASE,
)

# Regex that pulls which control type the user is asking about
_CONTROL_TYPE_RE = re.compile(
    r"\b(button|link|hyperlink|text|element|control|checkbox|"
    r"field|input|label|tab|option|menu\s*item)\b",
    re.IGNORECASE,
)

# Map user words → UIA control_type values
_USER_WORD_TO_UIA: Dict[str, Tuple[str, ...]] = {
    "button":    ("Button",),
    "link":      ("Hyperlink",),
    "hyperlink": ("Hyperlink",),
    "checkbox":  ("CheckBox",),
    "tab":       ("TabItem",),
    "menu item": ("MenuItem",),
    "menuitem":  ("MenuItem",),
    "field":     ("Edit",),
    "input":     ("Edit",),
    "option":    ("RadioButton", "ComboBox", "ListItem"),
    "label":     ("Text",),
    "text":      ("Text",),
    "element":   tuple(),  # match any type
    "control":   tuple(),
}


def _extract_search_term(user_text: str) -> Optional[str]:
    """Return the UI-element name the user is searching for, or None."""
    m = _SEARCH_TERM_RE.search(user_text or "")
    return m.group(1).strip() if m else None


def _extract_control_filter(user_text: str) -> Tuple[str, ...]:
    """Return UIA control_type filter from user wording, or empty (= any)."""
    m = _CONTROL_TYPE_RE.search(user_text or "")
    if not m:
        return ()
    word = m.group(1).strip().lower()
    return _USER_WORD_TO_UIA.get(word, ())


def _controls_matching(controls: List[Dict[str, Any]],
                       search_term: str,
                       type_filter: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Return controls whose name case-insensitively contains *search_term*."""
    needle = search_term.lower()
    hits: List[Dict[str, Any]] = []
    for c in controls:
        name = (c.get("name") or "").strip()
        if not name:
            continue
        ctype = c.get("control_type") or ""
        if type_filter and ctype not in type_filter:
            continue
        if needle in name.lower():
            hits.append(c)
    return hits


def _closest_matches(controls: List[Dict[str, Any]],
                     search_term: str,
                     type_filter: Tuple[str, ...],
                     limit: int = 5) -> List[str]:
    """Return up to *limit* control names that are 'close' to *search_term*.

    Simple heuristic: any interactive control whose name is non-empty.
    We filter by type when the user specified one.
    """
    needle = search_term.lower()
    seen: set = set()
    result: List[str] = []
    for c in controls:
        name = (c.get("name") or "").strip()
        ctype = c.get("control_type") or ""
        if not name or name.lower() in seen:
            continue
        if type_filter and ctype not in type_filter:
            continue
        # Prefer interactive controls, but include anything named
        if ctype in _INTERACTIVE_TYPES or not type_filter:
            seen.add(name.lower())
            result.append(f"{name} ({ctype})")
            if len(result) >= limit:
                break
    return result


def format_ui_content_reply(user_text: str,
                            result: Dict[str, Any]) -> str:
    """Build a deterministic spoken reply for a UI-content intercept.

    Parameters
    ----------
    user_text : str
        The raw user query (e.g. "Is there a button that says Install?").
    result : dict
        The JSON returned by ``perceive_uia_focused_window``.

    Returns
    -------
    str
        A natural-language reply suitable for TTS.
    """
    controls: List[Dict[str, Any]] = result.get("controls") or []
    window: Dict[str, Any] = result.get("window") or {}
    errors: List[str] = result.get("errors") or []

    # If perception produced errors and no controls, report that.
    if not controls and errors:
        return "I couldn't read the window controls. " + errors[0]

    search_term = _extract_search_term(user_text)
    type_filter = _extract_control_filter(user_text)

    if search_term:
        hits = _controls_matching(controls, search_term, type_filter)
        if hits:
            # Report the first hit, mention count if multiple
            first = hits[0]
            name = (first.get("name") or "").strip()
            ctype = first.get("control_type") or "control"
            if len(hits) == 1:
                return f"Yes, I found a {ctype} called \"{name}\"."
            return (
                f"Yes, I found {len(hits)} matches. "
                f"The first is a {ctype} called \"{name}\"."
            )
        else:
            # No exact hit — provide closest alternatives
            close = _closest_matches(controls, search_term, type_filter)
            if close:
                listing = ", ".join(close)
                return (
                    f"No, I didn't find a control matching \"{search_term}\". "
                    f"The closest I see: {listing}."
                )
            return f"No, I didn't find anything matching \"{search_term}\" on this window."

    # ── Generic UI-content query (no specific name search) ──
    # e.g. "what buttons are on the screen" / "read the window"
    win_title = (window.get("title") or "").strip()
    if not controls:
        if win_title:
            return f"I don't see any controls on \"{win_title}\"."
        return "I don't see any controls on the active window."

    # Summarise by type
    by_type: Dict[str, List[str]] = {}
    for c in controls:
        ctype = c.get("control_type") or "Unknown"
        name = (c.get("name") or "").strip()
        if name:
            by_type.setdefault(ctype, []).append(name)

    # Build short summary
    parts: List[str] = []
    for ctype in ("Button", "Hyperlink", "TabItem", "MenuItem",
                  "CheckBox", "Edit", "ComboBox", "Text"):
        names = by_type.get(ctype)
        if names:
            sample = ", ".join(names[:5])
            if len(names) > 5:
                sample += f" and {len(names) - 5} more"
            parts.append(f"{ctype}s: {sample}")

    prefix = f"On \"{win_title}\"" if win_title else "On the active window"
    if parts:
        return f"{prefix}, I see: " + "; ".join(parts) + "."
    return f"{prefix}, I found {len(controls)} controls but none had readable names."


# ═══════════════════════════════════════════════════════════════
# 2.  RECENT-EVENTS REPLY  (get_recent_events)
# ═══════════════════════════════════════════════════════════════

def _format_single_event(evt: Dict[str, Any]) -> Optional[str]:
    """Return a one-line human summary for one event dict, or None."""
    etype = evt.get("event") or evt.get("type") or ""
    etype_lower = etype.lower()

    # ── Tool events ──
    if etype_lower in ("tool_start", "tool_end"):
        tool = evt.get("tool") or "unknown tool"
        latency = evt.get("latency_ms")
        ok = evt.get("ok")
        if etype_lower == "tool_end":
            lat_str = f", latency {latency}ms" if latency is not None else ""
            status = "succeeded" if ok else ("failed" if ok is False else "")
            return f"Ran {tool} ({status}{lat_str})"
        return None  # skip tool_start, tool_end is enough

    # ── Perception events ──
    if etype_lower == "perception":
        source = evt.get("source") or "unknown"
        count = evt.get("found_controls_count")
        latency = evt.get("latency_ms")
        parts = [f"Perceived focused window via {source.upper()}"]
        if count is not None:
            parts.append(f"{count} controls")
        if latency is not None:
            parts.append(f"latency {latency}ms")
        return " (".join(parts[:1]) + (", ".join(parts[1:]) + ")" if len(parts) > 1 else "")

    # ── Window-watcher / world events ──
    if etype_lower in ("world_evt", "opened", "focus_changed", "title_changed"):
        sub = evt.get("type") or etype_lower
        title = evt.get("title") or ""
        app = evt.get("app") or evt.get("process") or ""
        if title and app:
            return f"{sub}: {app} — \"{title}\""
        if title:
            return f"{sub}: \"{title}\""
        if app:
            return f"{sub}: {app}"
        return sub

    # ── UI actions ──
    if etype_lower == "ui_action":
        action = evt.get("action") or "unknown action"
        target = evt.get("target") or ""
        if target:
            return f"UI action: {action} on \"{target}\""
        return f"UI action: {action}"

    # ── Warning ──
    if etype_lower == "warning":
        msg = evt.get("message") or evt.get("msg") or "warning"
        return f"Warning: {msg}"

    # ── Fallback ──
    return f"Event: {etype}" if etype else None


def format_recent_events_reply(result: Dict[str, Any]) -> str:
    """Build a deterministic spoken reply for a get_recent_events intercept.

    Parameters
    ----------
    result : dict
        The JSON returned by ``get_recent_events``.

    Returns
    -------
    str
        A natural-language reply suitable for TTS.
    """
    events: List[Dict[str, Any]] = result.get("events") or []
    count: int = result.get("count", len(events))

    if not events:
        return "Nothing has happened recently."

    lines: List[str] = []
    for evt in events:
        line = _format_single_event(evt)
        if line:
            lines.append(line)

    if not lines:
        return "Nothing notable has happened recently."

    if len(lines) == 1:
        return f"Most recently: {lines[0]}."

    # Keep it short for TTS — at most 5 lines
    trimmed = lines[-5:]  # newest last
    summary = "; ".join(trimmed)
    return f"Here's what happened recently: {summary}."
