"""
describe_screen — Deterministic screen description tool.

Internally calls get_active_window + perceive_uia_focused_window,
then formats a short human-readable summary suitable for voice output.

Returns:
  {
    summary: str,          # 1-3 sentence spoken reply
    highlights: [str],     # Up to 6 notable UI items
    window: {title, exe},  # Focused window info
    evidence: {...},       # Raw UIA data for debugging / LLM context
  }
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from wyzer.tools.tool_base import ToolBase

# Control types that are most relevant for a short summary.
_HIGHLIGHT_TYPES = frozenset({
    "Button", "TabItem", "MenuItem", "Edit", "ComboBox",
    "CheckBox", "RadioButton", "Hyperlink", "ListItem",
    "TreeItem", "Slider",
})

_MAX_HIGHLIGHTS = 6


def _format_screen_summary(window_info: Dict[str, Any],
                           uia_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build a short spoken summary from raw UIA + window data."""

    # ── Window identification ────────────────────────────────────────
    title = (window_info.get("title") or "").strip()
    exe = (window_info.get("exe") or "").strip()
    if not title:
        title = (uia_info.get("window") or {}).get("title", "").strip()
    if not exe:
        exe = (uia_info.get("window") or {}).get("exe", "").strip()

    # Friendly app name (drop .exe)
    app_display = exe.replace(".exe", "").capitalize() if exe else None

    # ── Collect highlights (meaningful interactive controls) ──────────
    controls: List[Dict[str, Any]] = uia_info.get("controls") or []
    highlights: List[str] = []
    seen: set = set()

    for ctrl in controls:
        if len(highlights) >= _MAX_HIGHLIGHTS:
            break
        ctype = ctrl.get("control_type", "")
        name = (ctrl.get("name") or "").strip()
        if not name:
            continue
        if ctype not in _HIGHLIGHT_TYPES:
            continue
        # Deduplicate by name (case-insensitive)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        highlights.append(name)

    # ── Progress bar ─────────────────────────────────────────────────
    progress = uia_info.get("progress")
    progress_str = ""
    if progress:
        pval = progress.get("value")
        ptxt = progress.get("text", "").strip()
        if pval is not None:
            progress_str = f" A progress bar is at {pval}%."
        elif ptxt:
            progress_str = f" A progress bar shows: {ptxt}."

    # ── Dialog overlay ───────────────────────────────────────────────
    dialogs: list = uia_info.get("dialogs") or []
    dialog_str = ""
    if dialogs:
        dtitle = (dialogs[0].get("title") or "").strip()
        if dtitle and dtitle != title:
            dialog_str = f' A dialog is open: "{dtitle}".'

    # ── Compose summary ──────────────────────────────────────────────
    parts: list[str] = []

    # Sentence 1: window identification
    if app_display and title and title != app_display:
        # Truncate very long titles for speech
        display_title = title if len(title) <= 70 else title[:67] + "..."
        parts.append(f"The focused window is {app_display}: \"{display_title}\".")
    elif app_display:
        parts.append(f"The focused window is {app_display}.")
    elif title:
        display_title = title if len(title) <= 70 else title[:67] + "..."
        parts.append(f"The focused window is \"{display_title}\".")
    else:
        parts.append("I could not identify the focused window.")

    # Sentence 2: highlights (or UIA failure notice)
    uia_errors = uia_info.get("errors") or []
    uia_failed = any(
        e in ("pywinauto_not_installed", "no_foreground_window",
              "no_top_windows", "foreground_not_in_uia_list")
        for e in uia_errors
    )
    if uia_failed:
        parts.append(
            "I couldn't read UI controls due to an accessibility/permission issue."
        )
    elif highlights:
        if len(highlights) == 1:
            parts.append(f"I found a control labeled {highlights[0]}.")
        else:
            items_str = ", ".join(highlights[:-1]) + f", and {highlights[-1]}"
            parts.append(f"Notable items: {items_str}.")
    else:
        parts.append("No readable interactive controls were found.")

    # Sentence 3 (optional): dialog / progress
    if dialog_str:
        parts.append(dialog_str.strip())
    if progress_str:
        parts.append(progress_str.strip())

    summary = " ".join(parts)

    # ── Errors ───────────────────────────────────────────────────────
    errors = uia_info.get("errors") or []

    return {
        "summary": summary,
        "highlights": highlights,
        "window": {"title": title, "exe": exe},
        "evidence": {
            "control_count": len(controls),
            "dialog_count": len(dialogs),
            "progress": progress,
            "errors": errors,
        },
    }


class DescribeScreenTool(ToolBase):
    """Deterministic screen description — calls UIA then formats a spoken summary."""

    def __init__(self):
        super().__init__()
        self._name = "describe_screen"
        self._description = (
            "Describe the focused window: title, exe, and up to 6 notable UI controls. "
            "Returns a short spoken summary. Deterministic, no LLM."
        )
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.tools.desktop.get_active_window import get_active_window_info
        from wyzer.tools.desktop.perceive_uia import perceive_uia_focused_window
        from wyzer.context.world_state import emit_event, update_last_perception

        start = time.perf_counter()

        # Step 1: fast Win32 window metadata
        window_info = get_active_window_info()

        # Step 2: UIA tree walk (public API — always returns full schema)
        uia_info = perceive_uia_focused_window(max_nodes=60)
        update_last_perception(uia_info)

        # Step 3: format
        result = _format_screen_summary(window_info, uia_info)

        latency_ms = int((time.perf_counter() - start) * 1000)
        result["latency_ms"] = latency_ms

        # Emit event
        emit_event("perception", {
            "source": "describe_screen",
            "window_title": result["window"].get("title", ""),
            "highlight_count": len(result["highlights"]),
            "latency_ms": latency_ms,
        })

        return result
