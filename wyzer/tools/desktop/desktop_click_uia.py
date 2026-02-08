"""
desktop_click_uia — Desktop Ground Truth Tool (Phase 14)

Find + invoke a named UI control in the focused window via UIA.

Input:  {name: str, control_type?: str, scope: "focused_window"}
Output: {clicked:bool, matched:{name,type,rect}?, reason?}
Emits:  ui_action event.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from wyzer.tools.tool_base import ToolBase


def _best_match(name: str, control_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Find the best-matching control in the focused window and invoke it.

    Returns {clicked, matched, reason}.
    """
    try:
        from pywinauto import Desktop
    except ImportError:
        return {"clicked": False, "reason": "pywinauto not installed"}

    import ctypes
    hwnd_fg = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd_fg:
        return {"clicked": False, "reason": "no_foreground_window", "uac_detected": True}

    try:
        desktop = Desktop(backend="uia")
        top_windows = desktop.windows()

        target_win = None
        for w in top_windows:
            try:
                if w.handle == hwnd_fg:
                    target_win = w
                    break
            except Exception:
                continue

        if target_win is None:
            return {"clicked": False, "reason": "foreground_not_in_uia_list"}

        # Search descendants for name match
        name_lower = name.strip().lower()
        candidates = []
        for elem in target_win.descendants():
            try:
                elem_name = (elem.window_text() or "").strip()
                elem_type = elem.element_info.control_type or ""
                if not elem_name:
                    continue
                if name_lower not in elem_name.lower():
                    continue
                # Optional control_type filter
                if control_type and control_type.lower() != elem_type.lower():
                    continue
                try:
                    r = elem.rectangle()
                    rect = {"l": r.left, "t": r.top, "r": r.right, "b": r.bottom}
                except Exception:
                    rect = None
                candidates.append({
                    "elem": elem,
                    "name": elem_name,
                    "type": elem_type,
                    "rect": rect,
                    "exact": elem_name.lower() == name_lower,
                })
            except Exception:
                continue

        if not candidates:
            return {"clicked": False, "reason": f"no_control_matching '{name}'"}

        # Prefer exact match, then first substring match
        candidates.sort(key=lambda c: (not c["exact"], 0))
        best = candidates[0]
        elem = best["elem"]

        # Try InvokePattern first, then click_input fallback
        clicked = False
        try:
            iface = elem.iface_invoke
            if iface:
                iface.Invoke()
                clicked = True
        except Exception:
            pass

        if not clicked:
            try:
                elem.click_input()
                clicked = True
            except Exception as click_err:
                return {
                    "clicked": False,
                    "matched": {"name": best["name"], "type": best["type"], "rect": best["rect"]},
                    "reason": f"click_failed: {click_err}",
                }

        return {
            "clicked": True,
            "matched": {"name": best["name"], "type": best["type"], "rect": best["rect"]},
        }

    except Exception as exc:
        return {"clicked": False, "reason": str(exc)}


class DesktopClickUIATool(ToolBase):
    """Click a named UI control in the focused window via UIA."""

    def __init__(self):
        super().__init__()
        self._name = "desktop_click_uia"
        self._description = (
            "Find and click a named control in the focused window using UI Automation. "
            "Deterministic, no LLM."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Text/name of the control to click.",
                },
                "control_type": {
                    "type": "string",
                    "description": "Optional UIA control type filter (e.g. 'Button').",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        name = kwargs.get("name", "")
        control_type = kwargs.get("control_type")

        start = time.perf_counter()
        result = _best_match(name, control_type)
        latency_ms = int((time.perf_counter() - start) * 1000)
        result["latency_ms"] = latency_ms

        # Deterministic summary for TTS
        if result.get("clicked"):
            matched_name = (result.get("matched") or {}).get("name", name)
            result["summary"] = f"Clicked {matched_name}."
        else:
            reason = result.get("reason", "unknown error")
            result["summary"] = f"I couldn't click {name}: {reason}."

        emit_event("ui_action", {
            "kind": "click_uia",
            "target": name,
            "matched": result.get("matched"),
            "success": result.get("clicked", False),
            "latency_ms": latency_ms,
        })

        return result
