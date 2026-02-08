"""
desktop_click_uia — Desktop Ground Truth Tool (Phase 14 + Phase 16+ upgrades)

Find + invoke a named UI control in the focused window via UIA.

Input:  {name: str, control_type?: str, scope: "focused_window"}
Output: Unified schema:
    {ok: bool, clicked: bool, method: "uia_invoke"|"uia_focus"|"rect_click",
     fallback_used: bool, matched: {name, type, rect}?, reason: str,
     latency_ms: int}
Emits:  ui_action event.

Retry chain (deterministic, max 4 attempts):
    1. InvokePattern on best match
    2. SetFocus + Enter
    3. Rect center click
    4. Return failure (no infinite retries)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from wyzer.tools.tool_base import ToolBase


def _best_match(name: str, control_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Find the best-matching control in the focused window and invoke it.

    Returns unified schema: {ok, clicked, method, fallback_used, matched, reason}.
    """
    try:
        from pywinauto import Desktop
    except ImportError:
        return {"ok": False, "clicked": False, "method": "none",
                "fallback_used": False, "reason": "pywinauto not installed"}

    import ctypes
    hwnd_fg = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd_fg:
        return {"ok": False, "clicked": False, "method": "none",
                "fallback_used": False, "reason": "no_foreground_window",
                "uac_detected": True}

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
            return {"ok": False, "clicked": False, "method": "none",
                    "fallback_used": False, "reason": "foreground_not_in_uia_list"}

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
            return {"ok": False, "clicked": False, "method": "none",
                    "fallback_used": False,
                    "reason": f"no_control_matching '{name}'"}

        # Prefer exact match, then first substring match
        candidates.sort(key=lambda c: (not c["exact"], 0))
        best = candidates[0]
        elem = best["elem"]
        matched_info = {"name": best["name"], "type": best["type"], "rect": best["rect"]}

        # ── Step 1: Try InvokePattern ───────────────────────────────
        try:
            iface = elem.iface_invoke
            if iface:
                iface.Invoke()
                return {
                    "ok": True, "clicked": True, "method": "uia_invoke",
                    "fallback_used": False, "matched": matched_info,
                    "reason": "invoke_pattern",
                }
        except Exception:
            pass

        # ── Step 2: SetFocus + Enter ────────────────────────────────
        try:
            elem.set_focus()
            time.sleep(0.05)
            import pyautogui
            pyautogui.press("enter")
            return {
                "ok": True, "clicked": True, "method": "uia_focus",
                "fallback_used": True, "matched": matched_info,
                "reason": "set_focus_enter",
            }
        except Exception:
            pass

        # ── Step 3: click_input fallback ────────────────────────────
        try:
            elem.click_input()
            return {
                "ok": True, "clicked": True, "method": "rect_click",
                "fallback_used": True, "matched": matched_info,
                "reason": "click_input",
            }
        except Exception as click_err:
            return {
                "ok": False, "clicked": False, "method": "rect_click",
                "fallback_used": True, "matched": matched_info,
                "reason": f"click_failed: {click_err}",
            }

    except Exception as exc:
        return {"ok": False, "clicked": False, "method": "none",
                "fallback_used": False, "reason": str(exc)}


def _best_match_with_retry(
    name: str,
    control_type: Optional[str] = None,
    candidate_rect: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Extended match with retry chain and optional rect fallback.

    Called by ``click_and_type._do_click`` when the resolver has already
    found a candidate with a known rect.  This enables an additional
    rect-center click if the UIA element's own click methods fail.

    Args:
        name: Control text/name.
        control_type: Optional type filter.
        candidate_rect: Pre-resolved rect from the resolver, used as
                        last-resort fallback.

    Returns:
        Unified click result dict.
    """
    result = _best_match(name, control_type)
    if result.get("clicked"):
        return result

    # ── Last-resort: rect center from resolver ──────────────────────
    rect = candidate_rect or (result.get("matched") or {}).get("rect")
    if rect:
        cx = (rect.get("l", 0) + rect.get("r", 0)) // 2
        cy = (rect.get("t", 0) + rect.get("b", 0)) // 2
        try:
            import pyautogui
            pyautogui.click(x=cx, y=cy)
            return {
                "ok": True, "clicked": True, "method": "rect_click",
                "fallback_used": True,
                "matched": result.get("matched") or {"name": name, "type": control_type, "rect": rect},
                "reason": "resolver_rect_center",
            }
        except Exception as e:
            return {
                "ok": False, "clicked": False, "method": "rect_click",
                "fallback_used": True, "reason": f"rect_fallback failed: {e}",
            }

    return result


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

        # Ensure unified schema fields
        result.setdefault("ok", result.get("clicked", False))
        result.setdefault("fallback_used", False)
        result.setdefault("method", "none")

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
            "method": result.get("method", "none"),
            "fallback_used": result.get("fallback_used", False),
            "latency_ms": latency_ms,
        })

        return result
