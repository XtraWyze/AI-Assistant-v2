"""
input_actions — Desktop Ground Truth Tool (Phase 14)

Minimal input tools: hotkey, type_text, click_xy, scroll, wait_ms.
Uses pyautogui (already in repo) for simplicity.
Each action emits a ui_action event.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from wyzer.tools.tool_base import ToolBase


class WaitMsTool(ToolBase):
    """Wait for a specified number of milliseconds."""

    def __init__(self):
        super().__init__()
        self._name = "wait_ms"
        self._description = "Wait/sleep for a given number of milliseconds."
        self._args_schema = {
            "type": "object",
            "properties": {
                "ms": {"type": "integer", "description": "Milliseconds to wait."},
            },
            "required": ["ms"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        ms = max(0, min(kwargs.get("ms", 0), 30000))  # cap at 30s
        time.sleep(ms / 1000.0)
        emit_event("ui_action", {"kind": "wait_ms", "ms": ms})
        return {"waited_ms": ms}


class HotkeyTool(ToolBase):
    """Press a keyboard hotkey (e.g., ctrl+c)."""

    def __init__(self):
        super().__init__()
        self._name = "hotkey"
        self._description = "Press a keyboard hotkey combo. E.g. 'ctrl+c', 'alt+f4'."
        self._args_schema = {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "string",
                    "description": "Hotkey string, e.g. 'ctrl+c', 'alt+tab'.",
                },
            },
            "required": ["keys"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        keys_str = kwargs.get("keys", "")
        parts = [k.strip() for k in keys_str.split("+") if k.strip()]
        if not parts:
            return {"error": "no_keys_specified"}

        try:
            import pyautogui
            pyautogui.hotkey(*parts)
            emit_event("ui_action", {"kind": "hotkey", "keys": keys_str, "success": True})
            return {"pressed": keys_str, "success": True}
        except Exception as exc:
            emit_event("ui_action", {"kind": "hotkey", "keys": keys_str, "success": False, "error": str(exc)})
            return {"pressed": keys_str, "success": False, "error": str(exc)}


class TypeTextTool(ToolBase):
    """Type text into the focused control."""

    def __init__(self):
        super().__init__()
        self._name = "type_text"
        self._description = "Type text into the currently focused input field."
        self._args_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type."},
                "interval": {"type": "number", "description": "Seconds between keystrokes (default 0.02)."},
            },
            "required": ["text"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        text = kwargs.get("text", "")
        interval = kwargs.get("interval", 0.02)

        try:
            import pyautogui
            pyautogui.typewrite(text, interval=interval) if text.isascii() else pyautogui.write(text)
            emit_event("ui_action", {"kind": "type_text", "length": len(text), "success": True})
            return {"typed_length": len(text), "success": True}
        except Exception as exc:
            emit_event("ui_action", {"kind": "type_text", "success": False, "error": str(exc)})
            return {"success": False, "error": str(exc)}


class ClickXYTool(ToolBase):
    """Click at absolute screen coordinates."""

    def __init__(self):
        super().__init__()
        self._name = "click_xy"
        self._description = "Click at absolute screen coordinates (x, y)."
        self._args_schema = {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate."},
                "y": {"type": "integer", "description": "Y coordinate."},
                "button": {"type": "string", "description": "'left', 'right', or 'middle'. Default 'left'."},
                "clicks": {"type": "integer", "description": "Number of clicks (default 1)."},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        x = kwargs.get("x", 0)
        y = kwargs.get("y", 0)
        button = kwargs.get("button", "left")
        clicks = kwargs.get("clicks", 1)

        try:
            import pyautogui
            pyautogui.click(x=x, y=y, button=button, clicks=clicks)
            emit_event("ui_action", {"kind": "click_xy", "x": x, "y": y, "button": button, "success": True})
            return {"x": x, "y": y, "button": button, "clicks": clicks, "success": True}
        except Exception as exc:
            emit_event("ui_action", {"kind": "click_xy", "x": x, "y": y, "success": False, "error": str(exc)})
            return {"success": False, "error": str(exc)}


class ScrollTool(ToolBase):
    """Scroll the mouse wheel."""

    def __init__(self):
        super().__init__()
        self._name = "scroll"
        self._description = "Scroll the mouse wheel at the current position or given coordinates."
        self._args_schema = {
            "type": "object",
            "properties": {
                "clicks": {"type": "integer", "description": "Scroll amount (positive=up, negative=down)."},
                "x": {"type": "integer", "description": "Optional X coordinate."},
                "y": {"type": "integer", "description": "Optional Y coordinate."},
            },
            "required": ["clicks"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        scroll_clicks = kwargs.get("clicks", 0)
        x = kwargs.get("x")
        y = kwargs.get("y")

        try:
            import pyautogui
            pyautogui.scroll(scroll_clicks, x=x, y=y)
            emit_event("ui_action", {"kind": "scroll", "clicks": scroll_clicks, "success": True})
            return {"scrolled": scroll_clicks, "success": True}
        except Exception as exc:
            emit_event("ui_action", {"kind": "scroll", "success": False, "error": str(exc)})
            return {"success": False, "error": str(exc)}
