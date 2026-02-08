"""
screenshot_focused_window — Desktop Ground Truth Tool (Phase 14)

Capture a screenshot of the focused window (or full screen fallback).
Returns: {image_path, hwnd, rect}

Uses mss (already in repo).
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict

from wyzer.tools.tool_base import ToolBase

# Temp directory for Wyzer screenshots
_SCREENSHOT_DIR: str | None = None


def _get_screenshot_dir() -> str:
    global _SCREENSHOT_DIR
    if _SCREENSHOT_DIR is None:
        _SCREENSHOT_DIR = os.path.join(tempfile.gettempdir(), "wyzer_screenshots")
        os.makedirs(_SCREENSHOT_DIR, exist_ok=True)
    return _SCREENSHOT_DIR


def _capture_focused_window() -> Dict[str, Any]:
    """Capture screenshot of the focused window using mss."""
    import ctypes
    import ctypes.wintypes

    hwnd = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd:
        return {"error": "no_foreground_window", "uac_detected": True}

    rect = ctypes.wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return {"error": "cannot_get_window_rect"}

    region = {
        "left": rect.left,
        "top": rect.top,
        "width": max(rect.right - rect.left, 1),
        "height": max(rect.bottom - rect.top, 1),
    }

    try:
        import mss
        with mss.mss() as sct:
            img = sct.grab(region)
            ts = int(time.time() * 1000)
            filename = f"wyzer_screenshot_{ts}.png"
            filepath = os.path.join(_get_screenshot_dir(), filename)

            # mss returns raw BGRA; convert via Pillow (already a dep)
            from PIL import Image
            pil_img = Image.frombytes("RGB", img.size, img.rgb)
            pil_img.save(filepath, "PNG")

        return {
            "image_path": filepath,
            "hwnd": hwnd,
            "rect": {"l": rect.left, "t": rect.top, "r": rect.right, "b": rect.bottom},
        }

    except ImportError as ie:
        return {"error": f"missing_dependency: {ie}"}
    except Exception as exc:
        return {"error": str(exc)}


class ScreenshotFocusedWindowTool(ToolBase):
    """Capture a screenshot of the focused window."""

    def __init__(self):
        super().__init__()
        self._name = "screenshot_focused_window"
        self._description = (
            "Take a screenshot of the currently focused window. "
            "Returns the path to the saved PNG. Deterministic."
        )
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        start = time.perf_counter()
        result = _capture_focused_window()
        latency_ms = int((time.perf_counter() - start) * 1000)
        result["latency_ms"] = latency_ms

        emit_event("perception", {
            "source": "screenshot",
            "image_path": result.get("image_path"),
            "success": "error" not in result,
            "latency_ms": latency_ms,
        })

        return result
