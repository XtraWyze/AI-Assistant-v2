"""
get_active_window — Desktop Ground Truth Tool (Phase 14)

Returns deterministic metadata about the foreground window:
  {title, exe, pid, hwnd, rect:{l,t,r,b}, monitor, timestamp}

Uses pywin32 (already in repo) with ctypes fallback.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import time
from typing import Any, Dict

from wyzer.tools.tool_base import ToolBase

# ── Windows API ────────────────────────────────────────────────────────────
user32 = ctypes.windll.user32

try:
    import win32gui
    import win32process
    import psutil
    _HAS_PYWIN32 = True
except ImportError:
    _HAS_PYWIN32 = False


def _rect_dict(left: int, top: int, right: int, bottom: int) -> Dict[str, int]:
    return {"l": left, "t": top, "r": right, "b": bottom}


def _monitor_index_for_rect(left: int, top: int) -> int:
    """Best-effort monitor index (1-based) via MonitorFromPoint."""
    try:
        MONITOR_DEFAULTTONEAREST = 2
        point = ctypes.wintypes.POINT(left, top)
        hmon = ctypes.windll.user32.MonitorFromPoint(point, MONITOR_DEFAULTTONEAREST)
        # We can't trivially map hmon→index cheaply without EnumDisplayMonitors,
        # so return 1 when there's a single monitor. For multi-monitor we fall back to
        # world_state.detected_monitor_count if needed.
        return 1  # safe default; multi-monitor enrichment happens elsewhere
    except Exception:
        return 1


def _get_exe_for_pid(pid: int) -> str | None:
    """Get exe name for a PID using psutil (already a dep)."""
    try:
        import psutil as _ps
        proc = _ps.Process(pid)
        return proc.name()
    except Exception:
        return None


def _detect_uac_secure_desktop(hwnd: int, title: str | None) -> bool:
    """Best-effort detection of UAC / secure desktop."""
    if not hwnd:
        return True  # No foreground window → probably secure desktop
    if title and "user account control" in title.lower():
        return True
    return False


def get_active_window_info() -> Dict[str, Any]:
    """
    Return deterministic metadata about the foreground window.

    Returns dict with: title, exe, pid, hwnd, rect, monitor, timestamp,
    and optionally uac_detected.
    """
    ts = time.time()
    result: Dict[str, Any] = {
        "title": None,
        "exe": None,
        "pid": None,
        "hwnd": None,
        "rect": None,
        "monitor": None,
        "timestamp": ts,
    }

    try:
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            result["error"] = "no_foreground_window"
            result["uac_detected"] = True
            return result

        result["hwnd"] = hwnd

        # Window rect
        rect = ctypes.wintypes.RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            result["rect"] = _rect_dict(rect.left, rect.top, rect.right, rect.bottom)
            result["monitor"] = _monitor_index_for_rect(rect.left, rect.top)

        # Title
        if _HAS_PYWIN32:
            try:
                result["title"] = win32gui.GetWindowText(hwnd) or None
            except Exception:
                pass
        else:
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, 512)
            result["title"] = buf.value or None

        # PID
        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        result["pid"] = pid.value if pid.value else None

        # Exe
        if result["pid"]:
            result["exe"] = _get_exe_for_pid(result["pid"])

        # UAC check
        if _detect_uac_secure_desktop(hwnd, result.get("title")):
            result["uac_detected"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ── ToolBase wrapper ───────────────────────────────────────────────────────

class GetActiveWindowTool(ToolBase):
    """Deterministic foreground-window metadata tool."""

    def __init__(self):
        super().__init__()
        self._name = "get_active_window"
        self._description = (
            "Return metadata about the current foreground window: "
            "title, exe, pid, hwnd, rect, monitor. Deterministic, no LLM."
        )
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event, update_last_perception

        info = get_active_window_info()

        # Emit event
        emit_event("perception", {
            "source": "get_active_window",
            "title": info.get("title"),
            "exe": info.get("exe"),
            "hwnd": info.get("hwnd"),
        })

        # Update last_perception
        update_last_perception(info)

        return info
