"""
wyzer.vision.window_context

Phase 9: Screen Awareness (READ-ONLY)

Provides detection of the foreground application and window title.
This is read-only awareness - NO OCR, NO screenshots, NO UI automation.

Returns:
    {
        "app": str | None,      # Process name (e.g., "chrome.exe")
        "title": str | None,    # Window title
        "pid": int | None       # Process ID
    }
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Any

# Logging
_logger = None
_last_log_time = 0.0
_LOG_THROTTLE_SECONDS = 2.0  # Avoid log spam - only log once per N seconds


def _get_logger():
    """Lazy-load logger to avoid circular imports."""
    global _logger
    if _logger is None:
        try:
            from wyzer.core.logger import get_logger
            _logger = get_logger()
        except Exception:
            _logger = None
    return _logger


def _log_debug(msg: str) -> None:
    """Log at DEBUG level, throttled to avoid spam."""
    global _last_log_time
    now = time.time()
    if now - _last_log_time < _LOG_THROTTLE_SECONDS:
        return
    _last_log_time = now
    
    logger = _get_logger()
    if logger:
        try:
            logger.debug(msg)
        except Exception:
            pass


# ============================================================================
# Windows-only implementation
# ============================================================================
# Try pywin32 first (preferred), fall back to ctypes

HAS_PYWIN32 = False
try:
    import win32gui
    import win32process
    import psutil
    HAS_PYWIN32 = True
except ImportError:
    pass

# ctypes fallback
HAS_CTYPES = False
try:
    import ctypes
    from ctypes import wintypes
    HAS_CTYPES = True
except ImportError:
    pass


def _get_foreground_pywin32() -> Dict[str, Any]:
    """Get foreground window using pywin32 + psutil."""
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return {"app": None, "title": None, "pid": None}
        
        # Get window title
        title = None
        try:
            title = win32gui.GetWindowText(hwnd)
            if not title:
                title = None
        except Exception:
            title = None
        
        # Get process ID
        pid = None
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            pid = None
        
        # Get process name via psutil
        app = None
        if pid:
            try:
                proc = psutil.Process(pid)
                app = proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                app = None
        
        return {"app": app, "title": title, "pid": pid}
    
    except Exception:
        return {"app": None, "title": None, "pid": None}


def _get_foreground_ctypes() -> Dict[str, Any]:
    """Get foreground window using ctypes (fallback)."""
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        
        # GetForegroundWindow
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return {"app": None, "title": None, "pid": None}
        
        # Get window title
        title = None
        try:
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if not title:
                    title = None
        except Exception:
            title = None
        
        # Get process ID
        pid = None
        try:
            pid_val = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_val))
            pid = pid_val.value if pid_val.value else None
        except Exception:
            pid = None
        
        # Get process name via psutil (if available) or ctypes
        app = None
        if pid:
            # Try psutil first
            try:
                import psutil
                proc = psutil.Process(pid)
                app = proc.name()
            except Exception:
                # Fall back to ctypes OpenProcess + QueryFullProcessImageName
                try:
                    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                    MAX_PATH = 260
                    
                    h_process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                    if h_process:
                        try:
                            buf = ctypes.create_unicode_buffer(MAX_PATH)
                            size = wintypes.DWORD(MAX_PATH)
                            if kernel32.QueryFullProcessImageNameW(h_process, 0, buf, ctypes.byref(size)):
                                full_path = buf.value
                                if full_path:
                                    app = os.path.basename(full_path)
                        finally:
                            kernel32.CloseHandle(h_process)
                except Exception:
                    app = None
        
        return {"app": app, "title": title, "pid": pid}
    
    except Exception:
        return {"app": None, "title": None, "pid": None}


def get_foreground_window() -> Dict[str, Any]:
    """
    Get information about the current foreground window.
    
    This is a READ-ONLY operation. No screenshots, OCR, or automation.
    
    Returns:
        dict with keys:
            - "app": str | None - Process name (e.g., "chrome.exe")
            - "title": str | None - Window title
            - "pid": int | None - Process ID
            
    On any failure, returns None values (never throws).
    """
    result = {"app": None, "title": None, "pid": None}
    
    # Windows-only
    if os.name != "nt":
        _log_debug("[WINDOW_CONTEXT] Non-Windows platform, returning None")
        return result
    
    # Try pywin32 first (preferred)
    if HAS_PYWIN32:
        try:
            result = _get_foreground_pywin32()
        except Exception:
            result = {"app": None, "title": None, "pid": None}
    
    # Fallback to ctypes if pywin32 failed or unavailable
    if result["app"] is None and result["title"] is None and HAS_CTYPES:
        try:
            result = _get_foreground_ctypes()
        except Exception:
            result = {"app": None, "title": None, "pid": None}
    
    # Log result (throttled)
    app = result.get("app") or "unknown"
    title = result.get("title") or "unknown"
    pid = result.get("pid") or "unknown"
    _log_debug(f"[WINDOW_CONTEXT] app={app} title={title} pid={pid}")
    
    return result


def get_visual_context_block() -> str:
    """
    Get a formatted visual context block for LLM prompt injection.
    
    Returns a small read-only block like:
    
        Visual Context (read-only):
        - Foreground app: chrome.exe
        - Window title: GitHub - Google Chrome
    
    This is purely informational and should not trigger tools.
    
    Returns:
        Formatted string block, or empty string if detection fails.
    """
    info = get_foreground_window()
    
    app = info.get("app") or "unknown"
    title = info.get("title") or "unknown"
    
    # Don't include block if both are unknown
    if app == "unknown" and title == "unknown":
        return ""
    
    # Truncate title if too long (keep prompts lean)
    if len(title) > 80:
        title = title[:77] + "..."
    
    return f"""
Visual Context (read-only):
- Foreground app: {app}
- Window title: {title}
"""
