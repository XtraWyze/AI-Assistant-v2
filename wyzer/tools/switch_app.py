"""
wyzer.tools.switch_app

Deterministic app switching tool using focus history.

This tool provides reliable app switching based on actual focus events,
not heuristic guessing. It uses the focus_stack maintained in world_state
to track which apps were recently focused.

Modes:
- named: Focus the most recent window matching an app name
- previous: Switch to the immediately previous focused app
- next: Cycle forward through focus_stack (round-robin)

Risk: LOW_RISK (just window focus, no data modification)
"""

import time
from typing import Any, Dict, Optional

from wyzer.tools.tool_base import ToolBase
from wyzer.context.world_state import (
    get_previous_focused_app,
    get_next_focused_app,
    find_app_in_focus_stack,
    get_current_focused_app,
    get_focus_stack,
    push_focus_stack,
)


# Try to import pywin32 for window focusing
HAS_PYWIN32 = False
try:
    import win32gui
    import win32con
    HAS_PYWIN32 = True
except ImportError:
    pass

# Fallback to ctypes
HAS_CTYPES = False
try:
    import ctypes
    user32 = ctypes.windll.user32
    HAS_CTYPES = True
except ImportError:
    pass

# Windows constants
SW_RESTORE = 9


def _is_window_valid(hwnd: int) -> bool:
    """Check if a window handle is still valid."""
    if not hwnd:
        return False
    
    if HAS_PYWIN32:
        try:
            return win32gui.IsWindow(hwnd)
        except Exception:
            return False
    elif HAS_CTYPES:
        try:
            return bool(user32.IsWindow(hwnd))
        except Exception:
            return False
    return False


def _focus_window(hwnd: int) -> bool:
    """
    Focus a window by handle.
    
    Args:
        hwnd: Window handle
        
    Returns:
        True if focus succeeded, False otherwise
    """
    if not _is_window_valid(hwnd):
        return False
    
    try:
        if HAS_PYWIN32:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            return True
        elif HAS_CTYPES:
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
            return True
    except Exception:
        pass
    
    return False


def _format_app_name(app: str) -> str:
    """Format app name for display (remove .exe, title case)."""
    if not app:
        return "Unknown"
    
    name = app.strip()
    if name.lower().endswith(".exe"):
        name = name[:-4]
    
    # Common app name mappings
    display_names = {
        "chrome": "Chrome",
        "firefox": "Firefox",
        "msedge": "Edge",
        "edge": "Edge",
        "code": "VS Code",
        "spotify": "Spotify",
        "discord": "Discord",
        "slack": "Slack",
        "notepad": "Notepad",
        "explorer": "File Explorer",
        "windowsterminal": "Terminal",
        "wt": "Terminal",
        "steam": "Steam",
    }
    
    lower = name.lower()
    if lower in display_names:
        return display_names[lower]
    
    return name.title() if name.islower() or name.isupper() else name


class SwitchAppTool(ToolBase):
    """
    Deterministic app switching tool.
    
    Uses focus_stack from world_state to switch between apps based on
    actual focus history, not heuristic guessing.
    
    Modes:
    - named: Focus the most recent window matching app name
    - previous: Switch to the immediately previous focused app
    - next: Cycle forward through focus_stack (round-robin)
    """
    
    def __init__(self):
        super().__init__()
        self._name = "switch_app"
        self._description = "Switch between applications using focus history"
        self._args_schema = {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["named", "previous", "next"],
                    "description": "Switching mode: 'named' (by app name), 'previous' (last app), 'next' (cycle)"
                },
                "app": {
                    "type": "string",
                    "description": "App name to switch to (only used with mode='named')"
                }
            },
            "required": ["mode"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        mode = kwargs.get("mode", "").lower().strip()
        app = kwargs.get("app", "").strip()
        
        # Validate mode
        if mode not in ("named", "previous", "next"):
            return {
                "error": {
                    "type": "invalid_args",
                    "message": f"Invalid mode '{mode}'. Must be 'named', 'previous', or 'next'."
                }
            }
        
        # Get current app for logging
        current = get_current_focused_app()
        current_app_name = _format_app_name(current.get("app", "")) if current else "Unknown"
        
        # Lazy import logger
        try:
            from wyzer.core.logger import get_logger
            logger = get_logger()
        except Exception:
            logger = None
        
        # Handle each mode
        if mode == "named":
            return self._switch_named(app, current_app_name, start_time, logger)
        elif mode == "previous":
            return self._switch_previous(current_app_name, start_time, logger)
        else:  # mode == "next"
            return self._switch_next(current_app_name, start_time, logger)
    
    def _switch_named(
        self,
        app: str,
        current_app: str,
        start_time: float,
        logger: Any,
    ) -> Dict[str, Any]:
        """Switch to a named app."""
        if not app:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "App name required for mode='named'"
                }
            }
        
        # First, try to find the app in focus stack (for fast switching)
        entry = find_app_in_focus_stack(app)
        
        if entry:
            # Found in focus stack - use the cached hwnd
            target_app = _format_app_name(entry.get("app", ""))
            hwnd = entry.get("hwnd")
            
            # Check if already on this app
            if target_app.lower() == current_app.lower():
                end_time = time.perf_counter()
                if logger:
                    logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_app}\" mode=named status=already_focused")
                return {
                    "status": "already_focused",
                    "app": target_app,
                    "message": f"Already on {target_app}.",
                    "spoken": f"You're already on {target_app}.",
                    "latency_ms": int((end_time - start_time) * 1000),
                }
            
            # Focus the window
            if _focus_window(hwnd):
                push_focus_stack(entry.get("app", ""), hwnd, entry.get("title", ""))
                end_time = time.perf_counter()
                if logger:
                    logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_app}\" mode=named status=success")
                return {
                    "status": "switched",
                    "from_app": current_app,
                    "to_app": target_app,
                    "hwnd": hwnd,
                    "message": f"Switched to {target_app}.",
                    "spoken": f"Switching to {target_app}.",
                    "latency_ms": int((end_time - start_time) * 1000),
                }
            # If focus failed, fall through to window_manager lookup
        
        # Not in focus stack or focus failed - use window_manager's proven window finding
        # This handles LocalLibrary resolution, process name matching, etc.
        try:
            from wyzer.tools.window_manager import _resolve_window_handle, _get_window_info, _enumerate_windows
            
            # Debug: log available windows for troubleshooting
            if logger:
                all_windows = _enumerate_windows()
                procs = [w.get("process", "?") for w in all_windows[:10]]
                logger.debug(f"[APP_SWITCH] Enumerated {len(all_windows)} windows, top procs: {procs}")
            
            hwnd, learned_alias, effective_process = _resolve_window_handle(title=app, process=None)
            
            if hwnd:
                window_info = _get_window_info(hwnd)
                target_app = _format_app_name(
                    window_info.get("process") or effective_process or app
                )
                
                # Check if already on this app
                if target_app.lower() == current_app.lower():
                    end_time = time.perf_counter()
                    if logger:
                        logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_app}\" mode=named status=already_focused")
                    return {
                        "status": "already_focused",
                        "app": target_app,
                        "message": f"Already on {target_app}.",
                        "spoken": f"You're already on {target_app}.",
                        "latency_ms": int((end_time - start_time) * 1000),
                    }
                
                # Focus the window
                if _focus_window(hwnd):
                    # Update focus stack
                    push_focus_stack(
                        window_info.get("process") or effective_process or app,
                        hwnd,
                        window_info.get("title", "")
                    )
                    end_time = time.perf_counter()
                    if logger:
                        logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_app}\" mode=named status=success")
                    return {
                        "status": "switched",
                        "from_app": current_app,
                        "to_app": target_app,
                        "hwnd": hwnd,
                        "matched": window_info,
                        "learned_alias": learned_alias,
                        "message": f"Switched to {target_app}.",
                        "spoken": f"Switching to {target_app}.",
                        "latency_ms": int((end_time - start_time) * 1000),
                    }
            
            # _resolve_window_handle didn't find it - try direct process name search
            # This fallback uses fresh window enumeration, not cached world_state
            if not hwnd:
                all_windows = _enumerate_windows()
                app_lower = app.lower().strip()
                if app_lower.endswith(".exe"):
                    app_lower = app_lower[:-4]
                
                for window in all_windows:
                    win_process = (window.get("process") or "").lower()
                    if win_process.endswith(".exe"):
                        win_process = win_process[:-4]
                    
                    # Match if app name is contained in process name or vice versa
                    if app_lower in win_process or win_process in app_lower:
                        hwnd = window.get("hwnd")
                        if _focus_window(hwnd):
                            target_app = _format_app_name(window.get("process") or app)
                            push_focus_stack(window.get("process", ""), hwnd, window.get("title", ""))
                            end_time = time.perf_counter()
                            if logger:
                                logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_app}\" mode=named status=success (enum fallback)")
                            return {
                                "status": "switched",
                                "from_app": current_app,
                                "to_app": target_app,
                                "hwnd": hwnd,
                                "message": f"Switched to {target_app}.",
                                "spoken": f"Switching to {target_app}.",
                                "latency_ms": int((end_time - start_time) * 1000),
                            }
                        break
                        
        except Exception as e:
            if logger:
                logger.warning(f"[APP_SWITCH] window_manager fallback failed: {e}")
        
        # Still not found - try open_windows as last resort
        from wyzer.context.world_state import get_all_open_windows
        open_windows = get_all_open_windows()
        
        app_lower = app.lower().strip()
        if app_lower.endswith(".exe"):
            app_lower = app_lower[:-4]
        
        for window in open_windows:
            win_process = (window.get("process") or "").lower()
            if win_process.endswith(".exe"):
                win_process = win_process[:-4]
            
            if app_lower in win_process or win_process in app_lower:
                hwnd = window.get("hwnd")
                if _focus_window(hwnd):
                    target_app = _format_app_name(window.get("process") or app)
                    push_focus_stack(window.get("process", ""), hwnd, window.get("title", ""))
                    end_time = time.perf_counter()
                    if logger:
                        logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_app}\" mode=named status=success")
                    return {
                        "status": "switched",
                        "from_app": current_app,
                        "to_app": target_app,
                        "hwnd": hwnd,
                        "message": f"Switched to {target_app}.",
                        "spoken": f"Switching to {target_app}.",
                        "latency_ms": int((end_time - start_time) * 1000),
                    }
        
        # App not found anywhere
        end_time = time.perf_counter()
        target_display = _format_app_name(app)
        
        if logger:
            logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_display}\" mode=named status=not_found")
        
        return {
            "status": "not_found",
            "error": {
                "type": "app_not_found",
                "message": f"Couldn't find {target_display} in recent apps or open windows."
            },
            "message": f"Couldn't find {target_display} in recent apps or open windows.",
            "spoken": f"I couldn't find {target_display}.",
            "latency_ms": int((end_time - start_time) * 1000),
        }
    
    def _switch_previous(
        self,
        current_app: str,
        start_time: float,
        logger: Any,
    ) -> Dict[str, Any]:
        """Switch to the previously focused app."""
        entry = get_previous_focused_app()
        
        if not entry:
            end_time = time.perf_counter()
            
            if logger:
                logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=None mode=previous status=no_history")
            
            return {
                "status": "no_history",
                "error": {
                    "type": "no_history",
                    "message": "No previous app to switch to."
                },
                "message": "No previous app to switch to.",
                "spoken": "No previous app to switch to.",
                "latency_ms": int((end_time - start_time) * 1000),
            }
        
        target_app = _format_app_name(entry.get("app", ""))
        hwnd = entry.get("hwnd")
        
        # Focus the window
        if not _focus_window(hwnd):
            end_time = time.perf_counter()
            
            if logger:
                logger.warning(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_app}\" mode=previous status=focus_failed")
            
            return {
                "error": {
                    "type": "focus_failed",
                    "message": f"Failed to focus {target_app}"
                },
                "latency_ms": int((end_time - start_time) * 1000),
            }
        
        # Update focus stack after successful switch
        push_focus_stack(entry.get("app", ""), hwnd, entry.get("title", ""))
        
        end_time = time.perf_counter()
        
        if logger:
            logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_app}\" mode=previous status=success")
        
        return {
            "status": "switched",
            "from_app": current_app,
            "to_app": target_app,
            "hwnd": hwnd,
            "message": f"Switched back to {target_app}.",
            "spoken": f"Switching back to {target_app}.",
            "latency_ms": int((end_time - start_time) * 1000),
        }
    
    def _switch_next(
        self,
        current_app: str,
        start_time: float,
        logger: Any,
    ) -> Dict[str, Any]:
        """Cycle to the next app in focus history."""
        entry = get_next_focused_app()
        
        if not entry:
            end_time = time.perf_counter()
            
            if logger:
                logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=None mode=next status=no_apps")
            
            return {
                "status": "no_apps",
                "error": {
                    "type": "no_apps",
                    "message": "Only one app in focus history."
                },
                "message": "Only one app in focus history.",
                "spoken": "Only one app available.",
                "latency_ms": int((end_time - start_time) * 1000),
            }
        
        target_app = _format_app_name(entry.get("app", ""))
        hwnd = entry.get("hwnd")
        
        # Focus the window
        if not _focus_window(hwnd):
            end_time = time.perf_counter()
            
            if logger:
                logger.warning(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_app}\" mode=next status=focus_failed")
            
            return {
                "error": {
                    "type": "focus_failed",
                    "message": f"Failed to focus {target_app}"
                },
                "latency_ms": int((end_time - start_time) * 1000),
            }
        
        # Note: For "next", we don't push to focus stack - the natural focus
        # change will be detected by window watcher
        
        end_time = time.perf_counter()
        
        if logger:
            logger.info(f"[APP_SWITCH] from=\"{current_app}\" to=\"{target_app}\" mode=next status=success")
        
        return {
            "status": "switched",
            "from_app": current_app,
            "to_app": target_app,
            "hwnd": hwnd,
            "message": f"Switched to {target_app}.",
            "spoken": f"Switching to {target_app}.",
            "latency_ms": int((end_time - start_time) * 1000),
        }
