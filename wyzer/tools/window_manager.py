"""
Window management tools for Windows desktop control.
"""
import time
import ctypes
from typing import Dict, Any, List, Optional, Tuple
from wyzer.tools.tool_base import ToolBase

# Windows API constants
SW_MINIMIZE = 6
SW_MAXIMIZE = 3
SW_RESTORE = 9
WM_CLOSE = 0x0010

# Try to import pywin32, fallback to ctypes
try:
    import win32gui
    import win32con
    import win32process
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False

# ctypes Windows API definitions
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


def _enumerate_windows() -> List[Dict[str, Any]]:
    """
    Enumerate all visible windows.
    
    Returns:
        List of window info dicts
    """
    windows = []
    
    if HAS_PYWIN32:
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        process_name = _get_process_name_pywin32(pid)
                    except:
                        process_name = ""
                    
                    windows.append({
                        "hwnd": hwnd,
                        "title": title,
                        "pid": pid,
                        "process": process_name
                    })
            return True
        
        win32gui.EnumWindows(callback, None)
    else:
        # ctypes fallback
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        
        def callback(hwnd, lParam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value
                    
                    if title:
                        pid = ctypes.c_ulong()
                        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                        process_name = _get_process_name_ctypes(pid.value)
                        
                        windows.append({
                            "hwnd": hwnd,
                            "title": title,
                            "pid": pid.value,
                            "process": process_name
                        })
            return True
        
        enum_func = EnumWindowsProc(callback)
        user32.EnumWindows(enum_func, 0)
    
    return windows


def _get_process_name_pywin32(pid: int) -> str:
    """Get process name from PID using pywin32"""
    try:
        import win32api
        import win32process
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
        try:
            exe = win32process.GetModuleFileNameEx(handle, 0)
            return exe.split("\\")[-1].lower()
        finally:
            win32api.CloseHandle(handle)
    except:
        return ""


def _get_process_name_ctypes(pid: int) -> str:
    """Get process name from PID using ctypes"""
    try:
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        
        if handle:
            try:
                buff = ctypes.create_unicode_buffer(260)
                size = ctypes.c_ulong(260)
                
                # QueryFullProcessImageNameW
                if kernel32.QueryFullProcessImageNameW(handle, 0, buff, ctypes.byref(size)):
                    return buff.value.split("\\")[-1].lower()
            finally:
                kernel32.CloseHandle(handle)
    except:
        pass
    
    return ""


def _find_window(title: Optional[str] = None, process: Optional[str] = None) -> Optional[int]:
    """
    Find window by title or process name.
    
    Args:
        title: Window title (substring match, case-insensitive)
        process: Process name (substring match, case-insensitive)
        
    Returns:
        Window handle (hwnd) or None if not found
    """
    windows = _enumerate_windows()
    
    # Filter by criteria
    for window in windows:
        if title and title.lower() in window["title"].lower():
            return window["hwnd"]
        if process and process.lower() in window["process"].lower():
            return window["hwnd"]
    
    return None


def _get_window_info(hwnd: int) -> Dict[str, Any]:
    """Get window information"""
    windows = _enumerate_windows()
    for window in windows:
        if window["hwnd"] == hwnd:
            return window
    return {}


class FocusWindowTool(ToolBase):
    """Tool to focus/activate a window"""
    
    def __init__(self):
        super().__init__()
        self._name = "focus_window"
        self._description = "Focus/activate a window by title or process name"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title (substring match)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name (e.g., 'chrome.exe', 'notepad')"
                }
            },
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        title = kwargs.get("title")
        process = kwargs.get("process")
        
        if not title and not process:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "Must specify either 'title' or 'process'"
                }
            }
        
        try:
            hwnd = _find_window(title, process)
            
            if not hwnd:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching criteria"
                    },
                    "latency_ms": int((end_time - start_time) * 1000)
                }
            
            # Focus window
            if HAS_PYWIN32:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
            else:
                user32.ShowWindow(hwnd, SW_RESTORE)
                user32.SetForegroundWindow(hwnd)
            
            window_info = _get_window_info(hwnd)
            end_time = time.perf_counter()
            
            return {
                "status": "focused",
                "matched": window_info,
                "latency_ms": int((end_time - start_time) * 1000)
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": int((end_time - start_time) * 1000)
            }


class MinimizeWindowTool(ToolBase):
    """Tool to minimize a window"""
    
    def __init__(self):
        super().__init__()
        self._name = "minimize_window"
        self._description = "Minimize a window by title or process name"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title (substring match)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name (e.g., 'chrome.exe', 'notepad')"
                }
            },
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        title = kwargs.get("title")
        process = kwargs.get("process")
        
        if not title and not process:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "Must specify either 'title' or 'process'"
                }
            }
        
        try:
            hwnd = _find_window(title, process)
            
            if not hwnd:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching criteria"
                    },
                    "latency_ms": int((end_time - start_time) * 1000)
                }
            
            # Minimize window
            if HAS_PYWIN32:
                win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            else:
                user32.ShowWindow(hwnd, SW_MINIMIZE)
            
            window_info = _get_window_info(hwnd)
            end_time = time.perf_counter()
            
            return {
                "status": "minimized",
                "matched": window_info,
                "latency_ms": int((end_time - start_time) * 1000)
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": int((end_time - start_time) * 1000)
            }


class MaximizeWindowTool(ToolBase):
    """Tool to maximize a window"""
    
    def __init__(self):
        super().__init__()
        self._name = "maximize_window"
        self._description = "Maximize a window by title or process name"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title (substring match)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name (e.g., 'chrome.exe', 'notepad')"
                }
            },
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        title = kwargs.get("title")
        process = kwargs.get("process")
        
        if not title and not process:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "Must specify either 'title' or 'process'"
                }
            }
        
        try:
            hwnd = _find_window(title, process)
            
            if not hwnd:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching criteria"
                    },
                    "latency_ms": int((end_time - start_time) * 1000)
                }
            
            # Maximize window
            if HAS_PYWIN32:
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            else:
                user32.ShowWindow(hwnd, SW_MAXIMIZE)
            
            window_info = _get_window_info(hwnd)
            end_time = time.perf_counter()
            
            return {
                "status": "maximized",
                "matched": window_info,
                "latency_ms": int((end_time - start_time) * 1000)
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": int((end_time - start_time) * 1000)
            }


class CloseWindowTool(ToolBase):
    """Tool to close a window"""
    
    def __init__(self):
        super().__init__()
        self._name = "close_window"
        self._description = "Close a window by title or process name (requires allowlist for force close)"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title (substring match)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name (e.g., 'chrome.exe', 'notepad')"
                },
                "force": {
                    "type": "boolean",
                    "description": "Force close (requires config flag and allowlist)",
                    "default": False
                }
            },
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.core.config import Config
        
        start_time = time.perf_counter()
        
        title = kwargs.get("title")
        process = kwargs.get("process")
        force = kwargs.get("force", False)
        
        if not title and not process:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "Must specify either 'title' or 'process'"
                }
            }
        
        # Check force close permission
        if force and not getattr(Config, "ENABLE_FORCE_CLOSE", False):
            return {
                "error": {
                    "type": "permission_denied",
                    "message": "Force close is disabled in configuration"
                },
                "latency_ms": 0
            }
        
        try:
            hwnd = _find_window(title, process)
            
            if not hwnd:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching criteria"
                    },
                    "latency_ms": int((end_time - start_time) * 1000)
                }
            
            window_info = _get_window_info(hwnd)
            
            # Check allowlist for close
            if process:
                allowed = getattr(Config, "ALLOWED_PROCESSES_TO_CLOSE", [])
                if allowed and process.lower() not in [p.lower() for p in allowed]:
                    end_time = time.perf_counter()
                    return {
                        "error": {
                            "type": "permission_denied",
                            "message": f"Process '{process}' not in allowed list"
                        },
                        "latency_ms": int((end_time - start_time) * 1000)
                    }
            
            # Close window
            if HAS_PYWIN32:
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            else:
                user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            
            end_time = time.perf_counter()
            
            return {
                "status": "closed",
                "matched": window_info,
                "latency_ms": int((end_time - start_time) * 1000)
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": int((end_time - start_time) * 1000)
            }


class MoveWindowToMonitorTool(ToolBase):
    """Tool to move window to specific monitor"""
    
    def __init__(self):
        super().__init__()
        self._name = "move_window_to_monitor"
        self._description = "Move a window to a specific monitor with positioning"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title (substring match)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name (e.g., 'chrome.exe', 'notepad')"
                },
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0-based)",
                    "minimum": 0
                },
                "position": {
                    "type": "string",
                    "enum": ["left", "right", "center", "maximize"],
                    "description": "Position on monitor",
                    "default": "maximize"
                }
            },
            "required": ["monitor"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        title = kwargs.get("title")
        process = kwargs.get("process")
        monitor_index = kwargs.get("monitor", 0)
        position = kwargs.get("position", "maximize")
        
        if not title and not process:
            return {
                "error": {
                    "type": "invalid_args",
                    "message": "Must specify either 'title' or 'process'"
                }
            }
        
        try:
            # Get monitors
            monitors = _enumerate_monitors()
            
            if monitor_index >= len(monitors):
                return {
                    "error": {
                        "type": "invalid_monitor",
                        "message": f"Monitor index {monitor_index} out of range (0-{len(monitors)-1})"
                    },
                    "latency_ms": int((time.perf_counter() - start_time) * 1000)
                }
            
            monitor = monitors[monitor_index]
            
            # Find window
            hwnd = _find_window(title, process)
            
            if not hwnd:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching criteria"
                    },
                    "latency_ms": int((end_time - start_time) * 1000)
                }
            
            # Calculate position
            mon_x = monitor["x"]
            mon_y = monitor["y"]
            mon_w = monitor["width"]
            mon_h = monitor["height"]
            
            if position == "maximize":
                # Restore first, move, then maximize
                if HAS_PYWIN32:
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                else:
                    user32.ShowWindow(hwnd, SW_RESTORE)
                
                user32.SetWindowPos(hwnd, 0, mon_x, mon_y, mon_w, mon_h, 0)
                
                if HAS_PYWIN32:
                    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                else:
                    user32.ShowWindow(hwnd, SW_MAXIMIZE)
            
            elif position == "left":
                win_w = mon_w // 2
                win_h = mon_h
                user32.SetWindowPos(hwnd, 0, mon_x, mon_y, win_w, win_h, 0)
            
            elif position == "right":
                win_w = mon_w // 2
                win_h = mon_h
                win_x = mon_x + win_w
                user32.SetWindowPos(hwnd, 0, win_x, mon_y, win_w, win_h, 0)
            
            elif position == "center":
                win_w = int(mon_w * 0.7)
                win_h = int(mon_h * 0.7)
                win_x = mon_x + (mon_w - win_w) // 2
                win_y = mon_y + (mon_h - win_h) // 2
                user32.SetWindowPos(hwnd, 0, win_x, win_y, win_w, win_h, 0)
            
            window_info = _get_window_info(hwnd)
            end_time = time.perf_counter()
            
            return {
                "status": "moved",
                "matched": window_info,
                "monitor": monitor,
                "position": position,
                "latency_ms": int((end_time - start_time) * 1000)
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": int((end_time - start_time) * 1000)
            }


def _enumerate_monitors() -> List[Dict[str, Any]]:
    """Enumerate all monitors"""
    monitors = []
    
    if HAS_PYWIN32:
        import win32api
        
        for i, monitor in enumerate(win32api.EnumDisplayMonitors()):
            info = win32api.GetMonitorInfo(monitor[0])
            mon_rect = info["Monitor"]
            
            monitors.append({
                "index": i,
                "x": mon_rect[0],
                "y": mon_rect[1],
                "width": mon_rect[2] - mon_rect[0],
                "height": mon_rect[3] - mon_rect[1],
                "primary": info.get("Flags", 0) == 1
            })
    else:
        # ctypes fallback
        MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(ctypes.c_long * 4), ctypes.c_double)
        
        def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
            rect = lprcMonitor.contents
            monitors.append({
                "index": len(monitors),
                "x": rect[0],
                "y": rect[1],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
                "primary": rect[0] == 0 and rect[1] == 0  # Heuristic
            })
            return 1
        
        enum_func = MonitorEnumProc(callback)
        user32.EnumDisplayMonitors(0, 0, enum_func, 0)
    
    return monitors
