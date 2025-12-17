"""
Get Window Monitor tool - determines which monitor a specific window/application is on.
"""
import time
import ctypes
from typing import Dict, Any, List, Optional, Tuple
from wyzer.tools.tool_base import ToolBase

# Try to import pywin32, fallback to ctypes
try:
    import win32gui
    import win32api
    import win32con
    import win32process
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False

# ctypes Windows API
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


class GetWindowMonitorTool(ToolBase):
    """Tool to get which monitor a specific window/application is on"""
    
    def __init__(self):
        super().__init__()
        self._name = "get_window_monitor"
        self._description = "Get which monitor a specific window or application is currently displayed on"
        self._args_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title substring to match (case-insensitive)"
                },
                "process": {
                    "type": "string",
                    "description": "Process name to match, e.g., 'spotify', 'chrome', 'discord'"
                }
            },
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Get which monitor a window is on.
        
        Args:
            title: Window title to search for (optional)
            process: Process name to search for (optional)
            
        Returns:
            Dict with monitor info for the matched window, or error
        """
        start_time = time.perf_counter()
        
        title = kwargs.get("title")
        process = kwargs.get("process")
        
        if not title and not process:
            return {
                "error": {
                    "type": "missing_argument",
                    "message": "Either 'title' or 'process' must be provided"
                },
                "latency_ms": 0
            }
        
        try:
            # Find the window
            hwnd, matched_info = self._find_window(title, process)
            
            if not hwnd:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "error": {
                        "type": "window_not_found",
                        "message": f"No window found matching title='{title}' or process='{process}'"
                    },
                    "latency_ms": latency_ms
                }
            
            # Get window position
            window_rect = self._get_window_rect(hwnd)
            if not window_rect:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "error": {
                        "type": "rect_error",
                        "message": "Could not get window position"
                    },
                    "latency_ms": latency_ms
                }
            
            # Get all monitors
            monitors = self._enumerate_monitors()
            
            # Determine which monitor the window is on
            monitor_info = self._get_window_monitor(window_rect, monitors)
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "matched_window": matched_info,
                "window_position": {
                    "x": window_rect[0],
                    "y": window_rect[1],
                    "width": window_rect[2] - window_rect[0],
                    "height": window_rect[3] - window_rect[1]
                },
                "monitor": monitor_info,
                "total_monitors": len(monitors),
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }
    
    def _get_window_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """Get window rectangle (left, top, right, bottom)"""
        try:
            if HAS_PYWIN32:
                rect = win32gui.GetWindowRect(hwnd)
                return rect
            else:
                rect = (ctypes.c_long * 4)()
                if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    return (rect[0], rect[1], rect[2], rect[3])
                return None
        except Exception:
            return None
    
    def _enumerate_monitors(self) -> List[Dict[str, Any]]:
        """Enumerate all monitors"""
        monitors = []
        
        if HAS_PYWIN32:
            for i, monitor in enumerate(win32api.EnumDisplayMonitors()):
                info = win32api.GetMonitorInfo(monitor[0])
                mon_rect = info["Monitor"]
                
                monitors.append({
                    "index": i,
                    "x": mon_rect[0],
                    "y": mon_rect[1],
                    "width": mon_rect[2] - mon_rect[0],
                    "height": mon_rect[3] - mon_rect[1],
                    "right": mon_rect[2],
                    "bottom": mon_rect[3],
                    "primary": info.get("Flags", 0) == 1
                })
        else:
            # ctypes fallback
            MonitorEnumProc = ctypes.WINFUNCTYPE(
                ctypes.c_int,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_long * 4),
                ctypes.c_double
            )
            
            def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
                rect = lprcMonitor.contents
                monitors.append({
                    "index": len(monitors),
                    "x": rect[0],
                    "y": rect[1],
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1],
                    "right": rect[2],
                    "bottom": rect[3],
                    "primary": rect[0] == 0 and rect[1] == 0  # Heuristic
                })
                return 1
            
            enum_func = MonitorEnumProc(callback)
            user32.EnumDisplayMonitors(0, 0, enum_func, 0)
        
        return monitors
    
    def _get_window_monitor(self, window_rect: Tuple[int, int, int, int], monitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine which monitor the window is primarily on based on overlap area"""
        win_left, win_top, win_right, win_bottom = window_rect
        win_center_x = (win_left + win_right) // 2
        win_center_y = (win_top + win_bottom) // 2
        
        best_monitor = None
        best_overlap = 0
        
        for monitor in monitors:
            mon_left = monitor["x"]
            mon_top = monitor["y"]
            mon_right = monitor["x"] + monitor["width"]
            mon_bottom = monitor["y"] + monitor["height"]
            
            # Calculate overlap area
            overlap_left = max(win_left, mon_left)
            overlap_top = max(win_top, mon_top)
            overlap_right = min(win_right, mon_right)
            overlap_bottom = min(win_bottom, mon_bottom)
            
            if overlap_right > overlap_left and overlap_bottom > overlap_top:
                overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
                if overlap_area > best_overlap:
                    best_overlap = overlap_area
                    best_monitor = monitor
            
            # Also check if window center is in this monitor
            if (mon_left <= win_center_x < mon_right and 
                mon_top <= win_center_y < mon_bottom):
                # Center-based match is a strong signal
                if best_monitor is None or overlap_area >= best_overlap:
                    best_monitor = monitor
        
        if best_monitor:
            # Return monitor number (1-indexed for user-friendly display)
            return {
                "number": best_monitor["index"] + 1,  # 1-indexed
                "index": best_monitor["index"],       # 0-indexed
                "resolution": f"{best_monitor['width']}x{best_monitor['height']}",
                "position": f"({best_monitor['x']}, {best_monitor['y']})",
                "primary": best_monitor.get("primary", False)
            }
        
        # Fallback: couldn't determine monitor
        return {
            "number": None,
            "index": None,
            "resolution": None,
            "position": None,
            "primary": None,
            "error": "Could not determine which monitor the window is on"
        }
    
    def _find_window(self, title: Optional[str], process: Optional[str]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """Find window by title or process name"""
        windows = self._enumerate_windows()
        
        title_norm = (title or "").strip().lower()
        process_norm = (process or "").strip().lower()
        
        # Build process hints from title if no process specified
        process_hints: List[str] = []
        if title_norm and not process_norm:
            base = title_norm
            if base.endswith(".exe"):
                base = base[:-4]
            compact = base.replace(" ", "")
            process_hints = [
                base,
                compact,
                f"{base}.exe",
                f"{compact}.exe",
            ]
        
        best_hwnd: Optional[int] = None
        best_info: Optional[Dict[str, Any]] = None
        best_score = -1
        
        for window in windows:
            window_title = (window.get("title") or "").lower()
            window_process = (window.get("process") or "").lower()
            
            score = -1
            
            # Check title match
            if title_norm and title_norm in window_title:
                score = max(score, 100)
            
            # Check process match
            if process_norm:
                if process_norm == window_process or f"{process_norm}.exe" == window_process:
                    score = max(score, 90)
                elif process_norm in window_process:
                    score = max(score, 80)
            
            # Check process hints from title
            for hint in process_hints:
                if hint == window_process or hint in window_process:
                    score = max(score, 70)
                    break
            
            if score > best_score:
                best_score = score
                best_hwnd = window.get("hwnd")
                best_info = {
                    "hwnd": window.get("hwnd"),
                    "title": window.get("title"),
                    "process": window.get("process"),
                    "pid": window.get("pid")
                }
        
        return best_hwnd, best_info
    
    def _enumerate_windows(self) -> List[Dict[str, Any]]:
        """Enumerate all visible windows"""
        windows = []
        
        if HAS_PYWIN32:
            def callback(hwnd, extra):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd) or ""
                    pid = 0
                    process_name = ""
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        process_name = self._get_process_name_pywin32(pid)
                    except:
                        pid = 0
                        process_name = ""
                    
                    if title or process_name:
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
                    title = ""
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        title = buff.value or ""
                    
                    pid = ctypes.c_ulong()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    process_name = self._get_process_name_ctypes(pid.value)
                    
                    if title or process_name:
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
    
    def _get_process_name_pywin32(self, pid: int) -> str:
        """Get process name from PID using pywin32"""
        try:
            import win32api
            handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
            try:
                exe = win32process.GetModuleFileNameEx(handle, 0)
                return exe.split("\\")[-1].lower()
            finally:
                win32api.CloseHandle(handle)
        except:
            return ""
    
    def _get_process_name_ctypes(self, pid: int) -> str:
        """Get process name from PID using ctypes"""
        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            PROCESS_QUERY_INFORMATION = 0x0400
            
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
            
            if handle:
                try:
                    buff = ctypes.create_unicode_buffer(260)
                    size = ctypes.c_ulong(260)
                    
                    if kernel32.QueryFullProcessImageNameW(handle, 0, buff, ctypes.byref(size)):
                        return buff.value.split("\\")[-1].lower()
                finally:
                    kernel32.CloseHandle(handle)
        except:
            pass
        
        return ""
