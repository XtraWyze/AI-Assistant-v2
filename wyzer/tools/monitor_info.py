"""
Monitor info tool - provides information about connected monitors.
"""
import time
import ctypes
from typing import Dict, Any, List
from wyzer.tools.tool_base import ToolBase

# Try to import pywin32, fallback to ctypes
try:
    import win32api
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False

# ctypes Windows API
user32 = ctypes.windll.user32


class MonitorInfoTool(ToolBase):
    """Tool to get information about connected monitors"""
    
    def __init__(self):
        super().__init__()
        self._name = "monitor_info"
        self._description = "Get information about all connected monitors (count, resolution, position)"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Get monitor information.
        
        Returns:
            Dict with monitors list and count, or error
        """
        start_time = time.perf_counter()
        
        try:
            monitors = self._enumerate_monitors()
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "monitors": monitors,
                "count": len(monitors),
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
                    "primary": rect[0] == 0 and rect[1] == 0  # Heuristic
                })
                return 1
            
            enum_func = MonitorEnumProc(callback)
            user32.EnumDisplayMonitors(0, 0, enum_func, 0)
        
        return monitors
