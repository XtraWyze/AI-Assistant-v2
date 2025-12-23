"""
Get Window Context tool - Phase 9: Screen Awareness (READ-ONLY)

Returns information about the current foreground window/application.
This is a stateless, read-only tool with NO side effects.

NO OCR, NO screenshots, NO UI automation.
"""

import time
from typing import Dict, Any
from wyzer.tools.tool_base import ToolBase


class GetWindowContextTool(ToolBase):
    """Tool to get the current foreground window context (read-only)"""
    
    def __init__(self):
        """Initialize get_window_context tool"""
        super().__init__()
        self._name = "get_window_context"
        self._description = "Get information about the current foreground window (app name, window title). Read-only, no side effects."
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Get information about the current foreground window.
        
        This is a READ-ONLY operation. No screenshots, OCR, or automation.
        
        Returns:
            Dict with:
                - app: str | None - Process name (e.g., "chrome.exe")
                - title: str | None - Window title
                - pid: int | None - Process ID
                - latency_ms: int - Execution time in milliseconds
        """
        start_time = time.perf_counter()
        
        try:
            from wyzer.vision.window_context import get_foreground_window
            
            result = get_foreground_window()
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "app": result.get("app"),
                "title": result.get("title"),
                "pid": result.get("pid"),
                "latency_ms": latency_ms
            }
        
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "app": None,
                "title": None,
                "pid": None,
                "error": {
                    "type": "detection_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }
