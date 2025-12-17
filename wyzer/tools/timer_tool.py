"""
Timer tool - Set, cancel, or check a countdown timer with alarm.

Uses file-based state persistence to work across the multiprocess architecture.
The brain worker polls check_timer_finished() to detect when timers complete.

Architecture:
- Tools execute in ToolWorkerPool worker processes
- Timer state is saved to a JSON file
- Brain worker polls check_timer_finished() every ~100ms
- When end_time is reached, brain worker triggers TTS
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Any

from wyzer.tools.tool_base import ToolBase


# ═══════════════════════════════════════════════════════════════════════════
# File-based timer state (works across processes)
# ═══════════════════════════════════════════════════════════════════════════

_TIMER_STATE_FILE = Path(__file__).parent.parent / "data" / "timer_state.json"
_file_lock = threading.Lock()


def _load_state() -> Dict[str, Any]:
    """Load timer state from file."""
    try:
        if _TIMER_STATE_FILE.exists():
            with open(_TIMER_STATE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "running": False,
        "start_time": 0.0,
        "duration": 0,
        "end_time": 0.0,
    }


def _save_state(state: Dict[str, Any]) -> None:
    """Save timer state to file."""
    try:
        _TIMER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_TIMER_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


def check_timer_finished() -> bool:
    """
    Check if a timer has finished and clear it. Returns True once when timer completes.
    
    This is called by the brain worker every ~100ms to detect timer completion.
    Safe to call from any process since it uses file-based state.
    
    Returns:
        True if a timer just finished (one-shot), False otherwise
    """
    with _file_lock:
        try:
            state = _load_state()
            
            # Check if a running timer has expired
            if state.get("running", False):
                end_time = state.get("end_time", 0)
                if end_time > 0 and time.time() >= end_time:
                    # Timer finished! Clear the state and return True
                    state["running"] = False
                    state["end_time"] = 0.0
                    _save_state(state)
                    return True
            
            return False
        except Exception:
            return False


class TimerTool(ToolBase):
    """Tool to set, cancel, or check a countdown timer with alarm."""
    
    def __init__(self):
        """Initialize the timer tool with metadata."""
        super().__init__()
        
        self._name = "timer"
        self._description = "Set, cancel, or check a countdown timer with alarm"
        self._args_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "cancel", "status"],
                    "description": "The timer action to perform"
                },
                "duration_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Duration in seconds (required for start action)"
                }
            },
            "required": ["action"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the timer tool.
        
        Args:
            action: One of "start", "cancel", or "status"
            duration_seconds: Duration in seconds (required for "start")
            
        Returns:
            Dict with timer status or error
        """
        try:
            action = kwargs.get("action", "").lower().strip()
            
            if not action:
                return {
                    "error": {
                        "type": "missing_argument",
                        "message": "action is required"
                    }
                }
            
            if action not in {"start", "cancel", "status"}:
                return {
                    "error": {
                        "type": "invalid_action",
                        "message": f"Unknown action: {action}. Use 'start', 'cancel', or 'status'."
                    }
                }
            
            # ═══════════════════════════════════════════════════════════════
            # ACTION: start
            # ═══════════════════════════════════════════════════════════════
            if action == "start":
                duration_seconds = kwargs.get("duration_seconds")
                
                if duration_seconds is None:
                    return {
                        "error": {
                            "type": "missing_argument",
                            "message": "duration_seconds is required for start action"
                        }
                    }
                
                try:
                    duration_seconds = int(duration_seconds)
                except (ValueError, TypeError):
                    return {
                        "error": {
                            "type": "invalid_duration",
                            "message": "duration_seconds must be a positive integer"
                        }
                    }
                
                if duration_seconds < 1:
                    return {
                        "error": {
                            "type": "invalid_duration",
                            "message": "duration_seconds must be at least 1"
                        }
                    }
                
                if duration_seconds > 86400:  # 24 hours max
                    return {
                        "error": {
                            "type": "invalid_duration",
                            "message": "Timer duration cannot exceed 24 hours"
                        }
                    }
                
                # Save timer state to file (works across processes)
                now = time.time()
                _save_state({
                    "running": True,
                    "start_time": now,
                    "duration": duration_seconds,
                    "end_time": now + duration_seconds,
                })
                
                return {
                    "status": "running",
                    "duration": duration_seconds
                }
            
            # ═══════════════════════════════════════════════════════════════
            # ACTION: cancel
            # ═══════════════════════════════════════════════════════════════
            if action == "cancel":
                state = _load_state()
                was_running = state.get("running", False)
                
                # Also check if end_time hasn't passed yet
                if not was_running:
                    end_time = state.get("end_time", 0)
                    if end_time > 0 and time.time() < end_time:
                        was_running = True
                
                # Clear timer state
                _save_state({
                    "running": False,
                    "start_time": 0.0,
                    "duration": 0,
                    "end_time": 0.0,
                })
                
                if was_running:
                    return {"status": "cancelled"}
                else:
                    return {
                        "error": {
                            "type": "no_timer",
                            "message": "No active timer to cancel"
                        }
                    }
            
            # ═══════════════════════════════════════════════════════════════
            # ACTION: status
            # ═══════════════════════════════════════════════════════════════
            if action == "status":
                state = _load_state()
                
                if not state.get("running", False):
                    return {"status": "idle"}
                
                end_time = state.get("end_time", 0)
                remaining = max(0, int(end_time - time.time()))
                
                if remaining <= 0:
                    return {
                        "status": "finished",
                        "remaining_seconds": 0
                    }
                
                return {
                    "status": "running",
                    "remaining_seconds": remaining,
                    "duration": state.get("duration", 0)
                }
            
            # Should never reach here
            return {
                "error": {
                    "type": "unknown_error",
                    "message": "Unexpected error in timer tool"
                }
            }
            
        except Exception as e:
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                }
            }
