"""wyzer.context.world_state

Phase 10 - World State Store (in-RAM only).

Provides a singleton WorldState dataclass that tracks:
- Last executed tool and target
- Active application and window title (from Phase 9 screen awareness)
- Timestamp of last update

HARD RULES:
- Readable from anywhere
- Writable ONLY by deterministic code paths (tools + state updates)
- Never written by the LLM

This module is intentionally simple and deterministic.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorldState:
    """
    In-RAM world state for reference resolution.
    
    Tracks recent actions and context for resolving vague follow-up phrases
    like "close it", "do that again", etc.
    
    Fields:
        last_tool: Name of the last executed tool (e.g., "open_app")
        last_target: Target of the last action (e.g., "Chrome", "Spotify")
        last_result_summary: Short summary of last result (optional)
        active_app: Currently focused application (from Phase 9)
        active_window_title: Current window title (from Phase 9)
        last_updated_ts: Unix timestamp of last update
    """
    last_tool: Optional[str] = None
    last_target: Optional[str] = None
    last_result_summary: Optional[str] = None
    active_app: Optional[str] = None
    active_window_title: Optional[str] = None
    last_updated_ts: float = field(default_factory=time.time)
    
    def clear(self) -> None:
        """Reset all state fields."""
        self.last_tool = None
        self.last_target = None
        self.last_result_summary = None
        self.active_app = None
        self.active_window_title = None
        self.last_updated_ts = time.time()
    
    def has_last_action(self) -> bool:
        """Check if there's a valid last action for reference."""
        return self.last_tool is not None and self.last_target is not None
    
    def get_age_seconds(self) -> float:
        """Get seconds since last update."""
        return time.time() - self.last_updated_ts


# Singleton instance and lock
_world_state: Optional[WorldState] = None
_world_state_lock = threading.Lock()


def get_world_state() -> WorldState:
    """
    Get the singleton WorldState instance.
    
    Thread-safe. Creates instance on first access.
    
    Returns:
        The global WorldState instance
    """
    global _world_state
    with _world_state_lock:
        if _world_state is None:
            _world_state = WorldState()
        return _world_state


def update_world_state(
    last_tool: Optional[str] = None,
    last_target: Optional[str] = None,
    last_result_summary: Optional[str] = None,
    active_app: Optional[str] = None,
    active_window_title: Optional[str] = None,
) -> None:
    """
    Update the world state with new values.
    
    Thread-safe. Only updates fields that are provided (not None).
    This is the ONLY function that should modify WorldState.
    
    Args:
        last_tool: Tool name that was just executed
        last_target: Target of the action (app name, file, etc.)
        last_result_summary: Short summary of the result
        active_app: Currently active application
        active_window_title: Current window title
    """
    ws = get_world_state()
    with _world_state_lock:
        if last_tool is not None:
            ws.last_tool = last_tool
        if last_target is not None:
            ws.last_target = last_target
        if last_result_summary is not None:
            ws.last_result_summary = last_result_summary
        if active_app is not None:
            ws.active_app = active_app
        if active_window_title is not None:
            ws.active_window_title = active_window_title
        ws.last_updated_ts = time.time()


def update_from_tool_execution(
    tool_name: str,
    tool_args: dict,
    tool_result: dict,
) -> None:
    """
    Update WorldState after a tool successfully executes.
    
    This function extracts the relevant target from tool args/result
    and updates the world state accordingly.
    
    Called by orchestrator after ANY tool successfully runs.
    Fails silently if fields cannot be derived.
    
    Args:
        tool_name: Name of the tool that was executed
        tool_args: Arguments passed to the tool
        tool_result: Result returned by the tool
    """
    if not tool_name:
        return
    
    # Don't update state on error results
    if tool_result and isinstance(tool_result, dict) and "error" in tool_result:
        return
    
    # Extract target from tool args (best effort)
    target = _extract_target_from_args(tool_name, tool_args)
    
    # Extract result summary (best effort)
    summary = _extract_result_summary(tool_name, tool_result)
    
    # Update state
    update_world_state(
        last_tool=tool_name,
        last_target=target,
        last_result_summary=summary,
    )


def _extract_target_from_args(tool_name: str, args: dict) -> Optional[str]:
    """
    Extract the primary target from tool arguments.
    
    Different tools have different arg structures. This function
    handles common patterns to extract the "target" of the action.
    
    Args:
        tool_name: Name of the tool
        args: Tool arguments dict
        
    Returns:
        Extracted target string or None
    """
    if not args or not isinstance(args, dict):
        return None
    
    # Common argument names that represent the target
    target_keys = [
        "target",       # Generic target
        "app",          # open_app, close_app, focus_window
        "app_name",     # Alternative app name
        "name",         # Generic name
        "process",      # Process-based tools
        "query",        # Search tools
        "url",          # Browser tools
        "path",         # File tools
        "file",         # File tools
        "folder",       # Folder tools
        "window",       # Window tools
        "window_title", # Window tools
    ]
    
    for key in target_keys:
        value = args.get(key)
        if value and isinstance(value, str):
            return value.strip()
    
    # Special handling for specific tools
    if tool_name == "volume":
        # Volume tool might have process/scope args
        process = args.get("process") or args.get("app")
        if process:
            return f"volume:{process}"
        return "volume:system"
    
    if tool_name == "timer":
        # Timer tool has duration
        duration = args.get("duration") or args.get("seconds")
        label = args.get("label") or args.get("name")
        if label:
            return f"timer:{label}"
        elif duration:
            return f"timer:{duration}s"
    
    return None


def _extract_result_summary(tool_name: str, result: dict) -> Optional[str]:
    """
    Extract a short summary from tool result.
    
    Args:
        tool_name: Name of the tool
        result: Tool result dict
        
    Returns:
        Short summary string or None
    """
    if not result or not isinstance(result, dict):
        return None
    
    # Check for error
    if "error" in result:
        return "failed"
    
    # Check for common success indicators
    if result.get("success") or result.get("ok"):
        return "success"
    
    # Check for message/status
    msg = result.get("message") or result.get("status")
    if msg and isinstance(msg, str):
        # Truncate to reasonable length
        return msg[:50] if len(msg) > 50 else msg
    
    return None


def clear_world_state() -> None:
    """Clear all world state. Useful for testing."""
    ws = get_world_state()
    with _world_state_lock:
        ws.clear()
