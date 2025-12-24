"""wyzer.policy.risk

Phase 11 - Risk Classification Module.

Deterministic risk classification for tool plans.
Classifies each tool as low/medium/high risk.

RISK LEVELS:
- LOW: Read-only tools (get_window_context, get_time, list, status queries)
- MEDIUM: Benign mutations (open app, close one window, volume, media controls)
- HIGH: Destructive/dangerous (delete, shutdown, restart, kill process, format)

HARD RULES:
- Keep HIGH list NARROW - do not over-classify
- replay_last_action inherits risk from the underlying tool
- Classification is deterministic (no LLM)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

# Type alias for risk levels
RiskLevel = Literal["low", "medium", "high"]


# ============================================================================
# RISK CLASSIFICATION CONSTANTS
# ============================================================================

# HIGH RISK: Destructive or dangerous actions
# Keep this list NARROW - only truly dangerous operations
HIGH_RISK_TOOLS = frozenset({
    # Destructive file operations (if such tools exist)
    "delete_file",
    "delete_files",
    "delete_folder",
    "delete_directory",
    "remove_file",
    "remove_files",
    "wipe",
    "format",
    "format_drive",
    "empty_trash",
    "empty_recycle_bin",
    
    # System power/state changes
    "shutdown",
    "restart",
    "reboot",
    "logoff",
    "log_off",
    "hibernate",
    "sleep_system",
    
    # Process termination
    "kill_process",
    "terminate_process",
    "kill_all",
    "end_task",
    "force_close_all",
})

# HIGH RISK: Patterns in tool names or args that indicate danger
HIGH_RISK_PATTERNS = (
    "delete",
    "remove",
    "wipe",
    "format",
    "shutdown",
    "restart",
    "reboot",
    "logoff",
    "kill",
    "terminate",
)

# MEDIUM RISK: Mutations that are reversible/benign
# These are common action tools
MEDIUM_RISK_TOOLS = frozenset({
    # App/window management
    "open_target",
    "open_website",
    "open_app",
    "launch_app",
    "close_window",
    "minimize_window",
    "maximize_window",
    "focus_window",
    "move_window_to_monitor",
    
    # Audio/media controls
    "volume_control",
    "volume_up",
    "volume_down",
    "volume_mute_toggle",
    "media_play_pause",
    "media_next",
    "media_previous",
    "set_audio_output_device",
    
    # Timer
    "timer",
    
    # Search
    "google_search_open",
    
    # Library management
    "local_library_refresh",
    
    # System storage operations (scanning, not deleting)
    "system_storage_scan",
    "system_storage_open",
})

# LOW RISK: Read-only tools
# Everything not in HIGH or MEDIUM defaults to LOW
LOW_RISK_TOOLS = frozenset({
    # Time/info queries
    "get_time",
    "get_system_info",
    "get_location",
    "get_weather_forecast",
    
    # Read-only window/screen context
    "get_window_context",
    "get_window_monitor",
    "monitor_info",
    "get_now_playing",
    
    # List/status queries
    "system_storage_list",
})


def classify_tool(tool_name: str, args: Optional[Dict[str, Any]] = None) -> RiskLevel:
    """
    Classify a single tool's risk level.
    
    Args:
        tool_name: Name of the tool
        args: Tool arguments (optional, for pattern matching)
        
    Returns:
        Risk level: "low", "medium", or "high"
    """
    if not tool_name:
        return "low"
    
    name_lower = tool_name.lower().strip()
    
    # Check explicit HIGH list first
    if name_lower in HIGH_RISK_TOOLS:
        return "high"
    
    # Check HIGH patterns in tool name
    for pattern in HIGH_RISK_PATTERNS:
        if pattern in name_lower:
            return "high"
    
    # Check args for dangerous patterns
    if args and isinstance(args, dict):
        for key, value in args.items():
            if isinstance(value, str):
                value_lower = value.lower()
                # Check if args suggest destructive action
                for pattern in ("delete", "remove", "wipe", "kill", "terminate"):
                    if pattern in value_lower:
                        return "high"
    
    # Check explicit MEDIUM list
    if name_lower in MEDIUM_RISK_TOOLS:
        return "medium"
    
    # Check explicit LOW list
    if name_lower in LOW_RISK_TOOLS:
        return "low"
    
    # Default: unknown tools are LOW (conservative for execution)
    # This means we won't block unknown read-only tools
    return "low"


def classify_plan(
    tool_plan: List[Dict[str, Any]],
    world_state: Optional[Any] = None,
) -> RiskLevel:
    """
    Classify the overall risk of a tool plan.
    
    Takes the MAXIMUM risk across all tools in the plan.
    
    For replay_last_action: inherits the risk of the stored last_action tool.
    
    Args:
        tool_plan: List of tool call dicts with "tool" and optional "args"
        world_state: Optional WorldState for resolving replay risk
        
    Returns:
        Maximum risk level across all tools: "low", "medium", or "high"
    """
    if not tool_plan:
        return "low"
    
    max_risk: RiskLevel = "low"
    risk_order = {"low": 0, "medium": 1, "high": 2}
    
    for intent in tool_plan:
        if not isinstance(intent, dict):
            continue
        
        tool_name = intent.get("tool", "")
        args = intent.get("args", {})
        
        # Special case: replay_last_action inherits underlying tool's risk
        if tool_name == "replay_last_action" or tool_name == "__REPLAY_LAST_ACTION__":
            underlying_risk = _get_replay_underlying_risk(world_state)
            if risk_order.get(underlying_risk, 0) > risk_order.get(max_risk, 0):
                max_risk = underlying_risk
            continue
        
        tool_risk = classify_tool(tool_name, args)
        if risk_order.get(tool_risk, 0) > risk_order.get(max_risk, 0):
            max_risk = tool_risk
    
    return max_risk


def _get_replay_underlying_risk(world_state: Optional[Any]) -> RiskLevel:
    """
    Get the risk level of the underlying last_action for replay.
    
    Args:
        world_state: WorldState object with last_action info
        
    Returns:
        Risk level of the stored last_action tool, or "medium" as safe default
    """
    if world_state is None:
        # Safe default if no state available
        return "medium"
    
    # Try to get last_action from WorldState
    last_action = getattr(world_state, "last_action", None)
    if last_action is None:
        # No last_action stored
        return "medium"
    
    # Get the tool name from LastAction
    tool_name = getattr(last_action, "tool", None)
    if not tool_name:
        return "medium"
    
    args = getattr(last_action, "args", {})
    return classify_tool(tool_name, args)


def get_risk_description(risk: RiskLevel) -> str:
    """
    Get a human-readable description of a risk level.
    
    Args:
        risk: Risk level
        
    Returns:
        Description string
    """
    descriptions = {
        "low": "read-only or informational",
        "medium": "modifies state but reversible",
        "high": "potentially destructive or dangerous",
    }
    return descriptions.get(risk, "unknown")
