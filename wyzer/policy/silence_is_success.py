"""wyzer.policy.silence_is_success

Phase 11.5 - Silence is Success Module.

Controls reply verbosity for successful tool execution:
- Deterministic tools return minimal confirmation ("Done.", "OK.")
- High-confidence actions do NOT trigger LLM chatter
- Only explain when tool result needs clarification or user asked

HARD RULES:
- Does NOT modify tool schemas or execution logic
- Does NOT change deterministic routing behavior
- Reduces verbosity, never increases it
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ============================================================================
# SILENT SUCCESS TOOLS
# ============================================================================
# Tools that should return minimal confirmation ("OK." or "Done.") on success.
# These are simple action tools where the user can observe the result directly.
SILENT_SUCCESS_TOOLS = frozenset({
    # Media controls - user can hear the result
    "media_play_pause",
    "media_next",
    "media_previous",
    
    # Volume controls - user can hear the result
    "volume_up",
    "volume_down",
    "volume_mute_toggle",
    
    # Window management - user can see the result
    "focus_window",
    "minimize_window",
    "maximize_window",
    "close_window",
    "move_window_to_monitor",
    
    # App launching - user can see the result
    "open_target",
    "open_website",
    
    # Library refresh - background operation
    "local_library_refresh",
})


# Tools that need explanation of their result (not silent)
INFO_TOOLS = frozenset({
    "get_time",
    "get_system_info",
    "get_location",
    "get_weather_forecast",
    "get_now_playing",
    "get_window_context",
    "get_window_monitor",
    "monitor_info",
    "system_storage_scan",
    "system_storage_list",
    "timer",  # Timer status needs feedback
})


# Tools that may need brief confirmation based on result
BRIEF_CONFIRMATION_TOOLS = frozenset({
    "volume_control",      # May need level confirmation
    "set_audio_output_device",  # May need device confirmation
    "system_storage_open",  # Brief "Opened X"
})


def should_be_silent(tool: str, success: bool) -> bool:
    """
    Determine if a tool execution should return minimal/no confirmation.
    
    Args:
        tool: Tool name
        success: Whether execution succeeded
        
    Returns:
        True if tool should be silent (minimal "OK." or "Done.")
    """
    if not success:
        return False  # Failed tools need explanation
    
    return tool in SILENT_SUCCESS_TOOLS


def needs_explanation(tool: str) -> bool:
    """
    Determine if a tool's result needs user-facing explanation.
    
    Args:
        tool: Tool name
        
    Returns:
        True if tool result should be explained to user
    """
    return tool in INFO_TOOLS


def needs_brief_confirmation(tool: str) -> bool:
    """
    Determine if a tool needs brief confirmation (not silent, but not verbose).
    
    Args:
        tool: Tool name
        
    Returns:
        True if tool needs brief confirmation
    """
    return tool in BRIEF_CONFIRMATION_TOOLS


def get_minimal_reply(tool: str, success: bool, error: Optional[str] = None) -> str:
    """
    Get minimal reply for a tool execution.
    
    Args:
        tool: Tool name
        success: Whether execution succeeded
        error: Error message if failed
        
    Returns:
        Minimal confirmation string
    """
    if not success:
        if error:
            return error
        return "Failed."
    
    # Media controls
    if tool == "media_play_pause":
        return "OK."
    if tool == "media_next":
        return "OK."
    if tool == "media_previous":
        return "OK."
    
    # Volume controls
    if tool in {"volume_up", "volume_down", "volume_mute_toggle"}:
        return "OK."
    
    # Window management
    if tool in {"focus_window", "minimize_window", "maximize_window"}:
        return "Done."
    if tool == "close_window":
        return "Closed."
    if tool == "move_window_to_monitor":
        return "Moved."
    
    # App launching
    if tool == "open_target":
        return "Opening."
    if tool == "open_website":
        return "Opening."
    
    # Library refresh
    if tool == "local_library_refresh":
        return "Done."
    
    return "Done."


def suppress_llm_chatter(
    tool_results: List[Dict[str, Any]],
    original_reply: str,
) -> str:
    """
    Suppress unnecessary LLM chatter after successful tool execution.
    
    If all tools executed successfully and are "silent success" tools,
    replace any verbose LLM reply with minimal confirmation.
    
    Args:
        tool_results: List of tool execution results
        original_reply: The LLM-generated reply
        
    Returns:
        Minimal reply if chatter should be suppressed, else original
    """
    if not tool_results:
        return original_reply
    
    # Check if ALL tools are silent success tools and succeeded
    all_silent = True
    all_success = True
    
    for result in tool_results:
        tool = result.get("tool", "")
        ok = result.get("ok", False)
        
        if not ok:
            all_success = False
            all_silent = False
            break
        
        if not should_be_silent(tool, ok):
            all_silent = False
    
    if all_silent and all_success:
        # All tools are silent success - use minimal reply
        # Get the last tool for the minimal reply
        last_tool = tool_results[-1].get("tool", "")
        return get_minimal_reply(last_tool, True)
    
    return original_reply


def is_verbose_reply(reply: str) -> bool:
    """
    Detect if a reply is unnecessarily verbose.
    
    Args:
        reply: The reply to check
        
    Returns:
        True if reply appears verbose (could be shortened)
    """
    if not reply:
        return False
    
    # Verbose indicators
    verbose_patterns = [
        r"\bI've\b",
        r"\bI have\b",
        r"\bI can\b",
        r"\bI will\b",
        r"\bI would\b",
        r"\blet me\b",
        r"\bfor you\b",
        r"\bif you\b",
        r"\bwould you like\b",
        r"\byou might want\b",
        r"\byou may want\b",
        r"\bfeel free to\b",
        r"\bplease note\b",
        r"\bkeep in mind\b",
        r"\bjust to clarify\b",
        r"\bjust a heads up\b",
        r"\bby the way\b",
    ]
    
    for pattern in verbose_patterns:
        if re.search(pattern, reply, re.IGNORECASE):
            return True
    
    # Long replies for simple actions are verbose
    if len(reply) > 100:
        return True
    
    return False


def truncate_verbose_reply(reply: str, max_length: int = 80) -> str:
    """
    Truncate an overly verbose reply to be more concise.
    
    Args:
        reply: The original reply
        max_length: Maximum length before truncation
        
    Returns:
        Truncated reply
    """
    if not reply or len(reply) <= max_length:
        return reply
    
    # Find the first sentence boundary
    sentence_end = None
    for i, char in enumerate(reply):
        if char in ".!?" and i < max_length:
            sentence_end = i + 1
    
    if sentence_end:
        return reply[:sentence_end].strip()
    
    # No sentence boundary found - truncate at word boundary
    truncated = reply[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]
    
    return truncated.rstrip() + "."
