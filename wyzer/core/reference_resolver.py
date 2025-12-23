"""wyzer.core.reference_resolver

Phase 10 - Reference Resolution for Continuity.

Deterministically rewrites vague follow-up phrases like:
- "close it" → "close <last_target>"
- "open it again" → "open <last_target>"
- "do that again" → repeat last_tool + last_target

HARD RULES:
- NO LLM-based guessing
- If resolution is uncertain, return original text unchanged
- All logic must be deterministic and testable

This module enables natural follow-ups without introducing autonomy.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
from wyzer.context.world_state import WorldState, get_world_state
from wyzer.core.logger import get_logger


# ============================================================================
# REFERENCE PATTERNS
# ============================================================================
# These patterns detect vague references that need resolution.
# Each pattern maps to a rewrite strategy.

# "close it", "close that", "shut it"
_CLOSE_IT_RE = re.compile(
    r"^(?:close|shut|quit|exit|end|kill|stop)\s+(?:it|that|this|the\s+app|the\s+window|the\s+application)\.?$",
    re.IGNORECASE
)

# "open it", "open that", "launch it", "start it", "open it again", "open that again"
_OPEN_IT_RE = re.compile(
    r"^(?:open|launch|start|run)\s+(?:it|that|this)(?:\s+again)?\.?$",
    re.IGNORECASE
)

# "minimize it", "maximize it", "focus on it"
_WINDOW_ACTION_IT_RE = re.compile(
    r"^(?:minimize|maximise|maximize|focus(?:\s+on)?|switch\s+to)\s+(?:it|that|this|the\s+window)\.?$",
    re.IGNORECASE
)

# "do that again", "repeat that", "do it again", "again"
_REPEAT_RE = re.compile(
    r"^(?:do\s+(?:that|it|this)\s+again|repeat\s+(?:that|it|this|the\s+last\s+(?:action|command))|again)\.?$",
    re.IGNORECASE
)

# "mute it", "unmute it" (for volume)
_MUTE_IT_RE = re.compile(
    r"^(?:mute|unmute)\s+(?:it|that|this)\.?$",
    re.IGNORECASE
)

# "pause it", "resume it", "play it" (for media)
_MEDIA_IT_RE = re.compile(
    r"^(?:pause|resume|play|stop)\s+(?:it|that|this)\.?$",
    re.IGNORECASE
)

# "what about the other one", "try the other one" (secondary target)
_OTHER_ONE_RE = re.compile(
    r"^(?:what\s+about|try|use|switch\s+to)\s+(?:the\s+)?other\s+(?:one|app|window)\.?$",
    re.IGNORECASE
)


# ============================================================================
# TOOL-TO-ACTION MAPPING
# ============================================================================
# Maps tool names to their opposite/related actions for reference resolution.

_TOOL_OPPOSITES = {
    "open_app": "close_app",
    "close_app": "open_app",
    "focus_window": "minimize_window",
    "minimize_window": "focus_window",
    "maximize_window": "minimize_window",
    "mute": "unmute",
    "unmute": "mute",
}

# Tools that operate on apps/windows (for "close it" style references)
_APP_TOOLS = {
    "open_app", "open_target", "close_app", "close_window", "focus_window", "minimize_window",
    "maximize_window", "switch_app", "launch_app"
}


def resolve_references(text: str, ws: Optional[WorldState] = None) -> str:
    """
    Deterministically rewrite text to resolve vague references.
    
    Uses WorldState to substitute "it", "that", "this" with actual targets.
    
    Args:
        text: User's input text (may contain vague references)
        ws: WorldState instance (if None, uses global singleton)
        
    Returns:
        Rewritten text with references resolved, OR original text if:
        - No reference patterns detected
        - WorldState lacks required context
        - Resolution would be ambiguous
    """
    if ws is None:
        ws = get_world_state()
    
    logger = get_logger()
    original = text.strip()
    
    if not original:
        return original
    
    # Try each pattern in order of specificity
    result = _try_resolve_close_it(original, ws)
    if result:
        logger.debug(f'[REF_RESOLVE] "{original}" → "{result}"')
        return result
    
    result = _try_resolve_open_it(original, ws)
    if result:
        logger.debug(f'[REF_RESOLVE] "{original}" → "{result}"')
        return result
    
    result = _try_resolve_window_action_it(original, ws)
    if result:
        logger.debug(f'[REF_RESOLVE] "{original}" → "{result}"')
        return result
    
    result = _try_resolve_repeat(original, ws)
    if result:
        logger.debug(f'[REF_RESOLVE] "{original}" → "{result}"')
        return result
    
    result = _try_resolve_mute_it(original, ws)
    if result:
        logger.debug(f'[REF_RESOLVE] "{original}" → "{result}"')
        return result
    
    result = _try_resolve_media_it(original, ws)
    if result:
        logger.debug(f'[REF_RESOLVE] "{original}" → "{result}"')
        return result
    
    # No rewrite needed
    return original


def _try_resolve_close_it(text: str, ws: WorldState) -> Optional[str]:
    """
    Resolve "close it" → "close <target>".
    
    Only resolves if:
    - Last tool was an app/window tool (we know what "it" refers to)
    - Or there's an active app in window context
    """
    if not _CLOSE_IT_RE.match(text):
        return None
    
    # Extract the action verb
    action = text.split()[0].lower()
    
    # Priority 1: Last target from recent app/window tool
    if ws.last_target and ws.last_tool in _APP_TOOLS:
        target = _clean_target(ws.last_target)
        return f"{action} {target}"
    
    # Priority 2: Active app from window context
    if ws.active_app:
        target = _clean_target(ws.active_app)
        return f"{action} {target}"
    
    # Can't resolve - return None to fall through
    return None


def _try_resolve_open_it(text: str, ws: WorldState) -> Optional[str]:
    """
    Resolve "open it again" → "open <target>".
    
    Only resolves if last tool was open/close/focus related.
    """
    if not _OPEN_IT_RE.match(text):
        return None
    
    # Extract the action verb
    action = text.split()[0].lower()
    
    # Need a last target from a relevant tool
    if ws.last_target and ws.last_tool in _APP_TOOLS:
        target = _clean_target(ws.last_target)
        return f"{action} {target}"
    
    return None


def _try_resolve_window_action_it(text: str, ws: WorldState) -> Optional[str]:
    """
    Resolve "minimize it" → "minimize <target>".
    """
    if not _WINDOW_ACTION_IT_RE.match(text):
        return None
    
    # Extract the action (may be multi-word like "focus on")
    lower = text.lower()
    if lower.startswith("focus on"):
        action = "focus"
    elif lower.startswith("switch to"):
        action = "switch to"
    else:
        action = text.split()[0].lower()
    
    # Priority 1: Last target from recent tool
    if ws.last_target and ws.last_tool in _APP_TOOLS:
        target = _clean_target(ws.last_target)
        return f"{action} {target}"
    
    # Priority 2: Active app
    if ws.active_app:
        target = _clean_target(ws.active_app)
        return f"{action} {target}"
    
    return None


def _try_resolve_repeat(text: str, ws: WorldState) -> Optional[str]:
    """
    Resolve "do that again" → full command reconstruction.
    
    Only works when we have both last_tool and last_target.
    """
    if not _REPEAT_RE.match(text):
        return None
    
    if not ws.has_last_action():
        return None
    
    # Reconstruct command based on tool and target
    tool = ws.last_tool
    target = _clean_target(ws.last_target or "")
    
    if not tool or not target:
        return None
    
    # Map tool names to natural commands
    tool_to_command = {
        "open_app": f"open {target}",
        "open_target": f"open {target}",
        "close_app": f"close {target}",
        "close_window": f"close {target}",
        "focus_window": f"focus {target}",
        "minimize_window": f"minimize {target}",
        "maximize_window": f"maximize {target}",
        "launch_app": f"launch {target}",
        "switch_app": f"switch to {target}",
    }
    
    cmd = tool_to_command.get(tool)
    if cmd:
        return cmd
    
    # For other tools, try a generic pattern
    if target.startswith("timer:"):
        # Timer was set
        return f"set timer {target[6:]}"
    
    if target.startswith("volume:"):
        # Volume was adjusted
        process = target[7:]
        if process == "system":
            return "check volume"
        return f"check {process} volume"
    
    # Can't reconstruct - return None
    return None


def _try_resolve_mute_it(text: str, ws: WorldState) -> Optional[str]:
    """
    Resolve "mute it" → "mute <app>".
    """
    if not _MUTE_IT_RE.match(text):
        return None
    
    action = text.split()[0].lower()
    
    # If last tool was volume-related with a process
    if ws.last_target and ws.last_target.startswith("volume:"):
        process = ws.last_target[7:]
        if process and process != "system":
            return f"{action} {process}"
    
    # If last tool was on an app
    if ws.last_target and ws.last_tool in _APP_TOOLS:
        target = _clean_target(ws.last_target)
        return f"{action} {target}"
    
    # Active app
    if ws.active_app:
        return f"{action} {_clean_target(ws.active_app)}"
    
    return None


def _try_resolve_media_it(text: str, ws: WorldState) -> Optional[str]:
    """
    Resolve "pause it" → "pause <app>" or just "pause".
    
    For media controls, we're more lenient - system-wide is often fine.
    """
    if not _MEDIA_IT_RE.match(text):
        return None
    
    action = text.split()[0].lower()
    
    # If there's a known media target
    if ws.last_target and ws.last_tool in _APP_TOOLS:
        target = _clean_target(ws.last_target)
        # Check if it's likely a media app
        media_apps = {"spotify", "vlc", "foobar", "itunes", "musicbee", "youtube", "netflix"}
        if target.lower() in media_apps:
            return f"{action} {target}"
    
    # For media, returning None will let the command pass through
    # (system-wide pause is valid)
    return None


def _clean_target(target: str) -> str:
    """
    Clean up a target string for use in commands.
    
    Removes prefixes like "volume:", normalizes casing.
    """
    if not target:
        return ""
    
    # Remove known prefixes
    for prefix in ["volume:", "timer:"]:
        if target.startswith(prefix):
            target = target[len(prefix):]
    
    return target.strip()


def get_resolution_context(ws: Optional[WorldState] = None) -> dict:
    """
    Get debug info about current resolution context.
    
    Useful for debugging and testing.
    
    Args:
        ws: WorldState instance (if None, uses global singleton)
        
    Returns:
        Dict with context information
    """
    if ws is None:
        ws = get_world_state()
    
    return {
        "last_tool": ws.last_tool,
        "last_target": ws.last_target,
        "last_result_summary": ws.last_result_summary,
        "active_app": ws.active_app,
        "active_window_title": ws.active_window_title,
        "age_seconds": ws.get_age_seconds(),
        "has_last_action": ws.has_last_action(),
    }
