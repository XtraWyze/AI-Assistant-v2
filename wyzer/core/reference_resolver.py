"""wyzer.core.reference_resolver

Phase 10 - Reference Resolution for Continuity.
Phase 10.1 - Deterministic Replay (replay_last_action).

Deterministically rewrites vague follow-up phrases like:
- "close it" → "close <last_target>"
- "open it again" → "open <last_target>"
- "do that again" → __REPLAY_LAST_ACTION__ (triggers deterministic replay)

HARD RULES:
- NO LLM-based guessing
- If resolution is uncertain, return original text unchanged
- All logic must be deterministic and testable
- Phase 10.1: "do that again" returns sentinel for deterministic replay,
  NOT reconstructed text

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

# Phase 10.1: Extended replay phrases for deterministic replay
# These phrases trigger __REPLAY_LAST_ACTION__ sentinel
# Supports optional prefixes like "can you", "could you", "please", etc.
_REPLAY_PREFIX_RE = r"(?:(?:can|could|would)\s+you\s+(?:please\s+)?|please\s+)?"
_REPLAY_PHRASES_RE = re.compile(
    r"^" + _REPLAY_PREFIX_RE + r"(?:"
    r"do\s+(?:that|it|this)\s+again|"
    r"repeat\s+(?:that|it|this|the\s+last\s+(?:action|command))|"
    r"do\s+it\s+again|"
    r"same\s+(?:thing|as\s+(?:last\s+time|before))|"
    r"again|"
    r"one\s+more\s+time|"
    r"repeat"
    r")\.?\??$",
    re.IGNORECASE
)

# Sentinel string returned when replay_last_action should be triggered
REPLAY_LAST_ACTION_SENTINEL = "__REPLAY_LAST_ACTION__"

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
    Resolve "do that again" / "repeat that" etc.
    
    Phase 10.1: Returns REPLAY_LAST_ACTION_SENTINEL when structured last_action
    is available, enabling deterministic replay without re-routing through text.
    
    Falls back to text-based command reconstruction when last_action is not available
    but last_tool/last_target are (legacy behavior).
    """
    # Check if this matches any replay phrase
    if not _REPLAY_PHRASES_RE.match(text):
        return None
    
    # Phase 10.1: Prefer structured last_action for deterministic replay
    if ws.has_replay_action():
        # Return sentinel - orchestrator will handle deterministic replay
        return REPLAY_LAST_ACTION_SENTINEL
    
    # Fallback: Legacy text-based reconstruction (for backwards compatibility)
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


def is_replay_request(text: str) -> bool:
    """
    Check if text is a replay/repeat request.
    
    Phase 10.1: Used by orchestrator to detect replay requests before routing.
    
    Args:
        text: User's input text
        
    Returns:
        True if text matches replay phrases like "do that again", "repeat that", etc.
    """
    return bool(_REPLAY_PHRASES_RE.match((text or "").strip()))


def is_replay_sentinel(text: str) -> bool:
    """
    Check if text is the replay sentinel.
    
    Phase 10.1: Used by orchestrator to detect when reference resolver
    returned the replay sentinel.
    
    Args:
        text: Resolved text from reference resolver
        
    Returns:
        True if text is the __REPLAY_LAST_ACTION__ sentinel
    """
    return text == REPLAY_LAST_ACTION_SENTINEL


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
        "has_replay_action": ws.has_replay_action(),
    }


# ============================================================================
# PHASE 10: ENHANCED REFERENCE RESOLUTION
# ============================================================================

# Pattern: "the other one", "switch to the other one", "use the other window"
_THE_OTHER_ONE_RE = re.compile(
    r"^(?:(?:switch\s+to|use|try|focus(?:\s+on)?|open|close|minimize|maximize)\s+)?(?:the\s+)?other\s+(?:one|app|window)\.?$",
    re.IGNORECASE
)

# Pattern: "move it to monitor X" - supports many variations
# Examples: "move it to monitor 2", "move it to the second monitor", "move it to the left monitor"
_MOVE_IT_TO_MONITOR_RE = re.compile(
    r"^move\s+(?:it|that|this|the\s+window)\s+to\s+"
    r"(?:(?:the\s+)?(?:monitor|screen|display)\s+)?"  # optional "the monitor" prefix
    r"(?:the\s+)?"  # optional "the" before ordinal/positional (e.g., "the second")
    r"(primary|main|secondary|other|\d+|one|two|three|four|five|"
    r"first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|"
    r"next|previous|left|right)"
    r"(?:\s+(?:monitor|screen|display))?\.?$",  # optional "monitor" suffix
    re.IGNORECASE
)

# Word-to-digit mapping for monitor numbers (mirrors hybrid_router)
_WORD_TO_DIGIT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    "primary": 1, "main": 1, "secondary": 2, "other": 2,
    # Positional - these get passed through as strings for the tool to handle
    "left": "left", "right": "right", "next": "next", "previous": "previous",
}

# Pronouns that can reference the last target
_PRONOUN_PATTERNS = {"it", "that", "this", "the app", "the window", "the application", "that one", "this one"}

# Pattern: pronoun-based action commands that need tool execution
# Examples: "close it", "close that", "minimize it", "maximize that", "shut it", "kill that"
_PRONOUN_ACTION_RE = re.compile(
    r"^(?:close|shut|kill|quit|exit|minimize|maximize|focus(?:\s+on)?)\s+"
    r"(?:it|that|this|the\s+(?:app|window|application))\.?$",
    re.IGNORECASE
)

# Window tool names for context-aware resolution
_WINDOW_TOOLS = {
    "close_window", "focus_window", "minimize_window", "maximize_window", 
    "move_window", "close_app", "switch_app"
}

# Open/launch tool names
_OPEN_TOOLS = {"open_app", "open_target", "open_website", "launch_app", "system_storage_open"}


def is_move_it_to_monitor_request(text: str) -> bool:
    """Check if text matches the 'move it to monitor X' pattern."""
    return bool(_MOVE_IT_TO_MONITOR_RE.match(text.strip()))


def is_pronoun_action_request(text: str) -> bool:
    """Check if text is a pronoun-based action like 'close it', 'minimize that'."""
    return bool(_PRONOUN_ACTION_RE.match(text.strip()))


def resolve_pronoun_target(
    text: str,
    ws: Optional[WorldState] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve pronoun references (it/this/that) to a concrete target.
    
    Phase 10: Deterministically resolves pronouns based on world state.
    
    Args:
        text: Text that may contain a pronoun (e.g., "close it")
        ws: WorldState instance
        
    Returns:
        Tuple of (resolved_target, reason) where:
        - resolved_target is the concrete target name or None if unresolved
        - reason is a human-readable explanation of the resolution
    """
    if ws is None:
        ws = get_world_state()
    
    logger = get_logger()
    
    # Check if text contains pronoun references
    text_lower = text.lower().strip()
    has_pronoun = any(p in text_lower for p in _PRONOUN_PATTERNS)
    
    if not has_pronoun:
        return None, None
    
    logger.debug(f"[REF] detected pattern=pronoun text=\"{text}\"")
    
    # Priority 1: Last target from a recent tool execution (within 5 minutes)
    if ws.last_target and ws.get_age_seconds() < 300:
        # Check if the last tool is relevant to the current command
        tool_type = _infer_tool_type_from_text(text)
        
        if tool_type == "window" and ws.last_tool in _WINDOW_TOOLS | _OPEN_TOOLS:
            logger.debug(f"[REF] resolved target={ws.last_target} reason=last_tool_match")
            return ws.last_target, f"Resolved from last action: {ws.last_tool}"
        
        if tool_type == "open" and ws.last_tool in _OPEN_TOOLS:
            logger.debug(f"[REF] resolved target={ws.last_target} reason=last_open_target")
            return ws.last_target, "Resolved from last opened target"
        
        if tool_type == "volume" and ws.last_tool in _APP_TOOLS:
            logger.debug(f"[REF] resolved target={ws.last_target} reason=last_app_target")
            return ws.last_target, "Resolved from last app target"
        
        # Default: use last target if it's from an app/window tool
        if ws.last_tool in _APP_TOOLS:
            logger.debug(f"[REF] resolved target={ws.last_target} reason=last_target_default")
            return ws.last_target, f"Resolved from last target ({ws.last_tool})"
    
    # Priority 2: Last active window (from window watcher)
    if ws.last_active_window:
        app_name = ws.last_active_window.get("app_name")
        if app_name:
            logger.debug(f"[REF] resolved target={app_name} reason=last_active_window")
            return app_name, "Resolved from active window"
    
    # Priority 3: Active app from legacy field
    if ws.active_app:
        logger.debug(f"[REF] resolved target={ws.active_app} reason=active_app")
        return ws.active_app, "Resolved from active app"
    
    # Priority 4: Check last_targets ring buffer
    from wyzer.context.world_state import get_last_targets
    targets = get_last_targets(1)
    if targets:
        target_name = targets[0].name
        logger.debug(f"[REF] resolved target={target_name} reason=last_targets_buffer")
        return target_name, "Resolved from recent target"
    
    logger.debug(f"[REF] resolution failed - no context available")
    return None, None


def resolve_other_one(
    text: str,
    ws: Optional[WorldState] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve "the other one" to toggle between last two targets.
    
    Phase 10: Enables "the other one" pattern for quick switching.
    
    Args:
        text: Text containing "the other one" pattern
        ws: WorldState instance
        
    Returns:
        Tuple of (resolved_target, reason) or (None, clarification_question)
    """
    if ws is None:
        ws = get_world_state()
    
    logger = get_logger()
    
    # Check if text matches "the other one" pattern
    if not _THE_OTHER_ONE_RE.match(text.strip()):
        return None, None
    
    logger.debug(f"[REF] detected pattern=other_one text=\"{text}\"")
    
    from wyzer.context.world_state import get_last_targets
    targets = get_last_targets(2)
    
    if len(targets) < 2:
        # Only one target known, ask for clarification
        if len(targets) == 1:
            clarification = f"I only have one recent target: {targets[0].name}. Which other one do you mean?"
        else:
            clarification = "I don't have any recent targets to switch between. Can you be more specific?"
        logger.debug(f"[REF] ambiguous candidates=[] reason=insufficient_targets")
        return None, clarification
    
    # Return the second most recent target (the "other" one)
    other_target = targets[1].name
    logger.debug(f"[REF] resolved target={other_target} reason=other_one_toggle")
    return other_target, f"Toggled from {targets[0].name} to {other_target}"


def resolve_repeat_last(
    text: str,
    ws: Optional[WorldState] = None,
) -> Tuple[Optional[list], Optional[str]]:
    """
    Resolve "do that again" to replay last intents.
    
    Phase 10: Returns the last intent list for deterministic replay.
    
    Args:
        text: Text matching repeat patterns
        ws: WorldState instance
        
    Returns:
        Tuple of (intent_list, reason) or (None, reason) if replay not possible
    """
    if ws is None:
        ws = get_world_state()
    
    logger = get_logger()
    
    # Check if text is a replay request
    if not is_replay_request(text):
        return None, None
    
    logger.debug(f"[REF] detected pattern=repeat text=\"{text}\"")
    
    # Check if last response was chat-only
    if ws.last_llm_reply_only:
        logger.debug(f"[REF] replay blocked reason=last_was_reply_only")
        return None, "I can only repeat actions, not chat responses. What would you like me to do?"
    
    # Get last intents from world state
    from wyzer.context.world_state import get_last_intents
    intents = get_last_intents()
    
    if not intents:
        # Fall back to last_action if available
        if ws.has_replay_action() and ws.last_action:
            intent = {
                "tool": ws.last_action.tool,
                "args": dict(ws.last_action.args) if ws.last_action.args else {},
            }
            logger.debug(f"[REF] resolved intents=[{intent['tool']}] reason=last_action")
            return [intent], f"Repeating: {ws.last_action.tool}"
        
        logger.debug(f"[REF] replay failed reason=no_last_intents")
        return None, "What should I repeat? I don't have a recent action to replay."
    
    tool_names = [i.get("tool", "?") for i in intents]
    logger.debug(f"[REF] resolved intents={tool_names} reason=last_intents")
    return intents, f"Repeating: {', '.join(tool_names)}"


def resolve_move_it_to_monitor(
    text: str,
    ws: Optional[WorldState] = None,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Resolve "move it to monitor X" to concrete window + monitor.
    
    Args:
        text: Text like "move it to monitor 2"
        ws: WorldState instance
        
    Returns:
        Tuple of (args_dict, reason) or (None, clarification_question)
        args_dict has keys: process (for tool), monitor, display_name (for reply)
    """
    if ws is None:
        ws = get_world_state()
    
    logger = get_logger()
    
    match = _MOVE_IT_TO_MONITOR_RE.match(text.strip())
    if not match:
        return None, None
    
    logger.debug(f"[REF] detected pattern=move_it_monitor text=\"{text}\"")
    
    # Parse monitor number using the shared mapping
    monitor_str = match.group(1).lower()
    if monitor_str in _WORD_TO_DIGIT:
        monitor = _WORD_TO_DIGIT[monitor_str]
    elif monitor_str.isdigit():
        monitor = int(monitor_str)
    else:
        monitor = 1  # Default fallback
    
    # Resolve the window target
    resolved_target, reason = resolve_pronoun_target("move it", ws)
    
    if not resolved_target:
        # Try to use focused window
        if ws.focused_window:
            resolved_target = ws.focused_window.get("process") or ws.focused_window.get("title")
        elif ws.last_active_window:
            resolved_target = ws.last_active_window.get("app_name")
    
    if not resolved_target:
        return None, "Which window do you want to move? I can't determine the target."
    
    # The move_window_to_monitor tool expects 'process' or 'title', not 'query'
    args = {
        "process": resolved_target,
        "monitor": monitor,
        "_display_name": resolved_target,  # For reply formatting
    }
    logger.debug(f"[REF] resolved target={resolved_target} monitor={monitor}")
    return args, f"Moving {resolved_target} to monitor {monitor}"


def _infer_tool_type_from_text(text: str) -> Optional[str]:
    """
    Infer what type of tool the user's command relates to.
    
    Returns: "window", "open", "volume", or None
    """
    text_lower = text.lower()
    
    if any(w in text_lower for w in ["close", "shut", "quit", "exit", "kill", "minimize", "maximize", "focus", "move"]):
        return "window"
    
    if any(w in text_lower for w in ["open", "launch", "start", "run"]):
        return "open"
    
    if any(w in text_lower for w in ["volume", "mute", "unmute", "sound", "audio", "louder", "quieter"]):
        return "volume"
    
    return None


def resolve_intent_args(
    intent: dict,
    ws: Optional[WorldState] = None,
) -> Tuple[dict, Optional[str]]:
    """
    Resolve missing/placeholder arguments in an intent using world state.
    
    Phase 10: Called by orchestrator before executing each intent.
    This fills in missing app_name, window_title, query, etc.
    
    Args:
        intent: Dict with "tool" and "args" keys
        ws: WorldState instance
        
    Returns:
        Tuple of (resolved_args, clarification_question)
        - resolved_args: Args dict with placeholders filled in
        - clarification_question: If not None, indicates resolution failed
    """
    if ws is None:
        ws = get_world_state()
    
    logger = get_logger()
    
    tool = intent.get("tool", "")
    args = dict(intent.get("args", {}))
    
    # Keys that might contain pronoun references needing resolution
    target_keys = ["app", "app_name", "query", "target", "window", "window_title", "process", "name"]
    
    for key in target_keys:
        value = args.get(key)
        if not value:
            continue
        
        value_lower = str(value).lower().strip()
        
        # Check if value is a pronoun that needs resolution
        if value_lower in _PRONOUN_PATTERNS or value_lower in {"", "it", "that", "this"}:
            # Resolve the pronoun
            resolved, reason = resolve_pronoun_target(f"{tool} {value}", ws)
            if resolved:
                logger.debug(f"[REF] resolved arg {key}={resolved} reason={reason}")
                args[key] = resolved
            else:
                # Can't resolve - need clarification
                candidates = _get_clarification_candidates(ws)
                if candidates:
                    question = f"Which one do you mean? {', '.join(candidates[:3])}"
                else:
                    question = "Which app or window do you mean?"
                return args, question
    
    return args, None


def _get_clarification_candidates(ws: WorldState) -> list:
    """
    Get a list of candidate targets for clarification.
    
    Returns up to 3 recent targets that the user might be referring to.
    """
    candidates = []
    
    # Add from last_targets
    from wyzer.context.world_state import get_last_targets
    targets = get_last_targets(3)
    for t in targets:
        if t.name and t.name not in candidates:
            candidates.append(t.name)
    
    # Add active app if not already included
    if ws.active_app and ws.active_app not in candidates:
        candidates.append(ws.active_app)
    
    return candidates[:3]


def is_other_one_request(text: str) -> bool:
    """
    Check if text is a "the other one" request.
    
    Args:
        text: User's input text
        
    Returns:
        True if text matches "the other one" patterns
    """
    return bool(_THE_OTHER_ONE_RE.match((text or "").strip()))


def has_unresolved_pronoun(intent: dict) -> bool:
    """
    Check if an intent has unresolved pronoun arguments.
    
    Args:
        intent: Dict with "tool" and "args" keys
        
    Returns:
        True if any argument is a pronoun needing resolution
    """
    args = intent.get("args", {})
    target_keys = ["app", "app_name", "query", "target", "window", "window_title", "process", "name"]
    
    for key in target_keys:
        value = args.get(key)
        if value:
            value_lower = str(value).lower().strip()
            if value_lower in _PRONOUN_PATTERNS or value_lower in {"it", "that", "this"}:
                return True
    
    return False