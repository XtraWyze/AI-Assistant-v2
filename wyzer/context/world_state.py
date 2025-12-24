"""wyzer.context.world_state

Phase 10 - World State Store (in-RAM only).
Phase 11 - Extended for Autonomy Policy.

Provides a singleton WorldState dataclass that tracks:
- Last executed tool and target
- Active application and window title (from Phase 9 screen awareness)
- Timestamp of last update
- Autonomy mode and pending confirmations (Phase 11)

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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wyzer.policy.autonomy_policy import AutonomyDecision


@dataclass
class LastAction:
    """
    Structured representation of the last successful tool execution.
    
    Used by Phase 10.1 replay_last_action for deterministic replays.
    
    Fields:
        tool: Name of the tool that was executed
        args: Original arguments passed to the tool
        resolved: Resolved/canonical target info from result (e.g., UWP path, matched window)
        ts: Unix timestamp when the action was executed
    """
    tool: str
    args: dict
    resolved: Optional[dict] = None
    ts: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dict for logging/debugging."""
        return {
            "tool": self.tool,
            "args": self.args,
            "resolved": self.resolved,
            "ts": self.ts,
        }


@dataclass
class PendingConfirmation:
    """
    Phase 11 - Pending confirmation for high-risk autonomy actions.
    
    When autonomy mode != off and a high-risk action is detected,
    the orchestrator stores a pending confirmation here instead of
    executing immediately.
    
    Fields:
        plan: The tool plan waiting for confirmation
        expires_ts: Unix timestamp when the confirmation window expires
        prompt: The confirmation question spoken to user
        decision: The autonomy decision that triggered this
    """
    plan: List[Dict[str, Any]]
    expires_ts: float
    prompt: str
    decision: Optional[Dict[str, Any]] = None
    
    def is_expired(self) -> bool:
        """Check if confirmation window has expired."""
        return time.time() > self.expires_ts
    
    def to_dict(self) -> dict:
        """Convert to dict for logging/debugging."""
        return {
            "plan_tools": [p.get("tool", "?") for p in self.plan] if self.plan else [],
            "expires_in": max(0.0, self.expires_ts - time.time()),
            "prompt": self.prompt,
        }


@dataclass
class LastAutonomyDecision:
    """
    Phase 11 - Record of the last autonomy policy decision.
    
    Used by "why did you do that" command to explain the last decision.
    
    Fields:
        mode: Autonomy mode at time of decision
        confidence: Router confidence value
        risk: Risk classification
        action: Decision action (execute/ask/deny)
        reason: Explanation of why this decision was made
        plan_summary: Short summary of the tool plan
        ts: Unix timestamp of decision
    """
    mode: str
    confidence: float
    risk: str
    action: str  # execute|ask|deny
    reason: str
    plan_summary: str
    ts: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dict for logging/debugging."""
        return {
            "mode": self.mode,
            "confidence": self.confidence,
            "risk": self.risk,
            "action": self.action,
            "reason": self.reason,
            "plan_summary": self.plan_summary,
            "ts": self.ts,
        }


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
        last_action: Phase 10.1 - Structured last action for deterministic replay
        
        # Phase 11 - Autonomy fields
        autonomy_mode: Current autonomy mode (off|low|normal|high)
        pending_confirmation: Pending high-risk confirmation, if any
        last_autonomy_decision: Last autonomy policy decision for "why" command
    """
    last_tool: Optional[str] = None
    last_target: Optional[str] = None
    last_result_summary: Optional[str] = None
    active_app: Optional[str] = None
    active_window_title: Optional[str] = None
    last_updated_ts: float = field(default_factory=time.time)
    last_action: Optional[LastAction] = None
    
    # Phase 11 - Autonomy fields
    autonomy_mode: str = "off"  # off|low|normal|high
    pending_confirmation: Optional[PendingConfirmation] = None
    last_autonomy_decision: Optional[LastAutonomyDecision] = None
    
    def clear(self) -> None:
        """Reset all state fields."""
        self.last_tool = None
        self.last_target = None
        self.last_result_summary = None
        self.active_app = None
        self.active_window_title = None
        self.last_updated_ts = time.time()
        self.last_action = None
        # Don't reset autonomy_mode on clear (user preference)
        self.pending_confirmation = None
        self.last_autonomy_decision = None
    
    def has_last_action(self) -> bool:
        """Check if there's a valid last action for reference."""
        return self.last_tool is not None and self.last_target is not None
    
    def has_replay_action(self) -> bool:
        """Check if there's a structured last_action available for replay."""
        return self.last_action is not None
    
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
    Initializes autonomy_mode from config on first creation.
    
    Returns:
        The global WorldState instance
    """
    global _world_state
    with _world_state_lock:
        if _world_state is None:
            _world_state = WorldState()
            # Initialize autonomy_mode from config
            from wyzer.core.config import Config
            default_mode = getattr(Config, "AUTONOMY_DEFAULT", "off")
            if default_mode in {"off", "low", "normal", "high"}:
                _world_state.autonomy_mode = default_mode
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
    
    # Extract target - prefer resolved name from result, fall back to args
    target = _extract_target_from_result(tool_name, tool_result)
    if not target:
        target = _extract_target_from_args(tool_name, tool_args)
    
    # Format target for nice display
    if target:
        target = _format_target_name(target)
    
    # Extract result summary (best effort)
    summary = _extract_result_summary(tool_name, tool_result)
    
    # ===========================================================================
    # Phase 10.1: Store structured last_action for deterministic replay
    # ===========================================================================
    resolved_info = _extract_resolved_info_for_replay(tool_name, tool_result)
    
    from wyzer.core.logger import get_logger
    logger = get_logger()
    
    last_action = LastAction(
        tool=tool_name,
        args=dict(tool_args) if tool_args else {},
        resolved=resolved_info,
        ts=time.time(),
    )
    
    # Log the last_action capture
    resolved_summary = _summarize_resolved(resolved_info) if resolved_info else "None"
    logger.info(f"[WORLD_STATE] last_action tool={tool_name} resolved={resolved_summary}")
    
    # Update state (including last_action)
    ws = get_world_state()
    with _world_state_lock:
        if tool_name:
            ws.last_tool = tool_name
        if target:
            ws.last_target = target
        if summary:
            ws.last_result_summary = summary
        ws.last_action = last_action
        ws.last_updated_ts = time.time()


def _extract_resolved_info_for_replay(tool_name: str, result: dict) -> Optional[dict]:
    """
    Extract resolved/canonical target info for deterministic replay.
    
    Phase 10.1: This extracts the stable launch target (e.g., UWP path, exe path,
    matched window info) so replays don't need to re-resolve.
    
    Args:
        tool_name: Name of the tool
        result: Tool result dict
        
    Returns:
        Dict with replay-stable resolved info, or None
    """
    if not result or not isinstance(result, dict):
        return None
    
    resolved_info = {}
    
    # For open_target: extract resolved dict with type, path, launch info
    if tool_name == "open_target":
        resolved = result.get("resolved")
        if resolved and isinstance(resolved, dict):
            # Copy key fields needed for stable replay
            resolved_info["type"] = resolved.get("type")
            resolved_info["path"] = resolved.get("path")
            resolved_info["matched_name"] = resolved.get("matched_name")
            # For games/UWP, include launch info
            launch = resolved.get("launch")
            if launch and isinstance(launch, dict):
                resolved_info["launch"] = dict(launch)
            # Include game_name/app_name if present
            if result.get("game_name"):
                resolved_info["game_name"] = result.get("game_name")
            if result.get("app_name"):
                resolved_info["app_name"] = result.get("app_name")
            return resolved_info if resolved_info.get("type") else None
    
    # For close_window/focus_window: extract matched window info
    if tool_name in ("close_window", "focus_window", "minimize_window", "maximize_window"):
        matched = result.get("matched")
        if matched and isinstance(matched, dict):
            resolved_info["title"] = matched.get("title")
            resolved_info["process"] = matched.get("process")
            resolved_info["hwnd"] = matched.get("hwnd")  # May be useful for some replays
            return resolved_info if resolved_info.get("process") or resolved_info.get("title") else None
    
    # For other tools, try to extract generic resolved/matched info
    if result.get("resolved") and isinstance(result.get("resolved"), dict):
        return dict(result["resolved"])
    if result.get("matched") and isinstance(result.get("matched"), dict):
        return dict(result["matched"])
    
    return None


def _summarize_resolved(resolved: Optional[dict]) -> str:
    """Create a short summary of resolved info for logging."""
    if not resolved:
        return "None"
    
    parts = []
    if resolved.get("type"):
        parts.append(f"type={resolved['type']}")
    if resolved.get("matched_name"):
        parts.append(f"name={resolved['matched_name']}")
    elif resolved.get("app_name"):
        parts.append(f"name={resolved['app_name']}")
    elif resolved.get("game_name"):
        parts.append(f"name={resolved['game_name']}")
    elif resolved.get("process"):
        parts.append(f"process={resolved['process']}")
    elif resolved.get("title"):
        title = resolved['title'][:30] + "..." if len(resolved.get('title', '')) > 30 else resolved.get('title', '')
        parts.append(f"title={title}")
    
    if resolved.get("launch") and isinstance(resolved.get("launch"), dict):
        launch_type = resolved["launch"].get("type")
        if launch_type:
            parts.append(f"launch={launch_type}")
    
    return " ".join(parts) if parts else str(resolved)[:50]


def _extract_target_from_result(tool_name: str, result: dict) -> Optional[str]:
    """
    Extract the canonical target name from tool result.
    
    Prefers resolved/matched names over raw query strings.
    This ensures aliases and fuzzy matches resolve to canonical names.
    
    Args:
        tool_name: Name of the tool
        result: Tool result dict
        
    Returns:
        Canonical target name or None
    """
    if not result or not isinstance(result, dict):
        return None
    
    # For open_target: use resolved.matched_name
    resolved = result.get("resolved")
    if resolved and isinstance(resolved, dict):
        matched_name = resolved.get("matched_name")
        if matched_name and isinstance(matched_name, str):
            return matched_name.strip()
        # Fall back to path basename for files/folders
        path = resolved.get("path")
        if path and isinstance(path, str):
            import os
            basename = os.path.basename(path)
            if basename:
                # Remove .exe extension for apps
                if basename.lower().endswith(".exe"):
                    basename = basename[:-4]
                return basename.strip()
    
    # For close_window/focus_window: use matched window info
    matched = result.get("matched")
    if matched and isinstance(matched, dict):
        # Prefer process name (cleaner than full title)
        process = matched.get("process")
        if process and isinstance(process, str):
            # Remove .exe extension
            if process.lower().endswith(".exe"):
                process = process[:-4]
            return process.strip()
        # Fall back to title
        title = matched.get("title")
        if title and isinstance(title, str):
            return title.strip()
    
    # For other tools, check common result fields
    target_keys = ["target", "name", "app", "window", "matched_name"]
    for key in target_keys:
        value = result.get(key)
        if value and isinstance(value, str):
            return value.strip()
    
    return None


def _format_target_name(target: str) -> str:
    """
    Format target name for nice display.
    
    Maps executable names to friendly display names.
    Handles casing and common patterns.
    
    Args:
        target: Raw target name
        
    Returns:
        Formatted display name
    """
    if not target:
        return target
    
    # Common app name mappings (lowercase key -> display name)
    app_display_names = {
        "chrome": "Chrome",
        "firefox": "Firefox",
        "edge": "Edge",
        "msedge": "Edge",
        "spotify": "Spotify",
        "discord": "Discord",
        "slack": "Slack",
        "code": "VS Code",
        "notepad": "Notepad",
        "notepad++": "Notepad++",
        "explorer": "File Explorer",
        "windowsterminal": "Windows Terminal",
        "wt": "Windows Terminal",
        "cmd": "Command Prompt",
        "powershell": "PowerShell",
        "pwsh": "PowerShell",
        "steam": "Steam",
        "vlc": "VLC",
        "obs64": "OBS",
        "obs": "OBS",
        "word": "Word",
        "excel": "Excel",
        "outlook": "Outlook",
        "teams": "Teams",
        "zoom": "Zoom",
        "gimp": "GIMP",
        "audacity": "Audacity",
        "blender": "Blender",
        "photoshop": "Photoshop",
        "premiere": "Premiere",
        "aftereffects": "After Effects",
    }
    
    # Check for exact match (case-insensitive)
    lower = target.lower().strip()
    if lower in app_display_names:
        return app_display_names[lower]
    
    # If it looks like an executable, try without .exe
    if lower.endswith(".exe"):
        base = lower[:-4]
        if base in app_display_names:
            return app_display_names[base]
    
    # Title case for unknown apps (but preserve if already looks good)
    if target.islower() or target.isupper():
        return target.title()
    
    return target


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


# ============================================================================
# PHASE 11: AUTONOMY STATE FUNCTIONS
# ============================================================================

def get_autonomy_mode() -> str:
    """
    Get the current autonomy mode.
    
    Returns:
        Current mode: "off", "low", "normal", or "high"
    """
    ws = get_world_state()
    with _world_state_lock:
        return ws.autonomy_mode


def set_autonomy_mode(mode: str) -> str:
    """
    Set the autonomy mode.
    
    Args:
        mode: One of "off", "low", "normal", "high"
        
    Returns:
        The new mode (normalized)
    """
    mode = mode.lower().strip() if mode else "off"
    valid_modes = {"off", "low", "normal", "high"}
    if mode not in valid_modes:
        mode = "off"
    
    ws = get_world_state()
    with _world_state_lock:
        ws.autonomy_mode = mode
        # Clear any pending confirmation when mode changes
        ws.pending_confirmation = None
    
    return mode


def set_pending_confirmation(
    plan: List[Dict[str, Any]],
    prompt: str,
    timeout_sec: float = 45.0,
    decision: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Set a pending confirmation for a high-risk action.
    
    Args:
        plan: The tool plan waiting for confirmation
        prompt: The confirmation question to speak
        timeout_sec: Seconds until confirmation expires (default 45s)
        decision: The autonomy decision that triggered this
    """
    from wyzer.core.logger import get_logger
    logger = get_logger()
    
    ws = get_world_state()
    with _world_state_lock:
        ws.pending_confirmation = PendingConfirmation(
            plan=plan,
            expires_ts=time.time() + timeout_sec,
            prompt=prompt,
            decision=decision,
        )
    
    # Log the pending plan setup
    tool_names = [p.get("tool", "?") for p in plan] if plan else []
    logger.info(f"[CONFIRM] pending_plan set expires_in={timeout_sec:.1f}s tools={tool_names}")


def get_pending_confirmation() -> Optional[PendingConfirmation]:
    """
    Get the pending confirmation, if any and not expired.
    
    Returns:
        PendingConfirmation or None if none exists or expired
    """
    ws = get_world_state()
    with _world_state_lock:
        pending = ws.pending_confirmation
        if pending is None:
            return None
        if pending.is_expired():
            ws.pending_confirmation = None
            return None
        return pending


def clear_pending_confirmation() -> None:
    """Clear any pending confirmation."""
    ws = get_world_state()
    with _world_state_lock:
        ws.pending_confirmation = None


def consume_pending_confirmation() -> Optional[List[Dict[str, Any]]]:
    """
    Consume and return the pending confirmation plan.
    
    Used when user confirms with "yes" - returns the plan and clears it.
    
    Returns:
        The tool plan or None if no valid pending confirmation
    """
    ws = get_world_state()
    with _world_state_lock:
        pending = ws.pending_confirmation
        if pending is None:
            return None
        if pending.is_expired():
            ws.pending_confirmation = None
            return None
        plan = pending.plan
        ws.pending_confirmation = None
        return plan


def set_last_autonomy_decision(
    mode: str,
    confidence: float,
    risk: str,
    action: str,
    reason: str,
    plan_summary: str,
) -> None:
    """
    Record the last autonomy policy decision.
    
    Used by "why did you do that" command.
    """
    ws = get_world_state()
    with _world_state_lock:
        ws.last_autonomy_decision = LastAutonomyDecision(
            mode=mode,
            confidence=confidence,
            risk=risk,
            action=action,
            reason=reason,
            plan_summary=plan_summary,
            ts=time.time(),
        )


def get_last_autonomy_decision() -> Optional[LastAutonomyDecision]:
    """
    Get the last autonomy decision for "why did you do that" command.
    
    Returns:
        LastAutonomyDecision or None if no decision has been made
    """
    ws = get_world_state()
    with _world_state_lock:
        return ws.last_autonomy_decision


def has_pending_confirmation() -> bool:
    """Check if there's a valid (non-expired) pending confirmation."""
    return get_pending_confirmation() is not None
