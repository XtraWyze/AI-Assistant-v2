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
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING

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
class TargetRecord:
    """
    Phase 10 - Record of a resolved target for reference resolution.
    
    Stored in last_targets ring buffer to enable "the other one" toggling
    and pronoun resolution across multiple entities.
    
    Fields:
        type: Type of target (window/app/folder/drive/url/file)
        name: Canonical name (e.g., "Chrome", "Notepad")
        app_name: Process/app name (for window targets)
        window_title: Full window title (for window targets)
        hwnd: Window handle (optional, for window targets)
        pid: Process ID (optional)
        path: File/folder path (for file system targets)
        url: URL (for website targets)
        ts: Timestamp when this target was recorded
    """
    type: str  # window|app|folder|drive|url|file
    name: str
    app_name: Optional[str] = None
    window_title: Optional[str] = None
    hwnd: Optional[int] = None
    pid: Optional[int] = None
    path: Optional[str] = None
    url: Optional[str] = None
    ts: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dict for logging/debugging."""
        return {
            "type": self.type,
            "name": self.name,
            "app_name": self.app_name,
            "window_title": self.window_title,
            "hwnd": self.hwnd,
            "pid": self.pid,
            "path": self.path,
            "url": self.url,
            "ts": self.ts,
        }
    
    def matches(self, other: "TargetRecord") -> bool:
        """Check if this target is effectively the same as another."""
        if self.type != other.type:
            return False
        # For windows, match by hwnd if available, else by app_name
        if self.type == "window":
            if self.hwnd and other.hwnd:
                return self.hwnd == other.hwnd
            return self.app_name and self.app_name.lower() == (other.app_name or "").lower()
        # For apps, match by name
        if self.type == "app":
            return self.name.lower() == other.name.lower()
        # For files/folders, match by path
        if self.type in ("file", "folder", "drive"):
            return self.path and self.path.lower() == (other.path or "").lower()
        # For URLs, match by URL
        if self.type == "url":
            return self.url and self.url.lower() == (other.url or "").lower()
        return self.name.lower() == other.name.lower()


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
        
        # Phase 10 - Enhanced reference resolution fields
        last_active_window: Dict with app_name, window_title, pid, hwnd for focused window
        last_targets: Ring buffer (max 5) of resolved targets for "the other one" toggle
        last_intents: Last executed intent list for "do that again"
        last_llm_reply_only: True if last response was chat-only (no tools)
        
        # Phase 11 - Autonomy fields
        autonomy_mode: Current autonomy mode (off|low|normal|high)
        pending_confirmation: Pending high-risk confirmation, if any
        last_autonomy_decision: Last autonomy policy decision for "why" command
        
        # Phase 12 - Window Watcher fields (Multi-Monitor Awareness)
        open_windows: Snapshot of all open windows
        windows_by_monitor: Windows grouped by monitor index (1..N)
        focused_window: Currently focused window record
        recent_window_events: Ring buffer of recent changes
        last_window_snapshot_ts: Timestamp of last window snapshot
        last_active_window_ts: Timestamp when last_active_window was last updated
    """
    last_tool: Optional[str] = None
    last_target: Optional[str] = None
    last_result_summary: Optional[str] = None
    active_app: Optional[str] = None
    active_window_title: Optional[str] = None
    last_updated_ts: float = field(default_factory=time.time)
    last_action: Optional[LastAction] = None
    
    # Phase 10 - Enhanced reference resolution fields
    last_active_window: Optional[Dict[str, Any]] = None  # {app_name, window_title, pid, hwnd}
    last_active_window_ts: float = 0.0  # Timestamp when last_active_window was updated
    last_targets: List["TargetRecord"] = field(default_factory=list)  # Ring buffer, max 5
    last_intents: Optional[List[Dict[str, Any]]] = None  # Last executed intent list
    last_llm_reply_only: bool = False  # True if last response was chat-only
    
    # Phase 11 - Autonomy fields
    autonomy_mode: str = "off"  # off|low|normal|high
    pending_confirmation: Optional[PendingConfirmation] = None
    last_autonomy_decision: Optional[LastAutonomyDecision] = None
    
    # Phase 12 - Window Watcher fields
    open_windows: List[Dict[str, Any]] = field(default_factory=list)
    windows_by_monitor: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    focused_window: Optional[Dict[str, Any]] = None
    recent_window_events: List[Dict[str, Any]] = field(default_factory=list)
    last_window_snapshot_ts: float = 0.0
    detected_monitor_count: int = 1  # Actual number of monitors detected by WindowWatcher
    
    # Focus stack for deterministic app switching (switch_app tool)
    # Ordered by actual focus changes, most recent first, no consecutive duplicates
    # Each entry: {app: str, hwnd: int, title: str, ts: float}
    focus_stack: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=10))
    # Current position in focus_stack for "next app" cycling (round-robin)
    focus_stack_index: int = 0
    
    def clear(self) -> None:
        """Reset all state fields."""
        self.last_tool = None
        self.last_target = None
        self.last_result_summary = None
        self.active_app = None
        self.active_window_title = None
        self.last_updated_ts = time.time()
        self.last_action = None
        # Phase 10: Clear enhanced reference resolution fields
        self.last_active_window = None
        self.last_active_window_ts = 0.0
        self.last_targets = []
        self.last_intents = None
        self.last_llm_reply_only = False
        # Don't reset autonomy_mode on clear (user preference)
        self.pending_confirmation = None
        self.last_autonomy_decision = None
        # Phase 12: Clear window watcher state
        self.open_windows = []
        self.windows_by_monitor = {}
        self.focused_window = None
        self.recent_window_events = []
        self.last_window_snapshot_ts = 0.0
        self.detected_monitor_count = 1
        # Clear focus stack
        self.focus_stack = deque(maxlen=10)
        self.focus_stack_index = 0
    
    def has_last_action(self) -> bool:
        """Check if there's a valid last action for reference."""
        return self.last_tool is not None and self.last_target is not None
    
    def has_replay_action(self) -> bool:
        """Check if there's a structured last_action available for replay."""
        return self.last_action is not None
    
    def get_age_seconds(self) -> float:
        """Get seconds since last update."""
        return time.time() - self.last_updated_ts
    
    def get_last_target(self, index: int = 0) -> Optional["TargetRecord"]:
        """Get target from history by index (0 = most recent)."""
        if 0 <= index < len(self.last_targets):
            return self.last_targets[index]
        return None
    
    def get_other_target(self) -> Optional["TargetRecord"]:
        """Get the 'other one' - second most recent distinct target."""
        if len(self.last_targets) >= 2:
            return self.last_targets[1]
        return None


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
    
    # ===========================================================================
    # Phase 10: Build TargetRecord for last_targets ring buffer
    # ===========================================================================
    target_record = _build_target_record(tool_name, tool_args, tool_result, target)
    
    # Update state (including last_action and last_targets)
    ws = get_world_state()
    with _world_state_lock:
        if tool_name:
            ws.last_tool = tool_name
        if target:
            ws.last_target = target
        if summary:
            ws.last_result_summary = summary
        ws.last_action = last_action
        ws.last_llm_reply_only = False  # This was a tool execution
        
        # Phase 10: Update last_targets ring buffer (max 5, no duplicates at front)
        if target_record is not None:
            _push_target_record(ws, target_record)
        
        ws.last_updated_ts = time.time()


# Maximum number of targets to keep in the ring buffer
_MAX_LAST_TARGETS = 5


def _push_target_record(ws: WorldState, record: TargetRecord) -> None:
    """
    Push a target record to the front of last_targets, avoiding duplicates.
    
    If the record matches the current front, skip adding.
    Maintains max size of _MAX_LAST_TARGETS.
    """
    # Skip if it matches the most recent target
    if ws.last_targets and ws.last_targets[0].matches(record):
        return
    
    # Insert at front
    ws.last_targets.insert(0, record)
    
    # Trim to max size
    if len(ws.last_targets) > _MAX_LAST_TARGETS:
        ws.last_targets = ws.last_targets[:_MAX_LAST_TARGETS]


def _build_target_record(
    tool_name: str,
    tool_args: dict,
    tool_result: dict,
    display_name: Optional[str],
) -> Optional[TargetRecord]:
    """
    Build a TargetRecord from tool execution info.
    
    Returns None if no meaningful target can be extracted.
    """
    if not display_name:
        return None
    
    # Determine target type based on tool
    target_type = "app"  # default
    app_name = None
    window_title = None
    hwnd = None
    pid = None
    path = None
    url = None
    
    # Window tools
    if tool_name in ("close_window", "focus_window", "minimize_window", "maximize_window", "move_window"):
        target_type = "window"
        matched = (tool_result or {}).get("matched", {})
        if isinstance(matched, dict):
            app_name = matched.get("process")
            window_title = matched.get("title")
            hwnd = matched.get("hwnd")
            pid = matched.get("pid")
    
    # App tools
    elif tool_name in ("open_app", "close_app", "launch_app", "switch_app"):
        target_type = "app"
        app_name = display_name
    
    # Open target - could be app, file, folder, game
    elif tool_name == "open_target":
        resolved = (tool_result or {}).get("resolved", {})
        if isinstance(resolved, dict):
            r_type = resolved.get("type", "app")
            if r_type in ("file", "folder", "drive"):
                target_type = r_type
                path = resolved.get("path")
            elif r_type == "game":
                target_type = "app"
                app_name = resolved.get("matched_name") or display_name
            else:
                target_type = "app"
                app_name = resolved.get("matched_name") or display_name
    
    # Website tools
    elif tool_name in ("open_website", "open_url"):
        target_type = "url"
        url = (tool_args or {}).get("url") or (tool_args or {}).get("query")
    
    # Storage tools
    elif tool_name == "system_storage_open":
        target_type = "folder"
        path = (tool_result or {}).get("path")
    
    # Volume tools - use app name
    elif tool_name == "volume_control":
        process = (tool_args or {}).get("process")
        if process and process != "master":
            target_type = "app"
            app_name = process
        else:
            return None  # Don't track master volume as a target
    
    return TargetRecord(
        type=target_type,
        name=display_name,
        app_name=app_name,
        window_title=window_title,
        hwnd=hwnd,
        pid=pid,
        path=path,
        url=url,
    )


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
    
    # For switch_app: extract to_app/from_app and matched window info
    # This is crucial for chained commands like "switch to Spotify and move it"
    if tool_name == "switch_app":
        # Handle both 'switched' status (has to_app) and 'already_focused' status (has app)
        target_app = result.get("to_app") or result.get("app")
        resolved_info["to_app"] = target_app
        resolved_info["from_app"] = result.get("from_app")
        resolved_info["hwnd"] = result.get("hwnd")
        # Also check matched dict if present
        matched = result.get("matched")
        if matched and isinstance(matched, dict):
            resolved_info["process"] = matched.get("process")
            resolved_info["title"] = matched.get("title")
        # Ensure we have at least to_app for reference resolution
        if resolved_info.get("to_app"):
            return resolved_info
        return None
    
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
# PHASE 10: ENHANCED REFERENCE RESOLUTION FUNCTIONS
# ============================================================================

def update_last_intents(intents: List[Dict[str, Any]]) -> None:
    """
    Store the last executed intent list for "do that again".
    
    Args:
        intents: List of intent dicts [{tool, args}, ...]
    """
    ws = get_world_state()
    with _world_state_lock:
        # Deep copy to avoid mutation issues
        ws.last_intents = [dict(i) for i in intents] if intents else None
        ws.last_llm_reply_only = False


def update_last_active_window(
    app_name: Optional[str] = None,
    window_title: Optional[str] = None,
    pid: Optional[int] = None,
    hwnd: Optional[int] = None,
) -> None:
    """
    Update the last active window info.
    
    Called by window watcher or after focus_window tool.
    """
    ws = get_world_state()
    with _world_state_lock:
        ws.last_active_window = {
            "app_name": app_name,
            "window_title": window_title,
            "pid": pid,
            "hwnd": hwnd,
        }
        ws.last_active_window_ts = time.time()  # Track when active window was updated
        # Also update legacy fields for compatibility
        if app_name:
            ws.active_app = app_name
        if window_title:
            ws.active_window_title = window_title
        ws.last_updated_ts = time.time()


def set_last_llm_reply_only(is_reply_only: bool) -> None:
    """
    Mark whether the last response was chat-only (no tools).
    
    Used to prevent "do that again" from repeating chat responses.
    """
    ws = get_world_state()
    with _world_state_lock:
        ws.last_llm_reply_only = is_reply_only


def get_last_intents() -> Optional[List[Dict[str, Any]]]:
    """
    Get the last executed intent list for replay.
    
    Returns:
        List of intent dicts or None
    """
    ws = get_world_state()
    with _world_state_lock:
        if ws.last_llm_reply_only:
            return None  # Don't replay chat-only responses
        return [dict(i) for i in ws.last_intents] if ws.last_intents else None


def get_last_active_window() -> Optional[Dict[str, Any]]:
    """
    Get the last active window info.
    
    Returns:
        Dict with app_name, window_title, pid, hwnd or None
    """
    ws = get_world_state()
    with _world_state_lock:
        return dict(ws.last_active_window) if ws.last_active_window else None


def get_last_targets(count: int = 2) -> List[TargetRecord]:
    """
    Get the most recent targets from the ring buffer.
    
    Args:
        count: Number of targets to return (default 2 for "the other one")
        
    Returns:
        List of TargetRecord objects (may be shorter than count)
    """
    ws = get_world_state()
    with _world_state_lock:
        return list(ws.last_targets[:count])


def update_after_tool(
    tool_name: str,
    args: Dict[str, Any],
    result: Dict[str, Any],
    success: bool,
) -> None:
    """
    Phase 10 contract: Single entry point for updating world state after tool execution.
    
    This is the canonical function to call after any tool runs.
    It updates all relevant fields including last_action, last_targets, etc.
    
    Args:
        tool_name: Name of the tool that was executed
        args: Arguments passed to the tool
        result: Result returned by the tool
        success: Whether the tool succeeded (no error)
    """
    if not success:
        return  # Don't update state on failures
    
    # Delegate to existing update function
    update_from_tool_execution(tool_name, args, result)


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


# ============================================================================
# PHASE 12: WINDOW WATCHER STATE FUNCTIONS
# ============================================================================

def update_window_watcher_state(
    open_windows: List[Dict[str, Any]],
    windows_by_monitor: Dict[int, List[Dict[str, Any]]],
    focused_window: Optional[Dict[str, Any]],
    recent_events: List[Dict[str, Any]],
    detected_monitor_count: int = 1,
) -> None:
    """
    Update world state with window watcher data.
    
    Called by the window watcher after each poll cycle.
    
    Args:
        open_windows: Snapshot of all open windows
        windows_by_monitor: Windows grouped by monitor index (1..N)
        focused_window: Currently focused window record
        recent_events: Recent change events (from ring buffer)
        detected_monitor_count: Number of monitors detected by WindowWatcher
    """
    ws = get_world_state()
    
    # Track if focus changed for focus_stack update (done outside the main lock)
    new_focus_app = None
    new_focus_hwnd = None
    new_focus_title = None
    
    with _world_state_lock:
        # Check if focus changed before updating
        old_focus_hwnd = ws.focused_window.get("hwnd") if ws.focused_window else None
        new_hwnd = focused_window.get("hwnd") if focused_window else None
        
        if new_hwnd and new_hwnd != old_focus_hwnd:
            # Focus changed - capture info for stack update
            new_focus_app = focused_window.get("process")
            new_focus_hwnd = new_hwnd
            new_focus_title = focused_window.get("title")
        
        ws.open_windows = list(open_windows)
        ws.windows_by_monitor = {k: list(v) for k, v in windows_by_monitor.items()}
        ws.focused_window = dict(focused_window) if focused_window else None
        ws.recent_window_events = list(recent_events)
        ws.last_window_snapshot_ts = time.time()
        ws.detected_monitor_count = max(1, detected_monitor_count)
        
        # Also update active_app and active_window_title for Phase 9 compatibility
        if focused_window:
            ws.active_app = focused_window.get("process")
            ws.active_window_title = focused_window.get("title")
    
    # Update focus_stack outside the main lock (push_focus_stack has its own locking)
    if new_focus_app and new_focus_hwnd:
        push_focus_stack(new_focus_app, new_focus_hwnd, new_focus_title or "")


def get_windows_on_monitor(monitor: int) -> List[Dict[str, Any]]:
    """
    Get windows on a specific monitor.
    
    Args:
        monitor: 1-based monitor index
        
    Returns:
        List of window records on that monitor
    """
    ws = get_world_state()
    with _world_state_lock:
        return list(ws.windows_by_monitor.get(monitor, []))


def get_focused_window_info() -> Optional[Dict[str, Any]]:
    """
    Get the currently focused window info.
    
    Returns:
        Dict with title, process, monitor, etc. or None
    """
    ws = get_world_state()
    with _world_state_lock:
        return dict(ws.focused_window) if ws.focused_window else None


def get_recent_window_events(event_type: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get recent window events, optionally filtered by type.
    
    Args:
        event_type: Filter by event type (opened/closed/moved/title_changed/focus_changed)
        limit: Maximum number of events to return
        
    Returns:
        List of event records (newest last)
    """
    ws = get_world_state()
    with _world_state_lock:
        events = list(ws.recent_window_events)
    
    if event_type:
        events = [e for e in events if e.get("type") == event_type]
    
    # Return last N events (newest)
    return events[-limit:] if len(events) > limit else events


def get_all_open_windows() -> List[Dict[str, Any]]:
    """
    Get all open windows.
    
    Returns:
        List of all window records
    """
    ws = get_world_state()
    with _world_state_lock:
        return list(ws.open_windows)


def get_monitor_count() -> int:
    """
    Get the number of detected monitors.
    
    Returns the actual number of monitors detected by the WindowWatcher,
    not just the number of monitors that have windows on them.
    
    Returns:
        Number of monitors (at least 1)
    """
    ws = get_world_state()
    with _world_state_lock:
        return max(1, ws.detected_monitor_count)


# ============================================================================
# FOCUS STACK - DETERMINISTIC APP SWITCHING
# ============================================================================

def push_focus_stack(app: str, hwnd: int, title: str) -> None:
    """
    Push a focus event onto the focus_stack.
    
    Called when a window gains focus. Deduplicates consecutive entries
    (same app won't appear twice in a row).
    
    Args:
        app: Application/process name (e.g., "chrome.exe", "notepad.exe")
        hwnd: Window handle
        title: Window title
    """
    if not app:
        return
    
    ws = get_world_state()
    with _world_state_lock:
        # Normalize app name for comparison
        app_norm = app.lower().strip()
        
        # Check if the top of the stack is the same app (avoid consecutive duplicates)
        if ws.focus_stack:
            top = ws.focus_stack[0]
            if top.get("app", "").lower() == app_norm:
                # Same app, just update hwnd/title/ts
                top["hwnd"] = hwnd
                top["title"] = title
                top["ts"] = time.time()
                return
        
        # Push new entry to front
        entry = {
            "app": app,
            "hwnd": hwnd,
            "title": title,
            "ts": time.time(),
        }
        ws.focus_stack.appendleft(entry)
        
        # Reset cycle index when a new app is focused
        ws.focus_stack_index = 0


def get_focus_stack() -> List[Dict[str, Any]]:
    """
    Get the current focus stack.
    
    Returns:
        List of focus records, most recent first
    """
    ws = get_world_state()
    with _world_state_lock:
        return list(ws.focus_stack)


def get_previous_focused_app() -> Optional[Dict[str, Any]]:
    """
    Get the previously focused app (for "go back" / "switch back").
    
    Returns:
        Focus record of the previous app, or None if only one app in history
    """
    ws = get_world_state()
    with _world_state_lock:
        if len(ws.focus_stack) < 2:
            return None
        return dict(ws.focus_stack[1])


def get_next_focused_app() -> Optional[Dict[str, Any]]:
    """
    Cycle forward through focus_stack (round-robin) for "next app".
    
    Returns:
        Focus record of the next app in the cycle, or None if not enough apps
    """
    ws = get_world_state()
    with _world_state_lock:
        if len(ws.focus_stack) < 2:
            return None
        
        # Increment index and wrap around
        ws.focus_stack_index = (ws.focus_stack_index + 1) % len(ws.focus_stack)
        return dict(ws.focus_stack[ws.focus_stack_index])


def find_app_in_focus_stack(app_name: str) -> Optional[Dict[str, Any]]:
    """
    Find the most recent entry for an app in the focus_stack.
    
    Args:
        app_name: Application name to find (case-insensitive, substring match)
        
    Returns:
        Focus record if found, None otherwise
    """
    if not app_name:
        return None
    
    app_norm = app_name.lower().strip()
    # Remove .exe if present for matching
    if app_norm.endswith(".exe"):
        app_norm = app_norm[:-4]
    
    ws = get_world_state()
    with _world_state_lock:
        for entry in ws.focus_stack:
            entry_app = (entry.get("app") or "").lower()
            # Remove .exe for comparison
            if entry_app.endswith(".exe"):
                entry_app = entry_app[:-4]
            
            # Match if the requested app name is contained in the entry app name
            if app_norm in entry_app or entry_app in app_norm:
                return dict(entry)
    
    return None


def get_current_focused_app() -> Optional[Dict[str, Any]]:
    """
    Get the currently focused app (top of focus_stack).
    
    Returns:
        Focus record of the current app, or None if stack is empty
    """
    ws = get_world_state()
    with _world_state_lock:
        if ws.focus_stack:
            return dict(ws.focus_stack[0])
        return None
