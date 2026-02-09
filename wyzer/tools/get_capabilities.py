"""
Get capabilities tool — returns a TTS-friendly summary of Wyzer's abilities.

Deterministic: reads the tool registry and returns names, short descriptions,
and a few example voice commands.
"""
from typing import Dict, Any, List
from wyzer.tools.tool_base import ToolBase


# Hand-curated example commands grouped by category.
# These are spoken aloud, so keep them short and natural.
_EXAMPLE_COMMANDS: List[str] = [
    "Open Chrome",
    "What time is it?",
    "Set a timer for 5 minutes",
    "Pause music",
    "Volume up",
    "What's on my screen?",
    "Google cats",
    "Switch to Notepad",
    "Close Discord",
    "What's the weather?",
    "Minimize all windows",
    "Move Spotify to monitor 2",
]

# Short, human-readable category labels for each tool.
_TOOL_CATEGORY: Dict[str, str] = {
    "get_time": "Time & Date",
    "get_system_info": "System Info",
    "open_website": "Web",
    "open_target": "Open Apps",
    "get_location": "Location",
    "get_weather_forecast": "Weather",
    "local_library_refresh": "App Library",
    "focus_window": "Window Management",
    "minimize_window": "Window Management",
    "maximize_window": "Window Management",
    "close_window": "Window Management",
    "move_window_to_monitor": "Window Management",
    "list_open_windows": "Window Management",
    "switch_app": "App Switching",
    "monitor_info": "Monitor Info",
    "get_window_monitor": "Monitor Info",
    "media_play_pause": "Media Controls",
    "media_next": "Media Controls",
    "media_previous": "Media Controls",
    "volume_up": "Media Controls",
    "volume_down": "Media Controls",
    "volume_mute_toggle": "Media Controls",
    "get_now_playing": "Media Controls",
    "volume_control": "Volume",
    "set_audio_output_device": "Audio Devices",
    "system_storage_scan": "Storage",
    "system_storage_list": "Storage",
    "system_storage_open": "Storage",
    "timer": "Timers",
    "google_search_open": "Google Search",
    "get_window_context": "Screen Awareness",
    "get_active_window": "Desktop Automation",
    "perceive_uia_focused_window": "Desktop Automation",
    "describe_screen": "Screen Awareness",
    "desktop_click_uia": "Desktop Automation",
    "screenshot_focused_window": "Desktop Automation",
    "ocr_region": "Desktop Automation",
    "ui_find_text": "Desktop Automation",
    "install_succeeded_check": "Desktop Automation",
    "wait_ms": "Desktop Automation",
    "hotkey": "Desktop Automation",
    "type_text": "Desktop Automation",
    "click_xy": "Desktop Automation",
    "scroll": "Desktop Automation",
    "get_recent_events": "Event Log",
    "perceive_ocr_focused_window": "Desktop Automation",
    "show_choice_overlay": "Desktop Automation",
    "wait_for_overlay_choice": "Desktop Automation",
    "assert_text_present": "Desktop Automation",
    "get_capabilities": "Help",
}


class GetCapabilitiesTool(ToolBase):
    """Deterministic tool that lists Wyzer's capabilities from the registry."""

    def __init__(self):
        super().__init__()
        self._name = "get_capabilities"
        self._description = "List all capabilities and available voice commands"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        """Return a JSON dict with categories, tool count, and example commands."""
        # Import here to avoid circular imports at module level.
        from wyzer.tools.registry import build_default_registry

        registry = build_default_registry()
        tools_list = registry.list_tools()

        # Deduplicate categories while preserving order.
        categories_seen: Dict[str, List[str]] = {}
        for t in tools_list:
            name = t["name"]
            cat = _TOOL_CATEGORY.get(name, "Other")
            categories_seen.setdefault(cat, []).append(name)

        category_summaries = []
        for cat, tool_names in categories_seen.items():
            category_summaries.append({
                "category": cat,
                "tool_count": len(tool_names),
            })

        return {
            "total_tools": len(tools_list),
            "categories": category_summaries,
            "examples": _EXAMPLE_COMMANDS[:6],
            "summary": _build_tts_summary(category_summaries, len(tools_list)),
        }


def _build_tts_summary(categories: List[Dict[str, Any]], total: int) -> str:
    """Build a short, TTS-friendly sentence summarising capabilities."""
    cat_names = [c["category"] for c in categories if c["category"] != "Other"]
    # Keep the spoken list short — pick the most interesting categories.
    highlights = cat_names[:8]
    joined = ", ".join(highlights)
    return (
        f"I can help with {joined}, and more. "
        f"I have {total} tools in total. "
        f"Try saying things like: Open Chrome, What time is it, "
        f"or Set a timer for 5 minutes."
    )
