"""
Tool registry for managing available tools.
"""
import logging
import os
from typing import Dict, List, Optional
from wyzer.tools.tool_base import ToolBase

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        """Initialize empty registry"""
        self._tools: Dict[str, ToolBase] = {}
    
    def register(self, tool: ToolBase) -> None:
        """
        Register a tool.
        
        If a tool with the same name is already registered, log a WARNING
        and keep the original (do NOT overwrite).  In debug mode
        (WYZER_DEBUG=1) raise instead, to surface accidental duplicates early.
        
        Args:
            tool: Tool instance to register
        """
        if tool.name in self._tools:
            existing = self._tools[tool.name]
            existing_loc = f"{type(existing).__module__}.{type(existing).__qualname__}"
            new_loc = f"{type(tool).__module__}.{type(tool).__qualname__}"
            msg = (
                f"Duplicate tool registration ignored: '{tool.name}' "
                f"(existing={existing_loc}, duplicate={new_loc})"
            )
            if os.environ.get("WYZER_DEBUG") == "1":
                raise ValueError(msg)
            logger.warning(msg)
            return
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[ToolBase]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """
        List all registered tools with their metadata.
        
        Returns:
            List of dicts with name and description
        """
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self._tools.values()
        ]
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool exists"""
        return name in self._tools


def build_default_registry() -> ToolRegistry:
    """
    Build registry with default tools.
    
    Returns:
        ToolRegistry with standard tools registered
    """
    from wyzer.tools.get_time import GetTimeTool
    from wyzer.tools.get_system_info import GetSystemInfoTool
    from wyzer.tools.open_website import OpenWebsiteTool

    # Location / Weather tools
    from wyzer.tools.get_location import GetLocationTool
    from wyzer.tools.get_weather_forecast import GetWeatherForecastTool
    
    # Phase 6 tools - LocalLibrary
    from wyzer.tools.local_library_refresh import LocalLibraryRefreshTool
    from wyzer.tools.open_target import OpenTargetTool
    
    # Phase 6 tools - Window management
    from wyzer.tools.window_manager import (
        FocusWindowTool,
        MinimizeWindowTool,
        MaximizeWindowTool,
        CloseWindowTool,
        MoveWindowToMonitorTool,
        ListOpenWindowsTool,
    )
    
    # Switch app tool (deterministic app switching using focus history)
    from wyzer.tools.switch_app import SwitchAppTool
    
    # Phase 6 tools - Monitor info
    from wyzer.tools.monitor_info import MonitorInfoTool
    
    # Get window monitor tool
    from wyzer.tools.get_window_monitor import GetWindowMonitorTool
    
    # Phase 6 tools - Media controls
    from wyzer.tools.media_controls import (
        MediaPlayPauseTool,
        MediaNextTool,
        MediaPreviousTool,
        VolumeUpTool,
        VolumeDownTool,
        VolumeMuteToggleTool,
        GetNowPlayingTool
    )

    # True volume control (pycaw)
    from wyzer.tools.volume_control import VolumeControlTool

    # Phase 6 tools - Audio device switching
    from wyzer.tools.audio_output_device import SetAudioOutputDeviceTool
    
    # System storage tools
    from wyzer.tools.system_storage import (
        SystemStorageScanTool,
        SystemStorageListTool,
        SystemStorageOpenTool
    )
    
    # Timer tool
    from wyzer.tools.timer_tool import TimerTool
    
    # Google search tool
    from wyzer.tools.google_search_open import GoogleSearchOpenTool
    
    # Phase 9 - Screen Awareness (READ-ONLY)
    from wyzer.tools.get_window_context import GetWindowContextTool
    
    # Phase 14 - Desktop Ground Truth Tools
    from wyzer.tools.desktop.get_active_window import GetActiveWindowTool
    from wyzer.tools.desktop.perceive_uia import PerceiveUIAFocusedWindowTool
    from wyzer.tools.desktop.describe_screen import DescribeScreenTool
    from wyzer.tools.desktop.desktop_click_uia import DesktopClickUIATool
    from wyzer.tools.desktop.screenshot_tool import ScreenshotFocusedWindowTool
    from wyzer.tools.desktop.ocr_tool import OCRRegionTool
    from wyzer.tools.desktop.assertions import UIFindTextTool, InstallSucceededCheckTool
    from wyzer.tools.desktop.input_actions import (
        WaitMsTool,
        HotkeyTool,
        TypeTextTool,
        ClickXYTool,
        ScrollTool,
    )
    from wyzer.tools.desktop.get_recent_events import GetRecentEventsTool
    
    # Phase 16 - Deterministic click-and-type tools
    from wyzer.tools.desktop.perceive_ocr_focused import PerceiveOCRFocusedWindowTool
    from wyzer.tools.desktop.overlay import ShowChoiceOverlayTool, WaitForOverlayChoiceTool
    from wyzer.tools.desktop.assert_text_present import AssertTextPresentTool
    
    registry = ToolRegistry()
    
    # Register default tools
    registry.register(GetTimeTool())
    registry.register(GetSystemInfoTool())
    registry.register(OpenWebsiteTool())

    # Register location/weather tools
    registry.register(GetLocationTool())
    registry.register(GetWeatherForecastTool())
    
    # Register LocalLibrary tools
    registry.register(LocalLibraryRefreshTool())
    registry.register(OpenTargetTool())
    
    # Register window management tools
    registry.register(FocusWindowTool())
    registry.register(MinimizeWindowTool())
    registry.register(MaximizeWindowTool())
    registry.register(CloseWindowTool())
    registry.register(MoveWindowToMonitorTool())
    registry.register(ListOpenWindowsTool())
    
    # Register switch app tool (deterministic app switching)
    registry.register(SwitchAppTool())
    
    # Register monitor info tool
    registry.register(MonitorInfoTool())
    
    # Register get window monitor tool
    registry.register(GetWindowMonitorTool())
    
    # Register media control tools
    registry.register(MediaPlayPauseTool())
    registry.register(MediaNextTool())
    registry.register(MediaPreviousTool())
    registry.register(VolumeUpTool())
    registry.register(VolumeDownTool())
    registry.register(VolumeMuteToggleTool())
    registry.register(GetNowPlayingTool())

    # Register true volume tool (preferred over VK-based volume_* tools)
    registry.register(VolumeControlTool())

    # Register audio device switching tool
    registry.register(SetAudioOutputDeviceTool())
    
    # Register system storage tools
    registry.register(SystemStorageScanTool())
    registry.register(SystemStorageListTool())
    registry.register(SystemStorageOpenTool())
    
    # Register timer tool
    registry.register(TimerTool())
    
    # Register Google search tool
    registry.register(GoogleSearchOpenTool())
    
    # Register Phase 9 - Screen Awareness tool (READ-ONLY)
    registry.register(GetWindowContextTool())
    
    # Register Phase 14 - Desktop Ground Truth Tools
    registry.register(GetActiveWindowTool())
    registry.register(PerceiveUIAFocusedWindowTool())
    registry.register(DescribeScreenTool())
    registry.register(DesktopClickUIATool())
    registry.register(ScreenshotFocusedWindowTool())
    registry.register(OCRRegionTool())
    registry.register(UIFindTextTool())
    registry.register(InstallSucceededCheckTool())
    registry.register(WaitMsTool())
    registry.register(HotkeyTool())
    registry.register(TypeTextTool())
    registry.register(ClickXYTool())
    registry.register(ScrollTool())
    
    # Register Phase 15 - Event log query tool
    registry.register(GetRecentEventsTool())
    
    # Register Phase 16 - Deterministic click-and-type tools
    registry.register(PerceiveOCRFocusedWindowTool())
    registry.register(ShowChoiceOverlayTool())
    registry.register(WaitForOverlayChoiceTool())
    registry.register(AssertTextPresentTool())
    
    return registry
