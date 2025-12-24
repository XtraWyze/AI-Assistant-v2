"""
Test suite for switch_app tool and focus_stack functionality.

Tests:
- focus_stack push/pop behavior
- Deduplication of consecutive focus events
- switch_app modes: named, previous, next
- Hybrid router patterns for app switching
"""

import pytest
from collections import deque
from unittest.mock import patch, MagicMock


class TestFocusStack:
    """Tests for focus_stack functionality in world_state."""
    
    def setup_method(self):
        """Reset world state before each test."""
        from wyzer.context.world_state import clear_world_state
        clear_world_state()
    
    def test_push_focus_stack_adds_entry(self):
        """Push adds a new entry to the focus stack."""
        from wyzer.context.world_state import push_focus_stack, get_focus_stack
        
        push_focus_stack("chrome.exe", 1001, "Google Chrome")
        
        stack = get_focus_stack()
        assert len(stack) == 1
        assert stack[0]["app"] == "chrome.exe"
        assert stack[0]["hwnd"] == 1001
        assert stack[0]["title"] == "Google Chrome"
    
    def test_push_focus_stack_deduplicates_consecutive(self):
        """Same app consecutive doesn't create new entry, just updates."""
        from wyzer.context.world_state import push_focus_stack, get_focus_stack
        
        push_focus_stack("chrome.exe", 1001, "Tab 1")
        push_focus_stack("chrome.exe", 1002, "Tab 2")
        
        stack = get_focus_stack()
        assert len(stack) == 1
        # Should have the updated hwnd/title
        assert stack[0]["hwnd"] == 1002
        assert stack[0]["title"] == "Tab 2"
    
    def test_push_focus_stack_adds_different_apps(self):
        """Different apps create new entries."""
        from wyzer.context.world_state import push_focus_stack, get_focus_stack
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        push_focus_stack("notepad.exe", 1002, "Notepad")
        push_focus_stack("discord.exe", 1003, "Discord")
        
        stack = get_focus_stack()
        assert len(stack) == 3
        # Most recent first
        assert stack[0]["app"] == "discord.exe"
        assert stack[1]["app"] == "notepad.exe"
        assert stack[2]["app"] == "chrome.exe"
    
    def test_push_focus_stack_maxlen_10(self):
        """Stack has maxlen of 10."""
        from wyzer.context.world_state import push_focus_stack, get_focus_stack
        
        for i in range(15):
            push_focus_stack(f"app{i}.exe", 1000 + i, f"App {i}")
        
        stack = get_focus_stack()
        assert len(stack) == 10
        # Most recent should be app14
        assert stack[0]["app"] == "app14.exe"
    
    def test_get_previous_focused_app(self):
        """Get the previously focused app."""
        from wyzer.context.world_state import (
            push_focus_stack,
            get_previous_focused_app,
        )
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        push_focus_stack("notepad.exe", 1002, "Notepad")
        
        prev = get_previous_focused_app()
        assert prev is not None
        assert prev["app"] == "chrome.exe"
    
    def test_get_previous_focused_app_no_history(self):
        """No previous when only one app."""
        from wyzer.context.world_state import (
            push_focus_stack,
            get_previous_focused_app,
        )
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        
        prev = get_previous_focused_app()
        assert prev is None
    
    def test_find_app_in_focus_stack(self):
        """Find an app by name in the focus stack."""
        from wyzer.context.world_state import (
            push_focus_stack,
            find_app_in_focus_stack,
        )
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        push_focus_stack("notepad.exe", 1002, "Notepad")
        push_focus_stack("discord.exe", 1003, "Discord")
        
        # Find by exact match
        result = find_app_in_focus_stack("chrome.exe")
        assert result is not None
        assert result["app"] == "chrome.exe"
        
        # Find by partial match (without .exe)
        result = find_app_in_focus_stack("notepad")
        assert result is not None
        assert result["app"] == "notepad.exe"
    
    def test_find_app_in_focus_stack_not_found(self):
        """Returns None when app not in stack."""
        from wyzer.context.world_state import (
            push_focus_stack,
            find_app_in_focus_stack,
        )
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        
        result = find_app_in_focus_stack("spotify")
        assert result is None
    
    def test_get_next_focused_app_cycles(self):
        """Get next app cycles through focus stack."""
        from wyzer.context.world_state import (
            push_focus_stack,
            get_next_focused_app,
        )
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        push_focus_stack("notepad.exe", 1002, "Notepad")
        push_focus_stack("discord.exe", 1003, "Discord")
        
        # Stack is: [discord, notepad, chrome]
        # First next should be notepad (index 1)
        next_app = get_next_focused_app()
        assert next_app["app"] == "notepad.exe"
        
        # Second next should be chrome (index 2)
        next_app = get_next_focused_app()
        assert next_app["app"] == "chrome.exe"
        
        # Third next wraps to discord (index 0)
        next_app = get_next_focused_app()
        assert next_app["app"] == "discord.exe"


class TestHybridRouterSwitchApp:
    """Tests for hybrid router switch_app patterns."""
    
    def test_switch_to_app(self):
        """'switch to chrome' routes to switch_app mode=named."""
        from wyzer.core.hybrid_router import decide
        
        decision = decide("switch to chrome")
        
        assert decision.mode == "tool_plan"
        assert len(decision.intents) == 1
        assert decision.intents[0]["tool"] == "switch_app"
        assert decision.intents[0]["args"]["mode"] == "named"
        assert decision.intents[0]["args"]["app"] == "chrome"
    
    def test_go_to_app(self):
        """'go to spotify' routes to switch_app mode=named."""
        from wyzer.core.hybrid_router import decide
        
        decision = decide("go to spotify")
        
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "switch_app"
        assert decision.intents[0]["args"]["mode"] == "named"
        assert decision.intents[0]["args"]["app"] == "spotify"
    
    def test_go_back(self):
        """'go back' routes to switch_app mode=previous."""
        from wyzer.core.hybrid_router import decide
        
        decision = decide("go back")
        
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "switch_app"
        assert decision.intents[0]["args"]["mode"] == "previous"
    
    def test_switch_back(self):
        """'switch back' routes to switch_app mode=previous."""
        from wyzer.core.hybrid_router import decide
        
        decision = decide("switch back")
        
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "switch_app"
        assert decision.intents[0]["args"]["mode"] == "previous"
    
    def test_previous_app(self):
        """'previous app' routes to switch_app mode=previous."""
        from wyzer.core.hybrid_router import decide
        
        decision = decide("previous app")
        
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "switch_app"
        assert decision.intents[0]["args"]["mode"] == "previous"
    
    def test_last_app(self):
        """'last app' routes to switch_app mode=previous."""
        from wyzer.core.hybrid_router import decide
        
        decision = decide("last app")
        
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "switch_app"
        assert decision.intents[0]["args"]["mode"] == "previous"
    
    def test_next_app(self):
        """'next app' routes to switch_app mode=next."""
        from wyzer.core.hybrid_router import decide
        
        decision = decide("next app")
        
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "switch_app"
        assert decision.intents[0]["args"]["mode"] == "next"
    
    def test_cycle_apps(self):
        """'cycle apps' routes to switch_app mode=next."""
        from wyzer.core.hybrid_router import decide
        
        decision = decide("cycle apps")
        
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "switch_app"
        assert decision.intents[0]["args"]["mode"] == "next"
    
    def test_switch_to_the_last_app(self):
        """'switch to the last app' routes to switch_app mode=previous."""
        from wyzer.core.hybrid_router import decide
        
        decision = decide("switch to the last app")
        
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "switch_app"
        assert decision.intents[0]["args"]["mode"] == "previous"
    
    def test_open_still_works(self):
        """'open chrome' still routes to open_target (not switch_app)."""
        from wyzer.core.hybrid_router import decide
        
        decision = decide("open chrome")
        
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "open_target"


class TestSwitchAppTool:
    """Tests for the switch_app tool itself."""
    
    def setup_method(self):
        """Reset world state before each test."""
        from wyzer.context.world_state import clear_world_state
        clear_world_state()
    
    @patch('wyzer.tools.switch_app._focus_window')
    def test_switch_previous_success(self, mock_focus):
        """Switch to previous app succeeds."""
        mock_focus.return_value = True
        
        from wyzer.context.world_state import push_focus_stack
        from wyzer.tools.switch_app import SwitchAppTool
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        push_focus_stack("notepad.exe", 1002, "Notepad")
        
        tool = SwitchAppTool()
        result = tool.run(mode="previous")
        
        assert result["status"] == "switched"
        assert result["to_app"] == "Chrome"
        mock_focus.assert_called_with(1001)
    
    def test_switch_previous_no_history(self):
        """Switch previous with only one app returns no_history."""
        from wyzer.context.world_state import push_focus_stack
        from wyzer.tools.switch_app import SwitchAppTool
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        
        tool = SwitchAppTool()
        result = tool.run(mode="previous")
        
        assert result["status"] == "no_history"
        assert "No previous app" in result["spoken"]
    
    @patch('wyzer.tools.switch_app._focus_window')
    def test_switch_named_success(self, mock_focus):
        """Switch to named app succeeds."""
        mock_focus.return_value = True
        
        from wyzer.context.world_state import push_focus_stack
        from wyzer.tools.switch_app import SwitchAppTool
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        push_focus_stack("notepad.exe", 1002, "Notepad")
        
        tool = SwitchAppTool()
        result = tool.run(mode="named", app="chrome")
        
        assert result["status"] == "switched"
        assert result["to_app"] == "Chrome"
        mock_focus.assert_called_with(1001)
    
    @patch('wyzer.tools.switch_app._focus_window')
    @patch('wyzer.tools.window_manager._resolve_window_handle')
    def test_switch_named_not_found(self, mock_resolve, mock_focus):
        """Switch to app not in focus history and not found anywhere returns not_found."""
        # Mock window_manager to return no match
        mock_resolve.return_value = (None, None, None)
        mock_focus.return_value = False
        
        from wyzer.context.world_state import push_focus_stack, get_world_state
        from wyzer.tools.switch_app import SwitchAppTool
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        
        # Mock open_windows to be empty
        ws = get_world_state()
        ws.open_windows = []
        
        tool = SwitchAppTool()
        result = tool.run(mode="named", app="nonexistent_app")
        
        assert result["status"] == "not_found"
        assert "error" in result, "not_found should include error key for orchestrator"
        assert result["error"]["type"] == "app_not_found"
        assert "couldn't find" in result["spoken"].lower()
    
    def test_switch_named_already_focused(self):
        """Switch to current app returns already_focused."""
        from wyzer.context.world_state import push_focus_stack
        from wyzer.tools.switch_app import SwitchAppTool
        
        push_focus_stack("chrome.exe", 1001, "Chrome")
        
        tool = SwitchAppTool()
        result = tool.run(mode="named", app="chrome")
        
        assert result["status"] == "already_focused"
        assert "already on" in result["spoken"].lower()
    
    def test_switch_invalid_mode(self):
        """Invalid mode returns error."""
        from wyzer.tools.switch_app import SwitchAppTool
        
        tool = SwitchAppTool()
        result = tool.run(mode="invalid")
        
        assert "error" in result
        assert result["error"]["type"] == "invalid_args"
    
    def test_switch_named_no_app(self):
        """Named mode without app returns error."""
        from wyzer.tools.switch_app import SwitchAppTool
        
        tool = SwitchAppTool()
        result = tool.run(mode="named")
        
        assert "error" in result
        assert result["error"]["type"] == "invalid_args"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
