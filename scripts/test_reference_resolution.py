"""Tests for Phase 10 - Continuity & Reference Resolution.

Tests the WorldState store and reference resolver components.
Run with: python -m pytest scripts/test_reference_resolution.py -v
"""

import pytest
import time

# Import the modules under test
from wyzer.context.world_state import (
    WorldState,
    get_world_state,
    update_world_state,
    update_from_tool_execution,
    clear_world_state,
    _extract_target_from_args,
    _extract_result_summary,
)
from wyzer.core.reference_resolver import (
    resolve_references,
    get_resolution_context,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def clean_world_state():
    """Ensure clean state before and after each test."""
    clear_world_state()
    yield
    clear_world_state()


@pytest.fixture
def populated_world_state():
    """Create a WorldState with typical values."""
    ws = get_world_state()
    update_world_state(
        last_tool="open_app",
        last_target="Chrome",
        last_result_summary="success",
        active_app="Chrome",
        active_window_title="Google - Chrome"
    )
    return ws


# ============================================================================
# WORLD STATE TESTS
# ============================================================================

class TestWorldState:
    """Tests for WorldState dataclass and singleton."""
    
    def test_get_world_state_returns_singleton(self):
        """get_world_state always returns the same instance."""
        ws1 = get_world_state()
        ws2 = get_world_state()
        assert ws1 is ws2
    
    def test_world_state_default_values(self):
        """New WorldState has None fields."""
        ws = get_world_state()
        assert ws.last_tool is None
        assert ws.last_target is None
        assert ws.last_result_summary is None
        assert ws.active_app is None
        assert ws.active_window_title is None
    
    def test_update_world_state_sets_fields(self):
        """update_world_state correctly updates fields."""
        update_world_state(
            last_tool="focus_window",
            last_target="Notepad",
        )
        ws = get_world_state()
        assert ws.last_tool == "focus_window"
        assert ws.last_target == "Notepad"
        assert ws.last_result_summary is None  # Not set
    
    def test_update_world_state_partial_update(self):
        """update_world_state only updates provided fields."""
        update_world_state(last_tool="open_app", last_target="Chrome")
        update_world_state(last_target="Firefox")  # Only update target
        
        ws = get_world_state()
        assert ws.last_tool == "open_app"  # Unchanged
        assert ws.last_target == "Firefox"  # Updated
    
    def test_world_state_clear(self):
        """clear() resets all fields."""
        update_world_state(last_tool="open_app", last_target="Chrome")
        clear_world_state()
        
        ws = get_world_state()
        assert ws.last_tool is None
        assert ws.last_target is None
    
    def test_has_last_action(self):
        """has_last_action returns True only when both tool and target set."""
        ws = get_world_state()
        assert not ws.has_last_action()
        
        update_world_state(last_tool="open_app")
        assert not ws.has_last_action()
        
        update_world_state(last_target="Chrome")
        assert ws.has_last_action()
    
    def test_get_age_seconds(self):
        """get_age_seconds returns time since last update."""
        ws = get_world_state()
        update_world_state(last_tool="test")
        
        # Age should be very small (just updated)
        assert ws.get_age_seconds() < 1.0
        
        # Simulate passage of time
        ws.last_updated_ts = time.time() - 10
        assert ws.get_age_seconds() >= 10.0


class TestUpdateFromToolExecution:
    """Tests for update_from_tool_execution helper."""
    
    def test_update_from_open_app(self):
        """Tool execution updates state correctly."""
        update_from_tool_execution(
            tool_name="open_app",
            tool_args={"app": "Chrome"},
            tool_result={"success": True}
        )
        
        ws = get_world_state()
        assert ws.last_tool == "open_app"
        assert ws.last_target == "Chrome"
        assert ws.last_result_summary == "success"
    
    def test_update_from_focus_window(self):
        """Focus window tool updates state."""
        update_from_tool_execution(
            tool_name="focus_window",
            tool_args={"window": "Visual Studio Code"},
            tool_result={"ok": True}
        )
        
        ws = get_world_state()
        assert ws.last_tool == "focus_window"
        assert ws.last_target == "Visual Studio Code"
    
    def test_no_update_on_error(self):
        """Error results don't update state."""
        update_world_state(last_tool="old_tool", last_target="old_target")
        
        update_from_tool_execution(
            tool_name="open_app",
            tool_args={"app": "Chrome"},
            tool_result={"error": {"type": "not_found"}}
        )
        
        ws = get_world_state()
        assert ws.last_tool == "old_tool"  # Unchanged
        assert ws.last_target == "old_target"  # Unchanged
    
    def test_extract_target_from_various_args(self):
        """Target extraction handles different arg structures."""
        # Common key: app
        assert _extract_target_from_args("open_app", {"app": "Chrome"}) == "Chrome"
        
        # Common key: target
        assert _extract_target_from_args("any_tool", {"target": "Something"}) == "Something"
        
        # Common key: name
        assert _extract_target_from_args("any_tool", {"name": "MyThing"}) == "MyThing"
        
        # Window-related
        assert _extract_target_from_args("focus_window", {"window": "Notepad"}) == "Notepad"
        
        # Empty args
        assert _extract_target_from_args("any_tool", {}) is None
        assert _extract_target_from_args("any_tool", None) is None
    
    def test_extract_result_summary(self):
        """Result summary extraction works."""
        assert _extract_result_summary("tool", {"success": True}) == "success"
        assert _extract_result_summary("tool", {"ok": True}) == "success"
        assert _extract_result_summary("tool", {"error": "failed"}) == "failed"
        assert _extract_result_summary("tool", {"message": "Done!"}) == "Done!"
        assert _extract_result_summary("tool", {}) is None


# ============================================================================
# REFERENCE RESOLVER TESTS
# ============================================================================

class TestReferenceResolver:
    """Tests for resolve_references function."""
    
    def test_close_it_with_last_target(self, populated_world_state):
        """'close it' resolves to 'close <last_target>'."""
        result = resolve_references("close it")
        assert result == "close Chrome"
    
    def test_close_that_with_last_target(self, populated_world_state):
        """'close that' resolves to 'close <last_target>'."""
        result = resolve_references("close that")
        assert result == "close Chrome"
    
    def test_shut_it_with_last_target(self, populated_world_state):
        """'shut it' resolves to 'shut <last_target>'."""
        result = resolve_references("shut it")
        assert result == "shut Chrome"
    
    def test_close_it_no_last_target(self):
        """'close it' unchanged when no last_target."""
        result = resolve_references("close it")
        assert result == "close it"  # Unchanged
    
    def test_close_it_with_active_app_fallback(self):
        """'close it' falls back to active_app when no app tool was last."""
        update_world_state(
            last_tool="timer",  # Not an app tool
            last_target="timer:30s",
            active_app="Firefox"
        )
        result = resolve_references("close it")
        assert result == "close Firefox"
    
    def test_open_it_again(self, populated_world_state):
        """'open it again' resolves correctly."""
        result = resolve_references("open it again")
        assert result == "open Chrome"
    
    def test_open_that(self, populated_world_state):
        """'open that' resolves correctly."""
        result = resolve_references("open that")
        assert result == "open Chrome"
    
    def test_open_it_no_context(self):
        """'open it' unchanged when no context."""
        result = resolve_references("open it")
        assert result == "open it"
    
    def test_minimize_it(self, populated_world_state):
        """'minimize it' resolves correctly."""
        result = resolve_references("minimize it")
        assert result == "minimize Chrome"
    
    def test_maximize_it(self, populated_world_state):
        """'maximize it' resolves correctly."""
        result = resolve_references("maximize it")
        assert result == "maximize Chrome"
    
    def test_focus_on_it(self, populated_world_state):
        """'focus on it' resolves correctly."""
        result = resolve_references("focus on it")
        assert result == "focus Chrome"
    
    def test_do_that_again_with_full_context(self, populated_world_state):
        """'do that again' reconstructs the last command."""
        result = resolve_references("do that again")
        assert result == "open Chrome"  # last_tool was open_app
    
    def test_repeat_that(self, populated_world_state):
        """'repeat that' reconstructs the last command."""
        result = resolve_references("repeat that")
        assert result == "open Chrome"
    
    def test_again_alone(self, populated_world_state):
        """'again' alone reconstructs the last command."""
        result = resolve_references("again")
        assert result == "open Chrome"
    
    def test_do_that_again_no_context(self):
        """'do that again' unchanged when no context."""
        result = resolve_references("do that again")
        assert result == "do that again"
    
    def test_do_that_again_partial_context(self):
        """'do that again' unchanged when only partial context."""
        update_world_state(last_tool="open_app")  # No target
        result = resolve_references("do that again")
        assert result == "do that again"
    
    def test_normal_command_unchanged(self):
        """Normal commands pass through unchanged."""
        result = resolve_references("open Chrome")
        assert result == "open Chrome"
        
        result = resolve_references("set timer for 5 minutes")
        assert result == "set timer for 5 minutes"
        
        result = resolve_references("what time is it")
        assert result == "what time is it"
    
    def test_empty_text(self):
        """Empty text returns empty."""
        assert resolve_references("") == ""
        assert resolve_references("  ") == ""
    
    def test_case_insensitive(self):
        """Patterns are case-insensitive."""
        update_world_state(last_tool="open_app", last_target="Chrome")
        
        assert resolve_references("CLOSE IT") == "close Chrome"
        assert resolve_references("Close It") == "close Chrome"
        assert resolve_references("cLoSe iT") == "close Chrome"
    
    def test_trailing_punctuation(self):
        """Trailing punctuation is handled."""
        update_world_state(last_tool="open_app", last_target="Chrome")
        
        assert resolve_references("close it.") == "close Chrome"
    
    def test_different_close_variations(self, populated_world_state):
        """Different close action verbs work."""
        assert resolve_references("quit it") == "quit Chrome"
        assert resolve_references("exit it") == "exit Chrome"
        assert resolve_references("kill it") == "kill Chrome"
        assert resolve_references("stop it") == "stop Chrome"


class TestReferenceResolverWithTools:
    """Tests for reference resolution with specific tool types."""
    
    def test_close_after_close_window(self):
        """'close it' after close_window uses that target."""
        update_world_state(
            last_tool="close_window",
            last_target="Notepad"
        )
        result = resolve_references("close it")
        assert result == "close Notepad"
    
    def test_focus_after_focus_window(self):
        """'focus on it' after focus_window uses that target."""
        update_world_state(
            last_tool="focus_window",
            last_target="VS Code"
        )
        result = resolve_references("focus on it")
        assert result == "focus VS Code"
    
    def test_repeat_focus_window(self):
        """'do that again' after focus_window."""
        update_world_state(
            last_tool="focus_window",
            last_target="Terminal"
        )
        result = resolve_references("do that again")
        assert result == "focus Terminal"
    
    def test_repeat_close_window(self):
        """'do that again' after close_window."""
        update_world_state(
            last_tool="close_window",
            last_target="Notepad"
        )
        result = resolve_references("do that again")
        assert result == "close Notepad"


class TestResolutionContext:
    """Tests for get_resolution_context helper."""
    
    def test_context_includes_all_fields(self, populated_world_state):
        """Context dict includes all relevant fields."""
        ctx = get_resolution_context()
        
        assert "last_tool" in ctx
        assert "last_target" in ctx
        assert "last_result_summary" in ctx
        assert "active_app" in ctx
        assert "active_window_title" in ctx
        assert "age_seconds" in ctx
        assert "has_last_action" in ctx
        
        assert ctx["last_tool"] == "open_app"
        assert ctx["last_target"] == "Chrome"
        assert ctx["has_last_action"] is True
    
    def test_context_empty_state(self):
        """Context with empty WorldState."""
        ctx = get_resolution_context()
        
        assert ctx["last_tool"] is None
        assert ctx["last_target"] is None
        assert ctx["has_last_action"] is False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegrationFlow:
    """Integration test stubs for full flow testing."""
    
    def test_tool_execution_updates_state(self):
        """
        Integration test stub: tool execution → WorldState update.
        
        Flow:
        1. Simulate tool execution (open_app with target="Chrome")
        2. Verify WorldState is updated
        3. Verify subsequent "close it" resolves correctly
        """
        # Simulate what orchestrator does after tool runs
        update_from_tool_execution(
            tool_name="open_app",
            tool_args={"app": "Chrome"},
            tool_result={"success": True, "message": "Opened Chrome"}
        )
        
        # Verify state was updated
        ws = get_world_state()
        assert ws.last_tool == "open_app"
        assert ws.last_target == "Chrome"
        
        # Verify reference resolution works
        result = resolve_references("close it")
        assert result == "close Chrome"
    
    def test_sequential_commands(self):
        """
        Integration test: sequential commands update state correctly.
        
        Flow:
        1. User: "open Chrome" → tool runs, state updates
        2. User: "open Notepad" → tool runs, state updates
        3. User: "close it" → should resolve to "close Notepad" (most recent)
        """
        # First command
        update_from_tool_execution(
            tool_name="open_app",
            tool_args={"app": "Chrome"},
            tool_result={"success": True}
        )
        
        # Second command
        update_from_tool_execution(
            tool_name="open_app",
            tool_args={"app": "Notepad"},
            tool_result={"success": True}
        )
        
        # Third command - should reference most recent
        result = resolve_references("close it")
        assert result == "close Notepad"  # Not Chrome
    
    def test_error_doesnt_update_state(self):
        """
        Integration test: failed tool execution doesn't update state.
        
        Flow:
        1. User: "open Chrome" → success, state updates
        2. User: "open BadApp" → error, state should NOT change
        3. User: "close it" → should still resolve to Chrome
        """
        # First command succeeds
        update_from_tool_execution(
            tool_name="open_app",
            tool_args={"app": "Chrome"},
            tool_result={"success": True}
        )
        
        # Second command fails
        update_from_tool_execution(
            tool_name="open_app",
            tool_args={"app": "BadApp"},
            tool_result={"error": {"type": "not_found", "message": "App not found"}}
        )
        
        # Should still reference Chrome (last successful)
        result = resolve_references("close it")
        assert result == "close Chrome"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
