"""Tests for Phase 10.1 - Replay Last Action (Deterministic Replay).

Tests the replay_last_action mechanism that enables stable replays of
the last successful tool execution.

Run with: python -m pytest scripts/test_replay_last_action.py -v
"""

import pytest
import time
from unittest.mock import patch, MagicMock

# Import the modules under test
from wyzer.context.world_state import (
    WorldState,
    LastAction,
    get_world_state,
    update_world_state,
    update_from_tool_execution,
    clear_world_state,
)
from wyzer.core.reference_resolver import (
    resolve_references,
    is_replay_request,
    is_replay_sentinel,
    REPLAY_LAST_ACTION_SENTINEL,
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
def ws_with_open_target_uwp():
    """WorldState with last_action set to open_target with UWP resolved info."""
    ws = get_world_state()
    ws.last_tool = "open_target"
    ws.last_target = "Spotify"
    ws.last_action = LastAction(
        tool="open_target",
        args={"query": "spotify"},
        resolved={
            "type": "uwp",
            "path": "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
            "matched_name": "Spotify",
        },
        ts=time.time(),
    )
    return ws


@pytest.fixture
def ws_with_close_window():
    """WorldState with last_action set to close_window with matched info."""
    ws = get_world_state()
    ws.last_tool = "close_window"
    ws.last_target = "Chrome"
    ws.last_action = LastAction(
        tool="close_window",
        args={"process": "chrome.exe"},
        resolved={
            "process": "chrome.exe",
            "title": "Google - Chrome",
            "hwnd": 12345,
        },
        ts=time.time(),
    )
    return ws


@pytest.fixture
def ws_with_open_target_game():
    """WorldState with last_action set to open_target with game resolved info."""
    ws = get_world_state()
    ws.last_tool = "open_target"
    ws.last_target = "Elden Ring"
    ws.last_action = LastAction(
        tool="open_target",
        args={"query": "elden ring"},
        resolved={
            "type": "game",
            "matched_name": "Elden Ring",
            "game_name": "Elden Ring",
            "launch": {
                "type": "steam_uri",
                "target": "steam://rungameid/1245620",
            },
        },
        ts=time.time(),
    )
    return ws


# ============================================================================
# LAST_ACTION DATACLASS TESTS
# ============================================================================

class TestLastActionDataclass:
    """Tests for the LastAction dataclass."""
    
    def test_create_last_action(self):
        """LastAction can be created with required fields."""
        action = LastAction(
            tool="open_target",
            args={"query": "spotify"},
        )
        assert action.tool == "open_target"
        assert action.args == {"query": "spotify"}
        assert action.resolved is None
        assert action.ts > 0
    
    def test_create_last_action_with_resolved(self):
        """LastAction can store resolved info."""
        action = LastAction(
            tool="open_target",
            args={"query": "spotify"},
            resolved={"type": "uwp", "path": "SpotifyAB.SpotifyMusic"},
        )
        assert action.resolved["type"] == "uwp"
        assert action.resolved["path"] == "SpotifyAB.SpotifyMusic"
    
    def test_to_dict(self):
        """to_dict() returns all fields as dict."""
        action = LastAction(
            tool="close_window",
            args={"process": "chrome.exe"},
            resolved={"process": "chrome.exe"},
            ts=12345.0,
        )
        d = action.to_dict()
        assert d["tool"] == "close_window"
        assert d["args"] == {"process": "chrome.exe"}
        assert d["resolved"] == {"process": "chrome.exe"}
        assert d["ts"] == 12345.0


# ============================================================================
# WORLD STATE LAST_ACTION TESTS
# ============================================================================

class TestWorldStateLastAction:
    """Tests for WorldState last_action field."""
    
    def test_world_state_last_action_none_by_default(self):
        """WorldState.last_action is None by default."""
        ws = get_world_state()
        assert ws.last_action is None
    
    def test_has_replay_action_false_when_none(self):
        """has_replay_action() returns False when last_action is None."""
        ws = get_world_state()
        assert ws.has_replay_action() is False
    
    def test_has_replay_action_true_when_set(self, ws_with_open_target_uwp):
        """has_replay_action() returns True when last_action is set."""
        ws = get_world_state()
        assert ws.has_replay_action() is True
    
    def test_clear_resets_last_action(self, ws_with_open_target_uwp):
        """clear() resets last_action to None."""
        ws = get_world_state()
        assert ws.last_action is not None
        ws.clear()
        assert ws.last_action is None


# ============================================================================
# UPDATE_FROM_TOOL_EXECUTION TESTS (LAST_ACTION CAPTURE)
# ============================================================================

class TestLastActionCapture:
    """Tests for capturing last_action in update_from_tool_execution."""
    
    def test_captures_last_action_for_open_target_uwp(self):
        """update_from_tool_execution captures last_action for UWP apps."""
        update_from_tool_execution(
            tool_name="open_target",
            tool_args={"query": "spotify"},
            tool_result={
                "status": "opened",
                "resolved": {
                    "type": "uwp",
                    "path": "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
                    "matched_name": "Spotify",
                },
            },
        )
        
        ws = get_world_state()
        assert ws.last_action is not None
        assert ws.last_action.tool == "open_target"
        assert ws.last_action.args == {"query": "spotify"}
        assert ws.last_action.resolved["type"] == "uwp"
        assert "SpotifyAB" in ws.last_action.resolved["path"]
    
    def test_captures_last_action_for_close_window(self):
        """update_from_tool_execution captures last_action for close_window."""
        update_from_tool_execution(
            tool_name="close_window",
            tool_args={"process": "chrome.exe"},
            tool_result={
                "status": "closed",
                "matched": {
                    "process": "chrome.exe",
                    "title": "Google - Chrome",
                    "hwnd": 12345,
                },
            },
        )
        
        ws = get_world_state()
        assert ws.last_action is not None
        assert ws.last_action.tool == "close_window"
        assert ws.last_action.resolved["process"] == "chrome.exe"
    
    def test_no_last_action_on_error(self):
        """update_from_tool_execution does NOT set last_action on error."""
        # First set a valid last_action
        update_from_tool_execution(
            tool_name="open_target",
            tool_args={"query": "spotify"},
            tool_result={"status": "opened", "resolved": {"type": "uwp"}},
        )
        
        ws = get_world_state()
        first_action = ws.last_action
        assert first_action is not None
        
        # Now try an error result - should NOT update last_action
        update_from_tool_execution(
            tool_name="open_target",
            tool_args={"query": "nonexistent"},
            tool_result={"error": {"type": "not_found", "message": "Not found"}},
        )
        
        # last_action should still be the first one
        assert ws.last_action is first_action
        assert ws.last_action.args == {"query": "spotify"}


# ============================================================================
# REPLAY REQUEST DETECTION TESTS
# ============================================================================

class TestReplayRequestDetection:
    """Tests for is_replay_request and replay phrase detection."""
    
    @pytest.mark.parametrize("phrase", [
        "do that again",
        "do it again",
        "do this again",
        "repeat that",
        "repeat it",
        "repeat this",
        "repeat the last action",
        "repeat the last command",
        "again",
        "same thing",
        "same as last time",
        "same as before",
        "one more time",
        "repeat",
        # Prefixed variants
        "can you do that again",
        "can you do that again?",
        "could you do that again",
        "would you repeat that",
        "please do that again",
        "can you please repeat that",
        "could you please do it again?",
    ])
    def test_is_replay_request_true(self, phrase):
        """is_replay_request returns True for replay phrases."""
        assert is_replay_request(phrase) is True
    
    @pytest.mark.parametrize("phrase", [
        "open spotify",
        "close chrome",
        "tell me about that again",
        "what did you do again?",
        "say that again",
        "again and again",
        "can you open chrome",
        "can you tell me more",
    ])
    def test_is_replay_request_false(self, phrase):
        """is_replay_request returns False for non-replay phrases."""
        assert is_replay_request(phrase) is False


# ============================================================================
# REFERENCE RESOLVER REPLAY TESTS
# ============================================================================

class TestReferenceResolverReplay:
    """Tests for resolve_references with replay phrases."""
    
    def test_replay_returns_sentinel_when_last_action_exists(self, ws_with_open_target_uwp):
        """resolve_references returns sentinel when last_action is available."""
        result = resolve_references("do that again")
        assert result == REPLAY_LAST_ACTION_SENTINEL
    
    def test_replay_returns_sentinel_for_various_phrases(self, ws_with_open_target_uwp):
        """Various replay phrases return the sentinel."""
        phrases = ["do that again", "repeat that", "again", "same thing"]
        for phrase in phrases:
            result = resolve_references(phrase)
            assert result == REPLAY_LAST_ACTION_SENTINEL, f"Failed for: {phrase}"
    
    def test_is_replay_sentinel_true(self):
        """is_replay_sentinel returns True for the sentinel."""
        assert is_replay_sentinel(REPLAY_LAST_ACTION_SENTINEL) is True
    
    def test_is_replay_sentinel_false(self):
        """is_replay_sentinel returns False for other strings."""
        assert is_replay_sentinel("open spotify") is False
        assert is_replay_sentinel("do that again") is False
    
    def test_replay_without_last_action_returns_none(self):
        """resolve_references returns original text when no last_action."""
        # With no last_action or last_tool/target, should return original
        result = resolve_references("do that again")
        # Since there's no has_last_action either, should return original
        assert result == "do that again"


# ============================================================================
# REPLAY ARG BUILDING TESTS
# ============================================================================

class TestBuildReplayArgs:
    """Tests for _build_replay_args function."""
    
    def test_open_target_uwp_uses_resolved_path(self):
        """For UWP apps, _build_replay_args adds _resolved_uwp_path."""
        from wyzer.core.orchestrator import _build_replay_args
        
        original_args = {"query": "spotify"}
        resolved_info = {
            "type": "uwp",
            "path": "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
        }
        
        replay_args = _build_replay_args("open_target", original_args, resolved_info)
        
        assert replay_args["query"] == "spotify"
        assert replay_args["_resolved_uwp_path"] == "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"
    
    def test_open_target_game_uses_resolved_launch(self):
        """For games, _build_replay_args adds _resolved_launch."""
        from wyzer.core.orchestrator import _build_replay_args
        
        original_args = {"query": "elden ring"}
        resolved_info = {
            "type": "game",
            "launch": {"type": "steam_uri", "target": "steam://rungameid/1245620"},
        }
        
        replay_args = _build_replay_args("open_target", original_args, resolved_info)
        
        assert replay_args["_resolved_launch"]["type"] == "steam_uri"
        assert "1245620" in replay_args["_resolved_launch"]["target"]
    
    def test_close_window_uses_resolved_process(self):
        """For close_window, _build_replay_args uses resolved process."""
        from wyzer.core.orchestrator import _build_replay_args
        
        original_args = {"title": "some window"}
        resolved_info = {"process": "chrome.exe", "title": "Google - Chrome"}
        
        replay_args = _build_replay_args("close_window", original_args, resolved_info)
        
        assert replay_args["process"] == "chrome.exe"
    
    def test_no_resolved_info_uses_original_args(self):
        """When resolved_info is None, uses original args."""
        from wyzer.core.orchestrator import _build_replay_args
        
        original_args = {"query": "something"}
        replay_args = _build_replay_args("open_target", original_args, None)
        
        assert replay_args == {"query": "something"}


# ============================================================================
# INTEGRATION-ISH TESTS
# ============================================================================

class TestReplayIntegration:
    """Integration-style tests for the replay mechanism."""
    
    @patch('wyzer.core.orchestrator._execute_tool')
    @patch('wyzer.core.orchestrator.get_registry')
    def test_handle_replay_last_action_with_uwp(self, mock_registry, mock_execute):
        """Replay of UWP app executes with resolved UWP path."""
        from wyzer.core.orchestrator import _handle_replay_last_action
        
        # Set up world state with UWP last_action
        ws = get_world_state()
        ws.last_action = LastAction(
            tool="open_target",
            args={"query": "spotify"},
            resolved={
                "type": "uwp",
                "path": "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
                "matched_name": "Spotify",
            },
            ts=time.time(),
        )
        
        # Mock the execute_tool to return success
        mock_execute.return_value = {
            "status": "opened",
            "resolved": {"type": "uwp", "path": "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"},
        }
        mock_registry.return_value = MagicMock()
        
        # Call the handler
        result = _handle_replay_last_action("do that again", time.perf_counter())
        
        # Verify _execute_tool was called with UWP path
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args
        assert call_args[0][1] == "open_target"  # tool name
        assert "_resolved_uwp_path" in call_args[0][2]  # args contain resolved path
        
        # Verify result
        assert "replay" in result.get("meta", {}) or "replayed_tool" in result.get("execution_summary", {})
    
    def test_handle_replay_no_last_action_asks_user(self):
        """When no last_action, replay returns clarification."""
        from wyzer.core.orchestrator import _handle_replay_last_action
        
        # Ensure no last_action
        clear_world_state()
        
        result = _handle_replay_last_action("do that again", time.perf_counter())
        
        assert "What should I repeat?" in result["reply"]
        assert result["meta"]["no_last_action"] is True
    
    @patch('wyzer.core.orchestrator._execute_tool')
    @patch('wyzer.core.orchestrator.get_registry')
    def test_full_flow_tool_success_then_replay(self, mock_registry, mock_execute):
        """Full flow: tool success updates last_action, then replay works."""
        # Step 1: Simulate successful tool execution updating world state
        update_from_tool_execution(
            tool_name="open_target",
            tool_args={"query": "spotify"},
            tool_result={
                "status": "opened",
                "resolved": {
                    "type": "uwp",
                    "path": "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
                    "matched_name": "Spotify",
                },
            },
        )
        
        # Verify last_action was captured
        ws = get_world_state()
        assert ws.last_action is not None
        assert ws.last_action.tool == "open_target"
        
        # Step 2: Resolve "do that again" - should return sentinel
        resolved = resolve_references("do that again")
        assert resolved == REPLAY_LAST_ACTION_SENTINEL
        
        # Step 3: Handle the replay
        from wyzer.core.orchestrator import _handle_replay_last_action
        
        mock_execute.return_value = {"status": "opened", "resolved": {"type": "uwp"}}
        mock_registry.return_value = MagicMock()
        
        result = _handle_replay_last_action("do that again", time.perf_counter())
        
        # Verify correct tool was replayed
        mock_execute.assert_called_once()
        assert mock_execute.call_args[0][1] == "open_target"


# ============================================================================
# OPEN_TARGET DIRECT LAUNCH TESTS
# ============================================================================

class TestOpenTargetDirectLaunch:
    """Tests for open_target's direct launch methods (Phase 10.1)."""
    
    @patch('subprocess.Popen')
    def test_run_uwp_direct(self, mock_popen):
        """_run_uwp_direct launches UWP app without resolution."""
        from wyzer.tools.open_target import OpenTargetTool
        
        tool = OpenTargetTool()
        result = tool.run(
            query="spotify",
            _resolved_uwp_path="SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
        )
        
        # Should have called Popen with explorer and shell:AppsFolder path
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "explorer.exe" in call_args
        assert "shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify" in call_args[1]
        
        # Result should indicate success
        assert result.get("status") == "opened"
        assert result.get("replay") is True
    
    @patch('webbrowser.open')
    def test_run_game_direct_steam(self, mock_webbrowser):
        """_run_game_direct launches Steam game without resolution."""
        from wyzer.tools.open_target import OpenTargetTool
        
        tool = OpenTargetTool()
        result = tool.run(
            query="elden ring",
            _resolved_launch={"type": "steam_uri", "target": "steam://rungameid/1245620"},
        )
        
        # Should have called webbrowser.open with Steam URI
        mock_webbrowser.assert_called_once_with("steam://rungameid/1245620")
        
        # Result should indicate success
        assert result.get("status") == "opened"
        assert result.get("replay") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
