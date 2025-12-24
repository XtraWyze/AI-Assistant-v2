"""Tests for Phase 11 - Risk Classification.

Tests the risk classification module that categorizes tools
as low/medium/high risk.

Run with: python -m pytest scripts/test_risk_classification.py -v
"""

import pytest
from unittest.mock import MagicMock
from wyzer.policy.risk import (
    classify_tool,
    classify_plan,
    get_risk_description,
    HIGH_RISK_TOOLS,
    MEDIUM_RISK_TOOLS,
    LOW_RISK_TOOLS,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_world_state_with_open_target():
    """Mock WorldState with last_action=open_target."""
    ws = MagicMock()
    ws.last_action = MagicMock()
    ws.last_action.tool = "open_target"
    ws.last_action.args = {"query": "chrome"}
    return ws


@pytest.fixture
def mock_world_state_with_delete():
    """Mock WorldState with last_action=delete_file."""
    ws = MagicMock()
    ws.last_action = MagicMock()
    ws.last_action.tool = "delete_file"
    ws.last_action.args = {"path": "/tmp/test.txt"}
    return ws


@pytest.fixture
def mock_world_state_no_action():
    """Mock WorldState with no last_action."""
    ws = MagicMock()
    ws.last_action = None
    return ws


# ============================================================================
# TOOL CLASSIFICATION TESTS
# ============================================================================

class TestToolClassification:
    """Tests for individual tool classification."""
    
    # LOW RISK - Read-only tools
    def test_get_window_context_is_low(self):
        """get_window_context should be LOW risk (read-only)."""
        assert classify_tool("get_window_context") == "low"
        
    def test_get_time_is_low(self):
        """get_time should be LOW risk."""
        assert classify_tool("get_time") == "low"
        
    def test_get_system_info_is_low(self):
        """get_system_info should be LOW risk."""
        assert classify_tool("get_system_info") == "low"
        
    def test_get_location_is_low(self):
        """get_location should be LOW risk."""
        assert classify_tool("get_location") == "low"
        
    def test_get_weather_forecast_is_low(self):
        """get_weather_forecast should be LOW risk."""
        assert classify_tool("get_weather_forecast") == "low"
        
    def test_get_now_playing_is_low(self):
        """get_now_playing should be LOW risk."""
        assert classify_tool("get_now_playing") == "low"
        
    def test_monitor_info_is_low(self):
        """monitor_info should be LOW risk."""
        assert classify_tool("monitor_info") == "low"
        
    def test_system_storage_list_is_low(self):
        """system_storage_list should be LOW risk."""
        assert classify_tool("system_storage_list") == "low"
    
    # MEDIUM RISK - Benign mutations
    def test_open_target_is_medium(self):
        """open_target should be MEDIUM risk."""
        assert classify_tool("open_target") == "medium"
        
    def test_close_window_is_medium(self):
        """close_window should be MEDIUM risk (NOT high)."""
        assert classify_tool("close_window") == "medium"
        
    def test_minimize_window_is_medium(self):
        """minimize_window should be MEDIUM risk."""
        assert classify_tool("minimize_window") == "medium"
        
    def test_maximize_window_is_medium(self):
        """maximize_window should be MEDIUM risk."""
        assert classify_tool("maximize_window") == "medium"
        
    def test_focus_window_is_medium(self):
        """focus_window should be MEDIUM risk."""
        assert classify_tool("focus_window") == "medium"
        
    def test_volume_control_is_medium(self):
        """volume_control should be MEDIUM risk."""
        assert classify_tool("volume_control") == "medium"
        
    def test_media_play_pause_is_medium(self):
        """media_play_pause should be MEDIUM risk."""
        assert classify_tool("media_play_pause") == "medium"
        
    def test_timer_is_medium(self):
        """timer should be MEDIUM risk."""
        assert classify_tool("timer") == "medium"
        
    def test_open_website_is_medium(self):
        """open_website should be MEDIUM risk."""
        assert classify_tool("open_website") == "medium"
    
    # HIGH RISK - Destructive actions
    def test_delete_file_is_high(self):
        """delete_file should be HIGH risk."""
        assert classify_tool("delete_file") == "high"
        
    def test_delete_files_is_high(self):
        """delete_files should be HIGH risk."""
        assert classify_tool("delete_files") == "high"
        
    def test_shutdown_is_high(self):
        """shutdown should be HIGH risk."""
        assert classify_tool("shutdown") == "high"
        
    def test_restart_is_high(self):
        """restart should be HIGH risk."""
        assert classify_tool("restart") == "high"
        
    def test_kill_process_is_high(self):
        """kill_process should be HIGH risk."""
        assert classify_tool("kill_process") == "high"
        
    def test_format_drive_is_high(self):
        """format_drive should be HIGH risk."""
        assert classify_tool("format_drive") == "high"
        
    def test_empty_recycle_bin_is_high(self):
        """empty_recycle_bin should be HIGH risk."""
        assert classify_tool("empty_recycle_bin") == "high"


class TestToolClassificationPatterns:
    """Tests for pattern-based classification."""
    
    def test_tool_with_delete_pattern_is_high(self):
        """Tool with 'delete' in name should be HIGH risk."""
        assert classify_tool("some_delete_tool") == "high"
        assert classify_tool("delete_something") == "high"
        
    def test_tool_with_kill_pattern_is_high(self):
        """Tool with 'kill' in name should be HIGH risk."""
        assert classify_tool("kill_all_processes") == "high"
        
    def test_tool_with_terminate_pattern_is_high(self):
        """Tool with 'terminate' in name should be HIGH risk."""
        assert classify_tool("terminate_session") == "high"
        
    def test_unknown_tool_defaults_to_low(self):
        """Unknown tools should default to LOW risk."""
        assert classify_tool("some_random_unknown_tool") == "low"
        
    def test_empty_tool_name_is_low(self):
        """Empty tool name should be LOW risk."""
        assert classify_tool("") == "low"


class TestToolClassificationWithArgs:
    """Tests for classification with arguments."""
    
    def test_args_with_delete_value_is_high(self):
        """Args containing 'delete' value should be HIGH risk."""
        result = classify_tool("generic_tool", {"action": "delete_all"})
        assert result == "high"
        
    def test_args_with_kill_value_is_high(self):
        """Args containing 'kill' value should be HIGH risk."""
        result = classify_tool("generic_tool", {"operation": "kill_process"})
        assert result == "high"
        
    def test_safe_args_keep_original_classification(self):
        """Safe args should not change classification."""
        result = classify_tool("open_target", {"query": "chrome"})
        assert result == "medium"


# ============================================================================
# PLAN CLASSIFICATION TESTS
# ============================================================================

class TestPlanClassification:
    """Tests for plan-level classification."""
    
    def test_empty_plan_is_low(self):
        """Empty plan should be LOW risk."""
        assert classify_plan([]) == "low"
        
    def test_single_low_tool_is_low(self):
        """Plan with single low-risk tool should be LOW."""
        plan = [{"tool": "get_time", "args": {}}]
        assert classify_plan(plan) == "low"
        
    def test_single_medium_tool_is_medium(self):
        """Plan with single medium-risk tool should be MEDIUM."""
        plan = [{"tool": "open_target", "args": {"query": "chrome"}}]
        assert classify_plan(plan) == "medium"
        
    def test_single_high_tool_is_high(self):
        """Plan with single high-risk tool should be HIGH."""
        plan = [{"tool": "delete_file", "args": {"path": "/tmp/test"}}]
        assert classify_plan(plan) == "high"
        
    def test_mixed_plan_takes_maximum(self):
        """Plan with mixed risks should take maximum."""
        plan = [
            {"tool": "get_time", "args": {}},          # low
            {"tool": "open_target", "args": {}},       # medium
        ]
        assert classify_plan(plan) == "medium"
        
    def test_high_risk_in_multi_plan_bubbles_up(self):
        """High risk in multi-tool plan should bubble up."""
        plan = [
            {"tool": "get_time", "args": {}},          # low
            {"tool": "open_target", "args": {}},       # medium
            {"tool": "delete_file", "args": {}},       # high
        ]
        assert classify_plan(plan) == "high"


# ============================================================================
# REPLAY RISK INHERITANCE TESTS
# ============================================================================

class TestReplayRiskInheritance:
    """Tests for replay_last_action risk inheritance."""
    
    def test_replay_inherits_medium_from_open_target(self, mock_world_state_with_open_target):
        """Replay inherits MEDIUM from open_target last_action."""
        plan = [{"tool": "replay_last_action", "args": {}}]
        result = classify_plan(plan, mock_world_state_with_open_target)
        assert result == "medium"
        
    def test_replay_inherits_high_from_delete(self, mock_world_state_with_delete):
        """Replay inherits HIGH from delete last_action."""
        plan = [{"tool": "replay_last_action", "args": {}}]
        result = classify_plan(plan, mock_world_state_with_delete)
        assert result == "high"
        
    def test_replay_sentinel_inherits_risk(self, mock_world_state_with_open_target):
        """__REPLAY_LAST_ACTION__ sentinel inherits risk."""
        plan = [{"tool": "__REPLAY_LAST_ACTION__", "args": {}}]
        result = classify_plan(plan, mock_world_state_with_open_target)
        assert result == "medium"
        
    def test_replay_with_no_last_action_is_medium(self, mock_world_state_no_action):
        """Replay with no last_action defaults to MEDIUM (safe default)."""
        plan = [{"tool": "replay_last_action", "args": {}}]
        result = classify_plan(plan, mock_world_state_no_action)
        assert result == "medium"
        
    def test_replay_with_no_world_state_is_medium(self):
        """Replay with no world_state defaults to MEDIUM (safe default)."""
        plan = [{"tool": "replay_last_action", "args": {}}]
        result = classify_plan(plan, None)
        assert result == "medium"


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestRiskDescriptions:
    """Tests for risk description utility."""
    
    def test_low_description(self):
        """LOW risk description."""
        desc = get_risk_description("low")
        assert "read" in desc.lower() or "informational" in desc.lower()
        
    def test_medium_description(self):
        """MEDIUM risk description."""
        desc = get_risk_description("medium")
        assert "modif" in desc.lower() or "reversible" in desc.lower()
        
    def test_high_description(self):
        """HIGH risk description."""
        desc = get_risk_description("high")
        assert "destruct" in desc.lower() or "dangerous" in desc.lower()
        
    def test_unknown_description(self):
        """Unknown risk description."""
        desc = get_risk_description("unknown")
        assert "unknown" in desc.lower()


# ============================================================================
# HIGH RISK LIST TESTS (Keep narrow)
# ============================================================================

class TestHighRiskListNarrow:
    """Tests to ensure HIGH risk list stays narrow."""
    
    def test_close_window_is_not_high(self):
        """close_window should NOT be high risk."""
        assert classify_tool("close_window") != "high"
        
    def test_volume_is_not_high(self):
        """volume tools should NOT be high risk."""
        assert classify_tool("volume_control") != "high"
        assert classify_tool("volume_up") != "high"
        assert classify_tool("volume_down") != "high"
        
    def test_media_controls_not_high(self):
        """Media controls should NOT be high risk."""
        assert classify_tool("media_play_pause") != "high"
        assert classify_tool("media_next") != "high"
        assert classify_tool("media_previous") != "high"
        
    def test_window_management_not_high(self):
        """Window management tools should NOT be high risk."""
        assert classify_tool("minimize_window") != "high"
        assert classify_tool("maximize_window") != "high"
        assert classify_tool("focus_window") != "high"
        assert classify_tool("move_window_to_monitor") != "high"
