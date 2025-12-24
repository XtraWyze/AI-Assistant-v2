"""Tests for autonomy voice commands (deterministic, no LLM).

These tests verify that autonomy mode can be controlled by exact voice phrases
without involving the LLM.

Run with: python -m pytest scripts/test_autonomy_voice_commands.py -v
"""

import pytest
from unittest.mock import patch
import time

from wyzer.context.world_state import (
    clear_world_state,
    get_autonomy_mode,
    set_autonomy_mode,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def clean_world_state():
    """Ensure clean state before and after each test."""
    clear_world_state()
    set_autonomy_mode("off")
    yield
    clear_world_state()
    set_autonomy_mode("off")


# ============================================================================
# REGEX PATTERN TESTS
# ============================================================================

class TestAutonomyCommandPatterns:
    """Tests for autonomy command regex patterns matching exact phrases only."""
    
    def test_autonomy_off_pattern_exact(self):
        """'autonomy off' matches exactly."""
        from wyzer.core.orchestrator import _AUTONOMY_OFF_RE
        assert _AUTONOMY_OFF_RE.match("autonomy off") is not None
        assert _AUTONOMY_OFF_RE.match("Autonomy off") is not None
        assert _AUTONOMY_OFF_RE.match("AUTONOMY OFF") is not None
        assert _AUTONOMY_OFF_RE.match("autonomy off.") is not None
        
    def test_autonomy_off_pattern_variations(self):
        """Natural variations of 'autonomy off' match."""
        from wyzer.core.orchestrator import _AUTONOMY_OFF_RE
        assert _AUTONOMY_OFF_RE.match("set autonomy off") is not None
        assert _AUTONOMY_OFF_RE.match("set autonomy to off") is not None
        assert _AUTONOMY_OFF_RE.match("set my autonomy to off") is not None
        assert _AUTONOMY_OFF_RE.match("set your autonomy to off") is not None
        assert _AUTONOMY_OFF_RE.match("set autonomy level to off") is not None
        assert _AUTONOMY_OFF_RE.match("set your autonomy level to off") is not None
        assert _AUTONOMY_OFF_RE.match("autonomy mode off") is not None
        assert _AUTONOMY_OFF_RE.match("my autonomy off") is not None
        
    def test_autonomy_off_pattern_no_extra_words(self):
        """Patterns with unrecognized extra words should NOT match."""
        from wyzer.core.orchestrator import _AUTONOMY_OFF_RE
        assert _AUTONOMY_OFF_RE.match("turn autonomy off") is None
        assert _AUTONOMY_OFF_RE.match("please autonomy off") is None
        assert _AUTONOMY_OFF_RE.match("switch autonomy off") is None
        
    def test_autonomy_low_pattern_exact(self):
        """'autonomy low' matches exactly."""
        from wyzer.core.orchestrator import _AUTONOMY_LOW_RE
        assert _AUTONOMY_LOW_RE.match("autonomy low") is not None
        assert _AUTONOMY_LOW_RE.match("Autonomy Low") is not None
        assert _AUTONOMY_LOW_RE.match("AUTONOMY LOW") is not None
        assert _AUTONOMY_LOW_RE.match("autonomy low.") is not None
        
    def test_autonomy_low_pattern_variations(self):
        """Natural variations of 'autonomy low' match."""
        from wyzer.core.orchestrator import _AUTONOMY_LOW_RE
        assert _AUTONOMY_LOW_RE.match("set autonomy low") is not None
        assert _AUTONOMY_LOW_RE.match("set autonomy to low") is not None
        assert _AUTONOMY_LOW_RE.match("set my autonomy to low") is not None
        assert _AUTONOMY_LOW_RE.match("set your autonomy level to low") is not None
        
    def test_autonomy_low_pattern_no_extra_words(self):
        """Patterns with unrecognized extra words should NOT match."""
        from wyzer.core.orchestrator import _AUTONOMY_LOW_RE
        assert _AUTONOMY_LOW_RE.match("turn autonomy low") is None
        assert _AUTONOMY_LOW_RE.match("switch autonomy low") is None
        
    def test_autonomy_normal_pattern_exact(self):
        """'autonomy normal' matches exactly."""
        from wyzer.core.orchestrator import _AUTONOMY_NORMAL_RE
        assert _AUTONOMY_NORMAL_RE.match("autonomy normal") is not None
        assert _AUTONOMY_NORMAL_RE.match("Autonomy Normal") is not None
        assert _AUTONOMY_NORMAL_RE.match("AUTONOMY NORMAL") is not None
        assert _AUTONOMY_NORMAL_RE.match("autonomy normal.") is not None
        
    def test_autonomy_normal_pattern_variations(self):
        """Natural variations of 'autonomy normal' match."""
        from wyzer.core.orchestrator import _AUTONOMY_NORMAL_RE
        assert _AUTONOMY_NORMAL_RE.match("set autonomy normal") is not None
        assert _AUTONOMY_NORMAL_RE.match("set autonomy to normal") is not None
        assert _AUTONOMY_NORMAL_RE.match("set my autonomy to normal") is not None
        assert _AUTONOMY_NORMAL_RE.match("set your autonomy mode to normal") is not None
        
    def test_autonomy_normal_pattern_no_extra_words(self):
        """Patterns with unrecognized extra words should NOT match."""
        from wyzer.core.orchestrator import _AUTONOMY_NORMAL_RE
        assert _AUTONOMY_NORMAL_RE.match("turn autonomy normal") is None
        assert _AUTONOMY_NORMAL_RE.match("switch autonomy normal") is None
        
    def test_autonomy_high_pattern_exact(self):
        """'autonomy high' matches exactly."""
        from wyzer.core.orchestrator import _AUTONOMY_HIGH_RE
        assert _AUTONOMY_HIGH_RE.match("autonomy high") is not None
        assert _AUTONOMY_HIGH_RE.match("Autonomy High") is not None
        assert _AUTONOMY_HIGH_RE.match("AUTONOMY HIGH") is not None
        assert _AUTONOMY_HIGH_RE.match("autonomy high.") is not None
        
    def test_autonomy_high_pattern_variations(self):
        """Natural variations of 'autonomy high' match."""
        from wyzer.core.orchestrator import _AUTONOMY_HIGH_RE
        assert _AUTONOMY_HIGH_RE.match("set autonomy high") is not None
        assert _AUTONOMY_HIGH_RE.match("set autonomy to high") is not None
        assert _AUTONOMY_HIGH_RE.match("set my autonomy to high") is not None
        assert _AUTONOMY_HIGH_RE.match("set your autonomy level to high") is not None
        assert _AUTONOMY_HIGH_RE.match("set your autonomy setting to high") is not None
        
    def test_autonomy_high_pattern_no_extra_words(self):
        """Patterns with unrecognized extra words should NOT match."""
        from wyzer.core.orchestrator import _AUTONOMY_HIGH_RE
        assert _AUTONOMY_HIGH_RE.match("turn autonomy high") is None
        assert _AUTONOMY_HIGH_RE.match("switch autonomy high") is None
        
    def test_autonomy_status_pattern_whats(self):
        """\"what's my autonomy\" matches."""
        from wyzer.core.orchestrator import _AUTONOMY_STATUS_RE
        assert _AUTONOMY_STATUS_RE.match("what's my autonomy") is not None
        assert _AUTONOMY_STATUS_RE.match("What's my autonomy") is not None
        assert _AUTONOMY_STATUS_RE.match("WHAT'S MY AUTONOMY") is not None
        assert _AUTONOMY_STATUS_RE.match("what's my autonomy?") is not None
        
    def test_autonomy_status_pattern_what_is(self):
        """\"what is my autonomy\" matches."""
        from wyzer.core.orchestrator import _AUTONOMY_STATUS_RE
        assert _AUTONOMY_STATUS_RE.match("what is my autonomy") is not None
        assert _AUTONOMY_STATUS_RE.match("What is my autonomy") is not None
        assert _AUTONOMY_STATUS_RE.match("WHAT IS MY AUTONOMY") is not None
        assert _AUTONOMY_STATUS_RE.match("what is my autonomy?") is not None
        
    def test_autonomy_status_pattern_your_variants(self):
        """\"what's your autonomy\" variants match."""
        from wyzer.core.orchestrator import _AUTONOMY_STATUS_RE
        assert _AUTONOMY_STATUS_RE.match("what's your autonomy") is not None
        assert _AUTONOMY_STATUS_RE.match("what is your autonomy") is not None
        assert _AUTONOMY_STATUS_RE.match("what's your autonomy level") is not None
        assert _AUTONOMY_STATUS_RE.match("what is your autonomy mode") is not None
        
    def test_autonomy_status_pattern_level_mode_setting(self):
        """Patterns with level/mode/setting suffix match."""
        from wyzer.core.orchestrator import _AUTONOMY_STATUS_RE
        assert _AUTONOMY_STATUS_RE.match("what's my autonomy level") is not None
        assert _AUTONOMY_STATUS_RE.match("what is my autonomy mode") is not None
        assert _AUTONOMY_STATUS_RE.match("what's my autonomy setting") is not None
        
    def test_autonomy_status_pattern_set_at_to(self):
        """Patterns with 'set at/to' or just 'at/to' suffix match."""
        from wyzer.core.orchestrator import _AUTONOMY_STATUS_RE
        assert _AUTONOMY_STATUS_RE.match("what's my autonomy set at") is not None
        assert _AUTONOMY_STATUS_RE.match("what is my autonomy set to") is not None
        assert _AUTONOMY_STATUS_RE.match("what's your autonomy level set at") is not None
        assert _AUTONOMY_STATUS_RE.match("what is your autonomy mode set to") is not None
        # Also match without 'set'
        assert _AUTONOMY_STATUS_RE.match("what's your autonomy level at") is not None
        assert _AUTONOMY_STATUS_RE.match("what is my autonomy at") is not None
        
    def test_autonomy_status_pattern_no_match(self):
        """Unrelated patterns should NOT match."""
        from wyzer.core.orchestrator import _AUTONOMY_STATUS_RE
        assert _AUTONOMY_STATUS_RE.match("check my autonomy") is None
        assert _AUTONOMY_STATUS_RE.match("show autonomy") is None
        assert _AUTONOMY_STATUS_RE.match("autonomy status") is None
        assert _AUTONOMY_STATUS_RE.match("tell me my autonomy") is None


# ============================================================================
# COMMAND HANDLER TESTS
# ============================================================================

class TestAutonomyCommandHandler:
    """Tests for _check_autonomy_commands function."""
    
    def test_autonomy_off_sets_mode(self):
        """'autonomy off' sets mode to off."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        set_autonomy_mode("high")  # Start with different mode
        
        result = _check_autonomy_commands("autonomy off", time.perf_counter())
        
        assert result is not None
        assert get_autonomy_mode() == "off"
        assert result["reply"] == "Autonomy set to off."
        assert result["meta"]["autonomy_command"] == "off"
        assert result["meta"]["previous_mode"] == "high"
        
    def test_autonomy_low_sets_mode(self):
        """'autonomy low' sets mode to low."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        
        result = _check_autonomy_commands("autonomy low", time.perf_counter())
        
        assert result is not None
        assert get_autonomy_mode() == "low"
        assert result["reply"] == "Autonomy set to low."
        assert result["meta"]["autonomy_command"] == "low"
        
    def test_autonomy_normal_sets_mode(self):
        """'autonomy normal' sets mode to normal."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        
        result = _check_autonomy_commands("autonomy normal", time.perf_counter())
        
        assert result is not None
        assert get_autonomy_mode() == "normal"
        assert result["reply"] == "Autonomy set to normal."
        assert result["meta"]["autonomy_command"] == "normal"
        
    def test_autonomy_high_sets_mode(self):
        """'autonomy high' sets mode to high."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        
        result = _check_autonomy_commands("autonomy high", time.perf_counter())
        
        assert result is not None
        assert get_autonomy_mode() == "high"
        assert result["reply"] == "Autonomy set to high."
        assert result["meta"]["autonomy_command"] == "high"
        
    def test_whats_my_autonomy_returns_mode(self):
        """\"what's my autonomy\" returns current mode with 'Your'."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        set_autonomy_mode("normal")
        
        result = _check_autonomy_commands("what's my autonomy", time.perf_counter())
        
        assert result is not None
        assert result["reply"] == "Your autonomy mode is normal."
        assert result["meta"]["autonomy_command"] == "status"
        assert result["meta"]["current_mode"] == "normal"
        
    def test_what_is_my_autonomy_returns_mode(self):
        """\"what is my autonomy\" returns current mode with 'Your'."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        set_autonomy_mode("high")
        
        result = _check_autonomy_commands("what is my autonomy", time.perf_counter())
        
        assert result is not None
        assert result["reply"] == "Your autonomy mode is high."
        assert result["meta"]["autonomy_command"] == "status"
        assert result["meta"]["current_mode"] == "high"
        
    def test_whats_your_autonomy_returns_mode_with_my(self):
        """\"what's your autonomy\" returns current mode with 'My'."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        set_autonomy_mode("low")
        
        result = _check_autonomy_commands("what's your autonomy", time.perf_counter())
        
        assert result is not None
        assert result["reply"] == "My autonomy mode is low."
        assert result["meta"]["autonomy_command"] == "status"
        assert result["meta"]["current_mode"] == "low"
        
    def test_case_insensitive(self):
        """Commands are case-insensitive."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        
        result = _check_autonomy_commands("AUTONOMY HIGH", time.perf_counter())
        
        assert result is not None
        assert get_autonomy_mode() == "high"
        
    def test_unrelated_text_returns_none(self):
        """Unrelated text returns None (passes through to LLM)."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        
        result = _check_autonomy_commands("open chrome", time.perf_counter())
        
        assert result is None
        
    def test_similar_but_not_exact_returns_none(self):
        """Phrases with unrecognized verbs should return None (pass to LLM)."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        
        # These should NOT match (unrecognized verbs/patterns)
        assert _check_autonomy_commands("turn autonomy to high", time.perf_counter()) is None
        assert _check_autonomy_commands("switch autonomy off", time.perf_counter()) is None
        assert _check_autonomy_commands("change autonomy to low", time.perf_counter()) is None
        assert _check_autonomy_commands("check my autonomy", time.perf_counter()) is None
        assert _check_autonomy_commands("tell me my autonomy", time.perf_counter()) is None


# ============================================================================
# INTEGRATION TESTS (via handle_user_text)
# ============================================================================

class TestAutonomyVoiceIntegration:
    """Integration tests for autonomy voice commands via handle_user_text."""
    
    def test_autonomy_command_bypasses_llm(self):
        """Autonomy commands should NOT call the LLM."""
        from wyzer.core.orchestrator import handle_user_text
        
        with patch("wyzer.core.orchestrator._ollama_request") as mock_llm:
            result = handle_user_text("autonomy high")
            
            # LLM should not be called
            mock_llm.assert_not_called()
            
            # Should get autonomy response
            assert result["reply"] == "Autonomy set to high."
            assert get_autonomy_mode() == "high"
            
    def test_autonomy_command_works_regardless_of_confirmation(self):
        """Autonomy commands bypass pending confirmations."""
        from wyzer.core.orchestrator import handle_user_text
        from wyzer.context.world_state import set_pending_confirmation
        
        # Set up a pending confirmation
        set_pending_confirmation(
            plan=[{"tool": "delete_file", "args": {"path": "/tmp/test"}}],
            prompt="Are you sure?",
        )
        
        with patch("wyzer.core.orchestrator._ollama_request") as mock_llm:
            result = handle_user_text("autonomy off")
            
            mock_llm.assert_not_called()
            assert result["reply"] == "Autonomy set to off."
            assert get_autonomy_mode() == "off"
            
    def test_status_query_does_not_change_mode(self):
        """\"what's my autonomy\" should only query, not change mode."""
        from wyzer.core.orchestrator import handle_user_text
        set_autonomy_mode("low")
        
        with patch("wyzer.core.orchestrator._ollama_request") as mock_llm:
            result = handle_user_text("what's my autonomy")
            
            mock_llm.assert_not_called()
            assert get_autonomy_mode() == "low"  # Unchanged
            assert result["reply"] == "Your autonomy mode is low."


# ============================================================================
# LOG OUTPUT TESTS
# ============================================================================

class TestAutonomyLogging:
    """Tests for log output format."""
    
    def test_log_format_on_mode_change(self, capsys):
        """Log should match: [AUTONOMY] mode changed to <mode> (source=voice)"""
        from wyzer.core.orchestrator import _check_autonomy_commands
        
        _check_autonomy_commands("autonomy high", time.perf_counter())
        
        captured = capsys.readouterr()
        # Normalize whitespace for comparison (log might wrap)
        normalized = " ".join(captured.out.split())
        assert "[AUTONOMY] mode changed to high (source=voice)" in normalized
        
    def test_log_format_for_each_mode(self, capsys):
        """All modes should have consistent log format."""
        from wyzer.core.orchestrator import _check_autonomy_commands
        
        modes = ["off", "low", "normal", "high"]
        
        for mode in modes:
            _check_autonomy_commands(f"autonomy {mode}", time.perf_counter())
            
            captured = capsys.readouterr()
            # Normalize whitespace for comparison (log might wrap)
            normalized = " ".join(captured.out.split())
            expected_log = f"[AUTONOMY] mode changed to {mode} (source=voice)"
            assert expected_log in normalized, f"Expected log for mode '{mode}'"
