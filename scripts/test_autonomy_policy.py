"""Tests for Phase 11 - Autonomy Policy.

Tests the autonomy policy assessment module that decides
whether to execute, ask, or deny based on confidence and risk.

Run with: python -m pytest scripts/test_autonomy_policy.py -v
"""

import pytest
from wyzer.policy.autonomy_policy import (
    assess,
    AutonomyDecision,
    AUTONOMY_MODES,
    LOW_THRESHOLD_EXECUTE,
    NORMAL_THRESHOLD_EXECUTE,
    NORMAL_THRESHOLD_ASK,
    HIGH_THRESHOLD_EXECUTE,
    HIGH_THRESHOLD_ASK,
    HIGH_MODE_SENSITIVE_THRESHOLD,
    format_decision_for_speech,
    summarize_plan,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_tool_plan():
    """A simple single-tool plan."""
    return [{"tool": "open_target", "args": {"query": "chrome"}}]


@pytest.fixture
def multi_tool_plan():
    """A multi-tool plan."""
    return [
        {"tool": "open_target", "args": {"query": "chrome"}},
        {"tool": "volume_control", "args": {"level": 50}},
    ]


# ============================================================================
# MODE OFF TESTS
# ============================================================================

class TestAutonomyModeOff:
    """Tests for autonomy mode=off (current behavior preserved)."""
    
    def test_off_mode_always_executes(self, simple_tool_plan):
        """OFF mode should always return execute with no confirmation."""
        decision = assess(simple_tool_plan, 0.5, "off", "low")
        assert decision["action"] == "execute"
        assert decision["needs_confirmation"] is False
        
    def test_off_mode_executes_even_low_confidence(self, simple_tool_plan):
        """OFF mode executes even with very low confidence."""
        decision = assess(simple_tool_plan, 0.1, "off", "medium")
        assert decision["action"] == "execute"
        assert decision["needs_confirmation"] is False
        
    def test_off_mode_executes_high_risk(self, simple_tool_plan):
        """OFF mode executes even high risk without confirmation."""
        decision = assess(simple_tool_plan, 0.5, "off", "high")
        assert decision["action"] == "execute"
        assert decision["needs_confirmation"] is False
        
    def test_off_mode_preserves_reason(self, simple_tool_plan):
        """OFF mode includes appropriate reason."""
        decision = assess(simple_tool_plan, 0.5, "off", "low")
        assert "off" in decision["reason"].lower() or "preserved" in decision["reason"].lower()


# ============================================================================
# MODE LOW TESTS
# ============================================================================

class TestAutonomyModeLow:
    """Tests for autonomy mode=low (very conservative)."""
    
    def test_low_mode_executes_high_confidence_low_risk(self, simple_tool_plan):
        """LOW mode executes when confidence >= 0.95 and risk is low."""
        decision = assess(simple_tool_plan, 0.96, "low", "low")
        assert decision["action"] == "execute"
        assert decision["needs_confirmation"] is False
        
    def test_low_mode_executes_high_confidence_medium_risk(self, simple_tool_plan):
        """LOW mode executes when confidence >= 0.95 and risk is medium."""
        decision = assess(simple_tool_plan, 0.95, "low", "medium")
        assert decision["action"] == "execute"
        assert decision["needs_confirmation"] is False
        
    def test_low_mode_asks_below_threshold_low_risk(self, simple_tool_plan):
        """LOW mode asks when confidence < 0.95 (low risk)."""
        decision = assess(simple_tool_plan, 0.94, "low", "low")
        assert decision["action"] == "ask"
        assert decision["needs_confirmation"] is False  # Clarification, not confirmation
        
    def test_low_mode_asks_below_threshold_medium_risk(self, simple_tool_plan):
        """LOW mode asks when confidence < 0.95 (medium risk)."""
        decision = assess(simple_tool_plan, 0.80, "low", "medium")
        assert decision["action"] == "ask"
        assert decision["needs_confirmation"] is False
        
    def test_low_mode_always_asks_high_risk(self, simple_tool_plan):
        """LOW mode always asks with confirmation for high risk."""
        decision = assess(simple_tool_plan, 0.99, "low", "high")
        assert decision["action"] == "ask"
        assert decision["needs_confirmation"] is True
        
    def test_low_mode_high_risk_needs_confirmation_regardless(self, simple_tool_plan):
        """LOW mode requires confirmation for high risk even at 100% confidence."""
        decision = assess(simple_tool_plan, 1.0, "low", "high")
        assert decision["action"] == "ask"
        assert decision["needs_confirmation"] is True


# ============================================================================
# MODE NORMAL TESTS
# ============================================================================

class TestAutonomyModeNormal:
    """Tests for autonomy mode=normal (balanced)."""
    
    def test_normal_mode_executes_high_confidence_low_risk(self, simple_tool_plan):
        """NORMAL mode executes when confidence >= 0.90 (low risk)."""
        decision = assess(simple_tool_plan, 0.90, "normal", "low")
        assert decision["action"] == "execute"
        assert decision["needs_confirmation"] is False
        
    def test_normal_mode_executes_high_confidence_medium_risk(self, simple_tool_plan):
        """NORMAL mode executes when confidence >= 0.90 (medium risk)."""
        decision = assess(simple_tool_plan, 0.92, "normal", "medium")
        assert decision["action"] == "execute"
        assert decision["needs_confirmation"] is False
        
    def test_normal_mode_asks_in_middle_range(self, simple_tool_plan):
        """NORMAL mode asks when 0.75 <= confidence < 0.90."""
        decision = assess(simple_tool_plan, 0.80, "normal", "low")
        assert decision["action"] == "ask"
        assert decision["needs_confirmation"] is False
        
    def test_normal_mode_asks_at_lower_boundary(self, simple_tool_plan):
        """NORMAL mode asks at lower boundary (0.75)."""
        decision = assess(simple_tool_plan, 0.75, "normal", "medium")
        assert decision["action"] == "ask"
        assert decision["needs_confirmation"] is False
        
    def test_normal_mode_asks_below_threshold(self, simple_tool_plan):
        """NORMAL mode asks when confidence < 0.75 (prefer ask over deny)."""
        decision = assess(simple_tool_plan, 0.50, "normal", "low")
        assert decision["action"] == "ask"
        
    def test_normal_mode_always_asks_high_risk(self, simple_tool_plan):
        """NORMAL mode always asks with confirmation for high risk."""
        decision = assess(simple_tool_plan, 0.99, "normal", "high")
        assert decision["action"] == "ask"
        assert decision["needs_confirmation"] is True


# ============================================================================
# MODE HIGH TESTS
# ============================================================================

class TestAutonomyModeHigh:
    """Tests for autonomy mode=high (more permissive)."""
    
    def test_high_mode_executes_at_threshold_low_risk(self, simple_tool_plan):
        """HIGH mode executes when confidence >= 0.85 (low risk)."""
        decision = assess(simple_tool_plan, 0.85, "high", "low")
        assert decision["action"] == "execute"
        assert decision["needs_confirmation"] is False
        
    def test_high_mode_executes_above_threshold_medium_risk(self, simple_tool_plan):
        """HIGH mode executes when confidence >= 0.85 (medium risk)."""
        decision = assess(simple_tool_plan, 0.90, "high", "medium")
        assert decision["action"] == "execute"
        assert decision["needs_confirmation"] is False
        
    def test_high_mode_asks_in_middle_range(self, simple_tool_plan):
        """HIGH mode asks when 0.70 <= confidence < 0.85."""
        decision = assess(simple_tool_plan, 0.75, "high", "low")
        assert decision["action"] == "ask"
        assert decision["needs_confirmation"] is False
        
    def test_high_mode_asks_below_threshold(self, simple_tool_plan):
        """HIGH mode asks when confidence < 0.70."""
        decision = assess(simple_tool_plan, 0.60, "high", "medium")
        assert decision["action"] == "ask"
        
    def test_high_mode_high_risk_asks_with_confirm_sensitive_true(self, simple_tool_plan):
        """HIGH mode asks with confirmation for high risk when confirm_sensitive=True."""
        decision = assess(simple_tool_plan, 0.99, "high", "high", confirm_sensitive=True)
        assert decision["action"] == "ask"
        assert decision["needs_confirmation"] is True
        
    def test_high_mode_high_risk_executes_very_high_confidence_confirm_sensitive_false(self, simple_tool_plan):
        """HIGH mode executes high risk when confidence >= 0.97 and confirm_sensitive=False."""
        decision = assess(simple_tool_plan, 0.97, "high", "high", confirm_sensitive=False)
        assert decision["action"] == "execute"
        assert decision["needs_confirmation"] is False
        
    def test_high_mode_high_risk_asks_below_097_confirm_sensitive_false(self, simple_tool_plan):
        """HIGH mode asks for high risk when confidence < 0.97 and confirm_sensitive=False."""
        decision = assess(simple_tool_plan, 0.96, "high", "high", confirm_sensitive=False)
        assert decision["action"] == "ask"
        assert decision["needs_confirmation"] is True


# ============================================================================
# EDGE CASES
# ============================================================================

class TestAutonomyEdgeCases:
    """Edge case and boundary tests."""
    
    def test_empty_plan(self):
        """Empty plan should still work."""
        decision = assess([], 0.5, "normal", "low")
        assert decision["action"] in ("execute", "ask", "deny")
        
    def test_invalid_mode_defaults_to_off(self, simple_tool_plan):
        """Invalid mode should default to off behavior."""
        decision = assess(simple_tool_plan, 0.5, "invalid_mode", "low")
        assert decision["action"] == "execute"
        
    def test_invalid_risk_defaults_to_low(self, simple_tool_plan):
        """Invalid risk should default to low."""
        decision = assess(simple_tool_plan, 0.5, "normal", "invalid_risk")
        # Should treat as low risk
        assert decision["risk"] == "low"
        
    def test_confidence_clamped_to_0_1(self, simple_tool_plan):
        """Confidence should be clamped to [0, 1]."""
        decision_high = assess(simple_tool_plan, 1.5, "normal", "low")
        assert decision_high["confidence"] == 1.0
        
        decision_low = assess(simple_tool_plan, -0.5, "normal", "low")
        assert decision_low["confidence"] == 0.0
        
    def test_question_generated_for_ask(self, simple_tool_plan):
        """Ask action should generate a question."""
        decision = assess(simple_tool_plan, 0.80, "normal", "low")
        assert decision["action"] == "ask"
        assert decision["question"] is not None
        assert len(decision["question"]) > 0


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_format_decision_execute(self):
        """Format execute decision for speech."""
        decision = AutonomyDecision(
            action="execute",
            reason="High confidence",
            question=None,
            confidence=0.95,
            risk="low",
            needs_confirmation=False,
        )
        speech = format_decision_for_speech(decision)
        assert "95%" in speech
        assert "low" in speech.lower()
        
    def test_format_decision_ask(self):
        """Format ask decision for speech."""
        decision = AutonomyDecision(
            action="ask",
            reason="Low confidence",
            question="What do you want?",
            confidence=0.50,
            risk="medium",
            needs_confirmation=False,
        )
        speech = format_decision_for_speech(decision)
        assert "50%" in speech
        
    def test_summarize_plan_single(self):
        """Summarize single-tool plan."""
        plan = [{"tool": "open_target", "args": {}}]
        summary = summarize_plan(plan)
        assert summary == "open_target"
        
    def test_summarize_plan_multiple(self):
        """Summarize multi-tool plan."""
        plan = [
            {"tool": "open_target", "args": {}},
            {"tool": "volume_control", "args": {}},
        ]
        summary = summarize_plan(plan)
        assert "open_target" in summary
        assert "volume_control" in summary
        
    def test_summarize_plan_empty(self):
        """Summarize empty plan."""
        summary = summarize_plan([])
        assert "empty" in summary.lower() or "no" in summary.lower()
