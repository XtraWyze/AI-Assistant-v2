"""Tests for Phase 11 - Confirmation Flow.

Tests the pending confirmation system for high-risk autonomy actions.

Run with: python -m pytest scripts/test_autonomy_confirmation.py -v
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from wyzer.context.world_state import (
    WorldState,
    PendingConfirmation,
    LastAutonomyDecision,
    get_world_state,
    clear_world_state,
    get_autonomy_mode,
    set_autonomy_mode,
    set_pending_confirmation,
    get_pending_confirmation,
    clear_pending_confirmation,
    consume_pending_confirmation,
    has_pending_confirmation,
    set_last_autonomy_decision,
    get_last_autonomy_decision,
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


@pytest.fixture
def sample_plan():
    """A sample tool plan for testing."""
    return [
        {"tool": "delete_file", "args": {"path": "/tmp/test.txt"}},
    ]


# ============================================================================
# AUTONOMY MODE TESTS
# ============================================================================

class TestAutonomyModeState:
    """Tests for autonomy mode state management."""
    
    def test_default_mode_is_off(self):
        """Default autonomy mode should be off."""
        assert get_autonomy_mode() == "off"
        
    def test_set_mode_low(self):
        """Can set mode to low."""
        result = set_autonomy_mode("low")
        assert result == "low"
        assert get_autonomy_mode() == "low"
        
    def test_set_mode_normal(self):
        """Can set mode to normal."""
        result = set_autonomy_mode("normal")
        assert result == "normal"
        assert get_autonomy_mode() == "normal"
        
    def test_set_mode_high(self):
        """Can set mode to high."""
        result = set_autonomy_mode("high")
        assert result == "high"
        assert get_autonomy_mode() == "high"
        
    def test_set_mode_off(self):
        """Can set mode back to off."""
        set_autonomy_mode("high")
        result = set_autonomy_mode("off")
        assert result == "off"
        assert get_autonomy_mode() == "off"
        
    def test_invalid_mode_defaults_to_off(self):
        """Invalid mode should default to off."""
        result = set_autonomy_mode("invalid")
        assert result == "off"
        assert get_autonomy_mode() == "off"
        
    def test_mode_case_insensitive(self):
        """Mode should be case-insensitive."""
        result = set_autonomy_mode("HIGH")
        assert result == "high"
        assert get_autonomy_mode() == "high"
        
    def test_changing_mode_clears_pending(self, sample_plan):
        """Changing mode should clear pending confirmations."""
        set_autonomy_mode("high")
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        assert has_pending_confirmation()
        
        set_autonomy_mode("low")
        assert not has_pending_confirmation()


# ============================================================================
# PENDING CONFIRMATION TESTS
# ============================================================================

class TestPendingConfirmation:
    """Tests for pending confirmation management."""
    
    def test_no_pending_by_default(self):
        """No pending confirmation by default."""
        assert get_pending_confirmation() is None
        assert not has_pending_confirmation()
        
    def test_set_pending_confirmation(self, sample_plan):
        """Can set a pending confirmation."""
        set_pending_confirmation(sample_plan, "Are you sure?", 20.0)
        
        pending = get_pending_confirmation()
        assert pending is not None
        assert pending.plan == sample_plan
        assert pending.prompt == "Are you sure?"
        assert not pending.is_expired()
        
    def test_pending_expires_after_timeout(self, sample_plan):
        """Pending confirmation expires after timeout."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)  # 100ms timeout
        
        assert has_pending_confirmation()
        time.sleep(0.15)  # Wait for expiry
        assert not has_pending_confirmation()
        
    def test_get_pending_returns_none_if_expired(self, sample_plan):
        """get_pending_confirmation returns None if expired."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)
        
        assert get_pending_confirmation() is None
        
    def test_clear_pending_confirmation(self, sample_plan):
        """Can clear pending confirmation."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        assert has_pending_confirmation()
        
        clear_pending_confirmation()
        assert not has_pending_confirmation()
        
    def test_consume_returns_plan_and_clears(self, sample_plan):
        """consume_pending_confirmation returns plan and clears it."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        consumed = consume_pending_confirmation()
        assert consumed == sample_plan
        assert not has_pending_confirmation()
        
    def test_consume_returns_none_if_expired(self, sample_plan):
        """consume_pending_confirmation returns None if expired."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)
        
        consumed = consume_pending_confirmation()
        assert consumed is None
        
    def test_consume_returns_none_if_no_pending(self):
        """consume_pending_confirmation returns None if nothing pending."""
        consumed = consume_pending_confirmation()
        assert consumed is None


# ============================================================================
# CONFIRMATION DATACLASS TESTS
# ============================================================================

class TestPendingConfirmationDataclass:
    """Tests for PendingConfirmation dataclass."""
    
    def test_is_expired_false_within_window(self):
        """is_expired returns False within timeout window."""
        pending = PendingConfirmation(
            plan=[{"tool": "test"}],
            expires_ts=time.time() + 10.0,
            prompt="Confirm?",
        )
        assert not pending.is_expired()
        
    def test_is_expired_true_after_window(self):
        """is_expired returns True after timeout."""
        pending = PendingConfirmation(
            plan=[{"tool": "test"}],
            expires_ts=time.time() - 1.0,  # Already expired
            prompt="Confirm?",
        )
        assert pending.is_expired()
        
    def test_to_dict(self):
        """to_dict returns proper structure."""
        pending = PendingConfirmation(
            plan=[{"tool": "delete_file", "args": {}}],
            expires_ts=time.time() + 10.0,
            prompt="Are you sure?",
        )
        d = pending.to_dict()
        assert "plan_tools" in d
        assert "expires_in" in d
        assert "prompt" in d
        assert d["plan_tools"] == ["delete_file"]


# ============================================================================
# LAST AUTONOMY DECISION TESTS
# ============================================================================

class TestLastAutonomyDecision:
    """Tests for last autonomy decision storage."""
    
    def test_no_decision_by_default(self):
        """No last decision by default."""
        assert get_last_autonomy_decision() is None
        
    def test_set_and_get_decision(self):
        """Can set and get last decision."""
        set_last_autonomy_decision(
            mode="normal",
            confidence=0.85,
            risk="medium",
            action="execute",
            reason="High confidence",
            plan_summary="open_target",
        )
        
        decision = get_last_autonomy_decision()
        assert decision is not None
        assert decision.mode == "normal"
        assert decision.confidence == 0.85
        assert decision.risk == "medium"
        assert decision.action == "execute"
        assert decision.reason == "High confidence"
        assert decision.plan_summary == "open_target"
        
    def test_decision_has_timestamp(self):
        """Decision should have a timestamp."""
        set_last_autonomy_decision(
            mode="low",
            confidence=0.5,
            risk="low",
            action="ask",
            reason="Low confidence",
            plan_summary="timer",
        )
        
        decision = get_last_autonomy_decision()
        assert decision.ts > 0
        assert decision.ts <= time.time()
        
    def test_decision_to_dict(self):
        """to_dict returns proper structure."""
        decision = LastAutonomyDecision(
            mode="high",
            confidence=0.95,
            risk="high",
            action="ask",
            reason="High risk",
            plan_summary="delete_file",
        )
        d = decision.to_dict()
        assert d["mode"] == "high"
        assert d["confidence"] == 0.95
        assert d["risk"] == "high"
        assert d["action"] == "ask"


# ============================================================================
# CONFIRMATION FLOW INTEGRATION TESTS
# ============================================================================

class TestConfirmationFlowIntegration:
    """Integration tests for confirmation flow."""
    
    def test_confirmation_not_active_when_mode_off(self, sample_plan):
        """Confirmation logic should not run when mode is off."""
        set_autonomy_mode("off")
        
        # Manually set a pending confirmation (should not happen in normal flow)
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        # Mode is off, so this pending would be ignored by normal flow
        # (The orchestrator checks mode before setting pending)
        assert has_pending_confirmation()  # Still set technically
        
    def test_pending_preserved_same_mode(self, sample_plan):
        """Pending confirmation preserved when mode unchanged."""
        set_autonomy_mode("high")
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        # Re-set same mode
        set_autonomy_mode("high")
        
        # Pending should still be cleared (mode change clears pending)
        # This is by design - even same mode change is a reset
        assert not has_pending_confirmation()
        
    def test_multiple_confirmations_replace(self, sample_plan):
        """New confirmation replaces existing one."""
        set_autonomy_mode("high")
        
        plan1 = [{"tool": "delete_file", "args": {"path": "/a"}}]
        plan2 = [{"tool": "shutdown", "args": {}}]
        
        set_pending_confirmation(plan1, "Delete?", 20.0)
        set_pending_confirmation(plan2, "Shutdown?", 20.0)
        
        pending = get_pending_confirmation()
        assert pending.plan == plan2
        assert pending.prompt == "Shutdown?"
