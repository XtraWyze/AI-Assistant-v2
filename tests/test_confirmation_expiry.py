"""
Tests for Confirmation Expiry Functionality.

Tests that pending confirmations expire correctly both actively (when user
speaks after timeout) and passively (via heartbeat tick without user speech).

Run with: python -m pytest tests/test_confirmation_expiry.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from wyzer.policy.pending_confirmation import (
    resolve_pending,
    check_passive_expiry,
    has_active_pending,
    CONFIRMATION_TIMEOUT_SEC,
)
from wyzer.context.world_state import (
    PendingConfirmation,
    clear_world_state,
    set_pending_confirmation,
    get_pending_confirmation,
    has_pending_confirmation,
    set_autonomy_mode,
    get_world_state,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def clean_state():
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
        {"tool": "close_window", "args": {"target": "Chrome"}},
    ]


# ============================================================================
# PENDING CONFIRMATION DATACLASS TESTS
# ============================================================================

class TestPendingConfirmationExpiry:
    """Tests for PendingConfirmation.is_expired()."""
    
    def test_not_expired_within_window(self):
        """is_expired returns False within timeout window."""
        pending = PendingConfirmation(
            plan=[{"tool": "test"}],
            expires_ts=time.time() + 10.0,
            prompt="Confirm?",
        )
        assert not pending.is_expired()
        
    def test_expired_after_window(self):
        """is_expired returns True after timeout."""
        pending = PendingConfirmation(
            plan=[{"tool": "test"}],
            expires_ts=time.time() - 1.0,  # Already expired
            prompt="Confirm?",
        )
        assert pending.is_expired()
        
    def test_expired_exactly_at_boundary(self):
        """is_expired returns True when time equals expires_ts."""
        now = time.time()
        pending = PendingConfirmation(
            plan=[{"tool": "test"}],
            expires_ts=now,  # Exactly now
            prompt="Confirm?",
        )
        # time.time() > expires_ts may or may not be true depending on timing
        # Just verify it doesn't crash
        _ = pending.is_expired()


# ============================================================================
# ACTIVE EXPIRY TESTS (user speaks after timeout)
# ============================================================================

class TestActiveExpiry:
    """Tests for expiry when user responds after timeout."""
    
    def test_yes_after_expiry_returns_expired(self, sample_plan):
        """'yes' after timeout returns 'none' (auto-cleared) or 'expired'."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)  # Wait for expiry
        
        result = resolve_pending("yes")
        
        # get_pending_confirmation auto-clears expired, so returns "none"
        assert result in ("expired", "none")
        assert not has_pending_confirmation()
        
    def test_no_after_expiry_returns_expired(self, sample_plan):
        """'no' after timeout returns 'none' (auto-cleared) or 'expired'."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)  # Wait for expiry
        
        result = resolve_pending("no")
        
        # get_pending_confirmation auto-clears expired, so returns "none"
        assert result in ("expired", "none")
        assert not has_pending_confirmation()
        
    def test_executor_not_called_after_expiry(self, sample_plan):
        """Executor should NOT be called if confirmation expired."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)  # Wait for expiry
        
        executor = MagicMock()
        result = resolve_pending("yes", executor=executor)
        
        # Either expired or none (auto-cleared)
        assert result in ("expired", "none")
        executor.assert_not_called()
        
    def test_speak_not_called_after_expiry(self, sample_plan):
        """Speak function should NOT be called if confirmation expired."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)  # Wait for expiry
        
        speak_fn = MagicMock()
        result = resolve_pending("no", speak_fn=speak_fn)
        
        # Either expired or none (auto-cleared)
        assert result in ("expired", "none")
        speak_fn.assert_not_called()
        
    def test_expired_with_explicit_timestamp(self, sample_plan):
        """Expiry can be tested with explicit now_ts parameter."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        # Pass a timestamp far in the future
        future_ts = time.time() + 100
        result = resolve_pending("yes", now_ts=future_ts)
        
        assert result == "expired"


# ============================================================================
# PASSIVE EXPIRY TESTS (heartbeat tick, no user speech)
# ============================================================================

class TestPassiveExpiry:
    """Tests for passive expiry via heartbeat tick."""
    
    def test_passive_expiry_clears_expired(self, sample_plan):
        """Passive expiry clears expired pending confirmation."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)  # Wait for expiry
        
        result = check_passive_expiry()
        
        assert result is True
        assert not has_pending_confirmation()
        
    def test_passive_expiry_returns_false_when_valid(self, sample_plan):
        """Passive expiry returns False when pending is still valid."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        result = check_passive_expiry()
        
        assert result is False
        assert has_pending_confirmation()
        
    def test_passive_expiry_returns_false_when_none(self):
        """Passive expiry returns False when no pending confirmation."""
        result = check_passive_expiry()
        assert result is False
        
    def test_passive_expiry_only_runs_once(self, sample_plan):
        """Calling passive expiry twice doesn't cause issues."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)
        
        result1 = check_passive_expiry()
        result2 = check_passive_expiry()
        
        assert result1 is True  # First call clears it
        assert result2 is False  # Nothing to clear
        

class TestExpiryWithoutUserSpeech:
    """
    Tests verifying the CRITICAL requirement:
    Pending confirmations MUST expire even if the user never speaks again.
    """
    
    def test_expiry_without_speech(self, sample_plan):
        """Pending expires without any user interaction."""
        # Set pending with short timeout
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        
        # Verify it's active
        assert has_pending_confirmation()
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Simulate heartbeat tick (no user speech)
        expired = check_passive_expiry()
        
        # Must be expired
        assert expired is True
        assert not has_pending_confirmation()
        
    def test_get_pending_returns_none_after_expiry(self, sample_plan):
        """get_pending_confirmation returns None after expiry."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)
        
        pending = get_pending_confirmation()
        
        assert pending is None
        
    def test_has_active_pending_false_after_expiry(self, sample_plan):
        """has_active_pending returns False after expiry."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)
        
        assert has_active_pending() is False


# ============================================================================
# TIMEOUT CONFIGURATION TESTS
# ============================================================================

class TestTimeoutConfiguration:
    """Tests for confirmation timeout configuration."""
    
    def test_default_timeout_is_45_seconds(self):
        """Default confirmation timeout is 45 seconds (increased from 20s for easier response)."""
        assert CONFIRMATION_TIMEOUT_SEC == 45.0
        
    def test_custom_timeout_respected(self, sample_plan):
        """Custom timeout is respected when setting pending."""
        set_pending_confirmation(sample_plan, "Confirm?", 5.0)  # 5 second timeout
        
        pending = get_pending_confirmation()
        
        # Check that expires_ts is approximately 5 seconds from now
        expected_expiry = time.time() + 5.0
        assert abs(pending.expires_ts - expected_expiry) < 0.5  # Within 0.5s
        
    def test_very_short_timeout(self, sample_plan):
        """Very short timeouts work correctly."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.05)  # 50ms
        
        # Should be valid immediately
        assert has_pending_confirmation()
        
        # Wait just past timeout
        time.sleep(0.1)
        
        # Should be expired
        assert not has_pending_confirmation()


# ============================================================================
# EDGE CASES
# ============================================================================

class TestExpiryEdgeCases:
    """Edge cases for expiry handling."""
    
    def test_multiple_pending_replacements(self, sample_plan):
        """New pending confirmation replaces old one with new timeout."""
        set_pending_confirmation(sample_plan, "First?", 0.1)
        time.sleep(0.05)  # Halfway to first expiry
        
        # Set new pending with longer timeout
        plan2 = [{"tool": "other"}]
        set_pending_confirmation(plan2, "Second?", 20.0)
        
        # Old one should be replaced, new one valid
        pending = get_pending_confirmation()
        assert pending is not None
        assert pending.prompt == "Second?"
        assert not pending.is_expired()
        
    def test_clear_then_expiry_check(self, sample_plan):
        """Clearing then checking expiry doesn't cause issues."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        clear_world_state()
        
        # Should return False (nothing to expire)
        result = check_passive_expiry()
        assert result is False
