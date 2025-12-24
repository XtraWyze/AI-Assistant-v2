"""
Tests for Exit Phrase Guard with Pending Confirmation.

Ensures that "No, cancel that" and similar phrases are NOT intercepted
by the exit phrase detector when a pending confirmation exists.

Run with: python -m pytest tests/test_exit_guard.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from wyzer.core.followup_manager import (
    FollowupManager,
    is_exit_sentinel,
    make_exit_sentinel,
)
from wyzer.policy.pending_confirmation import (
    resolve_pending,
    has_active_pending,
    is_no,
)
from wyzer.context.world_state import (
    clear_world_state,
    set_pending_confirmation,
    get_pending_confirmation,
    clear_pending_confirmation,
    has_pending_confirmation,
    set_autonomy_mode,
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
# EXIT PHRASE DETECTION TESTS (without pending)
# ============================================================================

class TestExitPhraseWithoutPending:
    """Exit phrase detection when NO pending confirmation exists."""
    
    def test_cancel_is_exit_phrase(self):
        """'cancel' alone is detected as exit phrase."""
        manager = FollowupManager()
        
        result = manager.check_exit_phrase("cancel", log_detection=False)
        
        assert is_exit_sentinel(result)
        
    def test_no_is_exit_phrase(self):
        """'no' alone is detected as exit phrase."""
        manager = FollowupManager()
        
        result = manager.check_exit_phrase("no", log_detection=False)
        
        assert is_exit_sentinel(result)
        
    def test_stop_is_exit_phrase(self):
        """'stop' is detected as exit phrase."""
        manager = FollowupManager()
        
        result = manager.check_exit_phrase("stop", log_detection=False)
        
        assert is_exit_sentinel(result)
        
    def test_nevermind_is_exit_phrase(self):
        """'never mind' and 'nevermind' are exit phrases."""
        manager = FollowupManager()
        
        result1 = manager.check_exit_phrase("never mind", log_detection=False)
        result2 = manager.check_exit_phrase("nevermind", log_detection=False)
        
        assert is_exit_sentinel(result1)
        assert is_exit_sentinel(result2)


# ============================================================================
# EXIT GUARD TESTS (with pending confirmation)
# ============================================================================

class TestExitGuardWithPending:
    """
    Tests that the exit guard prevents exit phrase detection
    when a pending confirmation exists.
    
    The key fix: "No, cancel that." should cancel the pending confirmation,
    NOT be treated as an exit phrase.
    """
    
    def test_no_pattern_matches_when_pending(self, sample_plan):
        """is_no() correctly matches 'No, cancel that.' pattern."""
        # This verifies our no pattern matches the problematic phrase
        assert is_no("No, cancel that.") is True
        assert is_no("no, cancel that") is True
        
    def test_resolve_handles_cancel_that(self, sample_plan):
        """resolve_pending handles 'No, cancel that.' as cancellation."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        speak_fn = MagicMock()
        result = resolve_pending("No, cancel that.", speak_fn=speak_fn)
        
        # Should be cancelled, not ignored
        assert result == "cancelled"
        assert not has_pending_confirmation()
        speak_fn.assert_called_once_with("Okay, cancelled.")
        
    def test_guard_logic(self, sample_plan):
        """Simulates the guard logic from brain_worker."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        # The brain_worker guard: only check exit if no active pending
        transcript = "No, cancel that."
        
        # Step 1: Check pending first
        assert has_active_pending() is True
        
        # Step 2: resolve_pending handles it
        result = resolve_pending(transcript)
        assert result == "cancelled"
        
        # Step 3: Guard would prevent exit check because we returned early
        # (In brain_worker, we `continue` after resolve_pending returns non-"none")
        
        # If we hadn't returned early, the guard would block exit check:
        # if not has_active_pending():
        #     check_exit_phrase(...)
        # Since pending was cleared, this would now pass, but we already returned
        
    def test_stop_cancels_when_pending(self, sample_plan):
        """'stop' cancels pending confirmation, not exit."""
        set_pending_confirmation(sample_plan, "Delete file?", 20.0)
        
        speak_fn = MagicMock()
        result = resolve_pending("stop", speak_fn=speak_fn)
        
        assert result == "cancelled"
        assert not has_pending_confirmation()
        
    def test_nope_cancels_when_pending(self, sample_plan):
        """'nope' cancels pending confirmation."""
        set_pending_confirmation(sample_plan, "Shutdown?", 20.0)
        
        speak_fn = MagicMock()
        result = resolve_pending("nope", speak_fn=speak_fn)
        
        assert result == "cancelled"
        
    def test_nevermind_cancels_when_pending(self, sample_plan):
        """'never mind' cancels pending confirmation."""
        set_pending_confirmation(sample_plan, "Close window?", 20.0)
        
        speak_fn = MagicMock()
        result = resolve_pending("never mind", speak_fn=speak_fn)
        
        assert result == "cancelled"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestExitGuardIntegration:
    """Integration tests simulating the full flow."""
    
    def test_exit_phrase_blocked_during_pending(self, sample_plan):
        """Exit phrase detection is blocked while pending confirmation exists."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        manager = FollowupManager()
        
        # Before: Pending exists, guard should block
        assert has_active_pending() is True
        
        # If brain_worker calls has_active_pending() check:
        # it would skip check_exit_phrase entirely
        
        # Simulate: resolve_pending handles "no" first
        speak_fn = MagicMock()
        result = resolve_pending("no", speak_fn=speak_fn)
        
        assert result == "cancelled"
        
        # After: Pending cleared, exit phrase detection would work normally
        assert has_active_pending() is False
        exit_result = manager.check_exit_phrase("no", log_detection=False)
        assert is_exit_sentinel(exit_result)  # Now it would match
        
    def test_exit_phrase_works_after_pending_cleared(self, sample_plan):
        """Exit phrase detection works normally after pending is cleared."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        # Clear the pending confirmation
        clear_pending_confirmation()
        
        manager = FollowupManager()
        
        # Now exit phrase should be detected
        assert not has_active_pending()
        result = manager.check_exit_phrase("cancel", log_detection=False)
        
        assert is_exit_sentinel(result)


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestRegressions:
    """Regression tests for specific bugs."""
    
    def test_no_cancel_that_not_exit(self, sample_plan):
        """
        REGRESSION: 'No, cancel that.' was being intercepted by exit phrase
        detector, causing pending confirmation to never cancel properly.
        
        This test verifies the fix: resolve_pending handles it FIRST.
        """
        set_pending_confirmation(sample_plan, "Close tab?", 20.0)
        
        # The problematic phrase
        transcript = "No, cancel that."
        
        # Step 1: resolve_pending MUST be called first
        speak_fn = MagicMock()
        result = resolve_pending(transcript, speak_fn=speak_fn)
        
        # Step 2: Should be cancelled, not fall through to exit detection
        assert result == "cancelled", \
            "'No, cancel that.' should cancel pending, not be ignored"
        
        # Step 3: Pending must be cleared
        assert not has_pending_confirmation(), \
            "Pending confirmation should be cleared after cancel"
        
        # Step 4: Should speak acknowledgment
        speak_fn.assert_called_once_with("Okay, cancelled.")
        
    def test_exact_exit_phrases_cancel_pending(self, sample_plan):
        """All exit phrases should cancel pending when it exists."""
        exit_phrases_to_test = [
            "no",
            "nope", 
            "cancel",
            "stop",
            "never mind",
            "nevermind",
        ]
        
        for phrase in exit_phrases_to_test:
            # Reset state
            clear_world_state()
            set_pending_confirmation(sample_plan, "Confirm?", 20.0)
            
            result = resolve_pending(phrase)
            
            assert result == "cancelled", \
                f"'{phrase}' should cancel pending confirmation"
            assert not has_pending_confirmation(), \
                f"Pending should be cleared after '{phrase}'"
