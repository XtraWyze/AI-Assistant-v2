"""
Tests for Phase 11 Pending Confirmation Flow.

Tests the deterministic yes/no/cancel pattern matching and resolution
for pending confirmations.

Run with: python -m pytest tests/test_confirmation_flow.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch, call

from wyzer.policy.pending_confirmation import (
    normalize,
    is_yes,
    is_no,
    is_confirmation_response,
    resolve_pending,
    check_passive_expiry,
    get_pending_prompt,
    has_active_pending,
    YES_PATTERN,
    NO_PATTERN,
    CONFIRMATION_TIMEOUT_SEC,
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
# PATTERN MATCHING TESTS
# ============================================================================

class TestNormalize:
    """Tests for text normalization."""
    
    def test_normalize_lowercase(self):
        """Normalizes to lowercase."""
        assert normalize("YES") == "yes"
        assert normalize("No") == "no"
        
    def test_normalize_strips_whitespace(self):
        """Strips leading/trailing whitespace."""
        assert normalize("  yes  ") == "yes"
        
    def test_normalize_collapses_whitespace(self):
        """Collapses multiple spaces to single space."""
        assert normalize("do   it") == "do it"
        
    def test_normalize_empty(self):
        """Handles empty input."""
        assert normalize("") == ""
        assert normalize(None) == ""


class TestIsYes:
    """Tests for yes pattern matching."""
    
    def test_exact_yes(self):
        """Matches exact 'yes' variants."""
        assert is_yes("yes") is True
        assert is_yes("yeah") is True
        assert is_yes("yep") is True
        assert is_yes("yup") is True
        assert is_yes("sure") is True
        assert is_yes("ok") is True
        assert is_yes("okay") is True
        
    def test_yes_case_insensitive(self):
        """Matches are case-insensitive."""
        assert is_yes("YES") is True
        assert is_yes("Yes") is True
        assert is_yes("YEAH") is True
        
    def test_yes_with_suffix(self):
        """Matches yes with additional words."""
        assert is_yes("yes, do it") is True
        assert is_yes("yes please") is True
        assert is_yes("yeah that's fine") is True
        
    def test_yes_with_punctuation(self):
        """Matches yes with punctuation."""
        assert is_yes("yes!") is True
        assert is_yes("yes.") is True
        assert is_yes("yes,") is True
        
    def test_do_it_proceed(self):
        """Matches 'do it', 'proceed', 'go ahead'."""
        assert is_yes("do it") is True
        assert is_yes("proceed") is True
        assert is_yes("go ahead") is True
        assert is_yes("confirm") is True
        
    def test_not_yes(self):
        """Rejects non-yes patterns."""
        assert is_yes("no") is False
        assert is_yes("nope") is False
        assert is_yes("maybe") is False
        assert is_yes("yesterday") is False  # Contains "yes" but not at start


class TestIsNo:
    """Tests for no pattern matching."""
    
    def test_exact_no(self):
        """Matches exact 'no' variants."""
        assert is_no("no") is True
        assert is_no("nope") is True
        assert is_no("nah") is True
        assert is_no("cancel") is True
        assert is_no("stop") is True
        
    def test_no_case_insensitive(self):
        """Matches are case-insensitive."""
        assert is_no("NO") is True
        assert is_no("No") is True
        assert is_no("CANCEL") is True
        
    def test_no_with_suffix(self):
        """Matches no with additional words."""
        assert is_no("no, cancel that") is True
        assert is_no("no thanks") is True
        assert is_no("nope, don't do it") is True
        
    def test_nevermind_variants(self):
        """Matches 'never mind' and 'nevermind'."""
        assert is_no("never mind") is True
        assert is_no("nevermind") is True
        
    def test_dont_variants(self):
        """Matches don't/do not variants."""
        assert is_no("don't") is True
        assert is_no("dont") is True
        assert is_no("do not") is True
        
    def test_not_no(self):
        """Rejects non-no patterns."""
        assert is_no("yes") is False
        assert is_no("maybe") is False
        assert is_no("know") is False  # Contains "no" but not at start


class TestIsConfirmationResponse:
    """Tests for is_confirmation_response."""
    
    def test_yes_is_confirmation(self):
        """Yes patterns are confirmation responses."""
        assert is_confirmation_response("yes") is True
        assert is_confirmation_response("yeah") is True
        
    def test_no_is_confirmation(self):
        """No patterns are confirmation responses."""
        assert is_confirmation_response("no") is True
        assert is_confirmation_response("cancel") is True
        
    def test_other_not_confirmation(self):
        """Other text is not a confirmation response."""
        assert is_confirmation_response("maybe") is False
        assert is_confirmation_response("what?") is False


# ============================================================================
# RESOLVE_PENDING TESTS
# ============================================================================

class TestResolvePending:
    """Tests for the resolve_pending function."""
    
    def test_resolve_none_when_no_pending(self):
        """Returns 'none' when no pending confirmation exists."""
        result = resolve_pending("yes")
        assert result == "none"
        
    def test_resolve_executed_on_yes(self, sample_plan):
        """Returns 'executed' and calls executor on yes."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        executor = MagicMock()
        result = resolve_pending("yes", executor=executor)
        
        assert result == "executed"
        executor.assert_called_once_with(sample_plan)
        assert not has_pending_confirmation()  # Cleared
        
    def test_resolve_cancelled_on_no(self, sample_plan):
        """Returns 'cancelled' and speaks on no."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        speak_fn = MagicMock()
        result = resolve_pending("no", speak_fn=speak_fn)
        
        assert result == "cancelled"
        speak_fn.assert_called_once_with("Okay, cancelled.")
        assert not has_pending_confirmation()  # Cleared
        
    def test_resolve_cancelled_on_cancel_that(self, sample_plan):
        """'No, cancel that' returns 'cancelled' (not exit phrase)."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        speak_fn = MagicMock()
        result = resolve_pending("No, cancel that.", speak_fn=speak_fn)
        
        assert result == "cancelled"
        speak_fn.assert_called_once()
        assert not has_pending_confirmation()
        
    def test_resolve_expired_when_timeout(self, sample_plan):
        """Returns 'expired' when confirmation has timed out."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 0.1)
        time.sleep(0.15)  # Wait for expiry
        
        # Note: get_pending_confirmation() auto-clears expired confirmations,
        # so resolve_pending returns "none" (nothing to confirm).
        # Use now_ts parameter to test the explicit expiry path.
        result = resolve_pending("yes")
        # Either "expired" (if checked before auto-clear) or "none" (after auto-clear)
        assert result in ("expired", "none")
        
    def test_resolve_expired_with_explicit_timestamp(self, sample_plan):
        """Returns 'expired' when now_ts is past expires_ts."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        # Pass a timestamp far in the future
        future_ts = time.time() + 100
        result = resolve_pending("yes", now_ts=future_ts)
        
        assert result == "expired"
        
    def test_resolve_ignored_for_other_text(self, sample_plan):
        """Returns 'ignored' for non-yes/no text while pending."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        result = resolve_pending("what do you mean?")
        
        assert result == "ignored"
        assert has_pending_confirmation()  # Still pending
        
    def test_clears_before_execution(self, sample_plan):
        """Clears pending BEFORE calling executor (prevent double-run)."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        execution_order = []
        
        def track_executor(plan):
            execution_order.append("execute")
            # At this point, pending should already be cleared
            assert not has_pending_confirmation()
        
        result = resolve_pending("yes", executor=track_executor)
        
        assert result == "executed"
        assert execution_order == ["execute"]


class TestCheckPassiveExpiry:
    """Tests for passive expiry check."""
    
    def test_passive_expiry_clears_expired(self, sample_plan):
        """Clears expired pending confirmation."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 0.1)
        time.sleep(0.15)
        
        result = check_passive_expiry()
        
        assert result is True
        assert not has_pending_confirmation()
        
    def test_passive_expiry_returns_false_when_valid(self, sample_plan):
        """Returns False when pending is still valid."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        result = check_passive_expiry()
        
        assert result is False
        assert has_pending_confirmation()
        
    def test_passive_expiry_returns_false_when_none(self):
        """Returns False when no pending confirmation."""
        result = check_passive_expiry()
        assert result is False


class TestGetPendingPrompt:
    """Tests for get_pending_prompt."""
    
    def test_returns_prompt_when_pending(self, sample_plan):
        """Returns the prompt text when pending exists."""
        set_pending_confirmation(sample_plan, "Are you sure you want to close Chrome?", 20.0)
        
        prompt = get_pending_prompt()
        
        assert prompt == "Are you sure you want to close Chrome?"
        
    def test_returns_none_when_no_pending(self):
        """Returns None when no pending confirmation."""
        prompt = get_pending_prompt()
        assert prompt is None


class TestHasActivePending:
    """Tests for has_active_pending."""
    
    def test_returns_true_when_pending(self, sample_plan):
        """Returns True when pending confirmation exists."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        assert has_active_pending() is True
        
    def test_returns_false_when_none(self):
        """Returns False when no pending confirmation."""
        assert has_active_pending() is False
        
    def test_returns_false_when_expired(self, sample_plan):
        """Returns False when pending has expired."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)
        
        assert has_active_pending() is False


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case and regression tests."""
    
    def test_yes_executes_pending_plan(self, sample_plan):
        """A) Yes should execute the pending plan."""
        set_pending_confirmation(sample_plan, "Close window?", 20.0)
        
        executor = MagicMock(return_value="Done")
        result = resolve_pending("yes", executor=executor)
        
        assert result == "executed"
        assert executor.call_count == 1
        assert not has_pending_confirmation()
        
    def test_cancel_not_intercepted_by_exit(self, sample_plan):
        """B) Cancel should NOT be intercepted by EXIT when pending exists."""
        set_pending_confirmation(sample_plan, "Close window?", 20.0)
        
        # This is the exact phrase that was being intercepted
        speak_fn = MagicMock()
        result = resolve_pending("No, cancel that.", speak_fn=speak_fn)
        
        # Should be handled as cancellation, not as exit phrase
        assert result == "cancelled"
        assert not has_pending_confirmation()
        speak_fn.assert_called_once_with("Okay, cancelled.")
        
    def test_expiry_without_user_speech(self, sample_plan):
        """C) Expiry should clear pending without user speech."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        
        # Simulate time passing without user speech
        time.sleep(0.15)
        
        # Passive expiry check (called from heartbeat)
        expired = check_passive_expiry()
        
        assert expired is True
        assert not has_pending_confirmation()
        
    def test_non_yes_no_re_asks(self, sample_plan):
        """D) Non yes/no should allow re-asking (return 'ignored')."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        result = resolve_pending("would you?")
        
        assert result == "ignored"
        # Pending still exists so prompt can be re-asked
        assert has_pending_confirmation()
        assert get_pending_prompt() == "Close Chrome?"
        
    def test_nothing_to_confirm_after_expiry(self, sample_plan):
        """After expiry, 'yes' should get 'none' response (nothing to confirm)."""
        set_pending_confirmation(sample_plan, "Confirm?", 0.1)
        time.sleep(0.15)  # Let it expire
        
        result = resolve_pending("yes")
        
        # After expiry + auto-clear, there's nothing to confirm
        # The brain worker translates this to "Nothing to confirm."
        assert result in ("expired", "none")


class TestAcceptanceScenario:
    """
    ACCEPTANCE SCENARIO (MUST PASS):
    1) "Close tab" -> asks confirmation, sets pending 20s
    2) Immediately: "No, cancel that." -> cancels pending (no EXIT), says "Okay, cancelled."
    3) "Close tab" -> asks confirmation
    4) Wait >20s -> pending auto-expires via tick (no user speech required)
    5) "Yes" -> responds "Nothing to confirm." (our case: "expired")
    """
    
    def test_scenario_cancel_does_not_exit(self, sample_plan):
        """Step 2: 'No, cancel that' cancels pending, does not exit."""
        # Set up pending (simulating step 1)
        set_pending_confirmation(sample_plan, "Close tab?", 20.0)
        
        speak_fn = MagicMock()
        result = resolve_pending("No, cancel that.", speak_fn=speak_fn)
        
        assert result == "cancelled"
        speak_fn.assert_called_once_with("Okay, cancelled.")
        assert not has_pending_confirmation()
        
    def test_scenario_auto_expiry(self, sample_plan):
        """Steps 3-4: Pending auto-expires without user speech."""
        # Step 3: Set new pending
        set_pending_confirmation(sample_plan, "Close tab?", 0.1)  # Short timeout for test
        
        # Step 4: Wait for expiry
        time.sleep(0.15)
        
        # Passive expiry (simulating heartbeat tick)
        expired = check_passive_expiry()
        assert expired is True
        assert not has_pending_confirmation()
        
    def test_scenario_yes_after_expiry(self, sample_plan):
        """Step 5: 'Yes' after expiry responds with nothing to confirm."""
        set_pending_confirmation(sample_plan, "Close tab?", 0.1)
        time.sleep(0.15)  # Let it expire
        
        result = resolve_pending("Yes")
        
        # After expiry + auto-clear, resolve_pending returns "none"
        # The brain worker would say "Nothing to confirm."
        assert result in ("expired", "none")
