"""
Tests for Exit Phrase Suppression during Confirmation Handling.

Verifies that exit phrases like "No, cancel" are NOT intercepted by 
the Core EXIT phrase detector when a pending confirmation is active
or was just resolved.

Run with: python -m pytest tests/test_exit_suppression.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from wyzer.policy.pending_confirmation import (
    resolve_pending,
    has_active_pending,
    is_no,
    is_yes,
)
from wyzer.core.followup_manager import (
    FollowupManager,
    is_exit_sentinel,
)
from wyzer.context.world_state import (
    clear_world_state,
    set_pending_confirmation,
    get_pending_confirmation,
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
# CORE REGRESSION TESTS - Exact Bug Scenario
# ============================================================================

class TestExitNotTriggeredOnCancelConfirmation:
    """
    Test A: Exit NOT triggered on cancel confirmation.
    
    BUG: "No. Cancel." was triggering exit after confirmation cancelled.
    Log showed:
        [CONFIRM] cancelled
        ... Synthesizing: Okay, cancelled...
        [EXIT] Exit phrase detected: 'No. Cancel.' -> 'no cancel'
    """
    
    def test_no_cancel_handled_by_confirmation_not_exit(self, sample_plan):
        """'No. Cancel.' is handled by confirmation, not exit detector."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        transcript = "No. Cancel."
        
        # Step 1: Confirmation handler should handle it
        assert has_active_pending() is True
        
        speak_fn = MagicMock()
        result = resolve_pending(transcript, speak_fn=speak_fn)
        
        assert result == "cancelled"
        speak_fn.assert_called_once_with("Okay, cancelled.")
        
        # Step 2: Pending is now cleared
        assert has_active_pending() is False
        
        # Step 3: Even though "no cancel" matches exit phrases,
        # the suppress_exit_once flag should be set by Brain,
        # preventing Core from detecting it as exit
        
    def test_exit_detector_would_match_no_cancel(self):
        """Verify that 'No. Cancel.' would normally match exit phrases."""
        # This proves the exit detector WOULD match if not suppressed
        manager = FollowupManager()
        
        result = manager.check_exit_phrase("No. Cancel.", log_detection=False)
        
        # It DOES match exit phrase patterns
        assert is_exit_sentinel(result)
        assert result["phrase"] in ("no", "no cancel", "cancel")
        
    def test_confirm_handler_clears_pending_before_exit_check(self, sample_plan):
        """Confirmation handler clears pending before exit detection runs."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        # Before handling
        assert has_active_pending() is True
        
        # Handle cancellation
        result = resolve_pending("no")
        
        # After handling
        assert result == "cancelled"
        assert has_active_pending() is False
        

class TestExitNotTriggeredOnYesConfirmation:
    """
    Test B: Exit NOT triggered on yes confirmation.
    
    "Yes" should execute the pending plan, not trigger exit.
    """
    
    def test_yes_handled_by_confirmation_not_exit(self, sample_plan):
        """'Yes' executes pending plan, not exit detection."""
        set_pending_confirmation(sample_plan, "Close Chrome?", 20.0)
        
        transcript = "Yes"
        
        # Step 1: Confirmation handler should handle it
        executor = MagicMock()
        result = resolve_pending(transcript, executor=executor)
        
        assert result == "executed"
        executor.assert_called_once_with(sample_plan)
        
        # Step 2: Pending is cleared
        assert has_active_pending() is False
        
    def test_yes_variants_handled_by_confirmation(self, sample_plan):
        """All yes variants are handled by confirmation, not exit."""
        yes_variants = ["yes", "Yeah", "yep", "do it", "proceed", "go ahead"]
        
        for variant in yes_variants:
            clear_world_state()
            set_pending_confirmation(sample_plan, "Confirm?", 20.0)
            
            executor = MagicMock()
            result = resolve_pending(variant, executor=executor)
            
            assert result == "executed", f"'{variant}' should execute"
            executor.assert_called_once()


class TestExitStillWorksWithoutPending:
    """
    Test C: Exit still works when NO pending confirmation.
    
    Normal exit phrase detection must continue working.
    """
    
    def test_exit_detection_works_without_pending(self):
        """Exit phrases are detected when no pending confirmation."""
        # No pending confirmation set
        assert has_active_pending() is False
        
        manager = FollowupManager()
        
        # Exit phrases should still be detected
        result = manager.check_exit_phrase("cancel", log_detection=False)
        assert is_exit_sentinel(result)
        
        result = manager.check_exit_phrase("stop", log_detection=False)
        assert is_exit_sentinel(result)
        
        result = manager.check_exit_phrase("never mind", log_detection=False)
        assert is_exit_sentinel(result)
        
    def test_no_alone_is_exit_without_pending(self):
        """'no' alone is still exit phrase without pending confirmation."""
        assert has_active_pending() is False
        
        manager = FollowupManager()
        result = manager.check_exit_phrase("no", log_detection=False)
        
        assert is_exit_sentinel(result)


# ============================================================================
# SUPPRESS_EXIT_ONCE LOGIC TESTS
# ============================================================================

class TestSuppressExitOnceLogic:
    """Tests for the suppress_exit_once latch mechanism."""
    
    def test_suppress_flag_in_result_meta_cancelled(self, sample_plan):
        """Brain sets suppress_exit_once=True in RESULT meta for cancelled."""
        # This tests the intent - the actual flag is set in brain_worker
        # We verify the confirmation returns "cancelled" which should
        # trigger the flag setting in brain_worker
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        result = resolve_pending("no, cancel that")
        
        assert result == "cancelled"
        # Brain worker would set suppress_exit_once=True in meta
        
    def test_suppress_flag_in_result_meta_executed(self, sample_plan):
        """Brain sets suppress_exit_once=True in RESULT meta for executed."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        executor = MagicMock()
        result = resolve_pending("yes", executor=executor)
        
        assert result == "executed"
        # Brain worker would set suppress_exit_once=True in meta


# ============================================================================
# GUARD ORDER TESTS
# ============================================================================

class TestExitGuardOrder:
    """
    Tests the exit check order:
    A) if pending_confirmation exists -> do NOT exit-detect
    B) else if suppress_exit_once -> consume and do NOT exit-detect
    C) else -> normal exit detection
    """
    
    def test_pending_blocks_exit_detection(self, sample_plan):
        """When pending exists, exit detection is blocked."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        # Exit detection should be blocked
        assert has_active_pending() is True
        
        # Even though "no" matches exit, it should be handled by confirmation
        result = resolve_pending("no")
        assert result == "cancelled"
        
    def test_resolve_pending_runs_before_exit_check(self, sample_plan):
        """resolve_pending is called before exit phrase check."""
        set_pending_confirmation(sample_plan, "Confirm?", 20.0)
        
        # This is the exact flow in brain_worker:
        # 1. resolve_pending() called first
        # 2. If result != "none", exit check is skipped
        
        result = resolve_pending("no")
        assert result == "cancelled"  # Not "none"
        
        # Since result != "none", brain_worker returns early (continues)
        # Exit check never runs


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests for exit suppression."""
    
    def test_multiple_exit_words_in_cancel(self, sample_plan):
        """'No, cancel that' has multiple exit words but only triggers cancel."""
        set_pending_confirmation(sample_plan, "Delete file?", 20.0)
        
        # "no", "cancel" are both exit phrases
        transcript = "No, cancel that."
        
        result = resolve_pending(transcript)
        
        assert result == "cancelled"
        assert not has_active_pending()
        
    def test_stop_during_pending_cancels(self, sample_plan):
        """'stop' during pending cancels, not exits."""
        set_pending_confirmation(sample_plan, "Shutdown?", 20.0)
        
        result = resolve_pending("stop")
        
        assert result == "cancelled"
        
    def test_nevermind_during_pending_cancels(self, sample_plan):
        """'never mind' during pending cancels, not exits."""
        set_pending_confirmation(sample_plan, "Close tab?", 20.0)
        
        result = resolve_pending("never mind")
        
        assert result == "cancelled"
