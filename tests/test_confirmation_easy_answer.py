"""
Tests for Phase 11 Confirmation Easy Answer Flow.

Tests the improvements to make confirmation responses easier:
- 45s TTL instead of 20s
- Confirmation works regardless of Core state (IDLE/FOLLOWUP)
- Confirmation doesn't require hotword
- TTS grace period for barge-in during prompt

Run with: python -m pytest tests/test_confirmation_easy_answer.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from wyzer.policy.pending_confirmation import (
    resolve_pending,
    has_active_pending,
    is_yes,
    is_no,
    CONFIRMATION_TIMEOUT_SEC,
    CONFIRMATION_GRACE_MS,
)
from wyzer.context.world_state import (
    clear_world_state,
    set_pending_confirmation,
    get_pending_confirmation,
    has_pending_confirmation,
    set_autonomy_mode,
)
from wyzer.core.config import Config


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
# TEST A: Confirmation accepted outside FOLLOWUP
# ============================================================================

class TestConfirmationAcceptedOutsideFollowup:
    """
    Test that pending_confirmation can be handled even after FOLLOWUP would have ended.
    
    Scenario: User triggers confirmation, FOLLOWUP times out (typically 2-3s),
    but user says "yes" at 10s - should still work because confirmation TTL is 45s.
    """
    
    def test_confirmation_accepted_after_followup_timeout(self, sample_plan):
        """'Yes' confirms even after FOLLOWUP would have timed out (simulated)."""
        # Set up pending confirmation with 45s timeout
        set_pending_confirmation(sample_plan, "Close Chrome?", timeout_sec=45.0)
        
        assert has_active_pending() is True
        
        # Simulate time passing - FOLLOWUP typically times out after 2-3 seconds
        # We'll test at 10 seconds after confirmation was set
        # The confirmation should still be active (45s TTL)
        simulated_now = time.time() + 10.0  # 10 seconds later
        
        executor = MagicMock()
        result = resolve_pending("yes", now_ts=simulated_now, executor=executor)
        
        # Should still execute because 10s < 45s TTL
        assert result == "executed"
        executor.assert_called_once_with(sample_plan)
        
        # Pending should be cleared after execution
        assert has_active_pending() is False
    
    def test_confirmation_accepted_at_30_seconds(self, sample_plan):
        """'Yes' confirms even 30 seconds after prompt (well within 45s TTL)."""
        set_pending_confirmation(sample_plan, "Delete file?", timeout_sec=45.0)
        
        # 30 seconds later - would have missed with old 20s timeout
        simulated_now = time.time() + 30.0
        
        executor = MagicMock()
        result = resolve_pending("yes", now_ts=simulated_now, executor=executor)
        
        assert result == "executed"
        executor.assert_called_once()
    
    def test_cancel_accepted_after_followup_timeout(self, sample_plan):
        """'No' cancels even after FOLLOWUP would have timed out."""
        set_pending_confirmation(sample_plan, "Close all windows?", timeout_sec=45.0)
        
        # 15 seconds later - after FOLLOWUP would have ended but within 45s TTL
        simulated_now = time.time() + 15.0
        
        speak_fn = MagicMock()
        result = resolve_pending("no", now_ts=simulated_now, speak_fn=speak_fn)
        
        assert result == "cancelled"
        speak_fn.assert_called_once_with("Okay, cancelled.")
        assert has_active_pending() is False


# ============================================================================
# TEST B: Confirmation does not require hotword
# ============================================================================

class TestConfirmationDoesNotRequireHotword:
    """
    Test that yes/no can be processed without hotword detection.
    
    This tests the policy layer - the actual hotword bypass is in Core,
    but the resolve_pending function should accept any transcript that
    matches yes/no patterns, regardless of whether hotword was detected.
    """
    
    def test_bare_yes_triggers_confirmation(self, sample_plan):
        """Bare 'yes' without any hotword prefix confirms."""
        set_pending_confirmation(sample_plan, "Confirm action?", timeout_sec=45.0)
        
        # Transcript is just "yes" - no hotword in the text
        transcript = "yes"
        
        executor = MagicMock()
        result = resolve_pending(transcript, executor=executor)
        
        assert result == "executed"
        executor.assert_called_once_with(sample_plan)
    
    def test_bare_no_triggers_cancellation(self, sample_plan):
        """Bare 'no' without any hotword prefix cancels."""
        set_pending_confirmation(sample_plan, "Confirm action?", timeout_sec=45.0)
        
        transcript = "no"
        
        speak_fn = MagicMock()
        result = resolve_pending(transcript, speak_fn=speak_fn)
        
        assert result == "cancelled"
        speak_fn.assert_called_once_with("Okay, cancelled.")
    
    def test_yes_variants_work_without_hotword(self, sample_plan):
        """All yes variants work without hotword prefix."""
        yes_variants = ["yes", "yeah", "yep", "yup", "do it", "proceed", "go ahead", "sure", "ok", "okay"]
        
        for variant in yes_variants:
            clear_world_state()
            set_pending_confirmation(sample_plan, "Confirm?", timeout_sec=45.0)
            
            executor = MagicMock()
            result = resolve_pending(variant, executor=executor)
            
            assert result == "executed", f"'{variant}' should trigger execution"
            executor.assert_called_once()
    
    def test_no_variants_work_without_hotword(self, sample_plan):
        """All no variants work without hotword prefix."""
        no_variants = ["no", "nope", "nah", "cancel", "stop", "don't", "never mind"]
        
        for variant in no_variants:
            clear_world_state()
            set_pending_confirmation(sample_plan, "Confirm?", timeout_sec=45.0)
            
            speak_fn = MagicMock()
            result = resolve_pending(variant, speak_fn=speak_fn)
            
            assert result == "cancelled", f"'{variant}' should trigger cancellation"
    
    def test_confirmation_marker_not_required(self, sample_plan):
        """
        Verify that no special 'hotword detected' marker is required in transcript.
        
        The resolve_pending function should work purely on the transcript text,
        not requiring any metadata about hotword detection.
        """
        set_pending_confirmation(sample_plan, "Confirm?", timeout_sec=45.0)
        
        # Just pass raw transcript - no hotword metadata
        executor = MagicMock()
        result = resolve_pending("yes please", executor=executor)
        
        assert result == "executed"


# ============================================================================
# TEST C: Confirmation expires at 45s
# ============================================================================

class TestConfirmationExpires45s:
    """
    Test that confirmation expires after 45 seconds (not 20s).
    """
    
    def test_confirmation_expires_after_45_seconds(self, sample_plan):
        """Confirmation should be expired after 45 seconds."""
        set_pending_confirmation(sample_plan, "Confirm?", timeout_sec=45.0)
        
        # Advance time past the 45s TTL
        simulated_now = time.time() + 46.0  # Just past 45s
        
        executor = MagicMock()
        result = resolve_pending("yes", now_ts=simulated_now, executor=executor)
        
        # Should be expired, not executed
        assert result == "expired"
        executor.assert_not_called()
        
        # Pending should be cleared
        assert has_active_pending() is False
    
    def test_confirmation_still_active_at_44_seconds(self, sample_plan):
        """Confirmation should still be active at 44 seconds."""
        set_pending_confirmation(sample_plan, "Confirm?", timeout_sec=45.0)
        
        # Just before expiry
        simulated_now = time.time() + 44.0
        
        executor = MagicMock()
        result = resolve_pending("yes", now_ts=simulated_now, executor=executor)
        
        # Should still work
        assert result == "executed"
        executor.assert_called_once()
    
    def test_yes_does_not_execute_after_expiry(self, sample_plan):
        """'Yes' should NOT execute after confirmation expires."""
        set_pending_confirmation(sample_plan, "Delete important file?", timeout_sec=45.0)
        
        # 60 seconds later - well past 45s TTL
        simulated_now = time.time() + 60.0
        
        executor = MagicMock()
        result = resolve_pending("yes", now_ts=simulated_now, executor=executor)
        
        # Should NOT execute
        assert result == "expired"
        executor.assert_not_called()
    
    def test_default_timeout_is_45_seconds(self):
        """Verify the default confirmation timeout is 45 seconds."""
        assert CONFIRMATION_TIMEOUT_SEC == 45.0
    
    def test_config_default_timeout_is_45_seconds(self):
        """Verify Config.AUTONOMY_CONFIRM_TIMEOUT_SEC default is 45 seconds."""
        # This tests the default value in config (may be overridden by env var)
        # The default should be 45.0 unless env var changes it
        assert Config.AUTONOMY_CONFIRM_TIMEOUT_SEC >= 45.0 or \
               Config.AUTONOMY_CONFIRM_TIMEOUT_SEC == float(
                   __import__('os').environ.get("WYZER_AUTONOMY_CONFIRM_TIMEOUT_SEC", "45.0")
               )


# ============================================================================
# ADDITIONAL TESTS: TTS Grace Period
# ============================================================================

class TestConfirmationGracePeriod:
    """
    Test the TTS grace period feature (CONFIRMATION_GRACE_MS).
    
    When user speaks yes/no while the confirmation prompt TTS is still playing,
    their response should still be accepted.
    """
    
    def test_grace_period_constant_exists(self):
        """Verify CONFIRMATION_GRACE_MS constant is defined."""
        assert CONFIRMATION_GRACE_MS == 1500  # 1.5 seconds
    
    def test_config_grace_period_exists(self):
        """Verify Config.CONFIRMATION_GRACE_MS exists."""
        assert hasattr(Config, 'CONFIRMATION_GRACE_MS')
        assert Config.CONFIRMATION_GRACE_MS == 1500


# ============================================================================
# LOGGING TESTS
# ============================================================================

class TestConfirmationLogging:
    """Test that proper logging occurs during confirmation handling."""
    
    def test_yes_logs_age_and_expired_flag(self, sample_plan):
        """'Yes' should log age_ms and expired=0."""
        set_pending_confirmation(sample_plan, "Confirm?", timeout_sec=45.0)
        
        # Execute after 5 seconds
        simulated_now = time.time() + 5.0
        
        with patch('wyzer.policy.pending_confirmation.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            resolve_pending("yes", now_ts=simulated_now, executor=MagicMock())
            
            # Check that INFO log was called with expected format
            info_calls = [call for call in mock_logger.info.call_args_list]
            log_messages = [str(call) for call in info_calls]
            
            # Should have a log entry with 'received reply="yes"'
            assert any('received reply="yes"' in msg for msg in log_messages)
    
    def test_expired_logs_cleared_message(self, sample_plan):
        """Expired confirmation should log 'expired -> cleared'."""
        set_pending_confirmation(sample_plan, "Confirm?", timeout_sec=45.0)
        
        # Well past expiry
        simulated_now = time.time() + 50.0
        
        with patch('wyzer.policy.pending_confirmation.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            resolve_pending("yes", now_ts=simulated_now, executor=MagicMock())
            
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            
            # Should have 'expired -> cleared' message
            assert any('expired -> cleared' in msg for msg in info_calls)
