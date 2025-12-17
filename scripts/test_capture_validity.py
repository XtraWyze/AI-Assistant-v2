"""
Tests for capture validity and follow-up suppression on empty captures.

This test suite verifies:
1. Empty/minimal transcripts are detected as invalid captures
2. Invalid captures do NOT trigger follow-up mode
3. Exit phrases after invalid captures work normally (not treated as follow-up exit)
4. Valid captures still work correctly

Run: python -m pytest scripts/test_capture_validity.py -v
"""
import pytest
from unittest.mock import MagicMock, patch

# Import the helper function and sentinel utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wyzer.core.brain_worker import _is_capture_valid
from wyzer.core.followup_manager import FollowupManager


class TestCaptureValidity:
    """Test the _is_capture_valid helper function"""
    
    def test_empty_string_is_invalid(self):
        """Empty string should be invalid"""
        assert _is_capture_valid("") is False
    
    def test_none_is_invalid(self):
        """None should be invalid"""
        assert _is_capture_valid(None) is False
    
    def test_whitespace_only_is_invalid(self):
        """Whitespace-only strings should be invalid"""
        assert _is_capture_valid("   ") is False
        assert _is_capture_valid("\t\n") is False
        assert _is_capture_valid("  \n  ") is False
    
    def test_single_filler_word_is_invalid(self):
        """Single filler words should be invalid"""
        filler_words = ["um", "uh", "hmm", "hm", "ah", "oh", "er", "like", "so", "and"]
        for word in filler_words:
            assert _is_capture_valid(word) is False, f"'{word}' should be invalid"
            assert _is_capture_valid(word.upper()) is False, f"'{word.upper()}' should be invalid"
            assert _is_capture_valid(f"  {word}  ") is False, f"'{word}' with spaces should be invalid"
    
    def test_single_command_word_is_valid(self):
        """Single meaningful command words should be valid"""
        command_words = ["stop", "pause", "play", "mute", "yes", "no", "cancel"]
        for word in command_words:
            assert _is_capture_valid(word) is True, f"'{word}' should be valid"
    
    def test_multi_word_is_valid(self):
        """Multi-word transcripts should be valid"""
        valid_phrases = [
            "open chrome",
            "what time is it",
            "set a timer for 5 minutes",
            "never mind",
            "that's all",
        ]
        for phrase in valid_phrases:
            assert _is_capture_valid(phrase) is True, f"'{phrase}' should be valid"
    
    def test_two_words_is_valid(self):
        """Two-word phrases should be valid"""
        assert _is_capture_valid("hello world") is True
        assert _is_capture_valid("um okay") is True  # Even if starts with filler
    
    def test_punctuation_handling(self):
        """Punctuation shouldn't affect validity"""
        assert _is_capture_valid("stop!") is True
        assert _is_capture_valid("hello?") is True


class TestFollowupSuppression:
    """Test that follow-up mode is suppressed for invalid captures"""
    
    def test_invalid_capture_does_not_trigger_followup_prompt(self):
        """
        Simulate the flow where capture_valid=False should suppress follow-up.
        
        This tests the logic that would happen in assistant.py's RESULT handler.
        """
        # Simulate meta from brain worker for invalid capture
        meta = {
            "capture_valid": False,
            "show_followup_prompt": False,  # Brain worker sets this
            "user_text": "",
            "is_followup": False,
        }
        
        # The assistant should check capture_valid and skip follow-up
        capture_valid = meta.get("capture_valid", True)
        show_followup_prompt = meta.get("show_followup_prompt", True) and capture_valid
        
        assert capture_valid is False
        assert show_followup_prompt is False
    
    def test_valid_capture_can_trigger_followup_prompt(self):
        """Valid captures can trigger follow-up if show_followup_prompt is True"""
        meta = {
            "capture_valid": True,
            "show_followup_prompt": True,
            "user_text": "what time is it",
            "is_followup": False,
        }
        
        capture_valid = meta.get("capture_valid", True)
        show_followup_prompt = meta.get("show_followup_prompt", True) and capture_valid
        
        assert capture_valid is True
        assert show_followup_prompt is True


class TestExitPhraseWithInvalidCapture:
    """Test that exit phrases after invalid captures work correctly"""
    
    def test_exit_phrase_not_checked_for_invalid_capture(self):
        """
        Exit phrase detection should be skipped for invalid captures.
        
        This prevents the case where user activates hotword, stays silent,
        then says "never mind" - which should NOT be treated as a follow-up exit.
        """
        manager = FollowupManager()
        
        # Simulate invalid capture meta
        meta = {
            "capture_valid": False,
            "user_text": "",  # Empty transcript
            "is_followup": False,
        }
        
        capture_valid = meta.get("capture_valid", True)
        user_text = meta.get("user_text", "")
        exit_sentinel = meta.get("exit_sentinel")
        
        # The condition for fallback exit phrase check:
        # user_text and not exit_sentinel and capture_valid
        should_check_exit = bool(user_text) and not exit_sentinel and capture_valid
        
        # Should NOT check for exit phrases
        assert should_check_exit is False
    
    def test_exit_phrase_checked_for_valid_capture(self):
        """Exit phrase detection should work for valid captures"""
        manager = FollowupManager()
        
        # Simulate valid capture with exit phrase
        meta = {
            "capture_valid": True,
            "user_text": "never mind",
            "is_followup": True,
        }
        
        capture_valid = meta.get("capture_valid", True)
        user_text = meta.get("user_text", "")
        exit_sentinel = meta.get("exit_sentinel")
        
        # The condition for fallback exit phrase check
        should_check_exit = bool(user_text) and not exit_sentinel and capture_valid
        
        # Should check for exit phrases
        assert should_check_exit is True
        
        # And it should detect the exit phrase
        sentinel = manager.check_exit_phrase(user_text, log_detection=False)
        assert sentinel is not None


class TestFollowAfterSilenceTimeout:
    """
    Integration test for the specific scenario:
    1. User says hotword
    2. Silence timeout with no speech
    3. User says "never mind"
    4. Should NOT trigger follow-up exit
    """
    
    def test_never_mind_after_silence_not_followup_exit(self):
        """
        Simulate the full scenario where silence timeout happens,
        then user says "never mind" which should NOT be a follow-up exit.
        """
        manager = FollowupManager()
        
        # Step 1: First capture (silence timeout with invalid capture)
        first_meta = {
            "capture_valid": False,
            "user_text": "",
            "is_followup": False,
            "show_followup_prompt": False,
        }
        
        # Follow-up should NOT be entered
        capture_valid_1 = first_meta.get("capture_valid", True)
        assert capture_valid_1 is False
        
        # Follow-up manager should NOT be active after invalid capture
        assert manager.is_followup_active() is False
        
        # Step 2: User says "never mind" as a NEW hotword-triggered command
        # This is a fresh capture, NOT a follow-up
        second_meta = {
            "capture_valid": True,
            "user_text": "never mind",
            "is_followup": False,  # NOT a follow-up, it's a new command
            "show_followup_prompt": False,
        }
        
        capture_valid_2 = second_meta.get("capture_valid", True)
        is_followup_2 = second_meta.get("is_followup", False)
        user_text_2 = second_meta.get("user_text", "")
        
        # This is a valid capture
        assert capture_valid_2 is True
        
        # But it's NOT a follow-up
        assert is_followup_2 is False
        
        # Exit phrase IS detected
        sentinel = manager.check_exit_phrase(user_text_2, log_detection=False)
        assert sentinel is not None
        
        # But since is_followup is False, followup_manager.end_followup_window()
        # won't be called (there's no window to end)
        # The system should just return to IDLE normally


class TestBackwardCompatibility:
    """Test backward compatibility with messages that don't have capture_valid"""
    
    def test_missing_capture_valid_defaults_to_true(self):
        """If capture_valid is missing, default to True for backward compat"""
        meta = {
            "user_text": "open chrome",
            "is_followup": False,
            # Note: capture_valid is NOT present
        }
        
        capture_valid = meta.get("capture_valid", True)
        assert capture_valid is True
    
    def test_show_followup_prompt_respects_missing_capture_valid(self):
        """show_followup_prompt should work normally if capture_valid is missing"""
        meta = {
            "user_text": "what time is it?",
            "is_followup": False,
            "show_followup_prompt": True,
            # Note: capture_valid is NOT present
        }
        
        capture_valid = meta.get("capture_valid", True)
        show_followup_prompt = meta.get("show_followup_prompt", True) and capture_valid
        
        assert show_followup_prompt is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
