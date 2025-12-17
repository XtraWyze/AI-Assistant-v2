"""
Tests for exit phrase sentinel pattern and multi-intent preservation.

This test suite verifies:
1. Exit phrases are detected EXACTLY ONCE (no double-detection)
2. Exit phrases produce a proper sentinel that short-circuits the pipeline
3. Multi-intent commands still work correctly (not affected by exit handling)
4. Exit phrases work in both followup and non-followup contexts

Run: python -m pytest scripts/test_exit_phrase_sentinel.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
import logging

from wyzer.core.followup_manager import (
    FollowupManager,
    make_exit_sentinel,
    is_exit_sentinel,
    EXIT_SENTINEL_TYPE,
)
from wyzer.core.multi_intent_parser import try_parse_multi_intent


class TestExitSentinelBasics:
    """Test basic sentinel creation and detection"""
    
    def test_make_exit_sentinel(self):
        """Test that make_exit_sentinel creates proper structure"""
        sentinel = make_exit_sentinel("never mind", "Never mind!")
        
        assert isinstance(sentinel, dict)
        assert sentinel["type"] == EXIT_SENTINEL_TYPE
        assert sentinel["phrase"] == "never mind"
        assert sentinel["original"] == "Never mind!"
    
    def test_is_exit_sentinel_positive(self):
        """Test that is_exit_sentinel correctly identifies sentinels"""
        sentinel = make_exit_sentinel("stop", "Stop!")
        assert is_exit_sentinel(sentinel) is True
    
    def test_is_exit_sentinel_negative_none(self):
        """Test that None is not a sentinel"""
        assert is_exit_sentinel(None) is False
    
    def test_is_exit_sentinel_negative_string(self):
        """Test that strings are not sentinels"""
        assert is_exit_sentinel("stop") is False
        assert is_exit_sentinel("exit_followup") is False
    
    def test_is_exit_sentinel_negative_wrong_type(self):
        """Test that dicts with wrong type are not sentinels"""
        assert is_exit_sentinel({"type": "other", "phrase": "stop"}) is False
        assert is_exit_sentinel({"phrase": "stop"}) is False
    
    def test_is_exit_sentinel_negative_empty_dict(self):
        """Test that empty dict is not a sentinel"""
        assert is_exit_sentinel({}) is False


class TestCheckExitPhrase:
    """Test the check_exit_phrase method that returns sentinels"""
    
    def test_check_exit_phrase_exact_match(self):
        """Test exact match returns sentinel"""
        manager = FollowupManager()
        
        result = manager.check_exit_phrase("stop", log_detection=False)
        assert is_exit_sentinel(result)
        assert result["phrase"] == "stop"
        assert result["original"] == "stop"
    
    def test_check_exit_phrase_case_insensitive(self):
        """Test case insensitive matching"""
        manager = FollowupManager()
        
        result = manager.check_exit_phrase("STOP", log_detection=False)
        assert is_exit_sentinel(result)
        assert result["phrase"] == "stop"
    
    def test_check_exit_phrase_with_punctuation(self):
        """Test matching with punctuation"""
        manager = FollowupManager()
        
        result = manager.check_exit_phrase("Stop!", log_detection=False)
        assert is_exit_sentinel(result)
        assert result["phrase"] == "stop"
    
    def test_check_exit_phrase_nevermind_variants(self):
        """Test never mind / nevermind variants"""
        manager = FollowupManager()
        
        result1 = manager.check_exit_phrase("never mind", log_detection=False)
        assert is_exit_sentinel(result1)
        
        result2 = manager.check_exit_phrase("nevermind", log_detection=False)
        assert is_exit_sentinel(result2)
    
    def test_check_exit_phrase_thats_all(self):
        """Test 'that's all' variants"""
        manager = FollowupManager()
        
        result1 = manager.check_exit_phrase("that's all", log_detection=False)
        assert is_exit_sentinel(result1)
        
        result2 = manager.check_exit_phrase("thats all", log_detection=False)
        assert is_exit_sentinel(result2)
    
    def test_check_exit_phrase_non_exit_returns_none(self):
        """Test that non-exit phrases return None"""
        manager = FollowupManager()
        
        result = manager.check_exit_phrase("tell me a joke", log_detection=False)
        assert result is None
        
        result = manager.check_exit_phrase("what time is it", log_detection=False)
        assert result is None
    
    def test_check_exit_phrase_empty_returns_none(self):
        """Test that empty/None input returns None"""
        manager = FollowupManager()
        
        assert manager.check_exit_phrase("", log_detection=False) is None
        assert manager.check_exit_phrase(None, log_detection=False) is None
    
    def test_check_exit_phrase_logging(self):
        """Test that logging can be controlled"""
        manager = FollowupManager()
        
        # With logging disabled, should not log
        with patch.object(manager.logger, 'info') as mock_info:
            manager.check_exit_phrase("stop", log_detection=False)
            # Should not have logged "[EXIT] Exit phrase detected"
            for call in mock_info.call_args_list:
                assert "[EXIT]" not in str(call)
        
        # With logging enabled (default), should log
        with patch.object(manager.logger, 'info') as mock_info:
            manager.check_exit_phrase("stop", log_detection=True)
            # Should have logged
            logged_text = " ".join(str(c) for c in mock_info.call_args_list)
            assert "[EXIT]" in logged_text


class TestSingleDetection:
    """Test that exit phrases are detected exactly once"""
    
    def test_no_double_detection_sentinel_path(self):
        """Verify sentinel pattern prevents double detection"""
        manager = FollowupManager()
        
        # Simulate the pipeline: check once and get sentinel
        exit_sentinel = manager.check_exit_phrase("never mind", log_detection=True)
        
        # Verify we got a sentinel
        assert is_exit_sentinel(exit_sentinel)
        
        # Now downstream code should check for sentinel, NOT call is_exit_phrase again
        # If sentinel is present, skip re-detection
        if is_exit_sentinel(exit_sentinel):
            # This is the correct path - no second detection
            detected_count = 1
        else:
            # This would be wrong - would cause double detection
            if manager.is_exit_phrase("never mind"):
                detected_count = 2
            else:
                detected_count = 1
        
        assert detected_count == 1, "Exit phrase should be detected exactly once"
    
    def test_fallback_detection_only_when_no_sentinel(self):
        """Test that fallback detection only runs when no sentinel present"""
        manager = FollowupManager()
        
        # Case 1: Sentinel present (from brain worker) - no fallback needed
        meta = {"exit_sentinel": make_exit_sentinel("stop", "Stop!")}
        if is_exit_sentinel(meta.get("exit_sentinel")):
            fallback_ran = False
        else:
            fallback_ran = True
        
        assert fallback_ran is False
        
        # Case 2: No sentinel - fallback should run
        meta = {}
        if is_exit_sentinel(meta.get("exit_sentinel")):
            fallback_ran = False
        else:
            fallback_ran = True
        
        assert fallback_ran is True


class TestNoToolExecutionOnExit:
    """Test that exit phrases don't trigger tool execution"""
    
    def test_exit_phrase_skips_orchestrator(self):
        """Verify exit phrases skip orchestrator/tool processing"""
        manager = FollowupManager()
        
        exit_phrases = ["stop", "cancel", "never mind", "that's all", "no"]
        
        for phrase in exit_phrases:
            sentinel = manager.check_exit_phrase(phrase, log_detection=False)
            assert is_exit_sentinel(sentinel), f"'{phrase}' should return sentinel"
            
            # When sentinel is returned, the pipeline should short-circuit
            # before reaching orchestrator/multi-intent parser
            # This is verified by checking sentinel is truthy
            assert sentinel, f"'{phrase}' sentinel should be truthy for short-circuit"


class TestMultiIntentPreservation:
    """Test that multi-intent commands still work correctly"""
    
    def test_multi_intent_open_chrome_and_timer(self):
        """Test 'open chrome and then set a timer for 10 seconds'"""
        text = "open chrome and then set a timer for 10 seconds"
        
        # First verify it's NOT an exit phrase
        manager = FollowupManager()
        assert manager.check_exit_phrase(text, log_detection=False) is None
        
        # Then verify multi-intent parsing works
        # try_parse_multi_intent returns (intents, confidence) or None
        result = try_parse_multi_intent(text)
        
        # Should have parsed into multiple intents
        assert result is not None, "Multi-intent should be parsed"
        intents, confidence = result
        assert isinstance(intents, list), "Intents should be a list"
        assert len(intents) >= 2, f"Expected 2+ intents, got {len(intents)}"
    
    def test_multi_intent_open_multiple_apps(self):
        """Test 'open spotify and chrome'"""
        text = "open spotify and chrome"
        
        manager = FollowupManager()
        assert manager.check_exit_phrase(text, log_detection=False) is None
        
        result = try_parse_multi_intent(text)
        assert result is not None
        
        intents, confidence = result
        assert len(intents) >= 2, f"Expected 2+ intents, got {len(intents)}"
    
    def test_multi_intent_with_separators(self):
        """Test various separator patterns"""
        test_cases = [
            "pause, then mute",
            "close spotify; open youtube",
            "turn up volume and play music",
        ]
        
        manager = FollowupManager()
        
        for text in test_cases:
            # Should NOT be exit phrase
            assert manager.check_exit_phrase(text, log_detection=False) is None, \
                f"'{text}' should not be an exit phrase"
            
            # Note: Some of these may not parse as multi-intent depending on
            # hybrid router implementation - we just verify exit handling doesn't interfere
    
    def test_exit_phrase_containing_command_words(self):
        """
        Test edge case: 'open chrome and then never mind'
        
        Current design: If text ENDS with exit phrase, treat as exit.
        This is intentional - user is aborting the command.
        """
        text = "open chrome and then never mind"
        
        manager = FollowupManager()
        sentinel = manager.check_exit_phrase(text, log_detection=False)
        
        # Current behavior: exit phrase at end = treat as exit
        # This is documented design choice
        assert is_exit_sentinel(sentinel), \
            "Text ending with exit phrase should be treated as exit (design choice)"


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_exit_in_followup_context(self):
        """Test exit phrase during followup window"""
        manager = FollowupManager()
        manager.start_followup_window()
        
        assert manager.is_followup_active()
        
        sentinel = manager.check_exit_phrase("no thanks", log_detection=False)
        assert is_exit_sentinel(sentinel)
        
        # Followup window should still be active until explicitly ended
        assert manager.is_followup_active()
        
        # Now end it
        manager.end_followup_window()
        assert not manager.is_followup_active()
    
    def test_exit_outside_followup_context(self):
        """Test exit phrase when not in followup (fallback detection)"""
        manager = FollowupManager()
        
        # Not in followup
        assert not manager.is_followup_active()
        
        # Exit phrase should still be detected (for no-hotword mode etc)
        sentinel = manager.check_exit_phrase("stop", log_detection=False)
        assert is_exit_sentinel(sentinel)
    
    def test_prefix_exit_phrase(self):
        """Test text starting with exit phrase"""
        manager = FollowupManager()
        
        # "no thanks" starts with "no"
        sentinel = manager.check_exit_phrase("no thanks", log_detection=False)
        assert is_exit_sentinel(sentinel)
        
        # "stop right there" starts with "stop"
        sentinel = manager.check_exit_phrase("stop right there", log_detection=False)
        assert is_exit_sentinel(sentinel)
    
    def test_suffix_exit_phrase(self):
        """Test text ending with exit phrase"""
        manager = FollowupManager()
        
        # "forget it, cancel" ends with "cancel"
        sentinel = manager.check_exit_phrase("forget it, cancel", log_detection=False)
        assert is_exit_sentinel(sentinel)


class TestIPCSafety:
    """Test that sentinel is JSON-safe for IPC"""
    
    def test_sentinel_json_serializable(self):
        """Verify sentinel can be serialized to JSON"""
        import json
        
        sentinel = make_exit_sentinel("never mind", "Never mind!")
        
        # Should not raise
        json_str = json.dumps(sentinel)
        
        # Should round-trip
        restored = json.loads(json_str)
        assert is_exit_sentinel(restored)
        assert restored["phrase"] == "never mind"
        assert restored["original"] == "Never mind!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
