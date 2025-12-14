"""
Unit tests for the FOLLOWUP listening window manager.
Tests exit phrase detection, timeout behavior, and chain counting.
"""
import time
import pytest
from wyzer.core.followup_manager import FollowupManager
from wyzer.core.config import Config


class TestExitPhraseDetection:
    """Test exit phrase normalization and matching"""
    
    def test_exact_exit_phrase(self):
        """Test exact match of exit phrase"""
        manager = FollowupManager()
        assert manager.is_exit_phrase("no") is True
        assert manager.is_exit_phrase("nope") is True
        assert manager.is_exit_phrase("stop") is True
        assert manager.is_exit_phrase("cancel") is True
    
    def test_exit_phrase_case_insensitive(self):
        """Test that exit phrases are case-insensitive"""
        manager = FollowupManager()
        assert manager.is_exit_phrase("NO") is True
        assert manager.is_exit_phrase("No") is True
        assert manager.is_exit_phrase("NOPE") is True
        assert manager.is_exit_phrase("Stop") is True
    
    def test_exit_phrase_with_punctuation(self):
        """Test exit phrases with punctuation"""
        manager = FollowupManager()
        assert manager.is_exit_phrase("no.") is True
        assert manager.is_exit_phrase("nope!") is True
        assert manager.is_exit_phrase("stop,") is True
        assert manager.is_exit_phrase("No?") is True
    
    def test_thats_all_variants(self):
        """Test variants of 'that's all' phrase"""
        manager = FollowupManager()
        assert manager.is_exit_phrase("that's all") is True
        assert manager.is_exit_phrase("thats all") is True
        assert manager.is_exit_phrase("THAT'S ALL") is True
        assert manager.is_exit_phrase("That's All") is True
    
    def test_nevermind_variants(self):
        """Test variants of 'nevermind' phrase"""
        manager = FollowupManager()
        assert manager.is_exit_phrase("never mind") is True
        assert manager.is_exit_phrase("nevermind") is True
        assert manager.is_exit_phrase("NEVER MIND") is True
    
    def test_nothing_else(self):
        """Test 'nothing else' phrase"""
        manager = FollowupManager()
        assert manager.is_exit_phrase("nothing else") is True
        assert manager.is_exit_phrase("Nothing Else") is True
    
    def test_all_good(self):
        """Test 'all good' phrase"""
        manager = FollowupManager()
        assert manager.is_exit_phrase("all good") is True
        assert manager.is_exit_phrase("All Good") is True
    
    def test_substring_matching(self):
        """Test that exit phrases match as substrings"""
        manager = FollowupManager()
        # "no" appears in the middle
        assert manager.is_exit_phrase("yeah no") is True
        # "stop" appears in longer sentence
        assert manager.is_exit_phrase("ok stop now") is True
    
    def test_non_exit_phrases(self):
        """Test that non-exit phrases are not detected"""
        manager = FollowupManager()
        assert manager.is_exit_phrase("yes please") is False
        assert manager.is_exit_phrase("tell me more") is False
        assert manager.is_exit_phrase("go on") is False
        assert manager.is_exit_phrase("what's next") is False
    
    def test_empty_string(self):
        """Test empty string"""
        manager = FollowupManager()
        assert manager.is_exit_phrase("") is False
        assert manager.is_exit_phrase(None) is False


class TestTimeoutBehavior:
    """Test FOLLOWUP timeout mechanisms"""
    
    def test_start_followup_window(self):
        """Test starting a FOLLOWUP window"""
        manager = FollowupManager()
        manager.start_followup_window()
        assert manager.is_followup_active() is True
    
    def test_timeout_after_silence(self):
        """Test that timeout triggers after silence"""
        # Save original timeout
        original_timeout = Config.FOLLOWUP_TIMEOUT_SEC
        try:
            # Set short timeout for testing
            Config.FOLLOWUP_TIMEOUT_SEC = 0.1
            
            manager = FollowupManager()
            manager.start_followup_window()
            
            # Should not timeout immediately
            assert manager.check_timeout() is False
            
            # Wait for timeout
            time.sleep(0.15)
            
            # Now should timeout
            assert manager.check_timeout() is True
            assert manager.is_followup_active() is False
        finally:
            # Restore original timeout
            Config.FOLLOWUP_TIMEOUT_SEC = original_timeout
    
    def test_reset_speech_timer(self):
        """Test that speech detection resets the timer"""
        original_timeout = Config.FOLLOWUP_TIMEOUT_SEC
        try:
            Config.FOLLOWUP_TIMEOUT_SEC = 0.2
            
            manager = FollowupManager()
            manager.start_followup_window()
            
            # Sleep briefly but not enough to timeout
            time.sleep(0.1)
            
            # Reset timer
            manager.reset_speech_timer()
            
            # Should still not timeout (timer reset)
            assert manager.check_timeout() is False
            
            # Sleep again
            time.sleep(0.15)
            
            # Now should timeout
            assert manager.check_timeout() is True
        finally:
            Config.FOLLOWUP_TIMEOUT_SEC = original_timeout
    
    def test_remaining_time(self):
        """Test getting remaining time in window"""
        original_timeout = Config.FOLLOWUP_TIMEOUT_SEC
        try:
            Config.FOLLOWUP_TIMEOUT_SEC = 1.0
            
            manager = FollowupManager()
            manager.start_followup_window()
            
            # Check remaining time
            remaining = manager.get_remaining_time()
            assert 0 < remaining <= 1.0
            
            # After timeout
            manager.end_followup_window()
            remaining = manager.get_remaining_time()
            assert remaining == 0.0
        finally:
            Config.FOLLOWUP_TIMEOUT_SEC = original_timeout


class TestChainBehavior:
    """Test FOLLOWUP chain counting and limiting"""
    
    def test_chain_counter_increment(self):
        """Test chain counter increments"""
        manager = FollowupManager()
        assert manager.get_chain_count() == 0
        
        manager.increment_chain()
        assert manager.get_chain_count() == 1
        
        manager.increment_chain()
        assert manager.get_chain_count() == 2
    
    def test_max_chain_limit(self):
        """Test that max chain limit is enforced"""
        original_max = Config.FOLLOWUP_MAX_CHAIN
        try:
            Config.FOLLOWUP_MAX_CHAIN = 2
            
            manager = FollowupManager()
            manager.start_followup_window()
            
            # First increment OK
            assert manager.increment_chain() is True
            assert manager.get_chain_count() == 1
            
            # Second increment OK
            assert manager.increment_chain() is True
            assert manager.get_chain_count() == 2
            
            # Third increment fails (exceeds max)
            assert manager.increment_chain() is False
            assert manager.is_followup_active() is False
        finally:
            Config.FOLLOWUP_MAX_CHAIN = original_max
    
    def test_chain_reset_on_end(self):
        """Test that chain counter resets when window ends"""
        manager = FollowupManager()
        manager.start_followup_window()
        
        manager.increment_chain()
        manager.increment_chain()
        assert manager.get_chain_count() == 2
        
        manager.end_followup_window()
        assert manager.get_chain_count() == 0


class TestNormalization:
    """Test text normalization for exit phrase matching"""
    
    def test_normalize_lowercase(self):
        """Test lowercasing in normalization"""
        normalized = FollowupManager._normalize_text("HELLO WORLD")
        assert normalized == "hello world"
    
    def test_normalize_punctuation_removal(self):
        """Test punctuation removal"""
        normalized = FollowupManager._normalize_text("Hello, world!")
        assert normalized == "hello world"
    
    def test_normalize_multiple_spaces(self):
        """Test multiple space collapse"""
        normalized = FollowupManager._normalize_text("hello   world")
        assert normalized == "hello world"
    
    def test_normalize_strip_whitespace(self):
        """Test strip leading/trailing whitespace"""
        normalized = FollowupManager._normalize_text("  hello world  ")
        assert normalized == "hello world"
    
    def test_normalize_combined(self):
        """Test combined normalization"""
        normalized = FollowupManager._normalize_text("  HELLO,   World!  ")
        assert normalized == "hello world"


class TestIntegration:
    """Integration tests combining multiple features"""
    
    def test_full_followup_flow(self):
        """Test complete FOLLOWUP flow"""
        original_timeout = Config.FOLLOWUP_TIMEOUT_SEC
        try:
            Config.FOLLOWUP_TIMEOUT_SEC = 0.3
            
            manager = FollowupManager()
            
            # Start FOLLOWUP
            manager.start_followup_window()
            assert manager.is_followup_active() is True
            
            # User says non-exit phrase
            transcript = "tell me a joke"
            assert manager.is_exit_phrase(transcript) is False
            
            # Reset timer on speech
            manager.reset_speech_timer()
            
            # Chain once
            assert manager.increment_chain() is True
            
            # User says exit phrase
            transcript = "that's all"
            assert manager.is_exit_phrase(transcript) is True
            
            # End window
            manager.end_followup_window()
            assert manager.is_followup_active() is False
            assert manager.get_chain_count() == 0
        finally:
            Config.FOLLOWUP_TIMEOUT_SEC = original_timeout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
