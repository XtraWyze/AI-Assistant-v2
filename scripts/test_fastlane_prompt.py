"""
Unit tests for fast-lane prompt optimization.

Tests cover:
1. Identity query detection
2. Identity query uses max_tokens <= 24
3. Voice_fast prompt is minimal (est_tokens <= 150)
4. Fast-lane prompt builder output
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestIdentityQueryDetection(unittest.TestCase):
    """Test identity query detection for ultra-short responses."""
    
    def test_whats_my_name_is_identity_query(self):
        """'What's my name?' should be detected as identity query."""
        from wyzer.brain.llm_engine import _is_identity_query
        
        identity_queries = [
            "what's my name",
            "What's my name?",
            "what is my name",
            "What is my name?",
            "who am i",
            "Who am I?",
            # STT variants without apostrophe
            "whats my name",
            "Whats my name",
            "whats my name?",
        ]
        
        for query in identity_queries:
            self.assertTrue(
                _is_identity_query(query),
                f"Expected '{query}' to be detected as identity query"
            )
    
    def test_family_queries_are_identity(self):
        """Family-related queries should be identity queries."""
        from wyzer.brain.llm_engine import _is_identity_query
        
        family_queries = [
            "what's my wife's name",
            "what is my husband's name",
            "what's my dog's name",
            "what is my cat's name",
            "what is my pet's name",
        ]
        
        for query in family_queries:
            self.assertTrue(
                _is_identity_query(query),
                f"Expected '{query}' to be detected as identity query"
            )
    
    def test_location_birthday_are_identity(self):
        """Location and birthday queries are identity queries."""
        from wyzer.brain.llm_engine import _is_identity_query
        
        queries = [
            "when's my birthday",
            "what's my birthday",
            "where do i live",
            "what's my address",
            "what is my city",
        ]
        
        for query in queries:
            self.assertTrue(
                _is_identity_query(query),
                f"Expected '{query}' to be detected as identity query"
            )
    
    def test_non_identity_queries_not_detected(self):
        """General questions should NOT be identity queries."""
        from wyzer.brain.llm_engine import _is_identity_query
        
        non_identity_queries = [
            "what's the weather",
            "open chrome",
            "tell me a story",
            "what time is it",
            "who is the president",
            "what's 2 + 2",
            "tell me something",
        ]
        
        for query in non_identity_queries:
            self.assertFalse(
                _is_identity_query(query),
                f"Expected '{query}' to NOT be detected as identity query"
            )


class TestIdentityQueryMaxTokens(unittest.TestCase):
    """Test that identity queries use ultra-low max_tokens."""
    
    def test_identity_query_uses_max_tokens_12(self):
        """'What's my name?' should use max_tokens=12 for ultra-fast identity response."""
        from wyzer.brain.llm_engine import get_voice_fast_options
        
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            options = get_voice_fast_options("what's my name", "llamacpp")
            
            self.assertEqual(options.get("num_predict"), 12)
            self.assertTrue(options.get("_use_fastlane_prompt", False))
    
    def test_smalltalk_uses_max_tokens_16(self):
        """'Tell me something' should use max_tokens=16 for ultra-short smalltalk."""
        from wyzer.brain.llm_engine import get_voice_fast_options
        
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            options = get_voice_fast_options("tell me something", "llamacpp")
            
            self.assertEqual(options.get("num_predict"), 16)
            self.assertTrue(options.get("_use_fastlane_prompt", False))
    
    def test_normal_query_uses_default_max_tokens(self):
        """Normal queries should use default max_tokens (64)."""
        from wyzer.brain.llm_engine import get_voice_fast_options
        
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            options = get_voice_fast_options("what's the weather", "llamacpp")
            
            self.assertEqual(options.get("num_predict"), 64)
            # Normal queries should NOT use fastlane prompt
            self.assertFalse(options.get("_use_fastlane_prompt", False))


class TestFastLanePromptBuilder(unittest.TestCase):
    """Test fast-lane prompt builder produces minimal prompts."""
    
    def test_fastlane_prompt_under_150_tokens_no_memories(self):
        """Fast-lane prompt for 'what's my name' with no memories <= 150 tokens."""
        from wyzer.brain.prompt_builder import build_fastlane_prompt, estimate_tokens
        
        prompt, mode, stats = build_fastlane_prompt(
            user_text="what's my name",
            memories_context="",
        )
        
        self.assertEqual(mode, "fastlane")
        self.assertLessEqual(stats["tokens_est"], 150)
        self.assertEqual(stats["mem_chars"], 0)
        
        # Double-check with our own estimate
        actual_tokens = estimate_tokens(prompt)
        self.assertLessEqual(actual_tokens, 150)
    
    def test_fastlane_prompt_with_short_memory_under_150_tokens(self):
        """Fast-lane prompt with short memory still <= 150 tokens."""
        from wyzer.brain.prompt_builder import build_fastlane_prompt, estimate_tokens
        
        prompt, mode, stats = build_fastlane_prompt(
            user_text="what's my name",
            memories_context="- name: Your name is Levi",
        )
        
        self.assertEqual(mode, "fastlane")
        self.assertLessEqual(stats["tokens_est"], 150)
        self.assertGreater(stats["mem_chars"], 0)
        
        # Memory should be in prompt
        self.assertIn("Levi", prompt)
    
    def test_fastlane_prompt_shorter_than_normal(self):
        """Fast-lane prompt should be much shorter than normal prompt."""
        from wyzer.brain.prompt_builder import (
            build_fastlane_prompt, 
            build_llm_prompt,
            estimate_tokens
        )
        
        user_text = "what's my name"
        memories = "- name: Your name is John"
        
        # Build fastlane prompt
        fastlane_prompt, _, fastlane_stats = build_fastlane_prompt(
            user_text=user_text,
            memories_context=memories,
        )
        
        # Build normal prompt
        normal_prompt, _ = build_llm_prompt(
            user_text=user_text,
            memories_context=memories,
        )
        
        fastlane_tokens = estimate_tokens(fastlane_prompt)
        normal_tokens = estimate_tokens(normal_prompt)
        
        # Fastlane should be at least 50% smaller
        self.assertLess(
            fastlane_tokens,
            normal_tokens * 0.5,
            f"Fastlane ({fastlane_tokens} tokens) should be much smaller than normal ({normal_tokens} tokens)"
        )
    
    def test_fastlane_prompt_has_wyzer_identity(self):
        """Fast-lane prompt should include Wyzer identity."""
        from wyzer.brain.prompt_builder import build_fastlane_prompt
        
        prompt, _, _ = build_fastlane_prompt(
            user_text="who am i",
            memories_context="",
        )
        
        self.assertIn("Wyzer", prompt)
    
    def test_fastlane_prompt_has_user_text(self):
        """Fast-lane prompt should include user's question."""
        from wyzer.brain.prompt_builder import build_fastlane_prompt
        
        prompt, _, _ = build_fastlane_prompt(
            user_text="what's my dog's name",
            memories_context="",
        )
        
        self.assertIn("what's my dog's name", prompt)


class TestFastLaneIntegration(unittest.TestCase):
    """Integration tests for fast-lane prompt flow."""
    
    def test_fastlane_system_prompt_is_minimal(self):
        """Fast-lane system prompt should be very short."""
        from wyzer.brain.prompt_builder import FASTLANE_SYSTEM_PROMPT, estimate_tokens
        
        tokens = estimate_tokens(FASTLANE_SYSTEM_PROMPT)
        
        # System prompt should be under 30 tokens
        self.assertLess(tokens, 30)
        
        # Should contain core identity
        self.assertIn("Wyzer", FASTLANE_SYSTEM_PROMPT)
        
        # Should emphasize brevity
        self.assertIn("short", FASTLANE_SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    print("Running fast-lane prompt tests...")
    unittest.main(verbosity=2)
