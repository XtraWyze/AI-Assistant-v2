"""
Unit tests for voice-fast preset and related optimizations.

Tests cover:
1. Voice-fast preset detection (story vs normal vs smalltalk)
2. Memory injection filtering for identity queries
3. Smalltalk memory gating
4. Topic-gating for likes/preferences memories
5. First-emit TTS optimization
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVoiceFastPreset(unittest.TestCase):
    """Test voice-fast preset detection and options."""
    
    def test_story_request_uses_higher_max_tokens(self):
        """'Tell me a story' should use story mode with higher max_tokens."""
        from wyzer.brain.llm_engine import _is_story_creative_request, get_voice_fast_options
        
        # These should be detected as story/creative requests
        story_queries = [
            "tell me a story",
            "Tell me a story about a dragon",
            "write a poem",
            "write me a poem about love",
            "compose a song",
            "continue the story",
            "write a detailed explanation",
        ]
        
        for query in story_queries:
            self.assertTrue(
                _is_story_creative_request(query),
                f"Expected '{query}' to be detected as story/creative request"
            )
        
        # Test that story mode returns higher max_tokens
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            options = get_voice_fast_options("tell me a story", "llamacpp")
            self.assertEqual(options.get("num_predict"), 320)
    
    def test_normal_query_uses_low_max_tokens(self):
        """Normal Q&A should use voice_fast mode with low max_tokens."""
        from wyzer.brain.llm_engine import _is_story_creative_request, get_voice_fast_options
        
        # These should NOT be detected as story/creative requests
        normal_queries = [
            "what's the weather",
            "what time is it",
            "how are you",
            "what's my name",
            "turn on the lights",
        ]
        
        for query in normal_queries:
            self.assertFalse(
                _is_story_creative_request(query),
                f"Expected '{query}' to NOT be detected as story/creative request"
            )
        
        # Test that normal mode returns lower max_tokens
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            options = get_voice_fast_options("what's the weather", "llamacpp")
            self.assertEqual(options.get("num_predict"), 64)
    
    def test_voice_fast_only_applies_to_llamacpp_by_default(self):
        """Voice-fast preset should only apply to llamacpp mode by default."""
        from wyzer.brain.llm_engine import get_voice_fast_options
        
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            with patch.dict(os.environ, {"WYZER_VOICE_FAST": "auto"}):
                # llamacpp mode should get voice_fast options
                options_llamacpp = get_voice_fast_options("hello", "llamacpp")
                self.assertIn("num_predict", options_llamacpp)
                
                # ollama mode with "auto" should NOT get voice_fast options
                options_ollama = get_voice_fast_options("hello", "ollama")
                self.assertEqual(options_ollama, {})

    def test_smalltalk_request_detected(self):
        """'Tell me something' should be detected as smalltalk."""
        from wyzer.brain.llm_engine import _is_smalltalk_request
        
        smalltalk_queries = [
            "tell me something",
            "Tell me something.",
            "say something",
            "talk to me",
            "what's up",
            "what's up?",
            "how are you",
            "how are you?",
            "hello",
            "hi",
            "hey",
        ]
        
        for query in smalltalk_queries:
            self.assertTrue(
                _is_smalltalk_request(query),
                f"Expected '{query}' to be detected as smalltalk"
            )
    
    def test_non_smalltalk_not_detected(self):
        """'Tell me something about X' should NOT be detected as smalltalk."""
        from wyzer.brain.llm_engine import _is_smalltalk_request
        
        non_smalltalk_queries = [
            "tell me something about python",
            "tell me something interesting",
            "Tell me something about One Piece",
            "what's up with the weather",
            "how are you doing today",
            "hello there how are you",
            "tell me a story",  # This is creative, not smalltalk
            "what's my name",
        ]
        
        for query in non_smalltalk_queries:
            self.assertFalse(
                _is_smalltalk_request(query),
                f"Expected '{query}' to NOT be detected as smalltalk"
            )
    
    def test_smalltalk_uses_strict_max_tokens(self):
        """'Tell me something' should use max_tokens <= 64 and include directive."""
        from wyzer.brain.llm_engine import get_voice_fast_options, SMALLTALK_SYSTEM_DIRECTIVE
        
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            options = get_voice_fast_options("tell me something", "llamacpp")
            
            # Smalltalk should use max_tokens <= 64 (we set it to 48)
            self.assertLessEqual(options.get("num_predict"), 64)
            
            # Should include the "no follow-up questions" directive
            self.assertEqual(
                options.get("_smalltalk_directive"),
                SMALLTALK_SYSTEM_DIRECTIVE
            )
            self.assertIn("follow-up", SMALLTALK_SYSTEM_DIRECTIVE.lower())


class TestMemoryInjectionFiltering(unittest.TestCase):
    """Test selective memory injection for identity queries."""
    
    def test_identity_query_filters_likes_memories(self):
        """'What's my name' should only inject name memory, not likes."""
        from wyzer.memory.memory_manager import (
            _get_identity_query_keys,
            _is_likes_preference_key
        )
        
        # Test identity query detection
        name_query = "what's my name"
        identity_keys = _get_identity_query_keys(name_query)
        self.assertIsNotNone(identity_keys)
        self.assertIn("name", identity_keys)
        
        # Test likes/preference key detection
        self.assertTrue(_is_likes_preference_key("likes_one_piece"))
        self.assertTrue(_is_likes_preference_key("favorite_anime"))
        self.assertTrue(_is_likes_preference_key("prefers_dark_mode"))
        
        # These should NOT be detected as likes/preference keys
        self.assertFalse(_is_likes_preference_key("name"))
        self.assertFalse(_is_likes_preference_key("wife"))
        self.assertFalse(_is_likes_preference_key("dog_name"))
    
    def test_dog_name_query_only_injects_dog_memory(self):
        """'What is my dog's name' should only inject dog-related memories."""
        from wyzer.memory.memory_manager import _get_identity_query_keys
        
        dog_queries = [
            "what's my dog's name",
            "what is my dog's name",
            "who is my dog",
        ]
        
        for query in dog_queries:
            identity_keys = _get_identity_query_keys(query)
            self.assertIsNotNone(identity_keys, f"Expected identity keys for '{query}'")
            # Should include dog-related keys
            self.assertTrue(
                any(k in identity_keys for k in ["dog", "dog_name", "pet", "pet_name"]),
                f"Expected dog-related keys for '{query}', got {identity_keys}"
            )
    
    def test_general_query_does_not_trigger_identity_filter(self):
        """General queries should not trigger identity-specific filtering."""
        from wyzer.memory.memory_manager import _get_identity_query_keys
        
        general_queries = [
            "what's the weather",
            "tell me a joke",
            "how does photosynthesis work",
            "open chrome",
        ]
        
        for query in general_queries:
            identity_keys = _get_identity_query_keys(query)
            self.assertIsNone(
                identity_keys,
                f"Expected no identity keys for general query '{query}'"
            )


class TestTopicGating(unittest.TestCase):
    """Test topic-gating for likes/preferences memories."""
    
    def test_topic_gate_keywords_for_one_piece(self):
        """likes_one_piece should be gated behind One Piece-related keywords."""
        from wyzer.memory.memory_manager import _get_topic_gate_keywords, _passes_topic_gate
        
        # Check that likes_one_piece has topic gate keywords
        keywords = _get_topic_gate_keywords("likes_one_piece")
        self.assertIsNotNone(keywords)
        self.assertIn("one piece", keywords)
        self.assertIn("anime", keywords)
        self.assertIn("luffy", keywords)
        
        # Check passes/fails topic gate
        self.assertTrue(_passes_topic_gate("likes_one_piece", "Tell me about One Piece"))
        self.assertTrue(_passes_topic_gate("likes_one_piece", "What anime should I watch?"))
        self.assertTrue(_passes_topic_gate("likes_one_piece", "Who is Luffy?"))
        self.assertFalse(_passes_topic_gate("likes_one_piece", "Tell me something"))
        self.assertFalse(_passes_topic_gate("likes_one_piece", "What's my name?"))
    
    def test_topic_gate_keywords_for_weed(self):
        """weed/dabs preferences should be gated behind substance keywords."""
        from wyzer.memory.memory_manager import _get_topic_gate_keywords, _passes_topic_gate
        
        # Check weed-related keys
        for key in ["likes_weed", "prefers_dabs", "favorite_dabs"]:
            keywords = _get_topic_gate_keywords(key)
            self.assertIsNotNone(keywords, f"Expected topic gate for {key}")
        
        # Check passes/fails topic gate
        self.assertTrue(_passes_topic_gate("likes_weed", "Do you smoke weed?"))
        self.assertTrue(_passes_topic_gate("likes_dabs", "Tell me about dabs"))
        self.assertTrue(_passes_topic_gate("prefers_dabs", "How do you smoke thc?"))
        self.assertFalse(_passes_topic_gate("likes_weed", "Tell me something"))
        self.assertFalse(_passes_topic_gate("likes_dabs", "What's my name?"))
    
    def test_non_gated_keys_always_pass(self):
        """Non-gated keys like 'name' should always pass topic gate."""
        from wyzer.memory.memory_manager import _passes_topic_gate
        
        # These keys should not be topic-gated
        non_gated_keys = ["name", "wife", "dog_name", "birthday", "favorite_color"]
        
        for key in non_gated_keys:
            self.assertTrue(
                _passes_topic_gate(key, "Tell me something"),
                f"Expected '{key}' to pass topic gate for generic query"
            )
            self.assertTrue(
                _passes_topic_gate(key, "What's my name?"),
                f"Expected '{key}' to pass topic gate for identity query"
            )


class TestSmalltalkMemoryGating(unittest.TestCase):
    """Test smalltalk detection for memory gating."""
    
    def test_smalltalk_detected_in_memory_manager(self):
        """Smalltalk should be detected using memory_manager's _is_smalltalk_request."""
        from wyzer.memory.memory_manager import _is_smalltalk_request
        
        smalltalk_queries = [
            "tell me something",
            "Tell me something.",
            "say something",
            "what's up",
            "how are you",
        ]
        
        for query in smalltalk_queries:
            self.assertTrue(
                _is_smalltalk_request(query),
                f"Expected '{query}' to be detected as smalltalk in memory_manager"
            )
    
    def test_tell_me_something_about_x_not_smalltalk(self):
        """'Tell me something about X' should NOT be smalltalk."""
        from wyzer.memory.memory_manager import _is_smalltalk_request
        
        queries = [
            "tell me something about One Piece",
            "tell me something about my dog",
            "tell me something interesting",
        ]
        
        for query in queries:
            self.assertFalse(
                _is_smalltalk_request(query),
                f"Expected '{query}' to NOT be smalltalk"
            )


class TestFirstEmitTTS(unittest.TestCase):
    """Test first-emit TTS optimization."""
    
    def test_first_emit_triggers_on_early_punctuation(self):
        """First segment should emit on first punctuation within threshold."""
        from wyzer.brain.stream_tts import ChunkBuffer
        
        buffer = ChunkBuffer(min_chars=150, first_emit_chars=24)
        segments = []
        
        # Simulate streaming "Hi there! How can I help?"
        tokens = ["Hi", " ", "there", "!", " ", "How", " ", "can", " ", "I", " ", "help", "?"]
        
        for token in tokens:
            segment = buffer.add(token)
            if segment:
                segments.append(segment)
        
        # First segment should emit on "!" before waiting for full sentence
        # The first emit should happen early (on the first punctuation)
        self.assertGreaterEqual(len(segments), 1, "Should have at least 1 segment from first-emit")
        self.assertEqual(segments[0], "Hi there!")
    
    def test_first_emit_respects_char_threshold(self):
        """First-emit should consider the character threshold."""
        from wyzer.brain.stream_tts import ChunkBuffer
        
        buffer = ChunkBuffer(min_chars=150, first_emit_chars=24)
        segments = []
        
        # Very short text with punctuation - should still emit because of punctuation
        tokens = ["OK", "."]
        
        for token in tokens:
            segment = buffer.add(token)
            if segment:
                segments.append(segment)
        
        # Flush to get remaining
        final = buffer.flush()
        if final:
            segments.append(final)
        
        # Should have emitted "OK." either via first-emit or flush
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], "OK.")
    
    def test_first_emit_only_applies_once(self):
        """First-emit optimization should only apply to the first segment."""
        from wyzer.brain.stream_tts import ChunkBuffer
        
        buffer = ChunkBuffer(min_chars=150, first_emit_chars=24)
        
        # Feed first sentence
        for char in "Hello! ":
            buffer.add(char)
        
        # First emit should have triggered
        self.assertTrue(buffer._first_emitted, "First segment should have been emitted")
        
        # Now the buffer should use normal rules for subsequent segments
        # Feed more text without punctuation
        for char in "This is a longer text without punctuation":
            result = buffer.add(char)
            # Should not emit until min_chars is reached
            if len(buffer._buffer) < buffer.min_chars:
                self.assertIsNone(result, "Should not emit before min_chars after first emit")


if __name__ == "__main__":
    unittest.main()
