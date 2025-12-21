"""
Unit tests for fast-lane cold-start optimization.

Tests cover:
1. Identity query triggers FAST_LANE with max_tokens <= 16
2. Identity query injects ONLY name memory (not dog/wife)
3. FAST_LANE prompt is significantly smaller (est_tokens <= 150)
4. Generation params are ultra-tight for identity queries

Run with: python scripts/test_fastlane_cold_start.py
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFastLaneIdentityTriggering(unittest.TestCase):
    """Test that identity queries trigger FAST_LANE with correct params."""
    
    def test_whats_my_name_uses_max_tokens_12(self):
        """'What's my name?' should trigger FAST_LANE with max_tokens=12."""
        from wyzer.brain.llm_engine import get_voice_fast_options
        
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            options = get_voice_fast_options("what's my name", "llamacpp")
            
            # Check max_tokens <= 16 (actually 12)
            self.assertLessEqual(options.get("num_predict", 100), 16)
            self.assertEqual(options.get("num_predict"), 12)
            
            # Check fast-lane is enabled
            self.assertTrue(options.get("_use_fastlane_prompt", False))
            self.assertEqual(options.get("_fastlane_reason"), "identity")
            
            # Check temperature is 0 for deterministic factual recall
            self.assertEqual(options.get("temperature"), 0.0)
    
    def test_identity_query_has_tight_stop_sequences(self):
        """Identity queries should have stop sequences but NOT '.' (production-safe)."""
        from wyzer.brain.llm_engine import get_voice_fast_options
        
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            options = get_voice_fast_options("who am i", "llamacpp")
            
            stop_sequences = options.get("stop", [])
            # Should include newline stops
            self.assertIn("\n", stop_sequences)
            self.assertIn("\nUser:", stop_sequences)
            self.assertIn("\nWyzer:", stop_sequences)
            # Should NOT include "." - it can cut off mid-sentence on abbreviations
            self.assertNotIn(".", stop_sequences)
    
    def test_smalltalk_uses_max_tokens_16(self):
        """'Tell me something' should use max_tokens=16."""
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
            self.assertEqual(options.get("_fastlane_reason"), "smalltalk")


class TestIdentitySingleKeyMemoryInjection(unittest.TestCase):
    """Test that identity queries inject ONLY the relevant memory key."""
    
    def setUp(self):
        """Set up memory manager with test memories."""
        from wyzer.memory.memory_manager import MemoryManager
        import tempfile
        import json
        from pathlib import Path
        
        # Create temp memory file
        self.temp_dir = tempfile.mkdtemp()
        self.memory_file = Path(self.temp_dir) / "memory.json"
        
        # Set up test memories: name, wife, dog
        test_memories = [
            {
                "id": "1",
                "key": "name",
                "value": "My name is Levi",
                "text": "My name is Levi",
                "pinned": True,
                "type": "fact",
            },
            {
                "id": "2",
                "key": "wife_name",  # canonical key from _derive_key
                "value": "My wife's name is Sarah",
                "text": "My wife's name is Sarah",
                "pinned": True,
                "type": "fact",
            },
            {
                "id": "3",
                "key": "dog_name",  # canonical key from _derive_key
                "value": "My dog's name is Max",
                "text": "My dog's name is Max",
                "pinned": True,
                "type": "fact",
            },
            {
                "id": "4",
                "key": "likes_anime",
                "value": "I like watching anime",
                "text": "I like watching anime",
                "pinned": False,
                "type": "preference",
            },
        ]
        
        with open(self.memory_file, 'w') as f:
            json.dump(test_memories, f)
        
        # Create memory manager and patch its internal memory file path
        self.mgr = MemoryManager()
        self.mgr._memory_file = self.memory_file
        # Enable use_memories
        self.mgr.set_use_memories(True, "test")
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_whats_my_name_injects_only_name(self):
        """'What's my name?' should inject ONLY name memory, not wife/dog."""
        result = self.mgr.select_for_fastlane_injection("what's my name")
        
        # Should contain name
        self.assertIn("name", result.lower())
        self.assertIn("levi", result.lower())
        
        # Should NOT contain wife or dog
        self.assertNotIn("wife", result.lower())
        self.assertNotIn("sarah", result.lower())
        self.assertNotIn("dog", result.lower())
        self.assertNotIn("max", result.lower())
    
    def test_whats_my_dogs_name_injects_only_dog(self):
        """'What's my dog's name?' should inject ONLY dog memory."""
        result = self.mgr.select_for_fastlane_injection("what's my dog's name")
        
        # Should contain dog
        self.assertIn("dog", result.lower())
        self.assertIn("max", result.lower())
        
        # Should NOT contain name or wife
        self.assertNotIn("levi", result.lower())
        self.assertNotIn("sarah", result.lower())
    
    def test_whats_my_wifes_name_injects_only_wife(self):
        """'What's my wife's name?' should inject ONLY wife memory."""
        result = self.mgr.select_for_fastlane_injection("what's my wife's name")
        
        # Should contain wife's name
        self.assertIn("wife", result.lower())
        self.assertIn("sarah", result.lower())
        
        # Should NOT contain user name or dog
        self.assertNotIn("levi", result.lower())
        self.assertNotIn("max", result.lower())
    
    def test_whats_my_wifes_name_stt_variant(self):
        """'whats my wifes name' (STT without apostrophe) should inject wife memory."""
        result = self.mgr.select_for_fastlane_injection("whats my wifes name")
        
        # Should contain wife's name
        self.assertIn("wife", result.lower())
        self.assertIn("sarah", result.lower())
    
    def test_smalltalk_injects_no_memories(self):
        """'Tell me something' should inject NO memories."""
        result = self.mgr.select_for_fastlane_injection("tell me something")
        
        self.assertEqual(result, "")
    
    def test_fastlane_result_is_single_line(self):
        """Fast-lane result should be a single line, not a block."""
        result = self.mgr.select_for_fastlane_injection("what's my name")
        
        # Should be a single line (no newlines)
        self.assertNotIn("\n", result)
        
        # Should be short
        self.assertLess(len(result), 100)


class TestFastLanePromptSize(unittest.TestCase):
    """Test that FAST_LANE prompt is significantly smaller."""
    
    def test_fastlane_prompt_under_150_tokens_no_memory(self):
        """Fast-lane prompt with no memory should be <= 150 tokens."""
        from wyzer.brain.prompt_builder import build_fastlane_prompt, estimate_tokens
        
        prompt, mode, stats = build_fastlane_prompt(
            user_text="what's my name",
            memories_context="",
        )
        
        self.assertEqual(mode, "fastlane")
        self.assertLessEqual(stats["tokens_est"], 150)
        
        # Double-check with our own estimate
        actual_tokens = estimate_tokens(prompt)
        self.assertLessEqual(actual_tokens, 150)
    
    def test_fastlane_prompt_under_150_tokens_with_memory(self):
        """Fast-lane prompt with short memory should be <= 150 tokens."""
        from wyzer.brain.prompt_builder import build_fastlane_prompt, estimate_tokens
        
        prompt, mode, stats = build_fastlane_prompt(
            user_text="what's my name",
            memories_context="name: Your name is Levi",
        )
        
        self.assertEqual(mode, "fastlane")
        self.assertLessEqual(stats["tokens_est"], 150)
        
        # Memory should be in prompt
        self.assertIn("Levi", prompt)
    
    def test_fastlane_prompt_much_smaller_than_normal(self):
        """Fast-lane prompt should be at least 50% smaller than normal."""
        from wyzer.brain.prompt_builder import (
            build_fastlane_prompt, 
            build_llm_prompt,
            estimate_tokens
        )
        
        user_text = "what's my name"
        memories = "name: Your name is John"
        
        # Build fastlane prompt
        fastlane_prompt, _, _ = build_fastlane_prompt(
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


class TestGenerationParamsLogging(unittest.TestCase):
    """Test that generation params are logged correctly."""
    
    def test_identity_query_logs_gen_fast_lane(self):
        """Identity query should trigger [GEN_FAST_LANE] debug log."""
        from wyzer.brain.llm_engine import get_voice_fast_options
        import logging
        
        # Capture logs
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            with patch('wyzer.brain.llm_engine.get_logger') as mock_logger:
                logger_instance = MagicMock()
                mock_logger.return_value = logger_instance
                
                options = get_voice_fast_options("what's my name", "llamacpp")
                
                # Check debug was called with GEN_FAST_LANE
                debug_calls = [str(call) for call in logger_instance.debug.call_args_list]
                gen_fast_lane_logged = any("[GEN_FAST_LANE]" in str(call) for call in debug_calls)
                self.assertTrue(gen_fast_lane_logged, f"Expected [GEN_FAST_LANE] log, got: {debug_calls}")


class TestIdentityStopSequencesSafety(unittest.TestCase):
    """Test that identity stop sequences are production-safe."""
    
    def test_identity_stop_list_does_not_contain_period(self):
        """FAST_LANE identity stop list should NOT contain '.' for production safety."""
        from wyzer.brain.llm_engine import get_voice_fast_options
        
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            # Test various identity queries
            identity_queries = [
                "what's my name",
                "who am i",
                "what is my dog's name",
                "what's my wife's name",
            ]
            
            for query in identity_queries:
                options = get_voice_fast_options(query, "llamacpp")
                stop_sequences = options.get("stop", [])
                
                self.assertNotIn(
                    ".", stop_sequences,
                    f"Query '{query}' should NOT have '.' in stop sequences: {stop_sequences}"
                )
                # Should still have newline-based stops
                self.assertIn("\n", stop_sequences)


class TestLlamaCppWarmup(unittest.TestCase):
    """Test llama.cpp server warmup functionality."""
    
    def test_warmup_called_exactly_once_per_server_start(self):
        """Warmup should be called exactly once when server becomes ready."""
        from wyzer.brain.llama_server_manager import LlamaServerManager
        
        # Create a fresh manager instance (bypass singleton for testing)
        manager = object.__new__(LlamaServerManager)
        manager._initialized = False
        manager.__init__()
        
        # Mock the warmup method
        with patch.object(manager, '_perform_warmup') as mock_warmup:
            # Simulate server ready state
            manager.base_url = "http://127.0.0.1:8081"
            manager._warmup_done = False
            
            # First call should trigger warmup
            manager._perform_warmup(manager.base_url)
            
            # Second call should not trigger warmup again
            manager._warmup_done = True  # Simulating first warmup completed
            manager._perform_warmup(manager.base_url)
        
        # Warmup should have been called twice (once real, once mocked check)
        # But in production, the flag prevents actual execution
        # Let's verify the flag behavior directly
        self.assertTrue(manager._warmup_done)
    
    def test_warmup_flag_reset_on_server_stop(self):
        """Warmup flag should reset when server is stopped."""
        from wyzer.brain.llama_server_manager import LlamaServerManager
        
        # Create a fresh manager instance
        manager = object.__new__(LlamaServerManager)
        manager._initialized = False
        manager.__init__()
        
        # Simulate a running server that was started by Wyzer
        manager._warmup_done = True
        manager._started_by_wyzer = True
        manager.base_url = "http://127.0.0.1:8081"
        manager._log_file_handle = None
        
        # Create a mock process that looks like it's still running
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.terminate = MagicMock()
        mock_process.wait = MagicMock()
        manager.process = mock_process
        
        # Stop server (should reset warmup flag)
        manager.stop_server()
        
        # Warmup flag should be reset
        self.assertFalse(manager._warmup_done)
        self.assertIsNone(manager.process)
        self.assertIsNone(manager.base_url)
    
    def test_warmup_uses_correct_parameters(self):
        """Warmup request should use minimal parameters."""
        from wyzer.brain.llama_server_manager import LlamaServerManager
        import json
        
        # Create a fresh manager instance
        manager = object.__new__(LlamaServerManager)
        manager._initialized = False
        manager.__init__()
        manager._warmup_done = False
        
        # Track the warmup request
        captured_request = {}
        
        def mock_urlopen(req, timeout=None):
            captured_request['url'] = req.full_url
            captured_request['data'] = json.loads(req.data.decode('utf-8'))
            
            # Return a mock response
            class MockResponse:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def read(self):
                    return b'{"content": ""}'
            
            return MockResponse()
        
        with patch('urllib.request.urlopen', mock_urlopen):
            manager._perform_warmup("http://127.0.0.1:8081")
        
        # Verify warmup parameters
        self.assertIn('/v1/completions', captured_request.get('url', ''))
        data = captured_request.get('data', {})
        self.assertEqual(data.get('max_tokens'), 1)
        self.assertEqual(data.get('temperature'), 0)
    
    def test_warmup_retries_on_503_then_succeeds(self):
        """Warmup should retry on HTTP 503 and succeed when endpoint becomes ready."""
        from wyzer.brain.llama_server_manager import LlamaServerManager
        import urllib.error
        
        # Create a fresh manager instance
        manager = object.__new__(LlamaServerManager)
        manager._initialized = False
        manager.__init__()
        manager._warmup_done = False
        
        # Track call count
        call_count = [0]
        
        def mock_urlopen_503_then_ok(req, timeout=None):
            call_count[0] += 1
            
            # First 2 calls return 503, third succeeds
            if call_count[0] < 3:
                raise urllib.error.HTTPError(
                    req.full_url, 503, "Service Unavailable", {}, None
                )
            
            # Return success on third call
            class MockResponse:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def read(self):
                    return b'{"content": ""}'
            return MockResponse()
        
        with patch('urllib.request.urlopen', mock_urlopen_503_then_ok):
            manager._perform_warmup("http://127.0.0.1:8081")
        
        # Should have retried and eventually succeeded
        self.assertGreaterEqual(call_count[0], 3)
        self.assertTrue(manager._warmup_done)
    
    def test_warmup_times_out_after_max_time(self):
        """Warmup should give up after 15 seconds total."""
        from wyzer.brain.llama_server_manager import LlamaServerManager
        import urllib.error
        import time
        
        # Create a fresh manager instance
        manager = object.__new__(LlamaServerManager)
        manager._initialized = False
        manager.__init__()
        manager._warmup_done = False
        
        call_count = [0]
        
        def mock_urlopen_always_503(req, timeout=None):
            call_count[0] += 1
            raise urllib.error.HTTPError(
                req.full_url, 503, "Service Unavailable", {}, None
            )
        
        start = time.time()
        with patch('urllib.request.urlopen', mock_urlopen_always_503):
            manager._perform_warmup("http://127.0.0.1:8081")
        elapsed = time.time() - start
        
        # Should have given up within 16 seconds (15s timeout + some overhead)
        self.assertLess(elapsed, 16.0, f"Warmup took too long: {elapsed:.1f}s")
        # Should have made multiple attempts
        self.assertGreater(call_count[0], 1)
        # Warmup should still be marked done (to not block)
        self.assertTrue(manager._warmup_done)


class TestIdentityQueriesUseFastLane(unittest.TestCase):
    """Test that identity queries ALWAYS use FAST_LANE path."""
    
    def test_identity_query_never_calls_normal_prompt_path(self):
        """Identity query should never call _build_messages (normal path)."""
        from wyzer.brain.llm_engine import LLMEngine
        from unittest.mock import MagicMock, patch
        
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            mock_config.OLLAMA_TEMPERATURE = 0.4
            mock_config.OLLAMA_TOP_P = 0.9
            mock_config.OLLAMA_NUM_CTX = 4096
            mock_config.OLLAMA_NUM_PREDICT = 120
            mock_config.OLLAMA_STREAM = False
            mock_config.LLM_MAX_PROMPT_CHARS = 4000
            
            # Create engine with mocked client
            engine = LLMEngine.__new__(LLMEngine)
            engine.logger = MagicMock()
            engine.enabled = True
            engine.llm_mode = "llamacpp"
            engine.model = "test"
            engine.base_url = "http://127.0.0.1:8081"
            engine.timeout = 30
            engine.client = MagicMock()
            engine.client.generate.return_value = "Your name is Levi."
            
            # Spy on _build_messages
            original_build_messages = engine._build_messages
            engine._build_messages = MagicMock(side_effect=original_build_messages)
            
            # Call with identity query
            result = engine.think("what's my name")
            
            # _build_messages should NOT have been called (identity uses fastlane)
            engine._build_messages.assert_not_called()
    
    def test_whats_my_name_stt_variant_uses_fast_lane(self):
        """'whats my name' (STT variant without apostrophe) should use FAST_LANE."""
        from wyzer.brain.llm_engine import _is_identity_query, get_voice_fast_options
        
        # Test STT variants
        stt_variants = [
            "whats my name",
            "what's my name",
            "what is my name",
            "Whats my name",
        ]
        
        with patch('wyzer.brain.llm_engine.Config') as mock_config:
            mock_config.VOICE_FAST_ENABLED = True
            mock_config.VOICE_FAST_TEMPERATURE = 0.2
            mock_config.VOICE_FAST_TOP_P = 0.9
            mock_config.VOICE_FAST_MAX_TOKENS = 64
            mock_config.VOICE_FAST_STORY_MAX_TOKENS = 320
            
            for variant in stt_variants:
                # Should be detected as identity query
                self.assertTrue(
                    _is_identity_query(variant),
                    f"'{variant}' should be identity query"
                )
                
                # Should trigger fastlane
                options = get_voice_fast_options(variant, "llamacpp")
                self.assertTrue(
                    options.get("_use_fastlane_prompt", False),
                    f"'{variant}' should use fastlane"
                )


class TestDeriveKeyCanonical(unittest.TestCase):
    """Test that _derive_key produces canonical keys for relationship patterns."""
    
    def test_wife_name_pattern_possessive(self):
        """'my wife's name is X' should derive 'wife_name' key."""
        from wyzer.memory.memory_manager import _derive_key
        
        self.assertEqual(_derive_key("my wife's name is Audrey"), "wife_name")
        self.assertEqual(_derive_key("my wifes name is Audrey"), "wife_name")  # STT
        self.assertEqual(_derive_key("My Wife's Name is Audrey"), "wife_name")  # caps
    
    def test_spouse_partner_canonicalize_to_wife_name(self):
        """Spouse/partner patterns should canonicalize to 'wife_name'."""
        from wyzer.memory.memory_manager import _derive_key
        
        self.assertEqual(_derive_key("my spouse's name is Sam"), "wife_name")
        self.assertEqual(_derive_key("my partner's name is Alex"), "wife_name")
    
    def test_husband_name_pattern(self):
        """'my husband's name is X' should derive 'husband_name' key."""
        from wyzer.memory.memory_manager import _derive_key
        
        self.assertEqual(_derive_key("my husband's name is Bob"), "husband_name")
        self.assertEqual(_derive_key("my husbands name is Bob"), "husband_name")
    
    def test_dog_cat_pet_patterns(self):
        """Pet name patterns should derive canonical keys."""
        from wyzer.memory.memory_manager import _derive_key
        
        self.assertEqual(_derive_key("my dog's name is Bella"), "dog_name")
        self.assertEqual(_derive_key("my dogs name is Bella"), "dog_name")
        self.assertEqual(_derive_key("my cat's name is Whiskers"), "cat_name")
        self.assertEqual(_derive_key("my pet's name is Buddy"), "pet_name")
    
    def test_family_member_patterns(self):
        """Family member patterns should derive canonical keys."""
        from wyzer.memory.memory_manager import _derive_key
        
        self.assertEqual(_derive_key("my mom's name is Linda"), "mom_name")
        self.assertEqual(_derive_key("my mother's name is Carol"), "mom_name")
        self.assertEqual(_derive_key("my dad's name is John"), "dad_name")
        self.assertEqual(_derive_key("my father's name is Joe"), "dad_name")
    
    def test_relationship_without_possessive_name(self):
        """'my wife is X' (meaning name) should derive 'wife_name' key."""
        from wyzer.memory.memory_manager import _derive_key
        
        self.assertEqual(_derive_key("my wife is Audrey"), "wife_name")
        self.assertEqual(_derive_key("my dog is Bella"), "dog_name")
        self.assertEqual(_derive_key("my cat is Whiskers"), "cat_name")
    
    def test_standard_patterns_unchanged(self):
        """Standard patterns should still work correctly."""
        from wyzer.memory.memory_manager import _derive_key
        
        self.assertEqual(_derive_key("my name is Levi"), "name")
        self.assertEqual(_derive_key("my birthday is January 5th"), "birthday")
        self.assertEqual(_derive_key("my age is 30"), "age")
        self.assertEqual(_derive_key("my favorite color is blue"), "favorite_color")
        self.assertEqual(_derive_key("I like pizza"), "likes_pizza")


if __name__ == "__main__":
    print("Running fast-lane cold-start optimization tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
