"""
Unit tests for stream-to-TTS chunk policy and integration.

Tests cover:
1. Chunk policy - sentence boundary emission
2. Chunk policy - buffer overflow emission
3. Full reply accumulation
4. Fallback on streaming failure
5. Tool path is unaffected by streaming
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Iterator


class TestChunkBuffer(unittest.TestCase):
    """Test ChunkBuffer sentence/paragraph detection and buffering."""
    
    def test_sentence_boundary_emits_segment(self):
        """Streaming "Hello world. How are you?" emits 2 segments."""
        from wyzer.brain.stream_tts import ChunkBuffer
        
        buffer = ChunkBuffer(min_chars=150)
        segments = []
        
        # Simulate streaming tokens
        tokens = ["Hello", " ", "world", ".", " ", "How", " ", "are", " ", "you", "?"]
        
        for token in tokens:
            segment = buffer.add(token)
            if segment:
                segments.append(segment)
        
        # Flush remaining
        final = buffer.flush()
        if final:
            segments.append(final)
        
        # Should have 2 segments: "Hello world." and "How are you?"
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], "Hello world.")
        self.assertEqual(segments[1], "How are you?")
    
    def test_buffer_overflow_emits_without_punctuation(self):
        """No punctuation: emits when buffer exceeds threshold."""
        from wyzer.brain.stream_tts import ChunkBuffer
        
        # Use small threshold for testing
        buffer = ChunkBuffer(min_chars=20)
        segments = []
        
        # Long text without punctuation
        text = "This is a long piece of text without any punctuation marks"
        for char in text:
            segment = buffer.add(char)
            if segment:
                segments.append(segment)
        
        final = buffer.flush()
        if final:
            segments.append(final)
        
        # Should have emitted at least one segment due to overflow
        self.assertGreater(len(segments), 1)
        
        # Concatenated should equal original (minus extra spaces from word boundary breaks)
        concatenated = " ".join(segments)
        # Words should all be present
        for word in text.split():
            self.assertIn(word, concatenated)
    
    def test_paragraph_boundary_emits(self):
        """Paragraph boundary (newline) triggers emission."""
        from wyzer.brain.stream_tts import ChunkBuffer
        
        buffer = ChunkBuffer(min_chars=150)
        segments = []
        
        tokens = ["First", " ", "paragraph", "\n\n", "Second", " ", "paragraph", "."]
        
        for token in tokens:
            segment = buffer.add(token)
            if segment:
                segments.append(segment)
        
        final = buffer.flush()
        if final:
            segments.append(final)
        
        # Should have 2 segments
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], "First paragraph")
        self.assertEqual(segments[1], "Second paragraph.")
    
    def test_flush_returns_remaining(self):
        """Flush returns any remaining buffered text."""
        from wyzer.brain.stream_tts import ChunkBuffer
        
        # Use high first_emit_chars to prevent early emission for this test
        buffer = ChunkBuffer(min_chars=150, first_emit_chars=200)
        
        # Add text without any triggers
        buffer.add("Some text without ending")
        
        # Should be empty from add (no triggers)
        self.assertIsNone(buffer.add(""))
        
        # Flush should return the buffered text
        final = buffer.flush()
        self.assertEqual(final, "Some text without ending")
        
        # Second flush should be empty
        self.assertIsNone(buffer.flush())


class TestAccumulateFullReply(unittest.TestCase):
    """Test that full reply is properly accumulated during streaming."""
    
    def test_full_reply_equals_concatenated_tokens(self):
        """Concatenated segments equals the final returned assistant reply."""
        from wyzer.brain.stream_tts import accumulate_full_reply
        
        # Simulate token stream
        tokens = ["Hello", " ", "world", ".", " ", "How", " ", "are", " ", "you", "?"]
        
        def token_generator() -> Iterator[str]:
            for t in tokens:
                yield t
        
        segments = []
        def on_segment(segment: str) -> None:
            segments.append(segment)
        
        full_reply = accumulate_full_reply(
            token_stream=token_generator(),
            on_segment=on_segment,
            min_chars=150
        )
        
        # Full reply should be complete text
        self.assertEqual(full_reply, "Hello world. How are you?")
        
        # Segments should also concatenate to same text
        concatenated = " ".join(segments)
        self.assertEqual(concatenated, "Hello world. How are you?")
    
    def test_cancellation_stops_stream(self):
        """Cancellation check stops stream consumption."""
        from wyzer.brain.stream_tts import accumulate_full_reply
        
        cancelled = [False]
        token_count = [0]
        
        def token_generator() -> Iterator[str]:
            for i in range(100):
                token_count[0] += 1
                if i == 5:
                    cancelled[0] = True
                yield f"token{i} "
        
        def cancel_check() -> bool:
            return cancelled[0]
        
        segments = []
        full_reply = accumulate_full_reply(
            token_stream=token_generator(),
            on_segment=lambda s: segments.append(s),
            min_chars=150,
            cancel_check=cancel_check
        )
        
        # Should have stopped early
        self.assertLess(token_count[0], 100)


class TestStreamingFallback(unittest.TestCase):
    """Test fallback to non-streaming on error."""
    
    @patch('wyzer.brain.stream_tts.get_logger')
    def test_exception_during_stream_propagates(self, mock_get_logger):
        """If stream generator raises, exception propagates to caller."""
        from wyzer.brain.stream_tts import accumulate_full_reply
        
        mock_get_logger.return_value = Mock()
        
        def failing_generator() -> Iterator[str]:
            yield "hello"
            raise ConnectionError("Network error")
        
        with self.assertRaises(ConnectionError):
            accumulate_full_reply(
                token_stream=failing_generator(),
                on_segment=lambda s: None,
                min_chars=150
            )


class TestStreamingTTSConfig(unittest.TestCase):
    """Test that streaming TTS respects configuration."""
    
    @patch('wyzer.core.orchestrator.Config')
    @patch('wyzer.core.orchestrator.hybrid_router')
    def test_streaming_disabled_when_config_off(self, mock_hybrid_router, mock_config):
        """Streaming returns False when OLLAMA_STREAM_TTS is False."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        
        mock_config.OLLAMA_STREAM_TTS = False
        mock_config.NO_OLLAMA = False
        
        result = should_use_streaming_tts("tell me about the weather")
        self.assertFalse(result)
    
    @patch('wyzer.core.orchestrator.Config')
    @patch('wyzer.core.orchestrator.hybrid_router')
    @patch('wyzer.core.orchestrator.is_informational_query')
    @patch('wyzer.core.orchestrator.is_continuation_phrase')
    @patch('wyzer.core.orchestrator.is_explicit_continuation')
    def test_streaming_enabled_for_info_query(
        self, 
        mock_explicit, 
        mock_continuation, 
        mock_info_query,
        mock_hybrid_router, 
        mock_config
    ):
        """Streaming returns True for informational queries when enabled."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        
        mock_config.OLLAMA_STREAM_TTS = True
        mock_config.NO_OLLAMA = False
        
        # Mock hybrid router to return LLM mode
        mock_decision = Mock()
        mock_decision.mode = "llm"
        mock_decision.intents = None
        mock_hybrid_router.decide.return_value = mock_decision
        
        # Mock as informational query
        mock_continuation.return_value = False
        mock_explicit.return_value = None
        mock_info_query.return_value = True
        
        result = should_use_streaming_tts("what is python")
        self.assertTrue(result)


class TestToolPathUnaffected(unittest.TestCase):
    """Test that tool commands do not invoke streaming."""
    
    @patch('wyzer.core.orchestrator.Config')
    @patch('wyzer.core.orchestrator.hybrid_router')
    def test_tool_command_does_not_stream(self, mock_hybrid_router, mock_config):
        """A deterministic tool command does not use streaming."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        
        mock_config.OLLAMA_STREAM_TTS = True
        mock_config.NO_OLLAMA = False
        
        # Mock hybrid router to return tool_plan mode
        mock_decision = Mock()
        mock_decision.mode = "tool_plan"
        mock_decision.intents = [{"tool": "open_app", "args": {"name": "notepad"}}]
        mock_hybrid_router.decide.return_value = mock_decision
        
        result = should_use_streaming_tts("open notepad")
        self.assertFalse(result)


class TestTTSControllerCancellation(unittest.TestCase):
    """Test TTS controller cancellation for streaming."""
    
    def test_is_cancelled_reflects_stop_event(self):
        """is_cancelled() returns True when stop event is set."""
        import threading
        import queue
        
        # Minimal mock setup
        mock_tts = None
        mock_queue = queue.Queue()
        
        # Import the class
        from wyzer.core.brain_worker import _TTSController
        
        # We need to patch to avoid starting the actual thread
        with patch.object(_TTSController, '_loop', return_value=None):
            controller = _TTSController(mock_tts, mock_queue, simulate=True)
            
            # Initially not cancelled
            self.assertFalse(controller.is_cancelled())
            
            # Set stop event
            controller._stop_event.set()
            
            # Now should be cancelled
            self.assertTrue(controller.is_cancelled())
            
            # Cleanup
            controller._running = False


if __name__ == '__main__':
    unittest.main()
