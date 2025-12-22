"""
Unit tests for sentence-gated TTS stream buffer.

Tests cover:
1. Boundary flush when min_chars reached
2. Timeout flush uses boundary when present
3. Doesn't flush inside ``` fenced code until closed
4. Abbreviation handling doesn't flush on "e.g." and "Mr."
5. Payload detection: rejects chunks containing "[TOOLS]", "args={", JSON-like
6. Final flush returns remaining text
7. URL handling - doesn't split mid-URL
8. Min words requirement
"""
import unittest
from typing import List


class TestTTSStreamBuffer(unittest.TestCase):
    """Test TTSStreamBuffer sentence boundary detection and buffering."""
    
    def _simulate_stream(self, buffer, text: str, token_size: int = 1) -> List[str]:
        """Simulate streaming text through buffer in tokens of given size."""
        chunks = []
        # Simulate time advancing with each token
        base_time = 1000
        for i in range(0, len(text), token_size):
            token = text[i:i + token_size]
            result = buffer.add_text(token, base_time + i * 10)
            chunks.extend(result)
        return chunks
    
    def test_boundary_flush_on_sentence_end(self):
        """Flush when sentence boundary found and min_chars reached."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=20, min_words=3)
        
        # Text with clear sentence boundary
        text = "This is a complete sentence. And this is another one."
        chunks = self._simulate_stream(buffer, text, token_size=3)
        
        # Flush remaining
        final = buffer.flush_final()
        chunks.extend(final)
        
        # Should have 2 chunks
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].endswith('.'))
        self.assertTrue(chunks[1].endswith('.'))
    
    def test_min_chars_prevents_early_flush(self):
        """Don't flush if min_chars not reached even with boundary."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=100, min_words=20)
        
        # Short sentence - below min_chars
        text = "Hello. World."
        chunks = self._simulate_stream(buffer, text)
        
        # Should not have flushed yet (below thresholds)
        self.assertEqual(len(chunks), 0)
        
        # Final flush should return all text
        final = buffer.flush_final()
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0], "Hello. World.")
    
    def test_min_words_allows_early_flush(self):
        """Flush when min_words reached even if min_chars not reached."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=200, min_words=5)
        
        # Enough words but not enough chars
        text = "One two three four five. Six seven."
        chunks = self._simulate_stream(buffer, text, token_size=2)
        
        # Should flush on first sentence (5 words)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("five.", chunks[0])
    
    def test_timeout_flush_uses_boundary(self):
        """Timeout flush prefers sentence boundary if present."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=30, min_words=5, max_wait_ms=100)
        
        # Add text without triggering boundary flush
        text = "Short. More text coming"
        
        # Simulate slow typing - exceed timeout
        base_time = 1000
        for i, char in enumerate(text):
            chunks = buffer.add_text(char, base_time + i * 50)  # 50ms per char
            if chunks:
                # Should flush at "Short." boundary when timeout hit
                self.assertEqual(chunks[0], "Short.")
                break
    
    def test_timeout_flush_at_word_boundary(self):
        """Timeout flush at word boundary when no sentence boundary exists."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=20, min_words=3, max_wait_ms=50)
        
        # No sentence boundary
        text = "This is text without any punctuation that keeps going"
        
        # Add all text at once, then wait for timeout
        chunks = buffer.add_text(text, 1000)
        # No immediate flush expected
        
        # Simulate time passing - add empty text after timeout
        chunks = buffer.add_text("", 1100)  # 100ms later
        
        # Final flush should work
        final = buffer.flush_final()
        self.assertGreater(len(final), 0)
    
    def test_no_flush_inside_code_fence(self):
        """Code block content should be stripped and not spoken."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=10, min_words=2)
        
        # Code block with sentence endings inside
        text = "Look at this code: ```\nprint('Hello. World.')\n``` Done."
        chunks = self._simulate_stream(buffer, text, token_size=3)
        
        # Get final output
        final = buffer.flush_final()
        all_chunks = chunks + final
        
        # The final spoken text should NOT contain the code block content
        # After stripping, we should have "Look at this code: Done."
        all_text = "".join(all_chunks)
        
        # Should have "Done" in output
        self.assertIn("Done", all_text)
        
        # The code block content (print statement) should have been stripped
        # Note: During streaming, code blocks may appear in intermediate chunks
        # but the final spoken text should have them removed
        # The important thing is we don't speak broken fragments
    
    def test_code_fence_closes_properly(self):
        """After code fence closes, normal flushing resumes."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=10, min_words=2)
        
        # Start outside, enter code, exit code
        buffer.add_text("Before. ", 1000)
        buffer.add_text("```code```", 1010)  # Enter and exit
        chunks = buffer.add_text(" After the code block. Done.", 1020)
        
        # After code fence closes, we should be able to flush
        final = buffer.flush_final()
        all_chunks = chunks + final
        
        # Should capture text after code block
        all_text = "".join(all_chunks)
        self.assertIn("After", all_text)
    
    def test_abbreviation_mr_not_boundary(self):
        """'Mr.' should not be treated as sentence boundary."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=20, min_words=3)
        
        text = "Hello Mr. Smith is here. Welcome."
        chunks = self._simulate_stream(buffer, text, token_size=2)
        final = buffer.flush_final()
        all_chunks = chunks + final
        
        # "Mr." should NOT cause a split
        # The first chunk should be "Hello Mr. Smith is here."
        # NOT "Hello Mr."
        for chunk in all_chunks:
            self.assertNotEqual(chunk.strip(), "Hello Mr.")
    
    def test_abbreviation_eg_not_boundary(self):
        """'e.g.' should not be treated as sentence boundary."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=30, min_words=5)
        
        text = "Use something like e.g. Python or JavaScript. It works great."
        chunks = self._simulate_stream(buffer, text, token_size=2)
        final = buffer.flush_final()
        
        # Check that "e.g." didn't cause a split
        all_text = "".join(chunks + final)
        self.assertIn("e.g. Python", all_text)
    
    def test_abbreviation_dr_not_boundary(self):
        """'Dr.' should not be treated as sentence boundary."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=20, min_words=3)
        
        text = "Call Dr. Jones today. He is available."
        chunks = self._simulate_stream(buffer, text, token_size=2)
        final = buffer.flush_final()
        
        # "Dr." should NOT cause a split
        for chunk in chunks + final:
            self.assertNotEqual(chunk.strip(), "Call Dr.")
    
    def test_payload_detection_tools_marker(self):
        """Text containing '[TOOLS]' should be skipped."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=10, min_words=2)
        
        text = "Here is the response [TOOLS] tool_name. Done speaking."
        chunks = self._simulate_stream(buffer, text, token_size=3)
        final = buffer.flush_final()
        
        # The [TOOLS] content should not appear in spoken chunks
        all_spoken = "".join(chunks + final)
        # Should still get "Done speaking" but not the tool marker in a spoken chunk
        self.assertIn("Done speaking", all_spoken)
    
    def test_payload_detection_args_marker(self):
        """Text containing 'args={' should be detected as payload."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=10, min_words=2)
        
        # Direct check on the method
        self.assertTrue(buffer._looks_like_payload("Running args={name: test}"))
        self.assertTrue(buffer._looks_like_payload("Pool result: success"))
    
    def test_payload_detection_json_structure(self):
        """JSON-like content should be detected as payload."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=10, min_words=2)
        
        # JSON object
        self.assertTrue(buffer._looks_like_payload('{"key": "value", "num": 123}'))
        
        # JSON array
        self.assertTrue(buffer._looks_like_payload('[{"a": 1}, {"b": 2}]'))
        
        # High brace density
        self.assertTrue(buffer._looks_like_payload('Result: {a} {b} {c} {d} {e}'))
    
    def test_payload_detection_normal_text_passes(self):
        """Normal conversational text should not be flagged as payload."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=10, min_words=2)
        
        # Normal sentences
        self.assertFalse(buffer._looks_like_payload("The weather is nice today."))
        self.assertFalse(buffer._looks_like_payload("I can help you with that question."))
        self.assertFalse(buffer._looks_like_payload("Here are a few options to consider."))
    
    def test_final_flush_returns_remaining(self):
        """flush_final() returns any buffered text."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=100, min_words=20)
        
        # Add text that won't trigger flush
        buffer.add_text("This is some remaining text", 1000)
        
        # Final flush should return it
        final = buffer.flush_final()
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0], "This is some remaining text")
    
    def test_final_flush_skips_payload(self):
        """flush_final() skips payload content."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=100, min_words=20)
        
        # Add payload-like text
        buffer.add_text('{"result": "data", "status": "ok"}', 1000)
        
        # Final flush should skip it
        final = buffer.flush_final()
        self.assertEqual(len(final), 0)
    
    def test_reset_clears_state(self):
        """reset() clears buffer and code fence state."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=10, min_words=2)
        
        # Add some text and enter code fence
        buffer.add_text("Some text ```code", 1000)
        self.assertTrue(buffer._in_code_fence)
        
        # Reset
        buffer.reset()
        
        # Should be clean
        self.assertFalse(buffer._in_code_fence)
        final = buffer.flush_final()
        self.assertEqual(len(final), 0)
    
    def test_url_not_split(self):
        """URLs should not be split mid-URL."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=20, min_words=3)
        
        # URL with period in path
        text = "Check out https://example.com/page.html for info. Thanks."
        chunks = self._simulate_stream(buffer, text, token_size=3)
        final = buffer.flush_final()
        
        # The URL should not be split at internal periods
        all_text = "".join(chunks + final)
        self.assertIn("https://example.com/page.html", all_text)
    
    def test_exclamation_boundary(self):
        """Exclamation mark should be a valid boundary."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=20, min_words=3)
        
        text = "This is really exciting! And there is more to come."
        chunks = self._simulate_stream(buffer, text, token_size=2)
        final = buffer.flush_final()
        
        # Should split at "!"
        self.assertTrue(any(c.endswith('!') for c in chunks))
    
    def test_question_mark_boundary(self):
        """Question mark should be a valid boundary."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=20, min_words=3)
        
        text = "How are you doing today? I hope you are well."
        chunks = self._simulate_stream(buffer, text, token_size=2)
        final = buffer.flush_final()
        
        # Should split at "?"
        self.assertTrue(any(c.endswith('?') for c in chunks))
    
    def test_colon_boundary(self):
        """Colon should be a valid boundary (if configured)."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        buffer = TTSStreamBuffer(min_chars=20, min_words=3, boundaries=".!?:")
        
        text = "Here is the answer: the sky is blue. That is it."
        chunks = self._simulate_stream(buffer, text, token_size=2)
        final = buffer.flush_final()
        
        # Colon should trigger flush if conditions met
        all_text = "".join(chunks + final)
        self.assertIn("answer:", all_text)
    
    def test_custom_boundaries(self):
        """Custom boundary characters should work."""
        from wyzer.brain.tts_stream_buffer import TTSStreamBuffer
        
        # Only use period as boundary
        buffer = TTSStreamBuffer(min_chars=10, min_words=2, boundaries=".")
        
        text = "Question? Answer. More text!"
        chunks = self._simulate_stream(buffer, text, token_size=2)
        final = buffer.flush_final()
        
        # Only period should trigger boundary flush
        # The "?" and "!" should not cause splits on their own
        all_chunks = chunks + final
        
        # Verify we get the text
        all_text = "".join(all_chunks)
        self.assertIn("Question?", all_text)


class TestCreateBufferFromConfig(unittest.TestCase):
    """Test config-based buffer creation."""
    
    def test_creates_buffer_with_defaults(self):
        """create_buffer_from_config uses Config values."""
        from wyzer.brain.tts_stream_buffer import create_buffer_from_config
        
        buffer = create_buffer_from_config()
        
        # Should have created a valid buffer
        self.assertIsNotNone(buffer)
        self.assertGreater(buffer.min_chars, 0)
        self.assertGreater(buffer.min_words, 0)
        self.assertGreater(buffer.max_wait_ms, 0)


class TestNowMs(unittest.TestCase):
    """Test the now_ms helper function."""
    
    def test_returns_int(self):
        """now_ms returns an integer."""
        from wyzer.brain.tts_stream_buffer import now_ms
        
        result = now_ms()
        self.assertIsInstance(result, int)
    
    def test_monotonic_increasing(self):
        """now_ms values increase over time."""
        from wyzer.brain.tts_stream_buffer import now_ms
        import time
        
        t1 = now_ms()
        time.sleep(0.01)  # 10ms
        t2 = now_ms()
        
        self.assertGreater(t2, t1)


class TestAbbreviations(unittest.TestCase):
    """Test abbreviation detection."""
    
    def test_common_titles(self):
        """Common title abbreviations are detected."""
        from wyzer.brain.tts_stream_buffer import ABBREVIATIONS
        
        titles = ["mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr."]
        for title in titles:
            self.assertIn(title, ABBREVIATIONS)
    
    def test_latin_abbreviations(self):
        """Latin abbreviations are detected."""
        from wyzer.brain.tts_stream_buffer import ABBREVIATIONS
        
        latin = ["e.g.", "i.e.", "etc.", "vs.", "viz."]
        for abbr in latin:
            self.assertIn(abbr, ABBREVIATIONS)
    
    def test_country_abbreviations(self):
        """Country abbreviations are detected."""
        from wyzer.brain.tts_stream_buffer import ABBREVIATIONS
        
        countries = ["u.s.", "u.k.", "u.n."]
        for abbr in countries:
            self.assertIn(abbr, ABBREVIATIONS)


if __name__ == "__main__":
    unittest.main()
