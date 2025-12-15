"""
Unit tests for prompt compaction functionality.
Tests the compact_prompt() function for safe deterministic prompt reduction.
"""
import unittest
import sys
import os

# Add wyzer to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wyzer.brain.prompt_compact import compact_prompt


class TestPromptCompaction(unittest.TestCase):
    """Test prompt compaction logic."""
    
    def test_no_compaction_short_prompt(self):
        """Test that short prompts are not compacted."""
        short_prompt = "This is a short prompt with less than 8000 characters."
        
        result, was_compacted = compact_prompt(short_prompt, max_chars=8000)
        
        self.assertEqual(result, short_prompt)
        self.assertFalse(was_compacted)
    
    def test_no_compaction_exact_limit(self):
        """Test that prompts at the limit are not compacted."""
        prompt = "x" * 8000
        
        result, was_compacted = compact_prompt(prompt, max_chars=8000)
        
        self.assertEqual(result, prompt)
        self.assertFalse(was_compacted)
    
    def test_compaction_long_prompt(self):
        """Test that long prompts are compacted."""
        # Create a prompt longer than the limit
        long_prompt = "A" * 1000 + "B" * 5000 + "C" * 3000  # 9000 total
        
        result, was_compacted = compact_prompt(long_prompt, max_chars=8000)
        
        self.assertTrue(was_compacted)
        self.assertLessEqual(len(result), 8000)
        # Should keep first section (A's) and last section (C's)
        self.assertIn("A", result)
        self.assertIn("C", result)
    
    def test_compaction_preserves_first_section(self):
        """Test that compaction preserves the first 1200 chars."""
        first_section = "FIRST" * 240  # 1200 chars
        middle_section = "M" * 3000
        last_section = "LAST" * 450  # 1800 chars
        
        prompt = first_section + middle_section + last_section
        
        result, was_compacted = compact_prompt(prompt, max_chars=8000)
        
        # First section should be preserved
        self.assertIn("FIRST", result)
        # Result should start with the first section
        self.assertTrue(result.startswith("FIRST"))
    
    def test_compaction_preserves_last_section(self):
        """Test that compaction preserves the last 1800 chars."""
        first_section = "F" * 1000
        middle_section = "M" * 5000
        last_section = "LAST" * 450  # 1800 chars
        
        prompt = first_section + middle_section + last_section
        
        result, was_compacted = compact_prompt(prompt, max_chars=8000)
        
        # Last section should be preserved
        self.assertIn("LAST", result)
        # Result should end with the last section
        self.assertTrue(result.endswith("LAST" * 450))
    
    def test_compaction_includes_marker(self):
        """Test that compaction includes the omission marker."""
        long_prompt = "A" * 2000 + "B" * 5000 + "C" * 2000
        
        result, was_compacted = compact_prompt(long_prompt, max_chars=8000)
        
        self.assertTrue(was_compacted)
        self.assertIn("...[omitted for length]...", result)
    
    def test_compaction_respects_max_chars(self):
        """Test that compacted result never exceeds max_chars."""
        long_prompt = "X" * 50000  # Very long
        max_chars = 1000
        
        result, was_compacted = compact_prompt(long_prompt, max_chars=max_chars)
        
        self.assertTrue(was_compacted)
        self.assertLessEqual(len(result), max_chars)
    
    def test_compaction_with_multiline(self):
        """Test compaction with multiline prompts."""
        first_section = "SYSTEM:\nKey rules\n" * 100  # ~1800 chars
        middle_section = "\n".join(["irrelevant line"] * 500)  # ~8000 chars
        last_section = "USER: What is the answer?\nAssistant:\n" * 100  # ~3900 chars
        
        prompt = first_section + middle_section + last_section
        
        result, was_compacted = compact_prompt(prompt, max_chars=8000)
        
        self.assertTrue(was_compacted)
        self.assertLessEqual(len(result), 8000)
        # Should preserve key parts
        self.assertIn("SYSTEM", result)
        self.assertIn("USER", result)
    
    def test_compaction_very_small_limit(self):
        """Test compaction with a very small character limit."""
        prompt = "A" * 10000
        max_chars = 200
        
        result, was_compacted = compact_prompt(prompt, max_chars=max_chars)
        
        self.assertTrue(was_compacted)
        self.assertLessEqual(len(result), max_chars)
        # Even with small limit, should have some content
        self.assertGreater(len(result), 0)
    
    def test_compaction_with_special_chars(self):
        """Test compaction preserves special characters."""
        first = "#!/usr/bin/env\n" * 100  # ~1500 chars
        middle = "x" * 7000
        last = "# JSON: {\"key\": \"value\"}" * 100  # ~2700 chars
        
        prompt = first + middle + last
        
        result, was_compacted = compact_prompt(prompt, max_chars=8000)
        
        self.assertTrue(was_compacted)
        self.assertLessEqual(len(result), 8000)
        # Special chars should be preserved
        self.assertIn("#!", result)
        self.assertIn("JSON", result)


class TestCompactionReturnValues(unittest.TestCase):
    """Test the return value format of compact_prompt."""
    
    def test_return_is_tuple(self):
        """Test that return value is a tuple."""
        prompt = "test"
        result = compact_prompt(prompt)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
    
    def test_return_first_element_is_string(self):
        """Test that first element is the compacted prompt string."""
        prompt = "test prompt"
        result, _ = compact_prompt(prompt)
        
        self.assertIsInstance(result, str)
    
    def test_return_second_element_is_boolean(self):
        """Test that second element is a boolean flag."""
        prompt = "test"
        _, was_compacted = compact_prompt(prompt)
        
        self.assertIsInstance(was_compacted, bool)
    
    def test_boolean_false_when_no_compaction(self):
        """Test that boolean is False when no compaction needed."""
        short = "short"
        _, was_compacted = compact_prompt(short, max_chars=1000)
        
        self.assertFalse(was_compacted)
    
    def test_boolean_true_when_compacted(self):
        """Test that boolean is True when compaction occurred."""
        long = "x" * 10000
        _, was_compacted = compact_prompt(long, max_chars=1000)
        
        self.assertTrue(was_compacted)


if __name__ == '__main__':
    unittest.main()
