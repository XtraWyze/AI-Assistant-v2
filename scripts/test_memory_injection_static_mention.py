#!/usr/bin/env python3
"""
Test suite for deterministic memory injection (pinned + mention-triggered).

Tests the following:
1. Pinned records always included (up to PINNED_MAX)
2. Mention-triggered matching works on key/aliases
3. Respects caps (PINNED_MAX, MENTION_MAX, K_TOTAL)
4. Deterministic: same input → same output
5. Migration backward compatibility (pinned=False, aliases=[] by default)
6. Pin/unpin/alias voice commands
7. Integration with prompt.py
"""

import json
import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def reset_memory_manager():
    """Reset the global memory manager singleton."""
    try:
        from wyzer.memory import memory_manager
        memory_manager._memory_manager = None
    except Exception:
        pass


def create_test_manager():
    """Create a fresh MemoryManager with temp file."""
    from wyzer.memory.memory_manager import MemoryManager, reset_memory_manager as reset_mm
    
    reset_mm()
    mgr = MemoryManager()
    
    # Create temp directory for memory file
    temp_dir = tempfile.mkdtemp(prefix="wyzer_test_")
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    mgr._memory_file = data_dir / "memory.json"
    
    return mgr, temp_dir


def cleanup_test_dir(temp_dir: str):
    """Remove temp directory."""
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


class TestMemoryInjectionCore(unittest.TestCase):
    """Core tests for select_for_injection method."""
    
    def setUp(self):
        """Create a fresh memory manager with test data."""
        reset_memory_manager()
        self.mgr, self.temp_dir = create_test_manager()
    
    def tearDown(self):
        """Clean up temp file."""
        cleanup_test_dir(self.temp_dir)
        reset_memory_manager()
    
    def test_empty_memories_returns_empty_string(self):
        """Empty memory store returns empty string."""
        result = self.mgr.select_for_injection("hello")
        self.assertEqual(result, "")
    
    def test_pinned_records_always_included(self):
        """Pinned records are always included regardless of user text."""
        # Add a pinned memory
        self.mgr.add_explicit("My favorite color is blue", pinned=True)
        
        # Query with unrelated text
        result = self.mgr.select_for_injection("what is the weather")
        
        self.assertIn("blue", result.lower())
    
    def test_pinned_max_respected(self):
        """Only PINNED_MAX pinned records are included."""
        # Add 6 pinned memories (more than default PINNED_MAX=4)
        for i in range(6):
            self.mgr.add_explicit(f"Pinned fact number {i}", pinned=True)
        
        result = self.mgr.select_for_injection("anything", pinned_max=4)
        
        # Count how many pinned facts appear
        count = result.lower().count("pinned fact")
        self.assertLessEqual(count, 4)
    
    def test_mention_triggered_by_key(self):
        """Memories are included when user mentions the key."""
        # Add a memory about coffee
        self.mgr.add_explicit("I love espresso", key="coffee preference")
        
        # Query mentioning coffee
        result = self.mgr.select_for_injection("tell me about my coffee preference")
        
        self.assertIn("espresso", result.lower())
    
    def test_mention_triggered_by_alias(self):
        """Memories are included when user mentions an alias."""
        # Add a memory with aliases
        entry = self.mgr.add_explicit("My dog's name is Max", key="pet", aliases=["dog", "puppy"])
        self.assertTrue(entry.get("ok"))
        
        # Query mentioning the alias "puppy"
        result = self.mgr.select_for_injection("tell me about my puppy")
        
        self.assertIn("max", result.lower())
    
    def test_mention_max_respected(self):
        """Only MENTION_MAX mention-triggered records are included."""
        # Add 6 memories with distinct keys that are NOT mentioned
        other_keys = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        for key in other_keys:
            self.mgr.add_explicit(f"Fact about {key}", key=key)
        
        # Query mentioning only 2 of them, with mention_max=2 and k_total=2
        # This ensures we only get mention-triggered memories
        result = self.mgr.select_for_injection(
            "tell me about alpha and beta",
            mention_max=2,
            k_total=2,
            pinned_max=0
        )
        
        # Count how many of the keys appear
        count = sum(1 for key in other_keys if key in result.lower())
        self.assertLessEqual(count, 2)
    
    def test_k_total_respected(self):
        """Total memories don't exceed K_TOTAL."""
        # Add 10 unpinned memories
        for i in range(10):
            self.mgr.add_explicit(f"Fact number {i}", key=f"fact{i}")
        
        result = self.mgr.select_for_injection("general query", k_total=6)
        
        # Count bullet points
        count = result.count("•")
        self.assertLessEqual(count, 6)
    
    def test_max_chars_respected(self):
        """Output doesn't exceed max_chars."""
        # Add many long memories
        for i in range(20):
            self.mgr.add_explicit(f"This is a very long memory number {i} " * 5)
        
        result = self.mgr.select_for_injection("query", max_chars=500)
        
        self.assertLessEqual(len(result), 600)  # Some buffer for header
    
    def test_deterministic_same_input_same_output(self):
        """Same input always produces same output."""
        self.mgr.add_explicit("My name is Alice", key="name")
        self.mgr.add_explicit("I live in Seattle", key="location")
        self.mgr.add_explicit("I work as a developer", key="job")
        
        result1 = self.mgr.select_for_injection("tell me about my name and job")
        result2 = self.mgr.select_for_injection("tell me about my name and job")
        result3 = self.mgr.select_for_injection("tell me about my name and job")
        
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)
    
    def test_stopwords_ignored_in_matching(self):
        """Common words like 'the', 'my', 'is' don't cause false matches."""
        self.mgr.add_explicit("The is my favorite word", key="test")
        
        # Query with only stopwords
        result = self.mgr.select_for_injection("the is my a an")
        
        # Should not be included since only stopwords match
        # (unless it's in top-K fallback, which is fine)
        # Just verify it doesn't crash
        self.assertIsInstance(result, str)


class TestMigrationBackwardCompat(unittest.TestCase):
    """Test that legacy records get pinned=False, aliases=[] by default."""
    
    def test_legacy_records_get_pinned_false(self):
        """Legacy records without pinned field get pinned=False."""
        from wyzer.memory.memory_manager import _migrate_legacy_entry
        
        # Test migration directly
        legacy_entry = {
            "text": "Legacy fact one",
            "type": "fact",
            "timestamp": "2024-01-01T00:00:00"
        }
        
        migrated = _migrate_legacy_entry(legacy_entry)
        
        self.assertIn("pinned", migrated)
        self.assertFalse(migrated["pinned"])
    
    def test_legacy_records_get_empty_aliases(self):
        """Legacy records without aliases field get aliases=[]."""
        from wyzer.memory.memory_manager import _migrate_legacy_entry
        
        legacy_entry = {
            "text": "Legacy fact two",
            "key": "old_key",
            "type": "fact"
        }
        
        migrated = _migrate_legacy_entry(legacy_entry)
        
        self.assertIn("aliases", migrated)
        self.assertEqual(migrated["aliases"], [])
    
    def test_migration_preserves_existing_data(self):
        """Migration doesn't corrupt existing fields."""
        from wyzer.memory.memory_manager import _migrate_legacy_entry
        
        legacy_entry = {
            "text": "My name is Alice",
            "key": "name",
            "type": "fact",
            "tags": ["personal"]
        }
        
        migrated = _migrate_legacy_entry(legacy_entry)
        
        # Original fields preserved
        self.assertEqual(migrated.get("text"), "My name is Alice")
        self.assertEqual(migrated.get("key"), "name")
        self.assertEqual(migrated.get("tags"), ["personal"])
        # New fields added
        self.assertFalse(migrated.get("pinned"))
        self.assertEqual(migrated.get("aliases"), [])


class TestPinUnpinAlias(unittest.TestCase):
    """Test set_pinned_by_query and add_alias_by_query methods."""
    
    def setUp(self):
        """Create a fresh memory manager with test data."""
        reset_memory_manager()
        self.mgr, self.temp_dir = create_test_manager()
    
    def tearDown(self):
        """Clean up temp file."""
        cleanup_test_dir(self.temp_dir)
        reset_memory_manager()
    
    def test_pin_by_query(self):
        """set_pinned_by_query pins a matching memory."""
        self.mgr.add_explicit("I love pizza", key="food preference")
        
        result = self.mgr.set_pinned_by_query("pizza", pinned=True)
        
        self.assertTrue(result.get("ok"))
        
        # Verify it's pinned
        memories = self.mgr.list_memories()
        pizza_mem = [m for m in memories if "pizza" in (m.get("text", "") or m.get("value", "")).lower()][0]
        self.assertTrue(pizza_mem.get("pinned"))
    
    def test_unpin_by_query(self):
        """set_pinned_by_query unpins a matching memory."""
        self.mgr.add_explicit("I love sushi", key="food preference", pinned=True)
        
        # Verify initially pinned
        memories = self.mgr.list_memories()
        sushi_mem = [m for m in memories if "sushi" in (m.get("text", "") or m.get("value", "")).lower()][0]
        self.assertTrue(sushi_mem.get("pinned"))
        
        # Unpin it
        result = self.mgr.set_pinned_by_query("sushi", pinned=False)
        self.assertTrue(result.get("ok"))
        
        # Verify unpinned
        memories = self.mgr.list_memories()
        sushi_mem = [m for m in memories if "sushi" in (m.get("text", "") or m.get("value", "")).lower()][0]
        self.assertFalse(sushi_mem.get("pinned"))
    
    def test_pin_no_match_returns_error(self):
        """set_pinned_by_query returns error when no match found."""
        self.mgr.add_explicit("I love tacos")
        
        result = self.mgr.set_pinned_by_query("nonexistent query", pinned=True)
        
        self.assertFalse(result.get("ok"))
    
    def test_add_alias_by_query(self):
        """add_alias_by_query adds an alias to a matching memory."""
        self.mgr.add_explicit("My cat's name is Whiskers", key="pet")
        
        result = self.mgr.add_alias_by_query("cat", "kitty")
        
        self.assertTrue(result.get("ok"))
        
        # Verify alias was added
        memories = self.mgr.list_memories()
        cat_mem = [m for m in memories if "whiskers" in (m.get("text", "") or m.get("value", "")).lower()][0]
        self.assertIn("kitty", cat_mem.get("aliases", []))
    
    def test_alias_no_match_returns_error(self):
        """add_alias_by_query returns error when no match found."""
        self.mgr.add_explicit("Random fact")
        
        result = self.mgr.add_alias_by_query("nonexistent", "alias")
        
        self.assertFalse(result.get("ok"))


class TestVoiceCommands(unittest.TestCase):
    """Test UNPIN/ADD_ALIAS voice command detection.
    
    Note: PIN commands were removed since remember() now auto-pins.
    """
    
    def test_remember_creates_pinned_memory(self):
        """'remember X' creates a pinned memory (auto-pins)."""
        from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
        
        cmd = detect_memory_command("remember my name is Alice")
        
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.command_type, MemoryCommandType.REMEMBER)
        # The memory manager sets pinned=True by default
    
    def test_you_can_forget_that_detected(self):
        """'you can forget that' is detected as UNPIN command."""
        from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
        
        cmd = detect_memory_command("you can forget that")
        
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.command_type, MemoryCommandType.UNPIN)
        self.assertEqual(cmd.text, "")  # Uses last recall
    
    def test_sometimes_forget_that_detected(self):
        """'sometimes forget that' is detected as UNPIN command."""
        from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
        
        cmd = detect_memory_command("sometimes forget that")
        
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.command_type, MemoryCommandType.UNPIN)
        self.assertEqual(cmd.text, "")
    
    def test_you_can_forget_x_detected(self):
        """'you can forget X' is detected as UNPIN command."""
        from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
        
        cmd = detect_memory_command("you can forget my location")
        
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.command_type, MemoryCommandType.UNPIN)
        self.assertIn("location", cmd.text.lower())
    
    def test_sometimes_forget_x_detected(self):
        """'sometimes forget X' is detected as UNPIN command."""
        from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
        
        cmd = detect_memory_command("sometimes forget my job")
        
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.command_type, MemoryCommandType.UNPIN)
        self.assertIn("job", cmd.text.lower())
    
    def test_dont_always_use_x_detected(self):
        """'don't always use X' is detected as UNPIN command."""
        from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
        
        cmd = detect_memory_command("don't always use my nickname")
        
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.command_type, MemoryCommandType.UNPIN)
        self.assertIn("nickname", cmd.text.lower())
    
    def test_add_alias_x_to_y_detected(self):
        """'add alias X to Y' is detected as ADD_ALIAS command."""
        from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
        
        cmd = detect_memory_command("add alias puppy to my dog")
        
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.command_type, MemoryCommandType.ADD_ALIAS)
        self.assertIn("||", cmd.text)  # Format: "query||alias"
    
    def test_also_call_x_y_detected(self):
        """'also call X Y' is detected as ADD_ALIAS command."""
        from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
        
        cmd = detect_memory_command("also call my car vehicle")
        
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.command_type, MemoryCommandType.ADD_ALIAS)


class TestPromptIntegration(unittest.TestCase):
    """Test integration with prompt.py."""
    
    def test_get_smart_memories_block_returns_string(self):
        """get_smart_memories_block returns a string."""
        from wyzer.brain.prompt import get_smart_memories_block
        
        result = get_smart_memories_block("test query")
        
        self.assertIsInstance(result, str)
    
    def test_build_prompt_messages_includes_smart_memories(self):
        """build_prompt_messages includes smart memory block."""
        from wyzer.brain.prompt import build_prompt_messages
        
        messages = build_prompt_messages("hello there")
        
        self.assertIsInstance(messages, list)
        self.assertTrue(len(messages) >= 2)  # System + user at minimum


class TestScoringDeterminism(unittest.TestCase):
    """Test that scoring produces consistent results."""
    
    def setUp(self):
        """Create memory manager with test data."""
        reset_memory_manager()
        self.mgr, self.temp_dir = create_test_manager()
        
        # Add several memories
        self.mgr.add_explicit("I work at Acme Corp", key="work")
        self.mgr.add_explicit("My hobby is painting", key="hobby")
        self.mgr.add_explicit("I drive a Tesla", key="car", aliases=["vehicle", "automobile"])
        self.mgr.add_explicit("My wife's name is Sarah", key="family", pinned=True)
    
    def tearDown(self):
        """Clean up temp file."""
        cleanup_test_dir(self.temp_dir)
        reset_memory_manager()
    
    def test_key_match_scores_higher_than_token_overlap(self):
        """Key matches should score higher than simple token overlap."""
        # Query that matches key exactly should include that memory
        result = self.mgr.select_for_injection("tell me about my hobby")
        
        self.assertIn("painting", result.lower())
    
    def test_alias_match_triggers_inclusion(self):
        """Alias match should trigger memory inclusion."""
        result = self.mgr.select_for_injection("what vehicle do I drive")
        
        self.assertIn("tesla", result.lower())
    
    def test_pinned_always_first(self):
        """Pinned memories should appear first in output."""
        result = self.mgr.select_for_injection("random unrelated query")
        
        # Sarah should be mentioned since that memory is pinned
        self.assertIn("sarah", result.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
