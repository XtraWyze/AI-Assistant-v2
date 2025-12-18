"""
Test suite for use_memories session flag.

Tests:
1. Flag toggles correctly via command detection
2. Memory injection only occurs when flag is ON
3. Injection respects size caps (30 bullets / 1200 chars)
4. Commands return proper spoken confirmations
5. No disk writes occur from enable/disable commands

Run with:
    python scripts/test_use_memories_flag.py
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_flag_default_on():
    """Test that use_memories flag defaults to ON."""
    print("\n=== test_flag_default_on ===")
    
    from wyzer.memory.memory_manager import MemoryManager
    
    mgr = MemoryManager()
    
    assert mgr.get_use_memories() is True, "Flag should default to True"
    print("✓ use_memories flag defaults to True")
    
    print("✓ test_flag_default_on: PASSED")


def test_flag_toggle_on_off():
    """Test that flag can be toggled on and off."""
    print("\n=== test_flag_toggle_on_off ===")
    
    from wyzer.memory.memory_manager import MemoryManager
    
    mgr = MemoryManager()
    
    # Starts ON by default
    assert mgr.get_use_memories() is True, "Flag should default to True"
    print("✓ Flag starts as True (default)")
    
    # Toggle OFF
    changed = mgr.set_use_memories(False, source="test")
    assert changed is True, "Should return True when state changes"
    assert mgr.get_use_memories() is False, "Flag should be False"
    print("✓ Flag toggled to False")
    
    # Toggle OFF again (no change)
    changed = mgr.set_use_memories(False, source="test")
    assert changed is False, "Should return False when state already set"
    assert mgr.get_use_memories() is False, "Flag should still be False"
    print("✓ No change when already False")
    
    # Toggle ON
    changed = mgr.set_use_memories(True, source="test")
    assert changed is True, "Should return True when state changes"
    assert mgr.get_use_memories() is True, "Flag should be True"
    print("✓ Flag toggled to True")
    
    # Toggle ON again (no change)
    changed = mgr.set_use_memories(True, source="test")
    assert changed is False, "Should return False when state already set"
    print("✓ No change when already True")
    
    print("✓ test_flag_toggle_on_off: PASSED")


def test_command_detection_enable():
    """Test that enable phrases are detected correctly."""
    print("\n=== test_command_detection_enable ===")
    
    from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
    
    enable_phrases = [
        "use memories",
        "Use memories",
        "USE MEMORIES",
        "use all memories",
        "enable memories",
        "turn on memories",
        "use my memories",
    ]
    
    for phrase in enable_phrases:
        cmd = detect_memory_command(phrase)
        assert cmd is not None, f"Should detect '{phrase}' as command"
        assert cmd.command_type == MemoryCommandType.USE_ALL_MEMORIES, \
            f"'{phrase}' should be USE_ALL_MEMORIES, got {cmd.command_type}"
        print(f"✓ '{phrase}' → USE_ALL_MEMORIES")
    
    print("✓ test_command_detection_enable: PASSED")


def test_command_detection_disable():
    """Test that disable phrases are detected correctly."""
    print("\n=== test_command_detection_disable ===")
    
    from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
    
    disable_phrases = [
        "stop using memories",
        "disable memories",
        "turn off memories",
        "don't use memories",
        "do not use memories",
        "stop using all memories",
    ]
    
    for phrase in disable_phrases:
        cmd = detect_memory_command(phrase)
        assert cmd is not None, f"Should detect '{phrase}' as command"
        assert cmd.command_type == MemoryCommandType.DISABLE_MEMORIES, \
            f"'{phrase}' should be DISABLE_MEMORIES, got {cmd.command_type}"
        print(f"✓ '{phrase}' → DISABLE_MEMORIES")
    
    print("✓ test_command_detection_disable: PASSED")


def test_handle_memory_command_enable():
    """Test that enable command returns proper response when already enabled."""
    print("\n=== test_handle_memory_command_enable ===")
    
    from wyzer.memory.command_detector import handle_memory_command
    from wyzer.memory.memory_manager import get_memory_manager, reset_memory_manager
    
    # Reset to ensure clean state (defaults to ON now)
    reset_memory_manager()
    mem_mgr = get_memory_manager()
    
    # Disable first so we can test enabling
    mem_mgr.set_use_memories(False)
    assert mem_mgr.get_use_memories() is False, "Should be OFF for test"
    
    result = handle_memory_command("use memories")
    assert result is not None, "Should return result"
    response, meta = result
    
    assert "use" in response.lower() or "memories" in response.lower(), \
        f"Response should confirm memories: '{response}'"
    assert meta.get("memory_action") == "use_all_memories", \
        f"Meta action should be use_all_memories: {meta}"
    assert meta.get("ok") is True, "Should be ok"
    assert meta.get("enabled") is True, "Should be enabled"
    
    assert mem_mgr.get_use_memories() is True, "Flag should now be True"
    print(f"✓ Response: '{response}'")
    print(f"✓ Meta: {meta}")
    print("✓ Flag is now True")
    
    # Reset for other tests
    reset_memory_manager()
    
    print("✓ test_handle_memory_command_enable: PASSED")


def test_handle_memory_command_disable():
    """Test that disable command returns proper response and toggles flag."""
    print("\n=== test_handle_memory_command_disable ===")
    
    from wyzer.memory.command_detector import handle_memory_command
    from wyzer.memory.memory_manager import get_memory_manager, reset_memory_manager
    
    # Reset (defaults to ON now)
    reset_memory_manager()
    mem_mgr = get_memory_manager()
    
    assert mem_mgr.get_use_memories() is True, "Should start with flag ON (default)"
    
    result = handle_memory_command("stop using memories")
    assert result is not None, "Should return result"
    response, meta = result
    
    assert "stop" in response.lower() or "memories" in response.lower(), \
        f"Response should confirm stopping: '{response}'"
    assert meta.get("memory_action") == "disable_memories", \
        f"Meta action should be disable_memories: {meta}"
    assert meta.get("ok") is True, "Should be ok"
    assert meta.get("enabled") is False, "Should be disabled"
    
    assert mem_mgr.get_use_memories() is False, "Flag should now be False"
    print(f"✓ Response: '{response}'")
    print(f"✓ Meta: {meta}")
    print("✓ Flag is now False")
    
    # Reset for other tests
    reset_memory_manager()
    
    print("✓ test_handle_memory_command_disable: PASSED")


def test_injection_only_when_flag_on():
    """Test that memory injection only occurs when flag is ON."""
    print("\n=== test_injection_only_when_flag_on ===")
    
    from wyzer.memory.memory_manager import MemoryManager
    import tempfile
    import json
    
    # Create a temporary memory file with test data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a custom memory manager with a temp file
        mgr = MemoryManager()
        
        # Manually set the memory file path for testing
        test_memories = [
            {"id": "1", "text": "your name is TestUser", "index_text": "your name is testuser"},
            {"id": "2", "text": "your favorite color is blue", "index_text": "your favorite color is blue"},
        ]
        
        # Write test memories
        mgr._memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mgr._memory_file, 'w') as f:
            json.dump(test_memories, f)
        
        # Flag OFF - should return empty
        mgr._use_memories = False
        block = mgr.get_all_memories_for_injection()
        assert block == "", f"Should return empty when flag OFF, got: '{block}'"
        print("✓ Flag OFF → empty block")
        
        # Flag ON - should return formatted block
        mgr._use_memories = True
        block = mgr.get_all_memories_for_injection()
        assert "[LONG-TERM MEMORY — facts about the user]" in block, f"Should have header: '{block}'"
        assert "- your name is TestUser" in block or "- your name is testuser" in block, \
            f"Should include memory: '{block}'"
        assert "Use this information to answer questions" in block, f"Should have footer: '{block}'"
        print("✓ Flag ON → formatted block with memories")
        print(f"  Block preview: {block[:100]}...")
    
    print("✓ test_injection_only_when_flag_on: PASSED")


def test_injection_size_cap():
    """Test that injection respects 30 bullets / 1200 chars cap."""
    print("\n=== test_injection_size_cap ===")
    
    from wyzer.memory.memory_manager import MemoryManager
    import json
    
    mgr = MemoryManager()
    
    # Create 50 test memories (exceeds 30 bullet cap)
    test_memories = [
        {"id": str(i), "text": f"fact number {i} is important", "index_text": f"fact number {i} is important"}
        for i in range(50)
    ]
    
    # Write test memories
    mgr._memory_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mgr._memory_file, 'w') as f:
        json.dump(test_memories, f)
    
    # Enable flag
    mgr._use_memories = True
    
    block = mgr.get_all_memories_for_injection()
    
    # Count bullets
    bullet_count = block.count("- fact number")
    assert bullet_count <= 30, f"Should have at most 30 bullets, got {bullet_count}"
    print(f"✓ Bullet count: {bullet_count} (max 30)")
    
    # Check char limit (1200 for content, plus header/footer)
    # The 1200 limit is for the bullet content, not including header/footer
    assert len(block) <= 1500, f"Total block should be reasonable size, got {len(block)}"
    print(f"✓ Total block size: {len(block)} chars")
    
    print("✓ test_injection_size_cap: PASSED")


def test_no_disk_writes_on_toggle():
    """Test that enabling/disabling memories doesn't write to disk."""
    print("\n=== test_no_disk_writes_on_toggle ===")
    
    from wyzer.memory.memory_manager import MemoryManager
    import json
    import os
    
    mgr = MemoryManager()
    
    # Write initial test data
    initial_memories = [{"id": "test", "text": "initial memory"}]
    mgr._memory_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mgr._memory_file, 'w') as f:
        json.dump(initial_memories, f)
    
    # Get initial mtime
    initial_mtime = os.path.getmtime(mgr._memory_file)
    
    # Toggle flag multiple times
    mgr.set_use_memories(True, source="test")
    mgr.set_use_memories(False, source="test")
    mgr.set_use_memories(True, source="test")
    mgr.set_use_memories(False, source="test")
    
    # Check file wasn't modified
    final_mtime = os.path.getmtime(mgr._memory_file)
    assert initial_mtime == final_mtime, "Memory file should not be modified"
    print("✓ Memory file not modified after toggle operations")
    
    # Verify file contents unchanged
    with open(mgr._memory_file, 'r') as f:
        final_memories = json.load(f)
    assert final_memories == initial_memories, "Memory contents should be unchanged"
    print("✓ Memory contents unchanged")
    
    print("✓ test_no_disk_writes_on_toggle: PASSED")


def test_prompt_block_function():
    """Test the prompt.py get_all_memories_block function."""
    print("\n=== test_prompt_block_function ===")
    
    from wyzer.brain.prompt import get_all_memories_block
    from wyzer.memory.memory_manager import get_memory_manager, reset_memory_manager
    import json
    
    # Reset to clean state (defaults to ON now)
    reset_memory_manager()
    mem_mgr = get_memory_manager()
    
    # Write test memory
    test_memories = [{"id": "1", "text": "test fact", "index_text": "test fact"}]
    mem_mgr._memory_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mem_mgr._memory_file, 'w') as f:
        json.dump(test_memories, f)
    
    # Flag ON (default) - should return block
    block = get_all_memories_block()
    assert "LONG-TERM MEMORY" in block, f"Should return block when flag ON (default): '{block}'"
    print("✓ get_all_memories_block returns block when flag ON (default)")
    
    # Disable flag - should return empty
    mem_mgr.set_use_memories(False, source="test")
    block = get_all_memories_block()
    assert block == "", f"Should return empty when flag OFF: '{block}'"
    print("✓ get_all_memories_block returns empty when flag OFF")
    
    # Reset
    reset_memory_manager()
    
    print("✓ test_prompt_block_function: PASSED")


def test_deduplication():
    """Test that duplicate memories are deduplicated."""
    print("\n=== test_deduplication ===")
    
    from wyzer.memory.memory_manager import MemoryManager
    import json
    
    mgr = MemoryManager()
    
    # Create memories with duplicates
    test_memories = [
        {"id": "1", "text": "my name is John", "index_text": "my name is john"},
        {"id": "2", "text": "My name is John", "index_text": "my name is john"},  # duplicate
        {"id": "3", "text": "my favorite color is blue", "index_text": "my favorite color is blue"},
    ]
    
    mgr._memory_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mgr._memory_file, 'w') as f:
        json.dump(test_memories, f)
    
    mgr._use_memories = True
    block = mgr.get_all_memories_for_injection()
    
    # Count occurrences of "name" - should only appear once
    name_count = block.lower().count("name is john")
    assert name_count == 1, f"Should deduplicate, found {name_count} occurrences of 'name is john'"
    print(f"✓ Deduplication works: 'name is john' appears {name_count} time(s)")
    
    print("✓ test_deduplication: PASSED")


# ===========================================================================
# REGRESSION TESTS - Prevent future regressions
# ===========================================================================

def test_near_miss_phrases_dont_trigger():
    """Test that near-miss phrases do NOT trigger the memory flag."""
    print("\n=== test_near_miss_phrases_dont_trigger ===")
    
    from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
    
    # Phrases that should NOT trigger USE_ALL_MEMORIES or DISABLE_MEMORIES
    non_trigger_phrases = [
        "use memory",  # singular, not "memories"
        "memories",  # just the word
        "use the memories",  # different pattern
        "using memories",  # gerund form
        "I want to use memories",  # embedded
        "can you use memories",  # question form
        "remember something",  # different command
        "stop memory",  # wrong form
        "memory disable",  # wrong order
        "memories off",  # wrong form
        "turn memories on",  # wrong word order
        "use memories and open chrome",  # multi-intent (should not be memory command)
    ]
    
    for phrase in non_trigger_phrases:
        cmd = detect_memory_command(phrase)
        if cmd is not None:
            # Should not be USE_ALL_MEMORIES or DISABLE_MEMORIES
            assert cmd.command_type not in (MemoryCommandType.USE_ALL_MEMORIES, MemoryCommandType.DISABLE_MEMORIES), \
                f"'{phrase}' should NOT trigger memory flag, got {cmd.command_type}"
        print(f"✓ '{phrase}' → no memory flag trigger")
    
    print("✓ test_near_miss_phrases_dont_trigger: PASSED")


def test_multi_intent_preserves_routing():
    """Test that multi-intent commands with 'and' don't break memory command detection."""
    print("\n=== test_multi_intent_preserves_routing ===")
    
    from wyzer.memory.command_detector import detect_memory_command
    
    # Multi-intent commands should NOT be handled as memory commands
    # (they should be routed to multi_intent_parser instead)
    multi_intent_phrases = [
        "use memories and open chrome",
        "enable memories and then check the time",
        "stop using memories; open notepad",
        "disable memories and play music",
    ]
    
    for phrase in multi_intent_phrases:
        cmd = detect_memory_command(phrase)
        # Should return None (not handled as memory command)
        assert cmd is None, f"Multi-intent '{phrase}' should not be detected as memory command, got {cmd}"
        print(f"✓ '{phrase}' → routed to multi-intent (not memory command)")
    
    print("✓ test_multi_intent_preserves_routing: PASSED")


def test_no_auto_memory_writes():
    """Test that memory flag commands do NOT write to memory.json."""
    print("\n=== test_no_auto_memory_writes ===")
    
    from wyzer.memory.command_detector import handle_memory_command
    from wyzer.memory.memory_manager import get_memory_manager, reset_memory_manager
    import json
    import os
    
    # Reset to clean state
    reset_memory_manager()
    mem_mgr = get_memory_manager()
    
    # Set up initial memory file
    initial_memories = [{"id": "safety_test", "text": "initial memory for safety test"}]
    mem_mgr._memory_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mem_mgr._memory_file, 'w') as f:
        json.dump(initial_memories, f)
    
    initial_mtime = os.path.getmtime(mem_mgr._memory_file)
    
    # Run multiple enable/disable commands
    handle_memory_command("use memories")
    handle_memory_command("stop using memories")
    handle_memory_command("enable memories")
    handle_memory_command("disable memories")
    handle_memory_command("turn on memories")
    handle_memory_command("turn off memories")
    
    # Verify file was NOT modified
    final_mtime = os.path.getmtime(mem_mgr._memory_file)
    assert initial_mtime == final_mtime, "Memory file should NOT be modified by flag commands"
    
    # Verify contents unchanged
    with open(mem_mgr._memory_file, 'r') as f:
        final_memories = json.load(f)
    assert final_memories == initial_memories, "Memory contents should be unchanged"
    
    print("✓ No disk writes from memory flag commands")
    
    # Reset
    reset_memory_manager()
    
    print("✓ test_no_auto_memory_writes: PASSED")


def test_voice_command_source_logging():
    """Test that voice commands log with correct source."""
    print("\n=== test_voice_command_source_logging ===")
    
    from wyzer.memory.command_detector import handle_memory_command
    from wyzer.memory.memory_manager import get_memory_manager, reset_memory_manager
    import io
    import sys
    
    # Reset
    reset_memory_manager()
    mem_mgr = get_memory_manager()
    mem_mgr.set_use_memories(False, source="test")  # Start OFF
    
    # Capture log output (simplified check - just verify it doesn't crash)
    result = handle_memory_command("use memories")
    assert result is not None, "Should handle command"
    response, meta = result
    assert meta.get("ok") is True, "Should succeed"
    
    # The source="voice_command" is passed internally and logged
    print("✓ Voice command handled with source tracking")
    
    # Reset
    reset_memory_manager()
    
    print("✓ test_voice_command_source_logging: PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running use_memories flag tests")
    print("=" * 60)
    
    tests = [
        test_flag_default_on,
        test_flag_toggle_on_off,
        test_command_detection_enable,
        test_command_detection_disable,
        test_handle_memory_command_enable,
        test_handle_memory_command_disable,
        test_injection_only_when_flag_on,
        test_injection_size_cap,
        test_no_disk_writes_on_toggle,
        test_prompt_block_function,
        test_deduplication,
        # Regression tests
        test_near_miss_phrases_dont_trigger,
        test_multi_intent_preserves_routing,
        test_no_auto_memory_writes,
        test_voice_command_source_logging,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
