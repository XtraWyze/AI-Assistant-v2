"""
Test suite for Phase 11 Long-Term Memory Expansion.

Tests:
1. Structured memory model migration (legacy format -> Phase 11)
2. add_explicit creates proper structured records
3. search finds correct items by value/key/tags
4. delete_by_query deletes matching and persists
5. export_to writes valid JSON
6. import_from loads and deduplicates
7. Command detection for new Phase 11 commands
8. Backward compatibility with Phase 7 format

Run with:
    python scripts/test_memory_phase11.py
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_test_memory_file(content: list) -> tuple:
    """Create a temp directory with a memory.json file for testing."""
    # Create temp data directory
    temp_dir = tempfile.mkdtemp(prefix="wyzer_test_")
    data_dir = Path(temp_dir) / "wyzer" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Write memory file
    memory_file = data_dir / "memory.json"
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=2)
    
    return temp_dir, memory_file


def cleanup_test_dir(temp_dir: str):
    """Remove temp directory."""
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


# =============================================================================
# Migration Tests
# =============================================================================

def test_migrate_plain_strings():
    """Test migration of plain string memories to Phase 11 format."""
    print("\n=== test_migrate_plain_strings ===")
    
    from wyzer.memory.memory_manager import _migrate_legacy_entry, MEMORY_RECORD_TYPES
    
    # Plain string
    entry = "my name is Levi"
    migrated = _migrate_legacy_entry(entry)
    
    assert migrated.get("value") == entry, f"Value should be preserved, got {migrated.get('value')}"
    assert migrated.get("type") in MEMORY_RECORD_TYPES, f"Type should be valid, got {migrated.get('type')}"
    assert migrated.get("key") == "name", f"Key should be derived as 'name', got {migrated.get('key')}"
    assert migrated.get("source") == "explicit_user", f"Source should be 'explicit_user'"
    assert "id" in migrated, "Should have an id"
    assert "created_at" in migrated, "Should have created_at"
    print("✓ Plain string migrated correctly")
    
    # Preference string
    entry2 = "I like pizza"
    migrated2 = _migrate_legacy_entry(entry2)
    assert migrated2.get("type") == "preference", f"Should detect preference type, got {migrated2.get('type')}"
    assert "likes_" in (migrated2.get("key") or ""), f"Key should start with 'likes_', got {migrated2.get('key')}"
    print("✓ Preference string migrated with correct type")
    
    print("✓ test_migrate_plain_strings PASSED")


def test_migrate_phase7_dict():
    """Test migration of Phase 7 dict format (with 'text', without 'value')."""
    print("\n=== test_migrate_phase7_dict ===")
    
    from wyzer.memory.memory_manager import _migrate_legacy_entry
    
    # Phase 7 format
    entry = {
        "id": "test-id-123",
        "created_at": "2025-01-01T00:00:00Z",
        "text": "my birthday is September 10th",
        "index_text": "my birthday is september 10th",
        "tags": ["personal"]
    }
    
    migrated = _migrate_legacy_entry(entry)
    
    assert migrated.get("id") == "test-id-123", "ID should be preserved"
    assert migrated.get("value") == "my birthday is September 10th", "Value should come from 'text'"
    assert migrated.get("text") == "my birthday is September 10th", "Text should be kept for compat"
    assert migrated.get("key") == "birthday", f"Key should be derived as 'birthday', got {migrated.get('key')}"
    assert migrated.get("tags") == ["personal"], "Tags should be preserved"
    assert migrated.get("type") == "fact", "Type should be 'fact' for birthday"
    print("✓ Phase 7 dict migrated correctly")
    
    print("✓ test_migrate_phase7_dict PASSED")


def test_migrate_already_phase11():
    """Test that Phase 11 format entries pass through unchanged."""
    print("\n=== test_migrate_already_phase11 ===")
    
    from wyzer.memory.memory_manager import _migrate_legacy_entry
    
    entry = {
        "id": "phase11-id",
        "type": "preference",
        "key": "favorite_color",
        "value": "blue",
        "tags": ["colors"],
        "pinned": False,
        "aliases": [],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": None,
        "source": "explicit_user"
    }
    
    migrated = _migrate_legacy_entry(entry)
    
    # Core fields should be preserved
    assert migrated.get("id") == entry.get("id"), "ID should match"
    assert migrated.get("type") == entry.get("type"), "Type should match"
    assert migrated.get("key") == entry.get("key"), "Key should match"
    assert migrated.get("value") == entry.get("value"), "Value should match"
    assert migrated.get("tags") == entry.get("tags"), "Tags should match"
    assert migrated.get("pinned") == entry.get("pinned"), "Pinned should match"
    assert migrated.get("aliases") == entry.get("aliases"), "Aliases should match"
    assert migrated.get("source") == entry.get("source"), "Source should match"
    print("✓ Phase 11 entry fields preserved correctly")
    
    print("✓ test_migrate_already_phase11 PASSED")


# =============================================================================
# API Tests
# =============================================================================

def test_add_explicit():
    """Test add_explicit creates proper structured records."""
    print("\n=== test_add_explicit ===")
    
    from wyzer.memory.memory_manager import MemoryManager, reset_memory_manager
    
    # Reset singleton
    reset_memory_manager()
    
    # Create manager with temp file
    mgr = MemoryManager()
    
    # Patch memory file to temp location
    temp_dir = tempfile.mkdtemp(prefix="wyzer_test_")
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    mgr._memory_file = data_dir / "memory.json"
    
    try:
        # Add a memory
        result = mgr.add_explicit("my dog's name is Kush", tags=["pets"])
        
        assert result.get("ok") is True, f"Should succeed, got {result}"
        record = result.get("record")
        assert record is not None, "Should return record"
        assert record.get("value") == "my dog's name is Kush"
        assert record.get("type") == "fact"
        assert record.get("tags") == ["pets"]
        assert record.get("source") == "explicit_user"
        print("✓ add_explicit created structured record")
        
        # Verify persistence
        memories = mgr.list_all()
        assert len(memories) == 1, f"Should have 1 memory, got {len(memories)}"
        print("✓ Memory persisted to disk")
        
        # Add another and verify deduplication
        result2 = mgr.add_explicit("My dog's name is Kush.")  # Same normalized
        assert result2.get("replaced") is True, "Should replace duplicate"
        
        memories = mgr.list_all()
        assert len(memories) == 1, f"Should still have 1 memory after dedupe, got {len(memories)}"
        print("✓ Deduplication working")
        
    finally:
        cleanup_test_dir(temp_dir)
        reset_memory_manager()
    
    print("✓ test_add_explicit PASSED")


def test_search():
    """Test search finds correct items."""
    print("\n=== test_search ===")
    
    from wyzer.memory.memory_manager import MemoryManager, reset_memory_manager
    
    reset_memory_manager()
    mgr = MemoryManager()
    
    temp_dir = tempfile.mkdtemp(prefix="wyzer_test_")
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    mgr._memory_file = data_dir / "memory.json"
    
    try:
        # Add test memories
        mgr.add_explicit("my name is Levi")
        mgr.add_explicit("my dog's name is Kush")
        mgr.add_explicit("I like pizza", record_type="preference")
        mgr.add_explicit("my favorite color is blue", tags=["colors"])
        
        # Search by value
        matches = mgr.search("Kush")
        assert len(matches) == 1, f"Should find 1 match for 'Kush', got {len(matches)}"
        assert "kush" in matches[0].get("value", "").lower()
        print("✓ Search by value works")
        
        # Search by partial match
        matches = mgr.search("name")
        assert len(matches) == 2, f"Should find 2 matches for 'name', got {len(matches)}"
        print("✓ Search by partial match works")
        
        # Search by tag
        matches = mgr.search("colors")
        assert len(matches) == 1, f"Should find 1 match for 'colors' tag, got {len(matches)}"
        print("✓ Search by tag works")
        
        # Search no matches
        matches = mgr.search("nonexistent")
        assert len(matches) == 0, f"Should find 0 matches, got {len(matches)}"
        print("✓ Search returns empty for no matches")
        
    finally:
        cleanup_test_dir(temp_dir)
        reset_memory_manager()
    
    print("✓ test_search PASSED")


def test_delete_by_query():
    """Test delete_by_query deletes matching and persists."""
    print("\n=== test_delete_by_query ===")
    
    from wyzer.memory.memory_manager import MemoryManager, reset_memory_manager
    
    reset_memory_manager()
    mgr = MemoryManager()
    
    temp_dir = tempfile.mkdtemp(prefix="wyzer_test_")
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    mgr._memory_file = data_dir / "memory.json"
    
    try:
        # Add test memories
        mgr.add_explicit("my name is Levi")
        mgr.add_explicit("my dog's name is Kush")
        mgr.add_explicit("I like pizza")
        
        assert len(mgr.list_all()) == 3
        print("✓ Added 3 test memories")
        
        # Delete by query
        deleted = mgr.delete_by_query("name")
        assert deleted == 2, f"Should delete 2 memories with 'name', got {deleted}"
        print("✓ Deleted 2 memories matching 'name'")
        
        # Verify persistence
        remaining = mgr.list_all()
        assert len(remaining) == 1, f"Should have 1 remaining, got {len(remaining)}"
        assert "pizza" in remaining[0].get("value", "").lower()
        print("✓ Deletion persisted correctly")
        
        # Delete non-matching
        deleted = mgr.delete_by_query("nonexistent")
        assert deleted == 0, f"Should delete 0, got {deleted}"
        print("✓ Delete returns 0 for no matches")
        
    finally:
        cleanup_test_dir(temp_dir)
        reset_memory_manager()
    
    print("✓ test_delete_by_query PASSED")


def test_export_to():
    """Test export_to writes valid JSON."""
    print("\n=== test_export_to ===")
    
    from wyzer.memory.memory_manager import MemoryManager, reset_memory_manager
    
    reset_memory_manager()
    mgr = MemoryManager()
    
    temp_dir = tempfile.mkdtemp(prefix="wyzer_test_")
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    mgr._memory_file = data_dir / "memory.json"
    
    try:
        # Add test memories
        mgr.add_explicit("my name is Test User")
        mgr.add_explicit("I like testing", record_type="preference")
        
        # Export
        export_path = mgr.export_to()
        assert os.path.exists(export_path), f"Export file should exist at {export_path}"
        print(f"✓ Exported to {export_path}")
        
        # Verify JSON structure
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data.get("version") == "phase11", "Should have version"
        assert "exported_at" in data, "Should have exported_at"
        assert data.get("count") == 2, f"Should have count=2, got {data.get('count')}"
        assert isinstance(data.get("memories"), list), "Should have memories list"
        assert len(data["memories"]) == 2
        print("✓ Export JSON structure is valid")
        
    finally:
        cleanup_test_dir(temp_dir)
        reset_memory_manager()
    
    print("✓ test_export_to PASSED")


def test_import_from():
    """Test import_from loads and deduplicates."""
    print("\n=== test_import_from ===")
    
    from wyzer.memory.memory_manager import MemoryManager, reset_memory_manager
    
    reset_memory_manager()
    mgr = MemoryManager()
    
    temp_dir = tempfile.mkdtemp(prefix="wyzer_test_")
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    mgr._memory_file = data_dir / "memory.json"
    
    try:
        # Add existing memory
        mgr.add_explicit("my name is Levi")
        
        # Create import file
        import_file = data_dir / "import_test.json"
        import_data = {
            "version": "phase11",
            "memories": [
                {"value": "my name is Levi", "type": "fact"},  # Duplicate
                {"value": "I like cats", "type": "preference"},  # New
                {"text": "my age is 28", "tags": []}  # Legacy format
            ]
        }
        with open(import_file, 'w', encoding='utf-8') as f:
            json.dump(import_data, f)
        
        # Import
        imported = mgr.import_from(str(import_file))
        assert imported == 2, f"Should import 2 new memories (1 dupe), got {imported}"
        print("✓ Imported 2 new memories, skipped 1 duplicate")
        
        # Verify total
        all_mems = mgr.list_all()
        assert len(all_mems) == 3, f"Should have 3 total, got {len(all_mems)}"
        print("✓ Total memories correct after import")
        
        # Test file not found
        try:
            mgr.import_from("/nonexistent/path.json")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            print("✓ FileNotFoundError raised for missing file")
        
    finally:
        cleanup_test_dir(temp_dir)
        reset_memory_manager()
    
    print("✓ test_import_from PASSED")


# =============================================================================
# Command Detection Tests
# =============================================================================

def test_command_detection_list_all():
    """Test LIST_ALL command detection."""
    print("\n=== test_command_detection_list_all ===")
    
    from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
    
    # Should match LIST_ALL
    test_cases = [
        "What do you remember about me?",
        "what do you remember about me",
        "What do you know about me?",
        "tell me everything you know about me",
    ]
    
    for text in test_cases:
        cmd = detect_memory_command(text)
        assert cmd is not None, f"Should detect command for '{text}'"
        assert cmd.command_type == MemoryCommandType.LIST_ALL, \
            f"Should be LIST_ALL for '{text}', got {cmd.command_type}"
    
    print("✓ LIST_ALL patterns detected correctly")
    print("✓ test_command_detection_list_all PASSED")


def test_command_detection_search():
    """Test SEARCH command detection."""
    print("\n=== test_command_detection_search ===")
    
    from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
    
    # Should match SEARCH
    test_cases = [
        ("What do you remember about my dog?", "my dog"),
        ("what do you know about pizza", "pizza"),
        ("search memories for birthday", "birthday"),
    ]
    
    for text, expected_query in test_cases:
        cmd = detect_memory_command(text)
        assert cmd is not None, f"Should detect command for '{text}'"
        assert cmd.command_type == MemoryCommandType.SEARCH, \
            f"Should be SEARCH for '{text}', got {cmd.command_type}"
        assert expected_query.lower() in cmd.text.lower(), \
            f"Query should contain '{expected_query}', got '{cmd.text}'"
    
    print("✓ SEARCH patterns detected correctly")
    print("✓ test_command_detection_search PASSED")


def test_command_detection_delete():
    """Test DELETE command detection."""
    print("\n=== test_command_detection_delete ===")
    
    from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
    
    # Should match DELETE
    test_cases = [
        ("forget everything about my dog", "my dog"),
        ("forget all about pizza", "pizza"),
        ("delete everything about work", "work"),
        ("delete all memories about cats", "cats"),
    ]
    
    for text, expected_query in test_cases:
        cmd = detect_memory_command(text)
        assert cmd is not None, f"Should detect command for '{text}'"
        assert cmd.command_type == MemoryCommandType.DELETE, \
            f"Should be DELETE for '{text}', got {cmd.command_type}"
        assert expected_query.lower() in cmd.text.lower(), \
            f"Query should contain '{expected_query}', got '{cmd.text}'"
    
    print("✓ DELETE patterns detected correctly")
    print("✓ test_command_detection_delete PASSED")


def test_command_detection_export():
    """Test EXPORT command detection."""
    print("\n=== test_command_detection_export ===")
    
    from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
    
    # Should match EXPORT
    test_cases = [
        "export my memory",
        "export memory",
        "export my memories",
        "backup my memories",
    ]
    
    for text in test_cases:
        cmd = detect_memory_command(text)
        assert cmd is not None, f"Should detect command for '{text}'"
        assert cmd.command_type == MemoryCommandType.EXPORT, \
            f"Should be EXPORT for '{text}', got {cmd.command_type}"
    
    print("✓ EXPORT patterns detected correctly")
    print("✓ test_command_detection_export PASSED")


def test_command_detection_import():
    """Test IMPORT command detection."""
    print("\n=== test_command_detection_import ===")
    
    from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
    
    # Should match IMPORT
    test_cases = [
        ("import memories from backup.json", "backup.json"),
        ("import memory from C:/data/export.json", "C:/data/export.json"),
        ("load memories from memory_export.json", "memory_export.json"),
    ]
    
    for text, expected_path in test_cases:
        cmd = detect_memory_command(text)
        assert cmd is not None, f"Should detect command for '{text}'"
        assert cmd.command_type == MemoryCommandType.IMPORT, \
            f"Should be IMPORT for '{text}', got {cmd.command_type}"
        assert expected_path.lower() in cmd.text.lower(), \
            f"Path should contain '{expected_path}', got '{cmd.text}'"
    
    print("✓ IMPORT patterns detected correctly")
    print("✓ test_command_detection_import PASSED")


def test_forget_vs_delete_distinction():
    """Test that 'forget X' (FORGET) is distinct from 'forget everything about X' (DELETE)."""
    print("\n=== test_forget_vs_delete_distinction ===")
    
    from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
    
    # "forget X" should be FORGET (legacy Phase 7 behavior)
    cmd1 = detect_memory_command("forget my name")
    assert cmd1 is not None
    assert cmd1.command_type == MemoryCommandType.FORGET, \
        f"'forget my name' should be FORGET, got {cmd1.command_type}"
    print("✓ 'forget X' -> FORGET")
    
    # "forget everything about X" should be DELETE (Phase 11)
    cmd2 = detect_memory_command("forget everything about my name")
    assert cmd2 is not None
    assert cmd2.command_type == MemoryCommandType.DELETE, \
        f"'forget everything about X' should be DELETE, got {cmd2.command_type}"
    print("✓ 'forget everything about X' -> DELETE")
    
    # "forget that" should be FORGET_LAST (legacy)
    cmd3 = detect_memory_command("forget that")
    assert cmd3 is not None
    assert cmd3.command_type == MemoryCommandType.FORGET_LAST, \
        f"'forget that' should be FORGET_LAST, got {cmd3.command_type}"
    print("✓ 'forget that' -> FORGET_LAST")
    
    print("✓ test_forget_vs_delete_distinction PASSED")


def test_multi_intent_not_captured():
    """Test that multi-intent commands are NOT treated as memory commands."""
    print("\n=== test_multi_intent_not_captured ===")
    
    from wyzer.memory.command_detector import detect_memory_command
    
    # These should NOT be memory commands (multi-intent separators present)
    test_cases = [
        "open chrome and then remember my name is Levi",
        "what's the weather; remember to bring umbrella",
        "search for cats and then export my memory",
    ]
    
    for text in test_cases:
        cmd = detect_memory_command(text)
        assert cmd is None, f"Should NOT detect memory command in multi-intent: '{text}'"
    
    print("✓ Multi-intent commands correctly bypass memory detection")
    print("✓ test_multi_intent_not_captured PASSED")


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

def test_backward_compat_phase7_list():
    """Test that Phase 7 list_memories still works."""
    print("\n=== test_backward_compat_phase7_list ===")
    
    from wyzer.memory.memory_manager import MemoryManager, reset_memory_manager
    
    reset_memory_manager()
    mgr = MemoryManager()
    
    temp_dir = tempfile.mkdtemp(prefix="wyzer_test_")
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    mgr._memory_file = data_dir / "memory.json"
    
    try:
        # Add using Phase 11 API
        mgr.add_explicit("test memory")
        
        # Use Phase 7 API
        memories = mgr.list_memories()
        assert len(memories) == 1
        assert memories[0].get("text") == "test memory"  # Phase 7 field
        assert memories[0].get("value") == "test memory"  # Phase 11 field
        print("✓ list_memories returns both Phase 7 and Phase 11 fields")
        
    finally:
        cleanup_test_dir(temp_dir)
        reset_memory_manager()
    
    print("✓ test_backward_compat_phase7_list PASSED")


def test_backward_compat_remember_command():
    """Test that 'remember X' still works and creates Phase 11 records."""
    print("\n=== test_backward_compat_remember_command ===")
    
    from wyzer.memory.memory_manager import MemoryManager, reset_memory_manager
    from wyzer.memory.command_detector import detect_memory_command, MemoryCommandType
    
    # Verify command detection
    cmd = detect_memory_command("remember my name is Test")
    assert cmd is not None
    assert cmd.command_type == MemoryCommandType.REMEMBER
    assert "my name is Test" in cmd.text
    print("✓ 'remember X' command still detected")
    
    # Verify manager.remember() works
    reset_memory_manager()
    mgr = MemoryManager()
    
    temp_dir = tempfile.mkdtemp(prefix="wyzer_test_")
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    mgr._memory_file = data_dir / "memory.json"
    
    try:
        result = mgr.remember("my name is Test")
        assert result.get("ok") is True
        
        memories = mgr.list_all()
        assert len(memories) == 1
        # Should have Phase 11 fields populated via migration
        mem = memories[0]
        assert mem.get("value") == "my name is Test"
        assert mem.get("type") == "fact"
        print("✓ manager.remember() creates Phase 11 compatible records")
        
    finally:
        cleanup_test_dir(temp_dir)
        reset_memory_manager()
    
    print("✓ test_backward_compat_remember_command PASSED")


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all Phase 11 tests."""
    print("=" * 60)
    print("Phase 11 Long-Term Memory Expansion Tests")
    print("=" * 60)
    
    tests = [
        # Migration tests
        test_migrate_plain_strings,
        test_migrate_phase7_dict,
        test_migrate_already_phase11,
        
        # API tests
        test_add_explicit,
        test_search,
        test_delete_by_query,
        test_export_to,
        test_import_from,
        
        # Command detection tests
        test_command_detection_list_all,
        test_command_detection_search,
        test_command_detection_delete,
        test_command_detection_export,
        test_command_detection_import,
        test_forget_vs_delete_distinction,
        test_multi_intent_not_captured,
        
        # Backward compatibility
        test_backward_compat_phase7_list,
        test_backward_compat_remember_command,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_fn.__name__} FAILED: {e}")
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
