"""
Tests for unknown tool handling - graceful degradation instead of hard failure.

These tests verify:
1. Unknown tool doesn't abort multi-intent
2. All intents unknown → reply fallback
3. Informational query bypasses tool-planning
4. Tool alias normalization works correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wyzer.core.intent_plan import Intent, filter_unknown_tools, validate_intents, normalize_tool_aliases
from wyzer.core.orchestrator import is_informational_query


class MockRegistry:
    """Mock tool registry for testing"""
    def __init__(self, known_tools: list):
        self._tools = set(known_tools)
    
    def has_tool(self, name: str) -> bool:
        return name in self._tools
    
    def list_tools(self) -> list:
        return [{"name": t, "description": ""} for t in self._tools]
    
    def get(self, name: str):
        if name in self._tools:
            return type('Tool', (), {'description': '', 'args_schema': {}})()
        return None


def test_filter_unknown_tools_mixed():
    """Unknown tool doesn't abort multi-intent - only unknown is filtered."""
    registry = MockRegistry(["open_website", "get_time"])
    
    intents = [
        Intent(tool="get_recommendations", args={}),  # Unknown
        Intent(tool="open_website", args={"url": "youtube.com"}),  # Valid
    ]
    
    valid, unknown = filter_unknown_tools(intents, registry)
    
    assert len(valid) == 1, f"Expected 1 valid intent, got {len(valid)}"
    assert valid[0].tool == "open_website"
    assert unknown == ["get_recommendations"]
    print("✓ Mixed intents: unknown filtered, valid kept")


def test_filter_unknown_tools_all_unknown():
    """All intents unknown → empty valid list, all in unknown."""
    registry = MockRegistry(["open_website", "get_time"])
    
    intents = [
        Intent(tool="get_recommendations", args={}),
        Intent(tool="search_web", args={}),
    ]
    
    valid, unknown = filter_unknown_tools(intents, registry)
    
    assert len(valid) == 0, f"Expected 0 valid intents, got {len(valid)}"
    assert set(unknown) == {"get_recommendations", "search_web"}
    print("✓ All unknown: empty valid list returned")


def test_filter_unknown_tools_all_valid():
    """All intents valid → no filtering."""
    registry = MockRegistry(["open_website", "get_time"])
    
    intents = [
        Intent(tool="open_website", args={"url": "google.com"}),
        Intent(tool="get_time", args={}),
    ]
    
    valid, unknown = filter_unknown_tools(intents, registry)
    
    assert len(valid) == 2
    assert len(unknown) == 0
    print("✓ All valid: no filtering")


def test_validate_intents_after_filter():
    """validate_intents works on filtered intents (no unknown tool error)."""
    registry = MockRegistry(["open_website"])
    
    intents = [
        Intent(tool="open_website", args={"url": "example.com"}),
    ]
    
    # Should not raise
    validate_intents(intents, registry)
    print("✓ validate_intents works after filtering")


def test_informational_query_anime_like():
    """'What's an anime like One Piece' is informational."""
    assert is_informational_query("What's an anime like One Piece?")
    assert is_informational_query("what is an anime like naruto")
    assert is_informational_query("What's a movie like Inception")
    print("✓ Anime/movie 'like X' detected as informational")


def test_informational_query_recommend():
    """'Recommend me...' is informational."""
    assert is_informational_query("recommend me a good anime")
    assert is_informational_query("Recommend a movie for tonight")
    print("✓ Recommend detected as informational")


def test_informational_query_what_should_watch():
    """'What should I watch' is informational (no action words)."""
    assert is_informational_query("what should I watch next")
    assert is_informational_query("What should I read tonight")
    print("✓ 'What should I watch' detected as informational")


def test_informational_query_what_should_with_action():
    """'What should I do about storage' has action word - NOT informational."""
    # These contain action words, so should NOT be purely informational
    assert not is_informational_query("what should I do to scan my storage")
    assert not is_informational_query("what should I set the volume to")
    assert not is_informational_query("what should I open")
    print("✓ 'What should I' with action words NOT informational")


def test_not_informational_action_queries():
    """Action queries should NOT be informational."""
    assert not is_informational_query("open chrome")
    assert not is_informational_query("set volume to 50")
    assert not is_informational_query("scan my drives")
    assert not is_informational_query("open chrome like last time")  # "like" in action context
    print("✓ Action queries NOT detected as informational")


# ============================================================================
# ALIAS NORMALIZATION TESTS
# ============================================================================

def test_alias_normalization_pause_media():
    """pause_media should normalize to media_play_pause."""
    registry = MockRegistry(["media_play_pause", "get_time"])
    
    intents = [
        Intent(tool="pause_media", args={}),
    ]
    
    normalized, logs = normalize_tool_aliases(intents, registry)
    
    assert len(normalized) == 1
    assert normalized[0].tool == "media_play_pause"
    assert len(logs) == 1
    assert "pause_media" in logs[0]
    assert "media_play_pause" in logs[0]
    print("✓ pause_media → media_play_pause normalization works")


def test_alias_normalization_multiple():
    """Multiple aliases in one request should all normalize."""
    registry = MockRegistry(["media_play_pause", "volume_control", "open_target"])
    
    intents = [
        Intent(tool="pause_music", args={}),
        Intent(tool="set_volume", args={"level": 50}),
        Intent(tool="launch", args={"query": "chrome"}),
    ]
    
    normalized, logs = normalize_tool_aliases(intents, registry)
    
    assert len(normalized) == 3
    assert normalized[0].tool == "media_play_pause"
    assert normalized[1].tool == "volume_control"
    assert normalized[2].tool == "open_target"
    assert len(logs) == 3
    print("✓ Multiple alias normalization works")


def test_alias_normalization_preserves_args():
    """Alias normalization should preserve args."""
    registry = MockRegistry(["volume_control"])
    
    intents = [
        Intent(tool="set_volume", args={"level": 75}),
    ]
    
    normalized, logs = normalize_tool_aliases(intents, registry)
    
    assert normalized[0].tool == "volume_control"
    assert normalized[0].args == {"level": 75}
    print("✓ Alias normalization preserves args")


def test_alias_normalization_unknown_canonical():
    """If canonical tool doesn't exist, don't normalize."""
    registry = MockRegistry(["get_time"])  # media_play_pause NOT registered
    
    intents = [
        Intent(tool="pause_media", args={}),
    ]
    
    normalized, logs = normalize_tool_aliases(intents, registry)
    
    # Should NOT normalize because media_play_pause isn't registered
    assert len(normalized) == 1
    assert normalized[0].tool == "pause_media"  # Unchanged
    assert len(logs) == 0  # No normalization logged
    print("✓ Alias normalization skips when canonical tool doesn't exist")


def test_alias_normalization_mixed_valid_and_unknown():
    """Mix of aliases and unknown tools should work correctly."""
    registry = MockRegistry(["media_play_pause", "open_target"])
    
    intents = [
        Intent(tool="pause_media", args={}),         # Alias → media_play_pause
        Intent(tool="open_target", args={}),          # Already valid
        Intent(tool="invented_tool", args={}),        # Unknown, stays unknown
    ]
    
    normalized, logs = normalize_tool_aliases(intents, registry)
    
    assert len(normalized) == 3
    assert normalized[0].tool == "media_play_pause"  # Normalized
    assert normalized[1].tool == "open_target"        # Unchanged
    assert normalized[2].tool == "invented_tool"      # Unchanged (stays unknown)
    assert len(logs) == 1  # Only one normalization
    print("✓ Mixed alias and unknown tool handling works")


def test_alias_then_filter_workflow():
    """Full workflow: normalize aliases then filter unknown."""
    registry = MockRegistry(["media_play_pause", "open_target"])
    
    intents = [
        Intent(tool="pause_media", args={}),         # Alias → media_play_pause (valid)
        Intent(tool="invented_tool", args={}),       # Unknown
    ]
    
    # Step 1: Normalize aliases
    normalized, logs = normalize_tool_aliases(intents, registry)
    
    # Step 2: Filter unknown tools
    valid, unknown = filter_unknown_tools(normalized, registry)
    
    assert len(valid) == 1
    assert valid[0].tool == "media_play_pause"
    assert unknown == ["invented_tool"]
    print("✓ Alias normalization + unknown filtering workflow works")


if __name__ == "__main__":
    test_filter_unknown_tools_mixed()
    test_filter_unknown_tools_all_unknown()
    test_filter_unknown_tools_all_valid()
    test_validate_intents_after_filter()
    test_informational_query_anime_like()
    test_informational_query_recommend()
    test_informational_query_what_should_watch()
    test_informational_query_what_should_with_action()
    test_not_informational_action_queries()
    
    # Alias normalization tests
    test_alias_normalization_pause_media()
    test_alias_normalization_multiple()
    test_alias_normalization_preserves_args()
    test_alias_normalization_unknown_canonical()
    test_alias_normalization_mixed_valid_and_unknown()
    test_alias_then_filter_workflow()
    
    print("\n✅ All tests passed!")
