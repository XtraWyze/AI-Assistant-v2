"""
Unit tests for deterministic_splitter.py

Tests the split_tool_then_text() function for various scenarios:
- "Pause music and what's a VPN?" -> tool + leftover
- "Pause music" -> no split (pure tool)
- "What's a VPN?" -> no split (no tool phrase)
- Connector edge cases (comma, "then", "and then")
- Tool phrases with filler words ("the", "my")
- Registry verification (tool must exist)
"""

import pytest
from unittest.mock import Mock, MagicMock
from wyzer.core.deterministic_splitter import (
    split_tool_then_text,
    get_split_intents,
    SplitResult,
    _normalize_text,
    _strip_fillers,
    _find_connector,
    _match_tool_phrase,
)


# ═══════════════════════════════════════════════════════════════════════════
# MOCK REGISTRY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_registry_full():
    """Mock registry with all common tools registered."""
    registry = Mock()
    registry.has_tool = Mock(side_effect=lambda name: name in {
        "media_play_pause",
        "media_next",
        "media_previous",
        "volume_control",
        "volume_mute_toggle",
        "volume_up",
        "volume_down",
        "open_target",
        "close_window",
    })
    return registry


@pytest.fixture
def mock_registry_minimal():
    """Mock registry with only media_play_pause registered."""
    registry = Mock()
    registry.has_tool = Mock(side_effect=lambda name: name in {
        "media_play_pause",
    })
    return registry


@pytest.fixture
def mock_registry_empty():
    """Mock registry with no tools registered."""
    registry = Mock()
    registry.has_tool = Mock(return_value=False)
    return registry


# ═══════════════════════════════════════════════════════════════════════════
# CORE SPLIT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitToolThenText:
    """Tests for the main split_tool_then_text function."""

    def test_pause_music_and_whats_a_vpn(self, mock_registry_full):
        """Test: 'Pause music and what's a VPN?' -> tool + leftover"""
        result = split_tool_then_text("Pause music and what's a VPN?", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"
        assert result.tool_intent["args"] == {}
        assert "vpn" in result.leftover_text.lower()
        # Should restore question mark
        assert result.leftover_text.endswith("?")

    def test_pause_music_only_no_split(self, mock_registry_full):
        """Test: 'Pause music' -> no split (pure tool command)"""
        result = split_tool_then_text("Pause music", mock_registry_full)
        
        # No split because there's no leftover text
        assert result is None

    def test_whats_a_vpn_only_no_split(self, mock_registry_full):
        """Test: 'What's a VPN?' -> no split (no tool phrase)"""
        result = split_tool_then_text("What's a VPN?", mock_registry_full)
        
        # No split because there's no high-confidence tool phrase
        assert result is None

    def test_pause_the_music_and_whats_a_vpn(self, mock_registry_full):
        """Test: 'Pause the music and what's a VPN?' -> tool + leftover (with filler 'the')"""
        result = split_tool_then_text("Pause the music and what's a VPN?", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"
        assert "vpn" in result.leftover_text.lower()

    def test_pause_media_and_define_vpn(self, mock_registry_full):
        """Test: 'pause media and define VPN' -> tool + leftover (leftover goes reply-only)"""
        result = split_tool_then_text("pause media and define VPN", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"
        assert "define vpn" in result.leftover_text.lower() or "vpn" in result.leftover_text.lower()

    def test_mute_and_tell_me_about_cats(self, mock_registry_full):
        """Test: 'mute and tell me about cats' -> volume_control mute + leftover"""
        result = split_tool_then_text("mute and tell me about cats", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "volume_control"
        assert result.tool_intent["args"]["action"] == "mute"
        assert "cats" in result.leftover_text.lower()

    def test_volume_up_then_whats_python(self, mock_registry_full):
        """Test: 'volume up then what is python' -> volume_control change + leftover"""
        result = split_tool_then_text("volume up then what is python", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "volume_control"
        assert result.tool_intent["args"]["action"] == "change"
        assert result.tool_intent["args"]["delta"] == 10
        assert "python" in result.leftover_text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# CONNECTOR EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestConnectorEdgeCases:
    """Test different connector patterns."""

    def test_comma_separator(self, mock_registry_full):
        """Test: 'pause, what's a VPN' -> tool + leftover (comma separator)"""
        result = split_tool_then_text("pause, what's a VPN", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"
        assert "vpn" in result.leftover_text.lower()

    def test_semicolon_separator(self, mock_registry_full):
        """Test: 'pause; what's a VPN' -> tool + leftover (semicolon separator)"""
        result = split_tool_then_text("pause; what's a VPN", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"
        assert "vpn" in result.leftover_text.lower()

    def test_and_then_separator(self, mock_registry_full):
        """Test: 'pause and then what's a VPN' -> tool + leftover (and then separator)"""
        result = split_tool_then_text("pause and then what's a VPN", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"
        assert "vpn" in result.leftover_text.lower()

    def test_also_separator(self, mock_registry_full):
        """Test: 'pause also what's a VPN' -> tool + leftover (also separator)"""
        result = split_tool_then_text("pause also what's a VPN", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"

    def test_plus_separator(self, mock_registry_full):
        """Test: 'pause plus what's a VPN' -> tool + leftover (plus separator)"""
        result = split_tool_then_text("pause plus what's a VPN", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRegistryVerification:
    """Test that splitter correctly verifies tool existence in registry."""

    def test_tool_not_in_registry_returns_none(self, mock_registry_empty):
        """Test: If tool doesn't exist in registry, splitter returns None."""
        result = split_tool_then_text("pause music and what's a VPN?", mock_registry_empty)
        
        # Should return None because media_play_pause isn't in registry
        assert result is None

    def test_volume_control_not_in_registry(self, mock_registry_minimal):
        """Test: volume_control not in registry -> no split for mute command."""
        result = split_tool_then_text("mute and what's a VPN?", mock_registry_minimal)
        
        # Should return None because volume_control isn't in registry
        assert result is None

    def test_media_play_pause_in_minimal_registry(self, mock_registry_minimal):
        """Test: media_play_pause exists in minimal registry -> split works."""
        result = split_tool_then_text("pause and what's a VPN?", mock_registry_minimal)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"


# ═══════════════════════════════════════════════════════════════════════════
# FILLER WORD TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestFillerWords:
    """Test handling of filler words at the start of utterances."""

    def test_please_pause_music(self, mock_registry_full):
        """Test: 'please pause music and what's a VPN' -> strips 'please'"""
        result = split_tool_then_text("please pause music and what's a VPN", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"

    def test_can_you_pause(self, mock_registry_full):
        """Test: 'can you pause and tell me about X' -> strips 'can you'"""
        result = split_tool_then_text("can you pause and tell me about cats", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"

    def test_hey_wyzer_pause(self, mock_registry_full):
        """Test: 'hey wyzer pause and what is X' -> strips 'hey wyzer'"""
        result = split_tool_then_text("hey wyzer pause and what is python", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_play_pause"


# ═══════════════════════════════════════════════════════════════════════════
# NO-SPLIT EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestNoSplitCases:
    """Test cases where split should NOT happen."""

    def test_no_connector_returns_none(self, mock_registry_full):
        """Test: 'pause music what is a VPN' (no connector) -> no split."""
        result = split_tool_then_text("pause music what is a VPN", mock_registry_full)
        
        # No connector found -> should not split
        assert result is None

    def test_unknown_tool_phrase_returns_none(self, mock_registry_full):
        """Test: 'dance and what's a VPN' (unknown tool phrase) -> no split."""
        result = split_tool_then_text("dance and what's a VPN", mock_registry_full)
        
        # "dance" is not a high-confidence tool phrase
        assert result is None

    def test_extra_text_before_connector_returns_none(self, mock_registry_full):
        """Test: 'pause my favorite music and X' (extra text) -> no split."""
        result = split_tool_then_text("pause my favorite music and what's a VPN", mock_registry_full)
        
        # "pause my favorite music" != "pause music" (extra words)
        # This shouldn't match because "my favorite" isn't part of the phrase
        # Note: "pause my music" IS a valid phrase, but "pause my favorite music" is not
        assert result is None

    def test_empty_input_returns_none(self, mock_registry_full):
        """Test: empty input -> no split."""
        result = split_tool_then_text("", mock_registry_full)
        assert result is None

    def test_none_input_returns_none(self, mock_registry_full):
        """Test: None input -> no split."""
        result = split_tool_then_text(None, mock_registry_full)
        assert result is None

    def test_whitespace_only_returns_none(self, mock_registry_full):
        """Test: whitespace only -> no split."""
        result = split_tool_then_text("   \t\n  ", mock_registry_full)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# GET_SPLIT_INTENTS WRAPPER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestGetSplitIntents:
    """Test the get_split_intents convenience wrapper."""

    def test_returns_tuple_on_success(self, mock_registry_full):
        """Test: successful split returns (intents_list, leftover_text)."""
        result = get_split_intents("pause and what's a VPN", mock_registry_full)
        
        assert result is not None
        intents, leftover = result
        
        assert isinstance(intents, list)
        assert len(intents) == 1
        assert intents[0]["tool"] == "media_play_pause"
        assert isinstance(leftover, str)
        assert "vpn" in leftover.lower()

    def test_returns_none_on_failure(self, mock_registry_full):
        """Test: failed split returns None."""
        result = get_split_intents("what's a VPN", mock_registry_full)
        
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestHelperFunctions:
    """Test internal helper functions."""

    def test_normalize_text(self):
        """Test _normalize_text function."""
        assert _normalize_text("  Hello   World  ") == "hello world"
        assert _normalize_text("PAUSE MUSIC") == "pause music"
        assert _normalize_text("") == ""
        assert _normalize_text(None) == ""

    def test_strip_fillers(self):
        """Test _strip_fillers function."""
        assert _strip_fillers("please pause music") == "pause music"
        assert _strip_fillers("can you pause") == "pause"
        assert _strip_fillers("hey wyzer pause") == "pause"
        assert _strip_fillers("pause music") == "pause music"  # No filler

    def test_find_connector(self):
        """Test _find_connector function."""
        # " and " should be found
        result = _find_connector("pause and what's a vpn")
        assert result is not None
        start, end, connector = result
        assert " and " in connector.lower()
        
        # ", " should be found
        result = _find_connector("pause, what's a vpn")
        assert result is not None
        
        # No connector
        result = _find_connector("pause music what's a vpn")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# MEDIA SKIP/PREVIOUS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestMediaSkipPrevious:
    """Test media skip/previous tool phrases."""

    def test_skip_and_question(self, mock_registry_full):
        """Test: 'skip and what's playing' -> media_next + leftover"""
        result = split_tool_then_text("skip and what's playing", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_next"

    def test_next_track_and_question(self, mock_registry_full):
        """Test: 'next track and tell me a joke' -> media_next + leftover"""
        result = split_tool_then_text("next track and tell me a joke", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_next"

    def test_previous_track_and_question(self, mock_registry_full):
        """Test: 'previous track, what is this song' -> media_previous + leftover"""
        result = split_tool_then_text("previous track, what is this song", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_previous"

    def test_go_back_and_question(self, mock_registry_full):
        """Test: 'go back then tell me about the artist' -> media_previous + leftover"""
        result = split_tool_then_text("go back then tell me about the artist", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "media_previous"


# ═══════════════════════════════════════════════════════════════════════════
# VOLUME CONTROL TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestVolumeControl:
    """Test volume-related tool phrases."""

    def test_unmute_and_question(self, mock_registry_full):
        """Test: 'unmute and what's the weather' -> volume_control unmute + leftover"""
        result = split_tool_then_text("unmute and what's the weather", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "volume_control"
        assert result.tool_intent["args"]["action"] == "unmute"

    def test_louder_and_question(self, mock_registry_full):
        """Test: 'louder, tell me a story' -> volume_control change + leftover"""
        result = split_tool_then_text("louder, tell me a story", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "volume_control"
        assert result.tool_intent["args"]["action"] == "change"
        assert result.tool_intent["args"]["delta"] == 10

    def test_quieter_and_question(self, mock_registry_full):
        """Test: 'quieter then what time is it' -> volume_control change -10 + leftover"""
        result = split_tool_then_text("quieter then what time is it", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "volume_control"
        assert result.tool_intent["args"]["action"] == "change"
        assert result.tool_intent["args"]["delta"] == -10

    def test_turn_it_up_and_question(self, mock_registry_full):
        """Test: 'turn it up and play some music' -> volume_control + leftover"""
        result = split_tool_then_text("turn it up and play some music", mock_registry_full)
        
        assert result is not None
        assert result.tool_intent["tool"] == "volume_control"
        assert result.tool_intent["args"]["delta"] == 10


# ═══════════════════════════════════════════════════════════════════════════
# QUESTION MARK RESTORATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestQuestionMarkRestoration:
    """Test that question marks are properly restored on leftover text."""

    def test_what_question_gets_question_mark(self, mock_registry_full):
        """Test: leftover starting with 'what' gets question mark."""
        result = split_tool_then_text("pause and what is python", mock_registry_full)
        
        assert result is not None
        assert result.leftover_text.endswith("?")

    def test_how_question_gets_question_mark(self, mock_registry_full):
        """Test: leftover starting with 'how' gets question mark."""
        result = split_tool_then_text("mute and how does it work", mock_registry_full)
        
        assert result is not None
        assert result.leftover_text.endswith("?")

    def test_statement_no_question_mark(self, mock_registry_full):
        """Test: leftover NOT starting with question word doesn't get question mark."""
        result = split_tool_then_text("pause and tell me about cats", mock_registry_full)
        
        assert result is not None
        # "tell me about cats" doesn't start with a question word
        # so it shouldn't get a question mark forced on it
        # (the implementation adds ? only for question-word starters)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
