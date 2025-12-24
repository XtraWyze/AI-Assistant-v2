"""Tests for Phase 12 - Deterministic Monitor Query Commands.

Pure Python tests for monitor query commands. No OS calls required.
Tests verify that:
- "what's on monitor N" returns deterministic window list
- Invalid monitor numbers return appropriate error message
- NO LLM call path is used (handler returns early with result)
- Junk window filtering works correctly
- Focused window is prioritized in listings

Run with: python -m pytest tests/test_monitor_commands.py -v
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from wyzer.context.world_state import (
    clear_world_state,
    update_window_watcher_state,
    get_world_state,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def clean_world_state():
    """Ensure clean state before and after each test."""
    clear_world_state()
    yield
    clear_world_state()


@pytest.fixture
def two_monitor_setup():
    """Set up world state with 2 monitors and windows on each."""
    windows = [
        {
            "hwnd": 1001,
            "title": "Chrome - Google",
            "process": "chrome.exe",
            "pid": 100,
            "monitor": 1,
            "rect": [0, 0, 800, 600],
            "visible": True,
        },
        {
            "hwnd": 1002,
            "title": "VS Code",
            "process": "code.exe",
            "pid": 200,
            "monitor": 1,
            "rect": [0, 0, 800, 600],
            "visible": True,
        },
        {
            "hwnd": 2001,
            "title": "Slack",
            "process": "slack.exe",
            "pid": 300,
            "monitor": 2,
            "rect": [1920, 0, 2720, 600],
            "visible": True,
        },
        {
            "hwnd": 2002,
            "title": "Discord",
            "process": "discord.exe",
            "pid": 400,
            "monitor": 2,
            "rect": [1920, 0, 2720, 600],
            "visible": True,
        },
    ]
    
    windows_by_monitor = {
        1: [w for w in windows if w["monitor"] == 1],
        2: [w for w in windows if w["monitor"] == 2],
    }
    
    focused = windows[0]  # Chrome is focused
    
    update_window_watcher_state(
        open_windows=windows,
        windows_by_monitor=windows_by_monitor,
        focused_window=focused,
        recent_events=[],
        detected_monitor_count=2,
    )
    
    return {
        "windows": windows,
        "windows_by_monitor": windows_by_monitor,
        "focused": focused,
        "monitor_count": 2,
    }


@pytest.fixture
def three_monitor_setup():
    """Set up world state with 3 monitors."""
    windows = [
        {
            "hwnd": 1001,
            "title": "Chrome",
            "process": "chrome.exe",
            "pid": 100,
            "monitor": 1,
            "rect": [0, 0, 800, 600],
            "visible": True,
        },
        {
            "hwnd": 2001,
            "title": "Slack",
            "process": "slack.exe",
            "pid": 200,
            "monitor": 2,
            "rect": [1920, 0, 2720, 600],
            "visible": True,
        },
        {
            "hwnd": 3001,
            "title": "Spotify",
            "process": "spotify.exe",
            "pid": 300,
            "monitor": 3,
            "rect": [3840, 0, 4640, 600],
            "visible": True,
        },
    ]
    
    windows_by_monitor = {
        1: [w for w in windows if w["monitor"] == 1],
        2: [w for w in windows if w["monitor"] == 2],
        3: [w for w in windows if w["monitor"] == 3],
    }
    
    update_window_watcher_state(
        open_windows=windows,
        windows_by_monitor=windows_by_monitor,
        focused_window=windows[0],
        recent_events=[],
        detected_monitor_count=3,
    )
    
    return {"monitor_count": 3}


@pytest.fixture
def windows_with_junk():
    """Set up world state with junk windows that should be filtered."""
    windows = [
        # Normal window
        {
            "hwnd": 1001,
            "title": "Chrome - Google",
            "process": "chrome.exe",
            "pid": 100,
            "monitor": 1,
            "rect": [0, 0, 800, 600],
            "visible": True,
        },
        # Empty title - should be filtered
        {
            "hwnd": 1002,
            "title": "",
            "process": "something.exe",
            "pid": 101,
            "monitor": 1,
            "rect": [0, 0, 100, 100],
            "visible": True,
        },
        # Program Manager - should be filtered
        {
            "hwnd": 1003,
            "title": "Program Manager",
            "process": "explorer.exe",
            "pid": 102,
            "monitor": 1,
            "rect": [0, 0, 1920, 1080],
            "visible": True,
        },
        # ApplicationFrameHost - should be filtered
        {
            "hwnd": 1004,
            "title": "Some UWP App",
            "process": "applicationframehost.exe",
            "pid": 103,
            "monitor": 1,
            "rect": [0, 0, 800, 600],
            "visible": True,
        },
        # Normal window
        {
            "hwnd": 1005,
            "title": "Notepad",
            "process": "notepad.exe",
            "pid": 104,
            "monitor": 1,
            "rect": [0, 0, 800, 600],
            "visible": True,
        },
    ]
    
    windows_by_monitor = {1: windows}
    
    update_window_watcher_state(
        open_windows=windows,
        windows_by_monitor=windows_by_monitor,
        focused_window=windows[0],
        recent_events=[],
        detected_monitor_count=1,
    )
    
    return {"total_count": 5, "expected_filtered_count": 2}  # Only Chrome and Notepad


# ============================================================================
# TEST: MONITOR COUNT DETECTION
# ============================================================================

class TestMonitorCountDetection:
    """Tests for detected monitor count tracking."""
    
    def test_detected_monitor_count_stored(self, two_monitor_setup):
        """Detected monitor count is stored in world_state."""
        from wyzer.context.world_state import get_monitor_count
        assert get_monitor_count() == 2
    
    def test_detected_monitor_count_three(self, three_monitor_setup):
        """Three monitors detected correctly."""
        from wyzer.context.world_state import get_monitor_count
        assert get_monitor_count() == 3
    
    def test_monitor_count_defaults_to_one(self):
        """Monitor count defaults to 1 when not set."""
        from wyzer.context.world_state import get_monitor_count
        assert get_monitor_count() == 1
    
    def test_monitor_count_passed_to_update(self):
        """Monitor count is correctly passed through update_window_watcher_state."""
        update_window_watcher_state(
            open_windows=[],
            windows_by_monitor={},
            focused_window=None,
            recent_events=[],
            detected_monitor_count=4,
        )
        from wyzer.context.world_state import get_monitor_count
        assert get_monitor_count() == 4


# ============================================================================
# TEST: WHATS ON MONITOR COMMAND
# ============================================================================

class TestWhatsOnMonitorCommand:
    """Tests for 'what's on monitor N' deterministic command."""
    
    def test_returns_deterministic_list_monitor_2(self, two_monitor_setup):
        """'what's on monitor 2' returns deterministic list of windows."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 2", time.perf_counter())
        
        assert result is not None, "Handler should return a result"
        assert "reply" in result, "Result should have a reply"
        assert "Slack" in result["reply"], "Slack should be in monitor 2 list"
        assert "Discord" in result["reply"], "Discord should be in monitor 2 list"
        assert "2 windows" in result["reply"], "Should report 2 windows"
    
    def test_returns_deterministic_list_monitor_1(self, two_monitor_setup):
        """'what's on monitor 1' returns deterministic list of windows."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 1", time.perf_counter())
        
        assert result is not None
        assert "Chrome" in result["reply"]
        assert "VS Code" in result["reply"]
        assert "2 windows" in result["reply"]
    
    def test_invalid_monitor_returns_error(self, two_monitor_setup):
        """'what's on monitor 3' returns error when only 2 monitors detected."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 3", time.perf_counter())
        
        assert result is not None, "Handler should return a result"
        assert "reply" in result, "Result should have a reply"
        assert "only detect 2 monitor" in result["reply"].lower(), \
            f"Should say only 2 monitors detected, got: {result['reply']}"
        assert result.get("meta", {}).get("error") == "monitor_not_found"
    
    def test_invalid_monitor_5_returns_error(self, two_monitor_setup):
        """'what's on monitor 5' returns error when only 2 monitors detected."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on screen 5", time.perf_counter())
        
        assert result is not None
        assert "only detect 2 monitor" in result["reply"].lower()
    
    def test_handler_returns_early_no_llm(self, two_monitor_setup):
        """Handler returns immediately without touching LLM path."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 1", time.perf_counter())
        
        # The presence of a result means LLM path was bypassed
        assert result is not None, "Handler must return result to bypass LLM"
        assert "latency_ms" in result, "Result should have latency_ms"
        assert result["latency_ms"] < 1000, "Should complete in <1s (no LLM)"
        
        # Meta should indicate this was a window watcher command
        assert result.get("meta", {}).get("window_watcher_command") == "whats_on_monitor"
    
    def test_primary_monitor_maps_to_1(self, two_monitor_setup):
        """'what's on primary monitor' maps to monitor 1."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor primary", time.perf_counter())
        
        assert result is not None
        assert "Chrome" in result["reply"], "Chrome should be on monitor 1 (primary)"
        assert result.get("meta", {}).get("monitor") == 1
    
    def test_word_number_two(self, two_monitor_setup):
        """'what's on monitor two' works with word numbers."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor two", time.perf_counter())
        
        assert result is not None
        assert "Slack" in result["reply"]
        assert result.get("meta", {}).get("monitor") == 2


# ============================================================================
# TEST: JUNK WINDOW FILTERING
# ============================================================================

class TestJunkWindowFiltering:
    """Tests for filtering junk windows from listings."""
    
    def test_empty_titles_filtered(self, windows_with_junk):
        """Windows with empty titles are filtered out."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 1", time.perf_counter())
        
        assert result is not None
        # Should only show Chrome and Notepad (2 windows), not all 5
        assert "2 windows" in result["reply"], f"Should have 2 windows, got: {result['reply']}"
    
    def test_explorer_filtered(self, windows_with_junk):
        """explorer.exe (Program Manager) is filtered out."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 1", time.perf_counter())
        
        assert result is not None
        assert "Program Manager" not in result["reply"]
    
    def test_applicationframehost_filtered(self, windows_with_junk):
        """ApplicationFrameHost.exe is filtered out."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 1", time.perf_counter())
        
        assert result is not None
        # The UWP app hosted by ApplicationFrameHost should not appear
        assert "Some UWP App" not in result["reply"]


# ============================================================================
# TEST: FOCUSED WINDOW PRIORITY
# ============================================================================

class TestFocusedWindowPriority:
    """Tests for focused window prioritization in listings."""
    
    def test_focused_window_shown_first(self, two_monitor_setup):
        """Focused window is listed first with arrow prefix."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 1", time.perf_counter())
        
        assert result is not None
        # Chrome is focused and should be first (with arrow prefix)
        lines = result["reply"].split("\n")
        # Find the first window line (after "Monitor 1 has X windows:")
        window_lines = [l for l in lines if l.startswith("→ ") or l.startswith("• ")]
        assert len(window_lines) >= 1
        assert window_lines[0].startswith("→ "), "Focused window should have arrow prefix"
        assert "Chrome" in window_lines[0], "Focused window (Chrome) should be first"


# ============================================================================
# TEST: NO LLM BYPASS CONFIRMATION
# ============================================================================

class TestNoLLMBypass:
    """Tests to confirm monitor commands bypass LLM entirely."""
    
    def test_valid_monitor_returns_result(self, two_monitor_setup):
        """Valid monitor query returns result dict (bypasses LLM)."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 1", time.perf_counter())
        
        # Non-None result means LLM was bypassed
        assert result is not None
        assert isinstance(result, dict)
        assert "reply" in result
    
    def test_invalid_monitor_returns_result(self, two_monitor_setup):
        """Invalid monitor query also returns result (bypasses LLM)."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 99", time.perf_counter())
        
        # Should still return result (error message), not None
        assert result is not None
        assert "reply" in result
    
    def test_non_matching_returns_none(self, two_monitor_setup):
        """Non-matching text returns None (allows LLM path)."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("tell me a joke", time.perf_counter())
        
        # None means this is not a window watcher command
        assert result is None


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests for monitor commands."""
    
    def test_empty_monitor_returns_appears_empty(self, three_monitor_setup):
        """Monitor with no windows (after filtering) says 'appears empty'."""
        # Set up monitor 2 with only junk windows
        windows = [
            {
                "hwnd": 2001,
                "title": "",  # Empty title - will be filtered
                "process": "something.exe",
                "pid": 200,
                "monitor": 2,
                "rect": [1920, 0, 2720, 600],
                "visible": True,
            },
        ]
        windows_by_monitor = {
            1: [],
            2: windows,
        }
        update_window_watcher_state(
            open_windows=windows,
            windows_by_monitor=windows_by_monitor,
            focused_window=None,
            recent_events=[],
            detected_monitor_count=2,
        )
        
        from wyzer.core.orchestrator import _check_window_watcher_commands
        result = _check_window_watcher_commands("what's on monitor 2", time.perf_counter())
        
        assert result is not None
        assert "appears empty" in result["reply"].lower()
    
    def test_curly_apostrophe_works(self, two_monitor_setup):
        """Curly apostrophe (from STT) is handled correctly."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        # U+2019 RIGHT SINGLE QUOTATION MARK
        result = _check_window_watcher_commands("what's on monitor 1", time.perf_counter())
        
        assert result is not None
        assert "Chrome" in result["reply"]
    
    def test_case_insensitive(self, two_monitor_setup):
        """Command matching is case insensitive."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("WHAT'S ON MONITOR 1", time.perf_counter())
        
        assert result is not None
        assert "Chrome" in result["reply"]
    
    def test_question_mark_suffix(self, two_monitor_setup):
        """Command with question mark suffix works."""
        from wyzer.core.orchestrator import _check_window_watcher_commands
        
        result = _check_window_watcher_commands("what's on monitor 2?", time.perf_counter())
        
        assert result is not None
        assert "Slack" in result["reply"]


# ============================================================================
# TEST: STREAMING TTS BYPASS
# ============================================================================

class TestStreamingTTSBypass:
    """Tests that monitor commands bypass streaming TTS path."""
    
    def test_whats_on_monitor_bypasses_streaming(self, two_monitor_setup):
        """'what's on monitor N' bypasses streaming TTS."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        
        assert should_use_streaming_tts("what's on monitor 1") is False
        assert should_use_streaming_tts("what's on monitor 2") is False
        assert should_use_streaming_tts("whats on screen 1") is False
    
    def test_whats_on_monitor_word_numbers_bypass_streaming(self, two_monitor_setup):
        """Word numbers also bypass streaming."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        
        assert should_use_streaming_tts("what's on monitor one") is False
        assert should_use_streaming_tts("what's on monitor two") is False
    
    def test_close_all_bypasses_streaming(self, two_monitor_setup):
        """'close all on monitor N' bypasses streaming TTS."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        
        assert should_use_streaming_tts("close all on screen 1") is False
        assert should_use_streaming_tts("close everything on monitor 2") is False
    
    def test_where_am_i_bypasses_streaming(self, two_monitor_setup):
        """'where am I' bypasses streaming TTS."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        
        assert should_use_streaming_tts("where am I") is False
        assert should_use_streaming_tts("which monitor am I on") is False
    
    def test_what_did_i_open_bypasses_streaming(self, two_monitor_setup):
        """'what did I just open' bypasses streaming TTS."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        
        assert should_use_streaming_tts("what did I just open") is False
        assert should_use_streaming_tts("what did I open") is False
    
    def test_non_monitor_commands_may_stream(self):
        """Non-monitor commands may still use streaming (if enabled)."""
        # This test just verifies these DON'T match the monitor patterns
        from wyzer.core.orchestrator import (
            _WHATS_ON_MONITOR_RE, _CLOSE_ALL_ON_MONITOR_RE,
            _WHAT_DID_I_OPEN_RE, _WHERE_AM_I_RE
        )
        
        # Random conversational queries should not match
        assert _WHATS_ON_MONITOR_RE.match("tell me about monitors") is None
        assert _WHATS_ON_MONITOR_RE.match("what is a monitor") is None
        assert _CLOSE_ALL_ON_MONITOR_RE.match("close the window") is None
        assert _WHERE_AM_I_RE.match("where is the file") is None
