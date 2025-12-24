"""Tests for Phase 12 - Window Watcher Voice Commands.

Tests for deterministic voice commands that query window state:
- "what's on monitor 2"
- "close all on screen 1"
- "what did I just open"
- "where am I"

Run with: python -m pytest scripts/test_window_watcher_commands.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
import time

from wyzer.context.world_state import (
    clear_world_state,
    update_window_watcher_state,
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
def mock_windows_multi_monitor():
    """Mock window data across multiple monitors."""
    return [
        {
            "hwnd": 1001,
            "title": "Chrome - Google",
            "process": "chrome.exe",
            "pid": 100,
            "monitor": 1,
            "rect": (0, 0, 800, 600),
            "visible": True,
        },
        {
            "hwnd": 1002,
            "title": "VS Code",
            "process": "code.exe",
            "pid": 200,
            "monitor": 1,
            "rect": (0, 0, 800, 600),
            "visible": True,
        },
        {
            "hwnd": 2001,
            "title": "Slack",
            "process": "slack.exe",
            "pid": 300,
            "monitor": 2,
            "rect": (1920, 0, 2720, 600),
            "visible": True,
        },
        {
            "hwnd": 3001,
            "title": "Discord",
            "process": "discord.exe",
            "pid": 400,
            "monitor": 3,
            "rect": (3840, 0, 4640, 600),
            "visible": True,
        },
    ]


@pytest.fixture
def mock_recent_events():
    """Mock recent window events."""
    return [
        {"ts": time.time() - 5, "type": "opened", "hwnd": 9999, "title": "Notepad", "process": "notepad.exe", "to_monitor": 1},
        {"ts": time.time() - 10, "type": "opened", "hwnd": 8888, "title": "Calculator", "process": "calc.exe", "to_monitor": 2},
        {"ts": time.time() - 15, "type": "closed", "hwnd": 7777, "title": "Old Window", "process": "old.exe", "from_monitor": 1},
    ]


# ============================================================================
# REGEX PATTERN TESTS - WHAT'S ON MONITOR
# ============================================================================

class TestWhatsOnMonitorPattern:
    """Tests for 'what's on monitor N' regex patterns."""

    def test_whats_on_monitor_2_matches(self):
        """'what's on monitor 2' matches."""
        from wyzer.core.orchestrator import _WHATS_ON_MONITOR_RE
        assert _WHATS_ON_MONITOR_RE.match("what's on monitor 2") is not None
        assert _WHATS_ON_MONITOR_RE.match("What's on monitor 2") is not None
        assert _WHATS_ON_MONITOR_RE.match("WHAT'S ON MONITOR 2") is not None

    def test_curly_apostrophe_matches(self):
        """Curly apostrophe (from STT) matches."""
        from wyzer.core.orchestrator import _WHATS_ON_MONITOR_RE
        # U+2019 RIGHT SINGLE QUOTATION MARK (curly apostrophe)
        assert _WHATS_ON_MONITOR_RE.match("What's on monitor 2") is not None
        assert _WHATS_ON_MONITOR_RE.match("what's on monitor 1?") is not None

    def test_whats_on_screen_1_matches(self):
        """'what's on screen 1' matches."""
        from wyzer.core.orchestrator import _WHATS_ON_MONITOR_RE
        assert _WHATS_ON_MONITOR_RE.match("what's on screen 1") is not None
        assert _WHATS_ON_MONITOR_RE.match("what's on screen 3") is not None

    def test_whats_on_my_monitor_matches(self):
        """'what's on my monitor N' variations match."""
        from wyzer.core.orchestrator import _WHATS_ON_MONITOR_RE
        assert _WHATS_ON_MONITOR_RE.match("what's on my monitor 2") is not None
        assert _WHATS_ON_MONITOR_RE.match("what is on monitor 1") is not None

    def test_whats_on_the_monitor_matches(self):
        """'what's on the monitor N' matches."""
        from wyzer.core.orchestrator import _WHATS_ON_MONITOR_RE
        assert _WHATS_ON_MONITOR_RE.match("what's on the monitor 2") is not None
        assert _WHATS_ON_MONITOR_RE.match("what is on the monitor 1") is not None
        assert _WHATS_ON_MONITOR_RE.match("what's on the screen 3") is not None

    def test_word_numbers_match(self):
        """Word numbers (one, two, three) match."""
        from wyzer.core.orchestrator import _WHATS_ON_MONITOR_RE
        assert _WHATS_ON_MONITOR_RE.match("what's on monitor one") is not None
        assert _WHATS_ON_MONITOR_RE.match("what's on monitor two") is not None
        assert _WHATS_ON_MONITOR_RE.match("what's on screen three") is not None

    def test_special_monitor_names_match(self):
        """Special monitor names (primary, left, right) match."""
        from wyzer.core.orchestrator import _WHATS_ON_MONITOR_RE
        assert _WHATS_ON_MONITOR_RE.match("what's on monitor primary") is not None
        assert _WHATS_ON_MONITOR_RE.match("what's on screen left") is not None
        assert _WHATS_ON_MONITOR_RE.match("what's on display right") is not None

    def test_extracts_monitor_number(self):
        """Monitor number can be extracted from match."""
        from wyzer.core.orchestrator import _WHATS_ON_MONITOR_RE, _parse_monitor_number
        match = _WHATS_ON_MONITOR_RE.match("what's on monitor 2")
        assert match is not None
        # The pattern uses group(1) not named group
        num = _parse_monitor_number(match.group(1))
        assert num == 2


# ============================================================================
# REGEX PATTERN TESTS - CLOSE ALL ON MONITOR
# ============================================================================

class TestCloseAllOnMonitorPattern:
    """Tests for 'close all on screen N' regex patterns."""

    def test_close_all_on_screen_1_matches(self):
        """'close all on screen 1' matches."""
        from wyzer.core.orchestrator import _CLOSE_ALL_ON_MONITOR_RE
        assert _CLOSE_ALL_ON_MONITOR_RE.match("close all on screen 1") is not None
        assert _CLOSE_ALL_ON_MONITOR_RE.match("Close all on screen 1") is not None

    def test_close_all_on_monitor_matches(self):
        """'close all on monitor N' matches."""
        from wyzer.core.orchestrator import _CLOSE_ALL_ON_MONITOR_RE
        assert _CLOSE_ALL_ON_MONITOR_RE.match("close all on monitor 2") is not None
        assert _CLOSE_ALL_ON_MONITOR_RE.match("close all on monitor 3") is not None

    def test_close_everything_on_matches(self):
        """'close everything on...' matches."""
        from wyzer.core.orchestrator import _CLOSE_ALL_ON_MONITOR_RE
        assert _CLOSE_ALL_ON_MONITOR_RE.match("close everything on screen 1") is not None
        assert _CLOSE_ALL_ON_MONITOR_RE.match("close everything on monitor 2") is not None

    def test_close_all_display_matches(self):
        """'close all on display N' matches."""
        from wyzer.core.orchestrator import _CLOSE_ALL_ON_MONITOR_RE
        assert _CLOSE_ALL_ON_MONITOR_RE.match("close all on display 1") is not None
        assert _CLOSE_ALL_ON_MONITOR_RE.match("close everything display 2") is not None

    def test_word_numbers_match(self):
        """Word numbers (one, two, three) match."""
        from wyzer.core.orchestrator import _CLOSE_ALL_ON_MONITOR_RE
        assert _CLOSE_ALL_ON_MONITOR_RE.match("close all on screen one") is not None
        assert _CLOSE_ALL_ON_MONITOR_RE.match("close all on monitor two") is not None


# ============================================================================
# REGEX PATTERN TESTS - WHAT DID I JUST OPEN
# ============================================================================

class TestWhatDidIJustOpenPattern:
    """Tests for 'what did I just open' regex patterns."""

    def test_what_did_i_just_open_matches(self):
        """'what did I just open' matches."""
        from wyzer.core.orchestrator import _WHAT_DID_I_OPEN_RE
        assert _WHAT_DID_I_OPEN_RE.match("what did I just open") is not None
        assert _WHAT_DID_I_OPEN_RE.match("What did I just open") is not None
        assert _WHAT_DID_I_OPEN_RE.match("WHAT DID I JUST OPEN") is not None

    def test_what_did_i_open_matches(self):
        """'what did I open' (without 'just') matches."""
        from wyzer.core.orchestrator import _WHAT_DID_I_OPEN_RE
        assert _WHAT_DID_I_OPEN_RE.match("what did I open") is not None

    def test_what_did_you_open_matches(self):
        """'what did you open' matches."""
        from wyzer.core.orchestrator import _WHAT_DID_I_OPEN_RE
        assert _WHAT_DID_I_OPEN_RE.match("what did you open") is not None
        assert _WHAT_DID_I_OPEN_RE.match("what did you just open") is not None

    def test_what_was_just_opened_matches(self):
        """'what was just opened' matches."""
        from wyzer.core.orchestrator import _WHAT_DID_I_OPEN_RE
        assert _WHAT_DID_I_OPEN_RE.match("what was just opened") is not None

    def test_what_were_opened_matches(self):
        """'what were opened' variations match."""
        from wyzer.core.orchestrator import _WHAT_DID_I_OPEN_RE
        assert _WHAT_DID_I_OPEN_RE.match("what were just opened") is not None
        assert _WHAT_DID_I_OPEN_RE.match("what were opened") is not None


# ============================================================================
# REGEX PATTERN TESTS - WHERE AM I
# ============================================================================

class TestWhereAmIPattern:
    """Tests for 'where am I' regex patterns."""

    def test_where_am_i_matches(self):
        """'where am I' matches."""
        from wyzer.core.orchestrator import _WHERE_AM_I_RE
        assert _WHERE_AM_I_RE.match("where am I") is not None
        assert _WHERE_AM_I_RE.match("Where am I") is not None
        assert _WHERE_AM_I_RE.match("WHERE AM I") is not None

    def test_which_monitor_am_i_on_matches(self):
        """'which monitor am I on' matches."""
        from wyzer.core.orchestrator import _WHERE_AM_I_RE
        assert _WHERE_AM_I_RE.match("which monitor am I on") is not None
        assert _WHERE_AM_I_RE.match("which screen am I on") is not None

    def test_what_monitor_is_this_matches(self):
        """'what monitor is this' matches."""
        from wyzer.core.orchestrator import _WHERE_AM_I_RE
        assert _WHERE_AM_I_RE.match("what monitor is this") is not None
        assert _WHERE_AM_I_RE.match("what screen is this") is not None

    def test_which_display_is_active_matches(self):
        """'which display is active' matches."""
        from wyzer.core.orchestrator import _WHERE_AM_I_RE
        assert _WHERE_AM_I_RE.match("which display is active") is not None
        assert _WHERE_AM_I_RE.match("what monitor is focused") is not None


# ============================================================================
# PARSE MONITOR NUMBER TESTS
# ============================================================================

class TestParseMonitorNumber:
    """Tests for _parse_monitor_number helper."""

    def test_digit_number(self):
        """Digit strings parse correctly."""
        from wyzer.core.orchestrator import _parse_monitor_number
        assert _parse_monitor_number("1") == 1
        assert _parse_monitor_number("2") == 2
        assert _parse_monitor_number("3") == 3

    def test_word_numbers(self):
        """Word numbers parse correctly."""
        from wyzer.core.orchestrator import _parse_monitor_number
        assert _parse_monitor_number("one") == 1
        assert _parse_monitor_number("two") == 2
        assert _parse_monitor_number("three") == 3
        assert _parse_monitor_number("four") == 4
        assert _parse_monitor_number("five") == 5

    def test_case_insensitive(self):
        """Word numbers are case insensitive."""
        from wyzer.core.orchestrator import _parse_monitor_number
        assert _parse_monitor_number("One") == 1
        assert _parse_monitor_number("TWO") == 2
        assert _parse_monitor_number("ThReE") == 3

    def test_special_names(self):
        """Special monitor names parse correctly."""
        from wyzer.core.orchestrator import _parse_monitor_number
        assert _parse_monitor_number("primary") == 1
        assert _parse_monitor_number("main") == 1
        assert _parse_monitor_number("secondary") == 2
        assert _parse_monitor_number("left") == 1
        assert _parse_monitor_number("right") == 2


# ============================================================================
# BULK CLOSE RISK CLASSIFICATION TESTS
# ============================================================================

class TestBulkCloseRiskClassification:
    """Tests for bulk close_window triggering HIGH risk."""

    def test_single_close_is_medium(self):
        """Single close_window is MEDIUM risk."""
        from wyzer.policy.risk import classify_plan
        plan = [{"tool": "close_window", "args": {"title": "Notepad"}}]
        assert classify_plan(plan) == "medium"

    def test_two_closes_is_medium(self):
        """Two close_window calls is still MEDIUM risk."""
        from wyzer.policy.risk import classify_plan
        plan = [
            {"tool": "close_window", "args": {"title": "Window 1"}},
            {"tool": "close_window", "args": {"title": "Window 2"}},
        ]
        assert classify_plan(plan) == "medium"

    def test_three_closes_is_high(self):
        """Three close_window calls triggers HIGH risk."""
        from wyzer.policy.risk import classify_plan
        plan = [
            {"tool": "close_window", "args": {"title": "Window 1"}},
            {"tool": "close_window", "args": {"title": "Window 2"}},
            {"tool": "close_window", "args": {"title": "Window 3"}},
        ]
        assert classify_plan(plan) == "high"

    def test_five_closes_is_high(self):
        """Five close_window calls is HIGH risk."""
        from wyzer.policy.risk import classify_plan
        plan = [{"tool": "close_window", "args": {"title": f"Window {i}"}} for i in range(5)]
        assert classify_plan(plan) == "high"

    def test_mixed_with_bulk_close(self):
        """Mixed plan with 3+ close_window is HIGH."""
        from wyzer.policy.risk import classify_plan
        plan = [
            {"tool": "get_time", "args": {}},  # low
            {"tool": "close_window", "args": {"title": "Window 1"}},
            {"tool": "close_window", "args": {"title": "Window 2"}},
            {"tool": "close_window", "args": {"title": "Window 3"}},
            {"tool": "open_target", "args": {"query": "chrome"}},  # medium
        ]
        assert classify_plan(plan) == "high"


# ============================================================================
# WORLD STATE INTEGRATION TESTS
# ============================================================================

class TestWorldStateIntegration:
    """Tests for window watcher world state integration."""

    def test_update_window_watcher_state(self, mock_windows_multi_monitor, mock_recent_events):
        """update_window_watcher_state populates world state correctly."""
        windows_by_monitor = {
            1: [w for w in mock_windows_multi_monitor if w["monitor"] == 1],
            2: [w for w in mock_windows_multi_monitor if w["monitor"] == 2],
            3: [w for w in mock_windows_multi_monitor if w["monitor"] == 3],
        }
        focused = mock_windows_multi_monitor[0]  # Chrome

        update_window_watcher_state(
            open_windows=mock_windows_multi_monitor,
            windows_by_monitor=windows_by_monitor,
            focused_window=focused,
            recent_events=mock_recent_events,
        )

        from wyzer.context.world_state import (
            get_windows_on_monitor,
            get_focused_window_info,
            get_recent_window_events,
            get_all_open_windows,
        )

        # Check monitor 1 has 2 windows
        mon1_windows = get_windows_on_monitor(1)
        assert len(mon1_windows) == 2

        # Check monitor 2 has 1 window
        mon2_windows = get_windows_on_monitor(2)
        assert len(mon2_windows) == 1
        assert mon2_windows[0]["title"] == "Slack"

        # Check focused window
        focused_info = get_focused_window_info()
        assert focused_info["title"] == "Chrome - Google"

        # Check recent events (filter by "opened", limit 5)
        events = get_recent_window_events("opened", 5)
        assert len(events) == 2  # 2 opened events in mock_recent_events

    def test_get_windows_on_nonexistent_monitor(self, mock_windows_multi_monitor):
        """get_windows_on_monitor returns empty list for nonexistent monitor."""
        windows_by_monitor = {
            1: [w for w in mock_windows_multi_monitor if w["monitor"] == 1],
        }

        update_window_watcher_state(
            open_windows=mock_windows_multi_monitor,
            windows_by_monitor=windows_by_monitor,
            focused_window=None,
            recent_events=[],
        )

        from wyzer.context.world_state import get_windows_on_monitor
        assert get_windows_on_monitor(99) == []
