"""Tests for Phase 12 - Window Diff Algorithm.

Pure Python tests for the window diff logic. No OS calls required.
These tests verify the snapshot comparison algorithm that detects
window open/close/move/title_change/focus_change events.

Event structure uses 'type' field: opened, closed, moved, title_changed, focus_changed
Monitor fields use from_monitor/to_monitor naming.

Run with: python -m pytest tests/test_window_diff.py -v
"""

import pytest
from wyzer.world.window_diff import diff_snapshots, build_hwnd_dict


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def empty_snapshot():
    """Empty window snapshot (no windows)."""
    return {}


@pytest.fixture
def single_window_snapshot():
    """Single window snapshot."""
    return {
        12345: {
            "hwnd": 12345,
            "title": "Notepad",
            "process": "notepad.exe",
            "pid": 1000,
            "monitor": 1,
            "rect": [0, 0, 800, 600],
            "visible": True,
        }
    }


@pytest.fixture
def two_window_snapshot():
    """Two window snapshot."""
    return {
        12345: {
            "hwnd": 12345,
            "title": "Notepad",
            "process": "notepad.exe",
            "pid": 1000,
            "monitor": 1,
            "rect": [0, 0, 800, 600],
            "visible": True,
        },
        67890: {
            "hwnd": 67890,
            "title": "Chrome",
            "process": "chrome.exe",
            "pid": 2000,
            "monitor": 2,
            "rect": [800, 0, 1600, 600],
            "visible": True,
        },
    }


# ============================================================================
# BUILD_HWND_DICT TESTS
# ============================================================================

class TestBuildHwndDict:
    """Tests for build_hwnd_dict helper function."""

    def test_empty_list(self):
        """Empty list produces empty dict."""
        result = build_hwnd_dict([])
        assert result == {}

    def test_single_window(self):
        """Single window list produces dict keyed by hwnd."""
        windows = [
            {
                "hwnd": 12345,
                "title": "Notepad",
                "process": "notepad.exe",
            }
        ]
        result = build_hwnd_dict(windows)
        assert 12345 in result
        assert result[12345]["title"] == "Notepad"

    def test_multiple_windows(self):
        """Multiple windows keyed by hwnd."""
        windows = [
            {"hwnd": 111, "title": "A"},
            {"hwnd": 222, "title": "B"},
            {"hwnd": 333, "title": "C"},
        ]
        result = build_hwnd_dict(windows)
        assert len(result) == 3
        assert result[111]["title"] == "A"
        assert result[222]["title"] == "B"
        assert result[333]["title"] == "C"

    def test_missing_hwnd_skipped(self):
        """Window without hwnd is skipped."""
        windows = [
            {"title": "No hwnd"},
            {"hwnd": 123, "title": "Has hwnd"},
        ]
        result = build_hwnd_dict(windows)
        assert len(result) == 1
        assert 123 in result


# ============================================================================
# DIFF_SNAPSHOTS - OPENED EVENTS
# ============================================================================

class TestDiffSnapshotsOpened:
    """Tests for detecting 'opened' window events."""

    def test_new_window_detected(self, empty_snapshot, single_window_snapshot):
        """Window appearing in new snapshot is detected as 'opened'."""
        events = diff_snapshots(empty_snapshot, single_window_snapshot, None, None)
        opened = [e for e in events if e["type"] == "opened"]
        assert len(opened) == 1
        assert opened[0]["hwnd"] == 12345
        assert opened[0]["title"] == "Notepad"

    def test_multiple_new_windows(self, empty_snapshot, two_window_snapshot):
        """Multiple new windows each produce 'opened' event."""
        events = diff_snapshots(empty_snapshot, two_window_snapshot, None, None)
        opened = [e for e in events if e["type"] == "opened"]
        assert len(opened) == 2
        hwnds = {e["hwnd"] for e in opened}
        assert hwnds == {12345, 67890}

    def test_no_change_no_opened(self, single_window_snapshot):
        """Same snapshot twice produces no 'opened' events."""
        events = diff_snapshots(
            single_window_snapshot, single_window_snapshot, None, None
        )
        opened = [e for e in events if e["type"] == "opened"]
        assert len(opened) == 0


# ============================================================================
# DIFF_SNAPSHOTS - CLOSED EVENTS
# ============================================================================

class TestDiffSnapshotsClosed:
    """Tests for detecting 'closed' window events."""

    def test_window_removed_detected(self, single_window_snapshot, empty_snapshot):
        """Window removed is detected as 'closed'."""
        events = diff_snapshots(single_window_snapshot, empty_snapshot, None, None)
        closed = [e for e in events if e["type"] == "closed"]
        assert len(closed) == 1
        assert closed[0]["hwnd"] == 12345

    def test_multiple_closed_windows(self, two_window_snapshot, empty_snapshot):
        """Multiple closed windows each produce 'closed' event."""
        events = diff_snapshots(two_window_snapshot, empty_snapshot, None, None)
        closed = [e for e in events if e["type"] == "closed"]
        assert len(closed) == 2

    def test_partial_close(self, two_window_snapshot, single_window_snapshot):
        """One of two windows closing produces single 'closed' event."""
        events = diff_snapshots(
            two_window_snapshot, single_window_snapshot, None, None
        )
        closed = [e for e in events if e["type"] == "closed"]
        assert len(closed) == 1
        assert closed[0]["hwnd"] == 67890  # Chrome was removed


# ============================================================================
# DIFF_SNAPSHOTS - TITLE CHANGED EVENTS
# ============================================================================

class TestDiffSnapshotsTitleChanged:
    """Tests for detecting 'title_changed' window events."""

    def test_title_change_detected(self, single_window_snapshot):
        """Window title change is detected."""
        new_snapshot = {
            12345: {
                **single_window_snapshot[12345],
                "title": "Notepad - document.txt",
            }
        }
        events = diff_snapshots(
            single_window_snapshot, new_snapshot, None, None
        )
        title_changes = [e for e in events if e["type"] == "title_changed"]
        assert len(title_changes) == 1
        # Event has "title" field with new title
        assert title_changes[0]["title"] == "Notepad - document.txt"

    def test_same_title_no_event(self, single_window_snapshot):
        """Same title produces no 'title_changed' event."""
        events = diff_snapshots(
            single_window_snapshot, single_window_snapshot, None, None
        )
        title_changes = [e for e in events if e["type"] == "title_changed"]
        assert len(title_changes) == 0


# ============================================================================
# DIFF_SNAPSHOTS - MOVED EVENTS
# ============================================================================

class TestDiffSnapshotsMoved:
    """Tests for detecting 'moved' window events (monitor change)."""

    def test_monitor_change_detected(self, single_window_snapshot):
        """Window moving to different monitor is detected."""
        new_snapshot = {
            12345: {
                **single_window_snapshot[12345],
                "monitor": 2,  # Changed from 1 to 2
                "rect": [1920, 0, 2720, 600],  # New position
            }
        }
        events = diff_snapshots(
            single_window_snapshot, new_snapshot, None, None
        )
        moved = [e for e in events if e["type"] == "moved"]
        assert len(moved) == 1
        assert moved[0]["from_monitor"] == 1
        assert moved[0]["to_monitor"] == 2

    def test_significant_rect_change_detected(self, single_window_snapshot):
        """Significant position change produces 'moved' event."""
        new_snapshot = {
            12345: {
                **single_window_snapshot[12345],
                "rect": [100, 100, 900, 700],  # Moved 100px in x and y
            }
        }
        events = diff_snapshots(
            single_window_snapshot, new_snapshot, None, None
        )
        moved = [e for e in events if e["type"] == "moved"]
        assert len(moved) == 1

    def test_minor_rect_change_ignored(self, single_window_snapshot):
        """Minor position change (< threshold) produces no 'moved' event."""
        new_snapshot = {
            12345: {
                **single_window_snapshot[12345],
                "rect": [1, 1, 801, 601],  # Just 1 pixel change
            }
        }
        events = diff_snapshots(
            single_window_snapshot, new_snapshot, None, None
        )
        moved = [e for e in events if e["type"] == "moved"]
        assert len(moved) == 0


# ============================================================================
# DIFF_SNAPSHOTS - FOCUS CHANGED EVENTS
# ============================================================================

class TestDiffSnapshotsFocusChanged:
    """Tests for detecting 'focus_changed' window events."""

    def test_focus_change_detected(self, single_window_snapshot, two_window_snapshot):
        """Focus change between windows is detected."""
        events = diff_snapshots(
            single_window_snapshot, two_window_snapshot, 12345, 67890
        )
        focus_changes = [e for e in events if e["type"] == "focus_changed"]
        assert len(focus_changes) == 1
        # focus_changed event has hwnd of new focused window
        assert focus_changes[0]["hwnd"] == 67890

    def test_same_focus_no_event(self, single_window_snapshot):
        """Same focus produces no 'focus_changed' event."""
        events = diff_snapshots(
            single_window_snapshot, single_window_snapshot, 12345, 12345
        )
        focus_changes = [e for e in events if e["type"] == "focus_changed"]
        assert len(focus_changes) == 0

    def test_focus_to_new_window(self, empty_snapshot, single_window_snapshot):
        """Focus to newly opened window is detected."""
        events = diff_snapshots(
            empty_snapshot, single_window_snapshot, None, 12345
        )
        focus_changes = [e for e in events if e["type"] == "focus_changed"]
        assert len(focus_changes) == 1
        assert focus_changes[0]["hwnd"] == 12345

    def test_focus_from_closed_window(self, single_window_snapshot, empty_snapshot):
        """Focus from closed window - no focus_changed since new focus is None."""
        events = diff_snapshots(
            single_window_snapshot, empty_snapshot, 12345, None
        )
        # No focus_changed when next_focus is None
        focus_changes = [e for e in events if e["type"] == "focus_changed"]
        assert len(focus_changes) == 0


# ============================================================================
# DIFF_SNAPSHOTS - EVENT RECORD STRUCTURE
# ============================================================================

class TestEventRecordStructure:
    """Tests for event record structure and fields."""

    def test_opened_event_has_required_fields(self, empty_snapshot, single_window_snapshot):
        """Opened event has all required fields."""
        events = diff_snapshots(empty_snapshot, single_window_snapshot, None, None)
        opened = [e for e in events if e["type"] == "opened"][0]
        assert "ts" in opened
        assert "type" in opened
        assert "hwnd" in opened
        assert "title" in opened
        assert "process" in opened
        assert "to_monitor" in opened

    def test_closed_event_has_required_fields(self, single_window_snapshot, empty_snapshot):
        """Closed event has all required fields."""
        events = diff_snapshots(single_window_snapshot, empty_snapshot, None, None)
        closed = [e for e in events if e["type"] == "closed"][0]
        assert "ts" in closed
        assert "type" in closed
        assert "hwnd" in closed
        assert "title" in closed

    def test_moved_event_has_monitor_fields(self, single_window_snapshot):
        """Moved event has from_monitor/to_monitor fields."""
        new_snapshot = {
            12345: {**single_window_snapshot[12345], "monitor": 2}
        }
        events = diff_snapshots(single_window_snapshot, new_snapshot, None, None)
        moved = [e for e in events if e["type"] == "moved"][0]
        assert "from_monitor" in moved
        assert "to_monitor" in moved

    def test_focus_changed_has_hwnd(self, single_window_snapshot, two_window_snapshot):
        """Focus changed event has hwnd of new focused window."""
        events = diff_snapshots(
            single_window_snapshot, two_window_snapshot, 12345, 67890
        )
        focus = [e for e in events if e["type"] == "focus_changed"][0]
        assert "hwnd" in focus
        assert focus["hwnd"] == 67890


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests for diff algorithm."""

    def test_both_empty_no_events(self, empty_snapshot):
        """Both snapshots empty produces no events."""
        events = diff_snapshots(empty_snapshot, empty_snapshot, None, None)
        assert len(events) == 0

    def test_none_snapshots_handled(self):
        """None snapshots are handled gracefully."""
        # diff_snapshots expects dict, passing None should handle gracefully
        # or throw, let's test the behavior
        try:
            events = diff_snapshots(None, None, None, None)
            # If it doesn't throw, we expect empty or error
        except (TypeError, AttributeError):
            # Expected - None.keys() would fail
            pass

    def test_hwnd_zero_in_snapshot(self):
        """hwnd of 0 produces events (it's a valid hwnd number)."""
        prev = {0: {"hwnd": 0, "title": "Invalid", "process": "x", "monitor": 1}}
        next_ = {}
        events = diff_snapshots(prev, next_, None, None)
        # hwnd 0 is in prev but not next, so closed event
        closed = [e for e in events if e["type"] == "closed"]
        assert len(closed) == 1

    def test_monitor_change_skips_title_check(self, single_window_snapshot):
        """When monitor changes, title_changed is NOT also emitted (per continue)."""
        new_snapshot = {
            12345: {
                **single_window_snapshot[12345],
                "title": "New Title",  # Title also changed
                "monitor": 2,  # Monitor changed
            }
        }
        events = diff_snapshots(single_window_snapshot, new_snapshot, None, None)
        # Should get moved, but NOT title_changed (per continue in code)
        moved = [e for e in events if e["type"] == "moved"]
        title = [e for e in events if e["type"] == "title_changed"]
        assert len(moved) == 1
        assert len(title) == 0
