"""Tests for wyzer.core.intercept_renderers — deterministic reply renderers.

Covers:
  - format_ui_content_reply: button-exists, no-match, generic listing
  - format_recent_events_reply: empty, single, multi-event summaries
  - _extract_search_term: regex parsing
  - _format_single_event: per-event formatting
"""

from __future__ import annotations

import pytest

from wyzer.core.intercept_renderers import (
    _extract_search_term,
    _format_single_event,
    format_recent_events_reply,
    format_ui_content_reply,
)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_controls(*specs):
    """Build a list of control dicts.  Each spec is (name, control_type)."""
    return [
        {"name": name, "control_type": ctype, "rect": None, "enabled": True}
        for name, ctype in specs
    ]


def _perception_result(controls=None, window=None, errors=None):
    return {
        "window": window or {"title": "Test Window", "exe": "test.exe"},
        "controls": controls or [],
        "dialogs": [],
        "progress": None,
        "errors": errors or [],
    }


# ═══════════════════════════════════════════════════════════════
# _extract_search_term
# ═══════════════════════════════════════════════════════════════

class TestExtractSearchTerm:
    def test_that_says(self):
        assert _extract_search_term("is there a button that says Install") == "Install"

    def test_called(self):
        assert _extract_search_term("do you see a link called Sign In?") == "Sign In"

    def test_named(self):
        assert _extract_search_term("can you find a button named OK?") == "OK"

    def test_labeled(self):
        assert _extract_search_term("is there a checkbox labeled Accept") == "Accept"

    def test_with_text(self):
        assert _extract_search_term("is there a button with text Submit") == "Submit"

    def test_no_match(self):
        assert _extract_search_term("what buttons are on the screen") is None

    def test_empty(self):
        assert _extract_search_term("") is None

    def test_quoted(self):
        assert _extract_search_term("is there a button that says 'Install'") == "Install"


# ═══════════════════════════════════════════════════════════════
# format_ui_content_reply — button exists
# ═══════════════════════════════════════════════════════════════

class TestUIContentReplyButtonExists:
    def test_button_found(self):
        controls = _make_controls(
            ("Install", "Button"),
            ("Cancel", "Button"),
        )
        result = _perception_result(controls=controls)
        reply = format_ui_content_reply("Is there a button that says Install?", result)
        assert "Yes" in reply
        assert "Install" in reply
        assert "Button" in reply

    def test_button_not_found(self):
        controls = _make_controls(
            ("Cancel", "Button"),
            ("OK", "Button"),
        )
        result = _perception_result(controls=controls)
        reply = format_ui_content_reply("Is there a button that says Install?", result)
        assert "No" in reply
        assert "install" in reply.lower()

    def test_button_not_found_shows_closest(self):
        controls = _make_controls(
            ("Cancel", "Button"),
            ("OK", "Button"),
            ("Help", "Button"),
        )
        result = _perception_result(controls=controls)
        reply = format_ui_content_reply("Is there a button that says Install?", result)
        assert "No" in reply
        assert "Cancel" in reply or "OK" in reply  # closest matches listed

    def test_case_insensitive(self):
        controls = _make_controls(
            ("INSTALL NOW", "Button"),
        )
        result = _perception_result(controls=controls)
        reply = format_ui_content_reply("is there a button that says install?", result)
        assert "Yes" in reply
        assert "INSTALL NOW" in reply

    def test_multiple_matches(self):
        controls = _make_controls(
            ("Install", "Button"),
            ("Install Updates", "Button"),
        )
        result = _perception_result(controls=controls)
        reply = format_ui_content_reply("Is there a button that says Install?", result)
        assert "Yes" in reply
        assert "2 matches" in reply

    def test_hyperlink_search(self):
        controls = _make_controls(
            ("Sign In", "Hyperlink"),
            ("Help", "Button"),
        )
        result = _perception_result(controls=controls)
        reply = format_ui_content_reply("Is there a link that says Sign In?", result)
        assert "Yes" in reply
        assert "Sign In" in reply

    def test_type_filter_excludes_wrong_type(self):
        controls = _make_controls(
            ("Install", "Text"),  # Text, not Button
            ("Cancel", "Button"),
        )
        result = _perception_result(controls=controls)
        reply = format_ui_content_reply("Is there a button that says Install?", result)
        assert "No" in reply  # "Install" is Text, not Button


# ═══════════════════════════════════════════════════════════════
# format_ui_content_reply — generic listing
# ═══════════════════════════════════════════════════════════════

class TestUIContentReplyGenericListing:
    def test_no_controls(self):
        result = _perception_result(controls=[])
        reply = format_ui_content_reply("what buttons are on the screen", result)
        assert "don't see" in reply.lower() or "no controls" in reply.lower()

    def test_with_controls(self):
        controls = _make_controls(
            ("OK", "Button"),
            ("Cancel", "Button"),
            ("Username", "Edit"),
        )
        result = _perception_result(controls=controls)
        reply = format_ui_content_reply("what buttons are on the screen", result)
        assert "OK" in reply
        assert "Cancel" in reply

    def test_error_no_controls(self):
        result = _perception_result(controls=[], errors=["no_foreground_window"])
        reply = format_ui_content_reply("read the window", result)
        assert "couldn't read" in reply.lower()

    def test_window_title_in_generic(self):
        controls = _make_controls(("OK", "Button"),)
        result = _perception_result(
            controls=controls,
            window={"title": "My App", "exe": "app.exe"},
        )
        reply = format_ui_content_reply("what buttons are on the screen", result)
        assert "My App" in reply


# ═══════════════════════════════════════════════════════════════
# _format_single_event
# ═══════════════════════════════════════════════════════════════

class TestFormatSingleEvent:
    def test_tool_end(self):
        evt = {"event": "tool_end", "tool": "open_target", "ok": True, "latency_ms": 250}
        line = _format_single_event(evt)
        assert "open_target" in line
        assert "succeeded" in line
        assert "250ms" in line

    def test_tool_start_skipped(self):
        evt = {"event": "tool_start", "tool": "open_target"}
        assert _format_single_event(evt) is None

    def test_perception(self):
        evt = {"event": "perception", "source": "uia", "found_controls_count": 42, "latency_ms": 800}
        line = _format_single_event(evt)
        assert "UIA" in line
        assert "42" in line
        assert "800ms" in line

    def test_world_evt(self):
        evt = {"event": "world_evt", "type": "focus_changed", "app": "chrome.exe", "title": "Google"}
        line = _format_single_event(evt)
        assert "focus_changed" in line
        assert "chrome.exe" in line
        assert "Google" in line

    def test_ui_action(self):
        evt = {"event": "ui_action", "action": "click", "target": "OK Button"}
        line = _format_single_event(evt)
        assert "click" in line
        assert "OK Button" in line

    def test_warning(self):
        evt = {"event": "warning", "message": "timeout reached"}
        line = _format_single_event(evt)
        assert "Warning" in line
        assert "timeout reached" in line

    def test_unknown(self):
        evt = {"event": "custom_event"}
        line = _format_single_event(evt)
        assert "custom_event" in line


# ═══════════════════════════════════════════════════════════════
# format_recent_events_reply
# ═══════════════════════════════════════════════════════════════

class TestRecentEventsReply:
    def test_empty_events(self):
        result = {"events": [], "count": 0}
        reply = format_recent_events_reply(result)
        assert "nothing" in reply.lower()

    def test_single_event(self):
        result = {
            "events": [
                {"event": "perception", "source": "uia", "found_controls_count": 60, "latency_ms": 1213}
            ],
            "count": 1,
        }
        reply = format_recent_events_reply(result)
        assert "UIA" in reply
        assert "60" in reply
        assert "1213ms" in reply
        assert "Most recently" in reply

    def test_multiple_events(self):
        result = {
            "events": [
                {"event": "tool_end", "tool": "open_target", "ok": True, "latency_ms": 100},
                {"event": "perception", "source": "uia", "found_controls_count": 30, "latency_ms": 500},
            ],
            "count": 2,
        }
        reply = format_recent_events_reply(result)
        assert "open_target" in reply
        assert "UIA" in reply
        assert "Here's what happened recently" in reply

    def test_skips_tool_start(self):
        result = {
            "events": [
                {"event": "tool_start", "tool": "get_time"},
                {"event": "tool_end", "tool": "get_time", "ok": True, "latency_ms": 5},
            ],
            "count": 2,
        }
        reply = format_recent_events_reply(result)
        assert "get_time" in reply
        assert "Most recently" in reply  # only one line after filtering tool_start

    def test_trims_to_five(self):
        events = [
            {"event": "tool_end", "tool": f"tool_{i}", "ok": True, "latency_ms": i * 10}
            for i in range(8)
        ]
        result = {"events": events, "count": 8}
        reply = format_recent_events_reply(result)
        # Should only mention last 5
        assert "tool_7" in reply
        assert "tool_3" in reply
        # tool_0, tool_1, tool_2 should be trimmed
        assert "tool_0" not in reply


# ═══════════════════════════════════════════════════════════════
# Integration: _format_fastpath_reply routing
# ═══════════════════════════════════════════════════════════════

class TestFastpathReplyRouting:
    """Verify that _format_fastpath_reply correctly delegates to intercept renderers."""

    def test_perceive_uia_returns_content_reply(self):
        from wyzer.core.orchestrator import _format_fastpath_reply

        class FakeIntent:
            def __init__(self, tool, args):
                self.tool = tool
                self.args = args
                self.meta = {}

        class FakeResult:
            def __init__(self, tool, result, ok=True, error=None):
                self.tool = tool
                self.result = result
                self.ok = ok
                self.error = error

        class FakeSummary:
            def __init__(self, ran):
                self.ran = ran

        controls = [
            {"name": "Install", "control_type": "Button", "rect": None, "enabled": True},
        ]
        uia_result = {
            "window": {"title": "Setup"},
            "controls": controls,
            "dialogs": [],
            "progress": None,
            "errors": [],
        }

        intents = [FakeIntent("perceive_uia_focused_window", {"max_nodes": 60})]
        summary = FakeSummary([FakeResult("perceive_uia_focused_window", uia_result)])

        reply = _format_fastpath_reply("Is there a button that says Install?", intents, summary)
        assert "Yes" in reply
        assert "Install" in reply
        assert reply != "Done."

    def test_get_recent_events_returns_summary(self):
        from wyzer.core.orchestrator import _format_fastpath_reply

        class FakeIntent:
            def __init__(self, tool, args):
                self.tool = tool
                self.args = args
                self.meta = {}

        class FakeResult:
            def __init__(self, tool, result, ok=True, error=None):
                self.tool = tool
                self.result = result
                self.ok = ok
                self.error = error

        class FakeSummary:
            def __init__(self, ran):
                self.ran = ran

        events_result = {
            "events": [
                {"event": "tool_end", "tool": "open_target", "ok": True, "latency_ms": 300},
            ],
            "count": 1,
        }

        intents = [FakeIntent("get_recent_events", {"limit": 10})]
        summary = FakeSummary([FakeResult("get_recent_events", events_result)])

        reply = _format_fastpath_reply("what did you just do?", intents, summary)
        assert "open_target" in reply
        assert reply != "Done."
