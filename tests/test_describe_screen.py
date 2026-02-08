"""
Tests for describe_screen tool and its response formatting.

Verifies:
- _format_screen_summary produces a spoken summary with window title/exe
- At least one control name appears when controls are available
- Response is never the generic "Done."
- Routing + formatting integration (mock UIA)
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# ── Unit tests for _format_screen_summary ────────────────────────────────

class TestFormatScreenSummary:
    """Test the deterministic screen summary formatter."""

    def test_includes_window_title_and_exe(self):
        from wyzer.tools.desktop.describe_screen import _format_screen_summary
        result = _format_screen_summary(
            window_info={"title": "Untitled - Notepad", "exe": "notepad.exe", "pid": 1234, "hwnd": 1},
            uia_info={
                "window": {"title": "Untitled - Notepad", "exe": "notepad.exe"},
                "controls": [
                    {"name": "File", "control_type": "MenuItem"},
                    {"name": "Edit", "control_type": "MenuItem"},
                ],
                "dialogs": [],
                "progress": None,
                "errors": [],
            },
        )
        assert "Notepad" in result["summary"]
        assert "Untitled" in result["summary"]
        assert result["window"]["exe"] == "notepad.exe"
        assert result["window"]["title"] == "Untitled - Notepad"

    def test_includes_control_names(self):
        from wyzer.tools.desktop.describe_screen import _format_screen_summary
        result = _format_screen_summary(
            window_info={"title": "My App", "exe": "myapp.exe"},
            uia_info={
                "window": {"title": "My App", "exe": "myapp.exe"},
                "controls": [
                    {"name": "Save", "control_type": "Button"},
                    {"name": "Cancel", "control_type": "Button"},
                    {"name": "Name", "control_type": "Edit"},
                ],
                "dialogs": [],
                "progress": None,
                "errors": [],
            },
        )
        assert "Save" in result["summary"]
        assert "Cancel" in result["summary"]
        assert len(result["highlights"]) >= 2

    def test_max_six_highlights(self):
        from wyzer.tools.desktop.describe_screen import _format_screen_summary
        controls = [{"name": f"Btn{i}", "control_type": "Button"} for i in range(20)]
        result = _format_screen_summary(
            window_info={"title": "Test", "exe": "test.exe"},
            uia_info={
                "window": {"title": "Test", "exe": "test.exe"},
                "controls": controls,
                "dialogs": [],
                "progress": None,
                "errors": [],
            },
        )
        assert len(result["highlights"]) <= 6

    def test_no_controls_says_so(self):
        from wyzer.tools.desktop.describe_screen import _format_screen_summary
        result = _format_screen_summary(
            window_info={"title": "Blank", "exe": "blank.exe"},
            uia_info={
                "window": {"title": "Blank", "exe": "blank.exe"},
                "controls": [],
                "dialogs": [],
                "progress": None,
                "errors": [],
            },
        )
        assert "no readable" in result["summary"].lower() or "not found" in result["summary"].lower()
        assert result["highlights"] == []

    def test_progress_bar_included(self):
        from wyzer.tools.desktop.describe_screen import _format_screen_summary
        result = _format_screen_summary(
            window_info={"title": "Installer", "exe": "setup.exe"},
            uia_info={
                "window": {"title": "Installer", "exe": "setup.exe"},
                "controls": [{"name": "Next", "control_type": "Button"}],
                "dialogs": [],
                "progress": {"value": 45, "text": ""},
                "errors": [],
            },
        )
        assert "45%" in result["summary"]

    def test_dialog_mentioned(self):
        from wyzer.tools.desktop.describe_screen import _format_screen_summary
        result = _format_screen_summary(
            window_info={"title": "App", "exe": "app.exe"},
            uia_info={
                "window": {"title": "App", "exe": "app.exe"},
                "controls": [{"name": "OK", "control_type": "Button"}],
                "dialogs": [{"title": "Save Changes?", "rect": None}],
                "progress": None,
                "errors": [],
            },
        )
        assert "Save Changes" in result["summary"]

    def test_never_returns_done(self):
        """The summary must never be literally 'Done.'"""
        from wyzer.tools.desktop.describe_screen import _format_screen_summary
        result = _format_screen_summary(
            window_info={"title": "", "exe": ""},
            uia_info={
                "window": {},
                "controls": [],
                "dialogs": [],
                "progress": None,
                "errors": [],
            },
        )
        assert result["summary"] != "Done."
        assert result["summary"].strip() != ""

    def test_deduplicates_highlights(self):
        from wyzer.tools.desktop.describe_screen import _format_screen_summary
        result = _format_screen_summary(
            window_info={"title": "X", "exe": "x.exe"},
            uia_info={
                "window": {"title": "X", "exe": "x.exe"},
                "controls": [
                    {"name": "Save", "control_type": "Button"},
                    {"name": "Save", "control_type": "Button"},
                    {"name": "save", "control_type": "Button"},
                ],
                "dialogs": [],
                "progress": None,
                "errors": [],
            },
        )
        assert result["highlights"].count("Save") == 1

    def test_long_title_truncated(self):
        from wyzer.tools.desktop.describe_screen import _format_screen_summary
        long_title = "A" * 200
        result = _format_screen_summary(
            window_info={"title": long_title, "exe": "app.exe"},
            uia_info={
                "window": {"title": long_title, "exe": "app.exe"},
                "controls": [],
                "dialogs": [],
                "progress": None,
                "errors": [],
            },
        )
        # Should NOT embed the full 200-char title
        assert len(result["summary"]) < 250


# ── Integration: format_info produces spoken reply ───────────────────────

class TestDescribeScreenFormatInfo:
    """Test that _format_fastpath_reply returns the summary, not 'Done.'"""

    def _make_execution_summary(self, tool, result):
        """Create a minimal ExecutionSummary + Intent list for testing."""
        from wyzer.core.intent_plan import ExecutionResult, ExecutionSummary, Intent
        er = ExecutionResult(tool=tool, ok=True, result=result, error=None)
        summary = ExecutionSummary(ran=[er], stopped_early=False)
        intents = [Intent(tool=tool, args={}, continue_on_error=False)]
        return summary, intents

    def test_format_fastpath_returns_summary_not_done(self):
        from wyzer.core.orchestrator import _format_fastpath_reply
        tool_result = {
            "summary": 'The focused window is Notepad: "Untitled - Notepad". Notable items: File, Edit, View.',
            "highlights": ["File", "Edit", "View"],
            "window": {"title": "Untitled - Notepad", "exe": "notepad.exe"},
            "evidence": {"control_count": 15, "dialog_count": 0, "progress": None, "errors": []},
        }
        summary, intents = self._make_execution_summary("describe_screen", tool_result)
        reply = _format_fastpath_reply("read the screen", intents, summary)
        assert reply != "Done."
        assert "Notepad" in reply
        assert "File" in reply

    def test_format_fastpath_no_controls(self):
        from wyzer.core.orchestrator import _format_fastpath_reply
        tool_result = {
            "summary": 'The focused window is App. No readable interactive controls were found.',
            "highlights": [],
            "window": {"title": "App", "exe": "app.exe"},
            "evidence": {"control_count": 0, "dialog_count": 0, "progress": None, "errors": []},
        }
        summary, intents = self._make_execution_summary("describe_screen", tool_result)
        reply = _format_fastpath_reply("describe the screen", intents, summary)
        assert reply != "Done."
        assert "no readable" in reply.lower() or "App" in reply


# ── Routing integration ─────────────────────────────────────────────────

class TestDescribeScreenRouting:
    """Test that screen-read phrases route to describe_screen."""

    def _route(self, text: str):
        from wyzer.core.hybrid_router import decide
        return decide(text)

    @pytest.mark.parametrize("phrase", [
        "read the screen",
        "read my screen",
        "describe the screen",
        "describe my screen",
        "what do you see",
        "what controls are on the screen?",
        "What can you see?",
        "what is on the screen",
    ])
    def test_routes_to_describe_screen(self, phrase):
        d = self._route(phrase)
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_whats_on_my_screen_still_window_context(self):
        """Anchored 'what's on my screen?' should still use describe_screen."""
        d = self._route("what's on my screen?")
        assert d.intents[0]["tool"] == "describe_screen"


# ── UIA noise reduction ─────────────────────────────────────────────────

class TestUiaNoseReduction:
    """Test that perceive_uia filters noisy text elements."""

    def test_long_text_filtered_in_format(self):
        """Very long Text names should not appear in highlights."""
        from wyzer.tools.desktop.describe_screen import _format_screen_summary
        long_text = "x" * 300
        result = _format_screen_summary(
            window_info={"title": "App", "exe": "app.exe"},
            uia_info={
                "window": {"title": "App", "exe": "app.exe"},
                "controls": [
                    {"name": long_text, "control_type": "Text"},
                    {"name": "OK", "control_type": "Button"},
                ],
                "dialogs": [],
                "progress": None,
                "errors": [],
            },
        )
        # The long text element should not be in highlights (it's type Text, not interactive)
        assert long_text not in result["highlights"]
        # OK button should be
        assert "OK" in result["highlights"]


# ── Tool registration ───────────────────────────────────────────────────

class TestDescribeScreenRegistered:
    def test_describe_screen_in_registry(self):
        from wyzer.tools.registry import build_default_registry
        registry = build_default_registry()
        assert registry.has_tool("describe_screen")

    def test_describe_screen_in_info_tools(self):
        from wyzer.policy.silence_is_success import INFO_TOOLS
        assert "describe_screen" in INFO_TOOLS
