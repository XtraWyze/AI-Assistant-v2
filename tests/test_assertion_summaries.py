"""
Tests for assertion tool summaries (ui_find_text, install_succeeded_check).

Verifies:
- ui_find_text returns a deterministic summary with "Yes" or "No"
- install_succeeded_check returns a deterministic summary (never "Done.")
- _format_fastpath_reply uses the summary field, not "Done."
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# ── ui_find_text summary tests ──────────────────────────────────────────

class TestUIFindTextSummary:
    """Test that ui_find_text produces a deterministic spoken summary."""

    def _make_uia_snapshot(self, controls):
        return {
            "controls": controls,
            "dialogs": [],
            "progress": None,
            "errors": [],
        }

    def test_found_includes_yes(self):
        """When text is found, summary starts with 'Yes'."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Save", "control_type": "Button", "rect": None},
            {"name": "Save As", "control_type": "MenuItem", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("save", method="uia")

        assert result["found"] is True
        assert "summary" in result
        assert result["summary"].startswith("Yes")
        assert "2" in result["summary"]  # 2 matches
        assert result["summary"] != "Done."

    def test_not_found_includes_no(self):
        """When text is not found, summary starts with 'No'."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([
            {"name": "File", "control_type": "MenuItem", "rect": None},
            {"name": "Edit", "control_type": "MenuItem", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("install", method="uia")

        assert result["found"] is False
        assert "summary" in result
        assert result["summary"].startswith("No")
        assert "install" in result["summary"]
        assert result["summary"] != "Done."

    def test_single_match_says_match(self):
        """Singular 'match' for 1 result."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Install", "control_type": "Button", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("install", method="uia")

        assert "1 match" in result["summary"]
        assert "matches" not in result["summary"]

    def test_multiple_matches_says_matches(self):
        """Plural 'matches' for >1 results."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Save", "control_type": "Button", "rect": None},
            {"name": "Save As", "control_type": "MenuItem", "rect": None},
            {"name": "Auto Save", "control_type": "CheckBox", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("save", method="uia")

        assert "3 matches" in result["summary"]

    def test_at_most_three_names_in_summary(self):
        """Summary lists at most 3 match names."""
        from wyzer.tools.desktop.assertions import ui_find_text

        controls = [{"name": f"Save {i}", "control_type": "Button", "rect": None} for i in range(10)]
        fake_snapshot = self._make_uia_snapshot(controls)
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("save", method="uia")

        # Count comma-separated names in summary
        assert result["found"] is True
        # Should mention 10 matches but only list 3 names
        assert "10 matches" in result["summary"]
        # The summary should not contain more than 3 control names
        names_part = result["summary"].split(": ", 1)[-1]
        assert names_part.count(",") <= 2  # at most 3 items = at most 2 commas


class TestUIFindTextOCRSummary:
    """Test OCR path also produces summary."""

    def test_ocr_found(self):
        from wyzer.tools.desktop.assertions import _find_text_ocr

        fake_ocr = {
            "full_text": "Welcome to the installer. Click Next to continue.",
            "lines": [
                {"text": "Welcome to the installer."},
                {"text": "Click Next to continue."},
            ],
        }
        with patch("wyzer.tools.desktop.ocr_tool._run_ocr", return_value=fake_ocr):
            result = _find_text_ocr("next")

        assert result["found"] is True
        assert "summary" in result
        assert result["summary"].startswith("Yes")
        assert result["summary"] != "Done."

    def test_ocr_not_found(self):
        from wyzer.tools.desktop.assertions import _find_text_ocr

        fake_ocr = {
            "full_text": "Hello world",
            "lines": [{"text": "Hello world"}],
        }
        with patch("wyzer.tools.desktop.ocr_tool._run_ocr", return_value=fake_ocr):
            result = _find_text_ocr("missing")

        assert result["found"] is False
        assert "summary" in result
        assert result["summary"].startswith("No")


# ── install_succeeded_check summary tests ────────────────────────────────

class TestInstallSucceededCheckSummary:
    """Test that install_succeeded_check produces a deterministic summary."""

    def _make_uia_snapshot(self, controls, dialogs=None, progress=None, errors=None):
        return {
            "controls": controls,
            "dialogs": dialogs or [],
            "progress": progress,
            "errors": errors or [],
        }

    def test_success_summary(self):
        from wyzer.tools.desktop.assertions import install_succeeded_check

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Play", "control_type": "Button"},
            {"name": "Installed", "control_type": "Text"},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = install_succeeded_check()

        assert result["status"] == "success"
        assert "summary" in result
        assert "succeeded" in result["summary"].lower()
        assert result["summary"] != "Done."

    def test_fail_summary(self):
        from wyzer.tools.desktop.assertions import install_succeeded_check

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Error occurred", "control_type": "Text"},
            {"name": "Retry", "control_type": "Button"},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = install_succeeded_check()

        assert result["status"] == "fail"
        assert "summary" in result
        assert "failed" in result["summary"].lower()
        assert result["summary"] != "Done."

    def test_unknown_summary(self):
        from wyzer.tools.desktop.assertions import install_succeeded_check

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Loading", "control_type": "Text"},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = install_succeeded_check()

        assert result["status"] == "unknown"
        assert "summary" in result
        assert "not sure" in result["summary"].lower()
        assert result["summary"] != "Done."

    def test_mixed_gives_not_sure(self):
        from wyzer.tools.desktop.assertions import install_succeeded_check

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Installed", "control_type": "Text"},
            {"name": "Error log", "control_type": "Text"},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = install_succeeded_check()

        assert result["status"] == "unknown"
        assert "not sure" in result["summary"].lower()


# ── format_fastpath_reply integration ────────────────────────────────────

class TestSummaryFieldInFormatFastpathReply:
    """Tool results with a 'summary' field should be spoken instead of 'Done.'"""

    def _make_execution_summary(self, tool, result):
        from wyzer.core.intent_plan import ExecutionResult, ExecutionSummary, Intent
        er = ExecutionResult(tool=tool, ok=True, result=result, error=None)
        summary = ExecutionSummary(ran=[er], stopped_early=False)
        intents = [Intent(tool=tool, args={}, continue_on_error=False)]
        return summary, intents

    def test_ui_find_text_found_reply(self):
        from wyzer.core.orchestrator import _format_fastpath_reply

        tool_result = {
            "found": True,
            "evidence": "UIA: found 2 control(s) matching 'save'",
            "method": "uia",
            "matches": [
                {"name": "Save", "control_type": "Button", "rect": None},
                {"name": "Save As", "control_type": "MenuItem", "rect": None},
            ],
            "summary": "Yes \u2014 I found 2 matches: Save, Save As.",
        }
        es, intents = self._make_execution_summary("ui_find_text", tool_result)
        reply = _format_fastpath_reply("is there a save button?", intents, es)
        assert reply != "Done."
        assert "Yes" in reply
        assert "Save" in reply

    def test_ui_find_text_not_found_reply(self):
        from wyzer.core.orchestrator import _format_fastpath_reply

        tool_result = {
            "found": False,
            "evidence": "UIA: no controls matching 'install' in 20 scanned",
            "method": "uia",
            "matches": [],
            "summary": "No \u2014 I didn't find any controls matching 'install'.",
        }
        es, intents = self._make_execution_summary("ui_find_text", tool_result)
        reply = _format_fastpath_reply("is there an install button?", intents, es)
        assert reply != "Done."
        assert "No" in reply

    def test_install_succeeded_check_reply(self):
        from wyzer.core.orchestrator import _format_fastpath_reply

        tool_result = {
            "status": "success",
            "evidence": "Success indicators found: ['Play']",
            "details": {},
            "summary": "The install succeeded. I see: Play.",
        }
        es, intents = self._make_execution_summary("install_succeeded_check", tool_result)
        reply = _format_fastpath_reply("did the install work?", intents, es)
        assert reply != "Done."
        assert "succeeded" in reply.lower()

    def test_unknown_tool_with_summary_uses_it(self):
        """Any tool that sets summary should have its summary spoken."""
        from wyzer.core.orchestrator import _format_fastpath_reply

        tool_result = {
            "ok": True,
            "summary": "Custom tool finished with 3 items processed.",
        }
        es, intents = self._make_execution_summary("some_future_tool", tool_result)
        reply = _format_fastpath_reply("do the thing", intents, es)
        assert reply != "Done."
        assert "3 items" in reply

    def test_tool_without_summary_still_says_done(self):
        """Tools with no summary field should still fall back to 'Done.'"""
        from wyzer.core.orchestrator import _format_fastpath_reply

        tool_result = {"ok": True}
        es, intents = self._make_execution_summary("some_silent_tool", tool_result)
        reply = _format_fastpath_reply("do something", intents, es)
        assert reply == "Done."
