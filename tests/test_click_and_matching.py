"""
Tests for improved ui_find_text matching and click-command routing.

Verifies:
- exact/word/contains match modes work correctly
- control_type filtering prevents false positives
- "Do you see an install button?" does NOT match "pre-installed"
- "Click the Maximize button" routes to maximize_window (not LLM)
- Generic "click X" routes to desktop_click_uia
- desktop_click_uia produces a deterministic summary
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# ═════════════════════════════════════════════════════════════════════════════
# A) Match-mode tests for ui_find_text
# ═════════════════════════════════════════════════════════════════════════════

class TestTextNormalization:
    """Verify _normalize_label and _text_matches."""

    def test_normalize_strips_punctuation(self):
        from wyzer.tools.desktop.assertions import _normalize_label
        assert _normalize_label("  Pre-Installed!  ") == "pre installed"

    def test_normalize_collapses_spaces(self):
        from wyzer.tools.desktop.assertions import _normalize_label
        assert _normalize_label("Save   As...") == "save as"

    def test_exact_match(self):
        from wyzer.tools.desktop.assertions import _text_matches
        assert _text_matches("install", "install", "exact") is True
        assert _text_matches("install", "pre installed", "exact") is False
        assert _text_matches("install", "install now", "exact") is False

    def test_word_match(self):
        from wyzer.tools.desktop.assertions import _text_matches
        assert _text_matches("install", "click install now", "word") is True
        assert _text_matches("install", "pre installed", "word") is False
        assert _text_matches("install", "reinstall", "word") is False

    def test_contains_match_legacy(self):
        from wyzer.tools.desktop.assertions import _text_matches
        assert _text_matches("install", "pre installed", "contains") is True
        assert _text_matches("install", "reinstall", "contains") is True
        assert _text_matches("install", "click here", "contains") is False


class TestUIFindTextExactMode:
    """Verify exact mode + control_type filter prevents false positives."""

    def _make_uia_snapshot(self, controls):
        return {
            "controls": controls,
            "dialogs": [],
            "progress": None,
            "errors": [],
        }

    def test_exact_does_not_match_pre_installed(self):
        """'install' with exact mode must NOT match 'Pre-Installed'."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Pre-Installed", "control_type": "Button", "rect": None},
            {"name": "Bookmarks", "control_type": "MenuItem", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("Install", method="uia", control_type="Button", match_mode="exact")

        assert result["found"] is False
        assert result["summary"].startswith("No")

    def test_exact_matches_exact_label(self):
        """'install' with exact mode must match a button labelled exactly 'Install'."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Install", "control_type": "Button", "rect": None},
            {"name": "Pre-Installed", "control_type": "Button", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("Install", method="uia", control_type="Button", match_mode="exact")

        assert result["found"] is True
        assert len(result["matches"]) == 1
        assert result["matches"][0]["name"] == "Install"
        assert result["summary"].startswith("Yes")

    def test_control_type_filter_skips_non_buttons(self):
        """control_type='Button' must skip MenuItem even if name matches."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Install", "control_type": "MenuItem", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("Install", method="uia", control_type="Button", match_mode="exact")

        assert result["found"] is False

    def test_word_mode_matches_button_install(self):
        """'install' with word mode matches 'Install Now' but not 'Pre-Installed'."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Install Now", "control_type": "Button", "rect": None},
            {"name": "Pre-Installed", "control_type": "Button", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("Install", method="uia", match_mode="word")

        assert result["found"] is True
        assert len(result["matches"]) == 1
        assert result["matches"][0]["name"] == "Install Now"

    def test_contains_mode_is_default(self):
        """Default contains mode matches substrings (backward compat)."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Pre-Installed", "control_type": "Button", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("install", method="uia")

        assert result["found"] is True  # contains is default

    def test_evidence_mentions_mode(self):
        """Evidence string includes match mode."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([
            {"name": "Save", "control_type": "Button", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("Save", method="uia", match_mode="exact")

        assert "mode=exact" in result["evidence"]

    def test_evidence_mentions_control_type(self):
        """Evidence string includes control_type filter."""
        from wyzer.tools.desktop.assertions import ui_find_text

        fake_snapshot = self._make_uia_snapshot([])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text("Save", method="uia", control_type="Button", match_mode="exact")

        assert "type=Button" in result["evidence"]


# ═════════════════════════════════════════════════════════════════════════════
# B) Click-command routing tests
# ═════════════════════════════════════════════════════════════════════════════

class TestClickCommandRouting:
    """Test that click commands route to deterministic tools, not LLM."""

    def _route(self, text: str):
        from wyzer.core.hybrid_router import decide
        return decide(text)

    @pytest.mark.parametrize("phrase,tool", [
        ("click the Maximize button", "maximize_window"),
        ("Click Maximize", "maximize_window"),
        ("press maximize", "maximize_window"),
        ("hit the maximize button", "maximize_window"),
        ("click the Minimize button", "minimize_window"),
        ("press minimize", "minimize_window"),
        ("click close", "close_window"),
        ("press the close button", "close_window"),
    ])
    def test_click_winctl_routes_to_native_tool(self, phrase, tool):
        d = self._route(phrase)
        assert d.mode == "tool_plan", f"'{phrase}' should be tool_plan, got {d.mode}"
        assert d.intents[0]["tool"] == tool, f"'{phrase}' should use {tool}, got {d.intents[0]['tool']}"

    @pytest.mark.parametrize("phrase", [
        "click the Save button",
        "press the OK button",
        "hit Apply",
        "tap Submit",
    ])
    def test_click_generic_routes_to_click_and_type(self, phrase):
        d = self._route(phrase)
        assert d.mode == "tool_plan", f"'{phrase}' should be tool_plan, got {d.mode}"
        assert d.intents[0]["tool"] == "__CLICK_AND_TYPE__", f"'{phrase}' should use __CLICK_AND_TYPE__, got {d.intents[0]['tool']}"

    def test_click_save_no_forced_control_type(self):
        """Generic clicks should NOT force control_type — let the resolver decide."""
        d = self._route("click the Save button")
        assert "control_type" not in d.intents[0]["args"], (
            "Generic click should not force control_type"
        )

    def test_click_maximize_does_not_use_llm(self):
        d = self._route("Click the Maximize button")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "maximize_window"


class TestButtonCheckRoutingWithExactMode:
    """Test that 'is there an install button' uses exact mode."""

    def _route(self, text: str):
        from wyzer.core.hybrid_router import decide
        return decide(text)

    def test_install_button_uses_exact_and_button_type(self):
        d = self._route("Do you see an install button?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"
        args = d.intents[0]["args"]
        assert args.get("match_mode") == "exact"
        assert args.get("control_type") == "Button"

    def test_button_that_says_install(self):
        d = self._route("Is there a button that says Install?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"
        args = d.intents[0]["args"]
        assert args.get("match_mode") == "exact"
        assert args.get("control_type") == "Button"

    def test_can_you_see_a_play_button(self):
        d = self._route("Can you see a play button?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"
        args = d.intents[0]["args"]
        # Should use exact mode since "button" is mentioned
        assert args.get("match_mode") == "exact"
        assert args.get("control_type") == "Button"


# ═════════════════════════════════════════════════════════════════════════════
# C) Integration: exact mode prevents "pre-installed" from matching
# ═════════════════════════════════════════════════════════════════════════════

class TestInstallButtonFalsePositive:
    """End-to-end: 'Do you see an install button?' should NOT match 'Pre-Installed'."""

    def _make_uia_snapshot(self, controls):
        return {
            "controls": controls,
            "dialogs": [],
            "progress": None,
            "errors": [],
        }

    def test_no_false_positive_pre_installed(self):
        from wyzer.core.hybrid_router import decide
        from wyzer.tools.desktop.assertions import ui_find_text

        # First verify the router sends exact mode
        d = decide("Do you see an install button?")
        assert d.intents[0]["args"].get("match_mode") == "exact"
        assert d.intents[0]["args"].get("control_type") == "Button"

        # Then verify the tool with those args rejects "Pre-Installed"
        fake_snapshot = self._make_uia_snapshot([
            {"name": "Pre-Installed", "control_type": "Button", "rect": None},
            {"name": "Bookmarks", "control_type": "MenuItem", "rect": None},
        ])
        with patch("wyzer.tools.desktop.perceive_uia._try_pywinauto", return_value=fake_snapshot):
            result = ui_find_text(
                text=d.intents[0]["args"]["text"],
                method=d.intents[0]["args"]["method"],
                control_type=d.intents[0]["args"]["control_type"],
                match_mode=d.intents[0]["args"]["match_mode"],
            )

        assert result["found"] is False
        assert result["summary"].startswith("No")


# ═════════════════════════════════════════════════════════════════════════════
# D) desktop_click_uia summary field
# ═════════════════════════════════════════════════════════════════════════════

class TestDesktopClickUIASummary:
    """Test that desktop_click_uia returns a deterministic summary."""

    def test_click_success_summary(self):
        from wyzer.tools.desktop.desktop_click_uia import DesktopClickUIATool

        fake_result = {
            "clicked": True,
            "matched": {"name": "Save", "type": "Button", "rect": None},
        }
        tool = DesktopClickUIATool()
        with patch("wyzer.tools.desktop.desktop_click_uia._best_match", return_value=fake_result):
            with patch("wyzer.context.world_state.emit_event"):
                result = tool.run(name="Save")

        assert "summary" in result
        assert "Clicked Save" in result["summary"]

    def test_click_failure_summary(self):
        from wyzer.tools.desktop.desktop_click_uia import DesktopClickUIATool

        fake_result = {
            "clicked": False,
            "reason": "no_control_matching 'Save'",
        }
        tool = DesktopClickUIATool()
        with patch("wyzer.tools.desktop.desktop_click_uia._best_match", return_value=fake_result):
            with patch("wyzer.context.world_state.emit_event"):
                result = tool.run(name="Save")

        assert "summary" in result
        assert "couldn't click" in result["summary"].lower()
        assert result["summary"] != "Done."
