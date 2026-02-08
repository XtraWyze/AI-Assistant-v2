"""
Tests for Desktop Ground Truth tools (Phase 14).

Tests are structured to run without a real desktop where possible
(mocking Win32 APIs), and to verify event emission + routing.
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import patch, MagicMock
from collections import deque


# ── WorldState event_log tests ─────────────────────────────────────────────

class TestWorldStateEventLog:
    """Test event_log + emit_event on WorldState."""

    def setup_method(self):
        from wyzer.context.world_state import clear_world_state
        clear_world_state()

    def test_emit_event_appends(self):
        from wyzer.context.world_state import emit_event, get_event_log
        ev = emit_event("test_event", {"key": "value"})
        assert ev["event"] == "test_event"
        assert ev["key"] == "value"
        assert "ts" in ev
        log = get_event_log()
        assert len(log) >= 1
        assert log[-1]["event"] == "test_event"

    def test_event_log_ring_buffer(self):
        from wyzer.context.world_state import emit_event, get_world_state
        ws = get_world_state()
        # Fill beyond maxlen
        for i in range(210):
            emit_event("fill", {"i": i})
        assert len(ws.event_log) == 200  # maxlen

    def test_update_last_perception(self):
        from wyzer.context.world_state import update_last_perception, get_last_perception
        update_last_perception({"source": "uia", "controls": []})
        p = get_last_perception()
        assert p is not None
        assert p["source"] == "uia"

    def test_clear_resets_event_log(self):
        from wyzer.context.world_state import emit_event, get_world_state, clear_world_state
        emit_event("before_clear", {})
        clear_world_state()
        ws = get_world_state()
        assert len(ws.event_log) == 0
        assert ws.last_perception is None


# ── get_active_window tests ───────────────────────────────────────────────

class TestGetActiveWindow:
    """Test get_active_window tool."""

    def setup_method(self):
        from wyzer.context.world_state import clear_world_state
        clear_world_state()

    def test_returns_required_fields(self):
        from wyzer.tools.desktop.get_active_window import get_active_window_info
        info = get_active_window_info()
        assert "title" in info
        assert "exe" in info
        assert "pid" in info
        assert "hwnd" in info
        assert "timestamp" in info
        # Should not error on a real desktop
        if info.get("hwnd"):
            assert info["rect"] is not None

    def test_tool_emits_event(self):
        from wyzer.tools.desktop.get_active_window import GetActiveWindowTool
        from wyzer.context.world_state import get_event_log
        tool = GetActiveWindowTool()
        result = tool.run()
        log = get_event_log()
        perception_events = [e for e in log if e["event"] == "perception"]
        assert len(perception_events) >= 1
        assert perception_events[-1]["source"] == "get_active_window"


# ── perceive_uia tests ───────────────────────────────────────────────────

class TestPerceiveUIA:
    """Test perceive_uia_focused_window tool."""

    def setup_method(self):
        from wyzer.context.world_state import clear_world_state
        clear_world_state()

    def test_returns_structure(self):
        from wyzer.tools.desktop.perceive_uia import _try_pywinauto
        result = _try_pywinauto(max_nodes=10)
        assert "window" in result
        assert "controls" in result
        assert "dialogs" in result
        assert isinstance(result["controls"], list)

    def test_tool_emits_perception_event(self):
        from wyzer.tools.desktop.perceive_uia import PerceiveUIAFocusedWindowTool
        from wyzer.context.world_state import get_event_log
        tool = PerceiveUIAFocusedWindowTool()
        result = tool.run()
        log = get_event_log()
        perception_events = [e for e in log if e["event"] == "perception" and e.get("source") == "uia"]
        assert len(perception_events) >= 1
        assert "found_controls_count" in perception_events[-1]

    def test_max_nodes_respected(self):
        from wyzer.tools.desktop.perceive_uia import _try_pywinauto
        result = _try_pywinauto(max_nodes=5)
        assert len(result.get("controls", [])) <= 5


# ── Assertion helpers tests ──────────────────────────────────────────────

class TestAssertions:
    """Test ui_find_text and install_succeeded_check."""

    def test_ui_find_text_empty_query(self):
        from wyzer.tools.desktop.assertions import ui_find_text
        result = ui_find_text("", method="uia")
        assert result["found"] is False
        assert "empty_query" in result["evidence"]

    def test_ui_find_text_uia_runs(self):
        from wyzer.tools.desktop.assertions import ui_find_text
        # Search for something likely to exist (e.g. "Close" button in VS Code)
        result = ui_find_text("Close", method="uia")
        assert "found" in result
        assert "method" in result
        assert result["method"] == "uia"

    def test_install_succeeded_check_runs(self):
        from wyzer.tools.desktop.assertions import install_succeeded_check
        result = install_succeeded_check()
        assert result["status"] in ("success", "fail", "unknown")
        assert "evidence" in result


# ── Screenshot tool tests ────────────────────────────────────────────────

class TestScreenshot:
    """Test screenshot_focused_window tool."""

    def test_screenshot_returns_path(self):
        from wyzer.tools.desktop.screenshot_tool import _capture_focused_window
        result = _capture_focused_window()
        if "error" not in result:
            assert "image_path" in result
            import os
            assert os.path.isfile(result["image_path"])


# ── OCR tool tests ───────────────────────────────────────────────────────

class TestOCR:
    """Test ocr_region tool (graceful degradation)."""

    def test_ocr_without_image_degrades_gracefully(self):
        from wyzer.tools.desktop.ocr_tool import _run_ocr
        result = _run_ocr()
        # Should either work or report missing_dependency / error
        assert "lines" in result or "error" in result or "missing_dependency" in result


# ── Hybrid router tests ─────────────────────────────────────────────────

class TestHybridRouterDesktopPatterns:
    """Test that Phase 14 patterns route deterministically."""

    def _route(self, text: str):
        from wyzer.core.hybrid_router import decide
        return decide(text)

    def test_whats_on_screen_right_now(self):
        d = self._route("describe the screen")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_whats_on_screen_right_now_no_apostrophe(self):
        d = self._route("what controls are on the screen?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_describe_whats_on_screen(self):
        d = self._route("describe what's on my screen")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_is_there_a_button_install(self):
        d = self._route("is there a button that says Install?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"
        assert d.intents[0]["args"]["text"] == "Install"

    def test_is_there_a_button_play(self):
        d = self._route("is there a button called Play?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"
        assert d.intents[0]["args"]["text"] == "Play"

    def test_is_there_a_play_button(self):
        d = self._route("is there a Play button?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"

    def test_did_install_succeed(self):
        d = self._route("did install succeed?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "install_succeeded_check"

    def test_did_the_install_finish(self):
        d = self._route("did the install finish?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "install_succeeded_check"

    def test_is_it_installed(self):
        d = self._route("is it installed?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "install_succeeded_check"

    def test_existing_screen_route_preserved(self):
        """Existing 'what's on my screen' route still works."""
        d = self._route("what's on my screen?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_existing_list_windows_preserved(self):
        """Existing 'what windows are open' route still works."""
        d = self._route("what windows are open?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "list_open_windows"


# ── Phase 14b: Broad screen-description & element-verify routing tests ──

class TestBroadScreenDescriptionRouting:
    """Test that natural-speech screen-description queries route to
    describe_screen via normalized phrase containment."""

    def _route(self, text: str):
        from wyzer.core.hybrid_router import decide
        return decide(text)

    def test_oh_whats_on_my_screen_describe(self):
        """The exact failing phrase from the logs."""
        d = self._route("Oh, what's on my screen? Can you describe it?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_what_do_you_see_right_now(self):
        d = self._route("What do you see right now?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_can_you_describe_it(self):
        d = self._route("Hey, can you describe it?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_whats_on_the_screen(self):
        d = self._route("whats on the screen")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_describe_my_screen(self):
        d = self._route("describe my screen")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_tell_me_whats_on_my_screen(self):
        d = self._route("tell me what's on my screen")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_whats_currently_on_my_screen(self):
        d = self._route("What's currently on my screen?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_screen_right_now(self):
        d = self._route("What about the screen right now?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_read_the_screen(self):
        d = self._route("read the screen")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_existing_whats_on_my_screen_still_works(self):
        """Anchored 'what's on my screen' should still route to describe_screen."""
        d = self._route("what's on my screen?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"


class TestBroadVerifyElementRouting:
    """Test that element-verify queries route to ui_find_text via
    normalized trigger + label matching."""

    def _route(self, text: str):
        from wyzer.core.hybrid_router import decide
        return decide(text)

    def test_you_see_install_button(self):
        """The exact failing phrase from the logs."""
        d = self._route("You see an install button.")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"
        assert d.intents[0]["args"]["text"] == "install"

    def test_is_there_button_says_play(self):
        d = self._route("Is there a button that says Play?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"

    def test_do_you_see_close_button(self):
        d = self._route("do you see a close button?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"
        assert d.intents[0]["args"]["text"] == "close"

    def test_can_you_see_the_submit_button(self):
        d = self._route("Can you see the submit button?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"

    def test_i_see_a_download_button(self):
        d = self._route("I see a download button")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"

    def test_is_there_an_ok_button(self):
        d = self._route("Is there an OK button?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"

    def test_do_you_see_cancel(self):
        d = self._route("Do you see cancel?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"
        assert d.intents[0]["args"]["text"] == "cancel"


class TestNormalizationHelper:
    """Verify _normalize_text_for_routing works as expected for screen queries."""

    def test_strips_apostrophes_and_punctuation(self):
        from wyzer.core.hybrid_router import _normalize_text_for_routing
        result = _normalize_text_for_routing("what's on my screen?")
        assert result == "whats on my screen"

    def test_collapses_whitespace(self):
        from wyzer.core.hybrid_router import _normalize_text_for_routing
        result = _normalize_text_for_routing("  what   is  on  my  screen ?  ")
        assert result == "what is on my screen"

    def test_lowercase(self):
        from wyzer.core.hybrid_router import _normalize_text_for_routing
        result = _normalize_text_for_routing("DESCRIBE MY SCREEN")
        assert result == "describe my screen"

    def test_multi_sentence(self):
        from wyzer.core.hybrid_router import _normalize_text_for_routing
        result = _normalize_text_for_routing("Oh, what's on my screen? Can you describe it?")
        assert "whats on my screen" in result
        assert "can you describe it" in result


# ── Tool registry tests ─────────────────────────────────────────────────

class TestToolRegistry:
    """Test that all Phase 14 tools are registered."""

    def test_all_tools_registered(self):
        from wyzer.tools.registry import build_default_registry
        registry = build_default_registry()
        expected_tools = [
            "get_active_window",
            "perceive_uia_focused_window",
            "describe_screen",
            "desktop_click_uia",
            "screenshot_focused_window",
            "ocr_region",
            "ui_find_text",
            "install_succeeded_check",
            "wait_ms",
            "hotkey",
            "type_text",
            "click_xy",
            "scroll",
        ]
        for tool_name in expected_tools:
            assert registry.has_tool(tool_name), f"Tool '{tool_name}' not registered"


# ── Input action tests ──────────────────────────────────────────────────

class TestInputActions:
    """Test input action tools."""

    def setup_method(self):
        from wyzer.context.world_state import clear_world_state
        clear_world_state()

    def test_wait_ms(self):
        from wyzer.tools.desktop.input_actions import WaitMsTool
        from wyzer.context.world_state import get_event_log
        tool = WaitMsTool()
        start = time.time()
        result = tool.run(ms=50)
        elapsed = time.time() - start
        assert result["waited_ms"] == 50
        assert elapsed >= 0.04  # allow some slack
        log = get_event_log()
        assert any(e.get("kind") == "wait_ms" for e in log)

    def test_wait_ms_capped(self):
        from wyzer.tools.desktop.input_actions import WaitMsTool
        tool = WaitMsTool()
        # Should cap at 30000
        result = tool.run(ms=999999)
        assert result["waited_ms"] == 30000
