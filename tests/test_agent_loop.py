"""
Tests for Phase 17 — Agent-Grade Behavior.

Covers:
  1. UI hallucination prevention (perception fails → deterministic refusal)
  2. Micro-loop (observe → act → observe → final)
  3. Follow-up: "Yes, it's Notepad" triggers close
  4. Prompt injection: orchestrator prompt includes [PERCEPTION SNAPSHOT]
"""

from __future__ import annotations

import json
import time
import types
import pytest
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canonical(
    *,
    fg_app: str | None = "notepad.exe",
    fg_title: str | None = "Untitled - Notepad",
    controls: list | None = None,
    ocr: list | None = None,
    errors: list | None = None,
    windows: list | None = None,
) -> Dict[str, Any]:
    """Build a canonical perception snapshot for tests."""
    return {
        "timestamp_ms": int(time.time() * 1000),
        "foreground": {"app": fg_app, "title": fg_title},
        "windows": windows or [{"title": fg_title or "", "app": fg_app}],
        "controls": controls or [],
        "ocr_text": ocr or [],
        "errors": errors or [],
    }


# =========================================================================
# 1) UI HALLUCINATION PREVENTION
# =========================================================================

class TestUIHallucinationPrevention:
    """When perception returns a fatal error, the response must be a
    deterministic refusal — never an LLM-generated answer with invented UI."""

    def test_fatal_perception_error_returns_deterministic_refusal(self):
        """Fatal perception errors → 'couldn't read UI controls…' message."""
        from wyzer.tools.desktop.truth_contract import has_fatal_perception_error

        canonical = _make_canonical(errors=["no_foreground_window"])
        assert has_fatal_perception_error(canonical) is True

    def test_no_fatal_error_passes(self):
        from wyzer.tools.desktop.truth_contract import has_fatal_perception_error

        canonical = _make_canonical(errors=[])
        assert has_fatal_perception_error(canonical) is False

    def test_agent_loop_bails_on_fatal_perception(self):
        """run_agent_loop must return the refusal string when perception
        has a fatal error, without calling the LLM."""
        from wyzer.core.orchestrator import run_agent_loop, _mark_obs_stale

        _mark_obs_stale()

        fatal_canonical = _make_canonical(errors=["pywinauto_not_installed"])

        with patch("wyzer.core.orchestrator._run_perception", return_value=fatal_canonical):
            result = run_agent_loop(
                user_text="What's on screen?",
                hybrid_decision=MagicMock(intents=[]),
                registry=MagicMock(),
                start_time=time.perf_counter(),
                original_text="What's on screen?",
            )

        assert "couldn\u2019t read ui controls" in result["reply"].lower() or "couldn't read ui controls" in result["reply"].lower()
        assert result["meta"].get("perception_fatal") is True


# =========================================================================
# 2) MICRO-LOOP: observe → click → observe → final
# =========================================================================

class TestMicroLoop:
    """Perception shows button "Install"; LLM says click it; next perception
    shows "Progress 50%"; LLM says final "Install started."  Loop ran twice."""

    def test_two_step_loop(self):
        from wyzer.core.orchestrator import run_agent_loop, _mark_obs_stale
        import wyzer.core.orchestrator as orch_mod

        _mark_obs_stale()

        # Two different perception snapshots
        snap_1 = _make_canonical(
            controls=[{"name": "Install", "type": "Button", "automation_id": None}],
        )
        snap_2 = _make_canonical(
            controls=[{"name": "Progress 50%", "type": "ProgressBar", "automation_id": None}],
        )
        perception_calls = iter([snap_1, snap_2])

        def fake_perception():
            return next(perception_calls, snap_2)

        # LLM responses: first returns tool_calls (click), second returns final
        llm_responses = iter([
            json.dumps({
                "type": "tool_calls",
                "tool_calls": [{"name": "desktop_click_uia", "args": {"target": "Install"}}],
            }),
            json.dumps({
                "type": "final",
                "final": "Install started.",
            }),
        ])

        mock_llm = MagicMock()
        mock_llm.generate_chat = MagicMock(side_effect=lambda **kw: next(llm_responses))

        with patch.object(orch_mod, "_run_perception", side_effect=fake_perception), \
             patch.object(orch_mod, "_get_llm_client", return_value=mock_llm), \
             patch.object(orch_mod, "_execute_tool", return_value={"ok": True}), \
             patch("wyzer.core.orchestrator.Config") as mock_cfg:
            mock_cfg.OLLAMA_MODEL = "test"
            mock_cfg.LLM_MODE = "llamacpp"
            mock_cfg.LLM_TIMEOUT = 30
            # Also mark stale after the tool call (simulating UI-affecting tool)
            result = run_agent_loop(
                user_text="Click install",
                hybrid_decision=MagicMock(intents=[]),
                registry=MagicMock(),
                start_time=time.perf_counter(),
                original_text="Click install",
            )

        assert result["reply"] == "Install started."
        assert result["meta"].get("agent_loop") is True
        assert result["meta"].get("steps") == 2


# =========================================================================
# 3) FOLLOW-UP: "Yes, it's Notepad." triggers close
# =========================================================================

class TestFollowUpResolution:
    """After perception showed Notepad, user says 'Yes, it's Notepad.'
    → resolve_yes_its_x returns {action: close_window, title: …}."""

    def test_yes_its_notepad_resolves(self):
        from wyzer.core.followup_manager import FollowupManager

        fm = FollowupManager()
        fm._agent_last_window_candidates = [
            {"title": "Untitled - Notepad", "app": "notepad.exe"},
            {"title": "Google Chrome", "app": "chrome.exe"},
        ]

        result = fm.resolve_yes_its_x("Yes, it's Notepad")
        assert result is not None
        assert result["action"] == "close_window"
        assert "notepad" in result["title"].lower()

    def test_yes_its_unknown_returns_none(self):
        from wyzer.core.followup_manager import FollowupManager

        fm = FollowupManager()
        fm._agent_last_window_candidates = [
            {"title": "Untitled - Notepad", "app": "notepad.exe"},
        ]

        result = fm.resolve_yes_its_x("Yes, it's Firefox")
        assert result is None

    def test_resolve_agent_followup_with_yes(self):
        """End-to-end: resolve_agent_followup picks up 'yes it's Notepad'."""
        from wyzer.core.reference_resolver import resolve_agent_followup
        from wyzer.core.followup_manager import FollowupManager

        fm = FollowupManager()
        fm._agent_last_window_candidates = [
            {"title": "Untitled - Notepad", "app": "notepad.exe"},
        ]

        # Patch the orchestrator's _get_followup_manager to return our fm
        with patch("wyzer.core.orchestrator._get_followup_manager", return_value=fm):
            # Need to ensure the module is in sys.modules
            import sys
            import wyzer.core.orchestrator
            result = resolve_agent_followup("Yes, it's Notepad")

        assert result is not None
        assert result["action"] == "close_window"


# =========================================================================
# 4) PROMPT INJECTION: verify prompt includes markers
# =========================================================================

class TestPromptInjection:
    """The agent-loop prompt MUST include [PERCEPTION SNAPSHOT] and
    [RECENT EVENTS] sections for UI-state queries."""

    def test_prompt_contains_perception_and_events(self):
        from wyzer.core.orchestrator import _build_agent_prompt

        canonical = _make_canonical(
            controls=[{"name": "Save", "type": "Button", "automation_id": None}],
        )
        events = [
            {"event": "tool_end", "ts": time.time(), "tool": "open_app", "success": True},
        ]

        prompt = _build_agent_prompt(
            user_text="What's on screen?",
            canonical=canonical,
            recent_events=events,
            step=0,
        )

        assert "[PERCEPTION SNAPSHOT]" in prompt
        assert "[RECENT EVENTS]" in prompt
        assert "[TRUTH RULE]" in prompt
        assert "[USER]" in prompt
        # The control from perception must appear in the prompt
        assert "Save" in prompt

    def test_prompt_contains_truth_rule(self):
        from wyzer.core.orchestrator import _build_agent_prompt

        canonical = _make_canonical()
        prompt = _build_agent_prompt("click OK", canonical, [], 0)

        assert "ONLY reference UI facts" in prompt or "Only use facts" in prompt


# =========================================================================
# BONUS: truth_contract unit tests
# =========================================================================

class TestTruthContract:
    """Unit tests for the new normalize_perception_multi and helpers."""

    def test_normalize_perception_multi_basic(self):
        from wyzer.tools.desktop.truth_contract import normalize_perception_multi

        raw_uia = {
            "window": {"title": "Test Window", "exe": "test.exe"},
            "controls": [
                {"name": "OK", "control_type": "Button"},
                {"name": "Cancel", "control_type": "Button"},
            ],
            "errors": [],
        }
        raw_ocr = {
            "lines": [{"text": "Hello World"}],
            "errors": [],
        }
        active_win = {"title": "Test Window", "exe": "test.exe"}

        result = normalize_perception_multi(raw_uia, raw_ocr, active_win, [])

        assert result["foreground"]["app"] == "test.exe"
        assert result["foreground"]["title"] == "Test Window"
        assert len(result["controls"]) == 2
        assert result["controls"][0]["name"] == "OK"
        assert result["controls"][0]["type"] == "Button"
        assert "Hello World" in result["ocr_text"]
        assert isinstance(result["timestamp_ms"], int)

    def test_normalize_perception_multi_with_errors(self):
        from wyzer.tools.desktop.truth_contract import normalize_perception_multi

        result = normalize_perception_multi(
            {"errors": ["no_foreground_window"]},
            None,
            {"error": "no_foreground_window"},
            [],
        )

        assert "no_foreground_window" in result["errors"]

    def test_canonical_to_prompt_block(self):
        from wyzer.tools.desktop.truth_contract import canonical_to_prompt_block

        canonical = _make_canonical(
            controls=[{"name": "OK", "type": "Button", "automation_id": None}],
            ocr=["Some text"],
        )

        block = canonical_to_prompt_block(canonical)
        assert "[PERCEPTION SNAPSHOT]" in block
        assert "OK" in block
        assert "notepad" in block.lower()

    def test_has_fatal_perception_error(self):
        from wyzer.tools.desktop.truth_contract import has_fatal_perception_error

        assert has_fatal_perception_error({"errors": ["pywinauto_not_installed"]}) is True
        assert has_fatal_perception_error({"errors": ["no_top_windows"]}) is True
        assert has_fatal_perception_error({"errors": ["some_other_error"]}) is False
        assert has_fatal_perception_error({"errors": []}) is False
        assert has_fatal_perception_error({}) is False


# =========================================================================
# BONUS: ui_state_patterns tests
# =========================================================================

class TestUIStatePatterns:
    """Test the new needs_perception_first function."""

    def test_needs_perception_for_screen_queries(self):
        from wyzer.core.ui_state_patterns import needs_perception_first

        assert needs_perception_first("what's on screen") is True
        assert needs_perception_first("what's on my screen") is True
        assert needs_perception_first("what windows are open") is True

    def test_needs_perception_for_tab_ui_actions(self):
        from wyzer.core.ui_state_patterns import needs_perception_first

        assert needs_perception_first("open the settings tab") is True
        assert needs_perception_first("chat history tab") is True

    def test_needs_perception_for_status_queries(self):
        from wyzer.core.ui_state_patterns import needs_perception_first

        assert needs_perception_first("is it still downloading") is True
        assert needs_perception_first("did it finish") is True
        assert needs_perception_first("is it still there") is True

    def test_no_perception_for_deterministic_commands(self):
        """Commands that have deterministic tool handlers must NOT
        trigger perception-first — they bypass the agent loop via the
        orchestrator guard."""
        from wyzer.core.ui_state_patterns import needs_perception_first

        assert needs_perception_first("switch to Chrome") is False
        assert needs_perception_first("switch to VS Code") is False
        assert needs_perception_first("close it") is False
        assert needs_perception_first("click Install") is False
        assert needs_perception_first("go to youtube.com") is False
        assert needs_perception_first("select the first option") is False

    def test_no_perception_for_general_queries(self):
        from wyzer.core.ui_state_patterns import needs_perception_first

        assert needs_perception_first("tell me about Python") is False
        assert needs_perception_first("what time is it") is False
        assert needs_perception_first("who is the president") is False
