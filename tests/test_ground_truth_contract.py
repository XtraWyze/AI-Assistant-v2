"""
test_ground_truth_contract.py — Phase 15 Acceptance Tests

Validates the ground-truth-only enforcement:
1. The assistant never mentions a button/text unless it exists in tool output.
2. On UI questions, it runs a perception tool first (hybrid router).
3. When UIA fails, it falls back to OCR (if available) or reports inability.
4. "Did install succeed?" requires verified success condition.
5. Broad UI-state queries route to perception, not LLM reply-only.
6. Truth contract schema normalization works correctly.
7. Event log is populated and queryable.

These tests use monkeypatching (no live desktop / LLM needed).
"""

import sys
import os
import time
import json
import re

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# ---------------------------------------------------------------------------
# Fixtures: reset world state between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_world_state():
    """Clear WorldState singleton between tests."""
    from wyzer.context.world_state import get_world_state
    ws = get_world_state()
    ws.clear()
    yield
    ws.clear()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Truth-contract schema normalization
# ═══════════════════════════════════════════════════════════════════════════

class TestTruthContractSchema:
    """Verify normalize_perception produces canonical keys."""

    def test_uia_output_normalizes(self):
        """UIA dict normalizes to truth-contract shape."""
        from wyzer.tools.desktop.truth_contract import normalize_perception

        uia_raw = {
            "window": {"title": "Notepad", "hwnd": 12345, "rect": {"l": 0, "t": 0, "r": 800, "b": 600}},
            "controls": [
                {"name": "File", "control_type": "MenuItem", "rect": None, "enabled": True},
                {"name": "Save", "control_type": "Button", "rect": None, "enabled": True},
            ],
            "dialogs": [{"title": "Save As", "rect": None}],
            "progress": {"value": 50, "text": "Loading..."},
            "errors": [],
            "timestamp": 1700000000.0,
        }
        norm = normalize_perception(uia_raw)

        # All top-level keys must exist
        for key in ("window", "controls", "text_lines", "dialogs", "progress", "errors", "timestamp"):
            assert key in norm, f"Missing key: {key}"

        # Window normalized
        assert norm["window"]["app"] is None  # no exe in raw
        assert norm["window"]["title"] == "Notepad"
        assert norm["window"]["hwnd"] == 12345

        # Controls normalized (control_type → type)
        assert len(norm["controls"]) == 2
        assert norm["controls"][0]["type"] == "MenuItem"
        assert norm["controls"][1]["name"] == "Save"

        # Progress normalized (value → percent)
        assert norm["progress"]["percent"] == 50
        assert norm["progress"]["text"] == "Loading..."

        # text_lines empty for UIA (no OCR)
        assert norm["text_lines"] == []

    def test_ocr_output_normalizes(self):
        """OCR dict normalizes to truth-contract shape with text_lines."""
        from wyzer.tools.desktop.truth_contract import normalize_perception

        ocr_raw = {
            "lines": [{"text": "Hello World"}, {"text": "Press OK"}],
            "full_text": "Hello World\nPress OK",
            "source_image": "/tmp/shot.png",
        }
        norm = normalize_perception(ocr_raw)

        assert "text_lines" in norm
        assert norm["text_lines"] == ["Hello World", "Press OK"]
        assert norm["window"]["app"] is None
        assert norm["controls"] == []
        assert norm["errors"] == []

    def test_empty_input_returns_defaults(self):
        """Empty/None input returns valid default schema."""
        from wyzer.tools.desktop.truth_contract import normalize_perception

        norm = normalize_perception({})
        for key in ("window", "controls", "text_lines", "dialogs", "progress", "errors", "timestamp"):
            assert key in norm
        assert norm["controls"] == []
        assert norm["errors"] == []

        norm2 = normalize_perception(None)
        assert norm2["controls"] == []

    def test_perception_to_prompt_block(self):
        """perception_to_prompt_block renders a compact text block."""
        from wyzer.tools.desktop.truth_contract import normalize_perception, perception_to_prompt_block

        raw = {
            "window": {"exe": "chrome.exe", "title": "GitHub"},
            "controls": [{"name": "Star", "control_type": "Button", "rect": None, "enabled": True}],
            "dialogs": [],
            "progress": None,
            "errors": [],
            "timestamp": 1700000000.0,
        }
        norm = normalize_perception(raw)
        block = perception_to_prompt_block(norm)

        assert "[PERCEPTION SNAPSHOT]" in block
        assert "Chrome" in block or "chrome" in block.lower()
        assert "Star (Button)" in block
        assert "Dialogs: none" in block


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: UI questions route to perception tools (hybrid router)
# ═══════════════════════════════════════════════════════════════════════════

class TestHybridRouterPerceptionRouting:
    """Verify screen/UI queries route to perception tools, not LLM."""

    def _decide(self, text):
        from wyzer.core.hybrid_router import decide
        return decide(text)

    def test_whats_on_screen_routes_to_describe_screen(self):
        d = self._decide("what's on my screen?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_describe_screen_routes(self):
        d = self._decide("describe the screen")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_is_there_install_button(self):
        d = self._decide("is there a button that says Install?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "ui_find_text"
        assert d.intents[0]["args"]["text"].lower() == "install"

    def test_did_install_succeed(self):
        d = self._decide("did the install succeed?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "install_succeeded_check"

    def test_what_am_i_looking_at(self):
        d = self._decide("what am I looking at?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_is_it_still_downloading(self):
        """Broad UI-state query should route to perception."""
        d = self._decide("is it still downloading?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_what_does_dialog_say(self):
        """Dialog-content query should route to perception."""
        d = self._decide("what does this dialog say?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"

    def test_whats_the_download_progress(self):
        """Progress query should route to perception."""
        d = self._decide("what's the download progress?")
        assert d.mode == "tool_plan"
        assert d.intents[0]["tool"] == "describe_screen"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Reply-only path blocks screen-state hallucination
# ═══════════════════════════════════════════════════════════════════════════

class TestReplyOnlyScreenBlocking:
    """
    Verify that if a screen/UI query somehow reaches the reply-only
    code path, it is blocked rather than letting the LLM hallucinate.
    """

    def test_is_ui_state_query_detects_variants(self):
        """_is_ui_state_query correctly matches broad UI queries."""
        from wyzer.core.hybrid_router import _is_ui_state_query

        positive_cases = [
            "is it still downloading?",
            "what does this dialog say?",
            "what's the download progress?",
            "did the update finish?",
            "is the download done?",
            "read the error message",
            "what's happening on screen?",
            "can you check the screen?",
        ]
        for text in positive_cases:
            assert _is_ui_state_query(text), f"Should match: {text!r}"

        negative_cases = [
            "what's the weather?",
            "open chrome",
            "tell me about python",
            "what time is it?",
            "play some music",
        ]
        for text in negative_cases:
            assert not _is_ui_state_query(text), f"Should NOT match: {text!r}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Perception failure handling (UIA fail → OCR fallback or report)
# ═══════════════════════════════════════════════════════════════════════════

class TestPerceptionFailure:
    """Verify assertions.ui_find_text falls back to OCR when UIA fails."""

    def test_uia_fail_ocr_fallback(self, monkeypatch):
        """When UIA returns zero controls, 'auto' method tries OCR."""
        from wyzer.tools.desktop import assertions

        def _fake_uia_snapshot(max_nodes=80):
            return {
                "window": {"title": "Test"},
                "controls": [],
                "dialogs": [],
                "progress": None,
                "errors": ["foreground_not_in_uia_list"],
                "timestamp": time.time(),
            }

        def _fake_ocr():
            return {
                "lines": [{"text": "Install Complete"}],
                "full_text": "Install Complete",
            }

        monkeypatch.setattr(assertions, "_find_text_uia", lambda *a, **kw: {
            "found": False, "evidence": "UIA: nothing", "method": "uia", "matches": [],
            "summary": "No",
        })
        monkeypatch.setattr(assertions, "_find_text_ocr", lambda *a, **kw: {
            "found": True, "evidence": "OCR: found", "method": "ocr",
            "matches": ["Install Complete"], "summary": "Yes",
        })

        result = assertions.ui_find_text("Install Complete", method="auto")
        assert result["found"] is True
        assert result["method"] == "ocr"

    def test_uia_fail_no_ocr_reports_failure(self, monkeypatch):
        """When both UIA and OCR fail, report inability."""
        from wyzer.tools.desktop import assertions

        monkeypatch.setattr(assertions, "_find_text_uia", lambda *a, **kw: {
            "found": False, "evidence": "UIA: nothing", "method": "uia", "matches": [],
            "summary": "No",
        })
        monkeypatch.setattr(assertions, "_find_text_ocr", lambda *a, **kw: {
            "found": False, "evidence": "OCR: not available", "method": "ocr",
            "matches": [], "missing_dependency": True,
        })

        result = assertions.ui_find_text("Install Complete", method="auto")
        assert result["found"] is False


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Install-succeeded check requires verified evidence
# ═══════════════════════════════════════════════════════════════════════════

class TestInstallSucceededCheck:
    """install_succeeded_check must base its verdict on actual UIA data."""

    def test_success_only_with_evidence(self, monkeypatch):
        """Status is 'success' only when success indicators exist."""
        from wyzer.tools.desktop.assertions import install_succeeded_check
        from wyzer.tools.desktop import perceive_uia

        monkeypatch.setattr(perceive_uia, "perceive_uia_focused_window", lambda **kw: {
            "window": {"title": "Steam"},
            "controls": [
                {"name": "Play", "control_type": "Button", "rect": None, "enabled": True},
                {"name": "Installed", "control_type": "Text", "rect": None, "enabled": True},
            ],
            "dialogs": [],
            "progress": None,
            "errors": [],
            "timestamp": time.time(),
        })
        result = install_succeeded_check()
        assert result["status"] == "success"
        assert result["evidence"]  # must have evidence string

    def test_unknown_without_evidence(self, monkeypatch):
        """Status is 'unknown' when no indicators found."""
        from wyzer.tools.desktop.assertions import install_succeeded_check
        from wyzer.tools.desktop import perceive_uia

        monkeypatch.setattr(perceive_uia, "perceive_uia_focused_window", lambda **kw: {
            "window": {"title": "Something"},
            "controls": [
                {"name": "Help", "control_type": "Button", "rect": None, "enabled": True},
            ],
            "dialogs": [],
            "progress": None,
            "errors": [],
            "timestamp": time.time(),
        })
        result = install_succeeded_check()
        assert result["status"] == "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Event log is populated
# ═══════════════════════════════════════════════════════════════════════════

class TestEventLog:
    """Verify event_log ring buffer works correctly."""

    def test_emit_event_adds_to_log(self):
        from wyzer.context.world_state import emit_event, get_event_log

        emit_event("test_event", {"key": "value"})
        events = get_event_log(limit=10)
        assert len(events) >= 1
        assert events[-1]["event"] == "test_event"
        assert events[-1]["key"] == "value"
        assert "ts" in events[-1]

    def test_event_log_ring_buffer(self):
        from wyzer.context.world_state import emit_event, get_event_log, get_world_state

        ws = get_world_state()
        # Fill beyond maxlen
        for i in range(210):
            emit_event("fill", {"i": i})
        events = get_event_log(limit=300)
        assert len(events) <= 200  # maxlen enforced

    def test_get_recent_events_tool(self):
        from wyzer.context.world_state import emit_event
        from wyzer.tools.desktop.get_recent_events import GetRecentEventsTool

        emit_event("tool_start", {"tool": "test_tool"})
        emit_event("tool_end", {"tool": "test_tool", "success": True})

        tool = GetRecentEventsTool()
        result = tool.run(limit=5)
        assert "events" in result
        assert result["count"] >= 2


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Prompt builder injects perception + events
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptBuilderPerceptionInjection:
    """Verify PromptBuilder injects perception and events into prompt."""

    def test_perception_injected_into_normal_prompt(self):
        """When last_perception is set, it appears in the prompt."""
        from wyzer.context.world_state import update_last_perception, emit_event
        from wyzer.tools.desktop.truth_contract import normalize_perception

        # Set up perception
        raw = {
            "window": {"exe": "notepad.exe", "title": "test.txt"},
            "controls": [{"name": "Edit", "control_type": "Edit"}],
            "dialogs": [],
            "progress": None,
            "errors": [],
            "timestamp": time.time(),
        }
        update_last_perception(normalize_perception(raw))
        emit_event("perception", {"source": "uia"})

        from wyzer.brain.prompt_builder import PromptBuilder
        builder = PromptBuilder(user_text="what's on screen?")
        prompt, mode = builder.build()

        assert "[PERCEPTION SNAPSHOT]" in prompt
        assert "notepad" in prompt.lower() or "Notepad" in prompt

    def test_anti_hallucination_rules_in_prompt(self):
        """The ground-truth UI rule appears in the prompt."""
        from wyzer.brain.prompt_builder import PromptBuilder
        builder = PromptBuilder(user_text="hello")
        prompt, mode = builder.build()

        # Check that some form of the ground-truth UI rule is present
        # (either the full version in normal mode or compact version)
        assert "GROUND-TRUTH" in prompt or "ground-truth" in prompt.lower() or "NEVER claim anything about the screen" in prompt
        # Verify anti-hallucination is present in some form
        assert "ANTI-HALLUCINATION" in prompt or "NEVER guess" in prompt or "never invent" in prompt.lower()
        # In compact mode the full "NON-NEGOTIABLE" heading is trimmed; accept either
        assert "NON-NEGOTIABLE" in prompt or "NEVER claim anything about the screen" in prompt or "Never invent" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# Run with: pytest tests/test_ground_truth_contract.py -v
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
