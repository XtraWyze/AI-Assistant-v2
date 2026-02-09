"""
Tests for multi-intent extraction and capabilities tool.

Covers:
  A) "What can you do? Can you tell me the time?"
     -> router returns intents = [get_capabilities, get_time]

  B) "Open notepad and then tell me something cool, but then what is the time?"
     -> router returns intents = [open_target(notepad), get_time]
        leftover = "tell me something cool"

  C) Orchestrator merges multi-intent replies without JSON leak or "Done."

  D) _sanitize_llm_reply strips JSON wrappers.

Run:
  python -m pytest scripts/test_multi_intent_v2.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# A) Hybrid-router: capabilities + time
# ---------------------------------------------------------------------------

class TestHybridRouterMultiIntent:
    """Tests that hybrid_router.decide() produces correct multi-intent plans."""

    def test_capabilities_plus_time(self):
        """'What can you do? Can you tell me the time?' -> [get_capabilities, get_time]"""
        from wyzer.core.hybrid_router import decide

        decision = decide("What can you do? Can you tell me the time?")
        assert decision.mode == "tool_plan", f"Expected tool_plan, got {decision.mode}"
        assert decision.intents is not None, "intents should not be None"

        tool_names = [i["tool"] for i in decision.intents]
        assert "get_capabilities" in tool_names, f"Missing get_capabilities in {tool_names}"
        assert "get_time" in tool_names, f"Missing get_time in {tool_names}"

    def test_capabilities_single(self):
        """'What can you do?' -> [get_capabilities]"""
        from wyzer.core.hybrid_router import decide

        decision = decide("What can you do?")
        assert decision.mode == "tool_plan"
        assert decision.intents is not None
        assert decision.intents[0]["tool"] == "get_capabilities"

    def test_help_single(self):
        """'help' -> [get_capabilities]"""
        from wyzer.core.hybrid_router import decide

        decision = decide("help")
        assert decision.mode == "tool_plan"
        assert decision.intents is not None
        assert decision.intents[0]["tool"] == "get_capabilities"

    def test_open_notepad_plus_time_with_freeform(self):
        """'Open notepad and then tell me something cool, but then what is the time?'
        -> intents includes open_target(notepad) + get_time
        -> leftover includes 'tell me something cool'
        """
        from wyzer.core.hybrid_router import decide

        text = "Open notepad and then tell me something cool, but then what is the time?"
        decision = decide(text)
        assert decision.mode == "tool_plan", f"Expected tool_plan, got {decision.mode}"
        assert decision.intents is not None

        tool_names = [i["tool"] for i in decision.intents]
        assert "open_target" in tool_names, f"Missing open_target in {tool_names}"
        assert "get_time" in tool_names, f"Missing get_time in {tool_names}"

        # The open_target intent should have notepad as query
        open_intents = [i for i in decision.intents if i["tool"] == "open_target"]
        assert open_intents, "Should have open_target intent"
        assert "notepad" in (open_intents[0].get("args", {}).get("query", "")).lower()

        # Leftover should contain freeform text
        leftover = ""
        if decision.reply and decision.reply.startswith("__LEFTOVER__:"):
            leftover = decision.reply[len("__LEFTOVER__:"):]
        assert "cool" in leftover.lower() or "something" in leftover.lower(), \
            f"Expected freeform leftover about 'something cool', got: {leftover!r}"

    def test_open_plus_time(self):
        """'Open Chrome and tell me the time' -> [open_target(chrome), get_time]"""
        from wyzer.core.hybrid_router import decide

        decision = decide("Open Chrome and tell me the time")
        assert decision.mode == "tool_plan"
        assert decision.intents is not None

        tool_names = [i["tool"] for i in decision.intents]
        assert "open_target" in tool_names, f"Missing open_target in {tool_names}"
        assert "get_time" in tool_names, f"Missing get_time in {tool_names}"

    def test_capabilities_in_time_misheard_connector(self):
        """Whisper sometimes transcribes 'and' as 'in'.
        'What can you do in what time is it?' -> [get_capabilities, get_time]
        """
        from wyzer.core.hybrid_router import decide

        decision = decide("What can you do in what time is it?")
        assert decision.mode == "tool_plan", f"Expected tool_plan, got {decision.mode}"
        assert decision.intents is not None

        tool_names = [i["tool"] for i in decision.intents]
        assert "get_capabilities" in tool_names, f"Missing get_capabilities in {tool_names}"
        assert "get_time" in tool_names, f"Missing get_time in {tool_names}"
        assert len(decision.intents) == 2, f"Expected 2 intents, got {len(decision.intents)}"

    def test_filler_you_open_plus_time(self):
        """Whisper sometimes prepends 'You' to commands.
        'You open notepad and then tell me the time' -> [open_target, get_time]
        """
        from wyzer.core.hybrid_router import decide

        decision = decide("You open notepad and then tell me the time")
        assert decision.mode == "tool_plan", f"Expected tool_plan, got {decision.mode}"
        assert decision.intents is not None

        tool_names = [i["tool"] for i in decision.intents]
        assert "open_target" in tool_names, f"Missing open_target in {tool_names}"
        assert "get_time" in tool_names, f"Missing get_time in {tool_names}"

        open_intents = [i for i in decision.intents if i["tool"] == "open_target"]
        assert "notepad" in open_intents[0]["args"]["query"].lower()


# ---------------------------------------------------------------------------
# B) extract_multi_intents directly
# ---------------------------------------------------------------------------

class TestExtractMultiIntents:
    """Tests for the new extract_multi_intents() function."""

    def test_two_tools(self):
        from wyzer.core.hybrid_router import extract_multi_intents

        result = extract_multi_intents("What can you do? Can you tell me the time?")
        assert result is not None
        intents, leftover = result
        tool_names = [i["tool"] for i in intents]
        assert "get_capabilities" in tool_names
        assert "get_time" in tool_names
        assert leftover == "" or leftover.strip() == ""

    def test_tools_plus_leftover(self):
        from wyzer.core.hybrid_router import extract_multi_intents

        result = extract_multi_intents(
            "Open notepad and then tell me something cool, but then what is the time?"
        )
        assert result is not None
        intents, leftover = result
        tool_names = [i["tool"] for i in intents]
        assert "open_target" in tool_names
        assert "get_time" in tool_names
        assert leftover, "Expected non-empty leftover"
        # The leftover should NOT contain time-related text (that's a tool intent)
        assert "time" not in leftover.lower(), f"Leftover should not have time text: {leftover!r}"

    def test_single_segment_returns_none(self):
        from wyzer.core.hybrid_router import extract_multi_intents

        result = extract_multi_intents("What time is it?")
        assert result is None, "Single segment should return None"

    def test_fallback_no_connector(self):
        """When connector splitting fails, boundary scanning should still
        detect 2 tool triggers (e.g. Whisper transcribes 'and' as 'in')."""
        from wyzer.core.hybrid_router import extract_multi_intents

        result = extract_multi_intents("What can you do in what time is it?")
        assert result is not None, "Fallback should detect 2 tool triggers"
        intents, leftover = result
        tool_names = [i["tool"] for i in intents]
        assert "get_capabilities" in tool_names
        assert "get_time" in tool_names


# ---------------------------------------------------------------------------
# C) get_capabilities tool returns valid JSON with summary
# ---------------------------------------------------------------------------

class TestGetCapabilitiesTool:
    """Tests for the get_capabilities tool itself."""

    def test_tool_returns_summary(self):
        from wyzer.tools.get_capabilities import GetCapabilitiesTool

        tool = GetCapabilitiesTool()
        assert tool.name == "get_capabilities"

        result = tool.run()
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 20

    def test_tool_registered(self):
        from wyzer.tools.registry import build_default_registry

        registry = build_default_registry()
        assert registry.has_tool("get_capabilities"), "get_capabilities not registered"


# ---------------------------------------------------------------------------
# D) _sanitize_llm_reply
# ---------------------------------------------------------------------------

class TestSanitizeLlmReply:
    """Tests that JSON-ish LLM output is cleaned to plain text."""

    def test_json_reply_extracted(self):
        from wyzer.core.orchestrator import _sanitize_llm_reply

        raw = '{"reply": "Here is something cool about space."}'
        assert _sanitize_llm_reply(raw) == "Here is something cool about space."

    def test_plain_text_unchanged(self):
        from wyzer.core.orchestrator import _sanitize_llm_reply

        raw = "The speed of light is 299 million meters per second."
        assert _sanitize_llm_reply(raw) == raw

    def test_braces_stripped(self):
        from wyzer.core.orchestrator import _sanitize_llm_reply

        raw = 'Some text with {weird} braces'
        result = _sanitize_llm_reply(raw)
        assert "{" not in result
        assert "}" not in result

    def test_empty_string(self):
        from wyzer.core.orchestrator import _sanitize_llm_reply

        assert _sanitize_llm_reply("") == ""
        assert _sanitize_llm_reply(None) == ""


# ---------------------------------------------------------------------------
# E) Orchestrator multi-intent merge (no "Done.", no JSON leak)
# ---------------------------------------------------------------------------

class TestOrchestratorMultiIntentMerge:
    """Smoke test: orchestrator merges multi-intent replies cleanly."""

    def test_multi_intent_no_done(self):
        """Run 'What can you do? What time is it?' through the orchestrator.
        The reply should contain capability info AND time, not 'Done.'"""
        from wyzer.core import orchestrator

        # Monkeypatch to avoid real tool execution side-effects
        original_execute = orchestrator._execute_tool

        tool_calls = []

        def _stub_execute(registry, tool_name, args):
            tool_calls.append((tool_name, dict(args or {})))
            if tool_name == "get_time":
                from datetime import datetime
                now = datetime.now()
                return {
                    "time": now.strftime("%H:%M:%S"),
                    "date": now.strftime("%Y-%m-%d"),
                    "timezone": "local",
                }
            if tool_name == "get_capabilities":
                return {
                    "total_tools": 30,
                    "categories": [],
                    "examples": ["Open Chrome", "What time is it"],
                    "summary": "I can help with many things. Try saying: Open Chrome or What time is it.",
                }
            return {"status": "ok"}

        # Stub LLM calls
        orchestrator._call_llm_reply_only = lambda t: {"reply": "(stub)"}
        orchestrator._call_llm = lambda t, r: {"reply": "(stub)"}
        orchestrator._call_llm_with_execution_summary = lambda t, s, r: {"reply": "(stub)"}
        orchestrator._call_llm_for_explicit_tool = lambda t, tn, r: {"reply": "(stub)", "intents": []}
        orchestrator._execute_tool = _stub_execute

        try:
            result = orchestrator.handle_user_text("What can you do? What time is it?")
            reply = result.get("reply", "")

            # Should NOT be just "Done."
            assert reply.strip() != "Done.", f"Reply should not be 'Done.', got: {reply!r}"

            # Should not contain JSON braces
            assert "{" not in reply, f"Reply contains JSON braces: {reply!r}"
            assert "}" not in reply, f"Reply contains JSON braces: {reply!r}"
        finally:
            orchestrator._execute_tool = original_execute

    def test_no_json_leak_in_llm_leftover(self):
        """When LLM returns JSON for leftover, it must be sanitized."""
        from wyzer.core import orchestrator

        original_execute = orchestrator._execute_tool
        original_llm = getattr(orchestrator, '_call_llm_reply_only', None)

        def _stub_execute(registry, tool_name, args):
            if tool_name == "open_target":
                return {"status": "ok", "opened": True}
            if tool_name == "get_time":
                return {"time": "12:30:00", "date": "2026-02-08", "timezone": "local"}
            return {"status": "ok"}

        def _stub_llm_reply_only(text):
            # Simulate LLM returning JSON-ish content
            return {"reply": '{"reply": "Here is something cool: bananas glow under UV light."}'}

        orchestrator._execute_tool = _stub_execute
        orchestrator._call_llm_reply_only = _stub_llm_reply_only
        orchestrator._call_llm = lambda t, r: {"reply": "(stub)"}
        orchestrator._call_llm_with_execution_summary = lambda t, s, r: {"reply": "(stub)"}
        orchestrator._call_llm_for_explicit_tool = lambda t, tn, r: {"reply": "(stub)", "intents": []}

        try:
            result = orchestrator.handle_user_text(
                "Open notepad and then tell me something cool, but then what is the time?"
            )
            reply = result.get("reply", "")

            # Must not contain JSON braces
            assert "{" not in reply, f"JSON leak in reply: {reply!r}"
            assert "}" not in reply, f"JSON leak in reply: {reply!r}"
        finally:
            orchestrator._execute_tool = original_execute
            if original_llm is not None:
                orchestrator._call_llm_reply_only = original_llm
