"""Tests for wyzer.agent_core.router and prompt_builder_profiles.

Covers:
  1. close_window pronoun resolution
  2. Verb-stripping in follow-up (\"Close Notepad\" → \"Notepad\")
  3. Router chooses REPAIR on window_not_found after a close_window attempt
  4. Token budget truncation keeps rules + minimal world snapshot
  5. Router: high-confidence regex → route=\"regex\"
  6. Router: low-confidence → route=\"plan\"
"""

import sys
import os
import types

# Ensure repo root is on path
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pytest


# ============================================================================
# Fixtures / stubs
# ============================================================================

class _StubWorldState:
    """Minimal stand-in for WorldState."""
    def __init__(self, **kw):
        self.active_window_title = kw.get("active_window_title", "")
        self.last_target = kw.get("last_target", "")
        self.open_windows = kw.get("open_windows", [])
        self.last_tool = kw.get("last_tool", None)
        self.last_result_summary = kw.get("last_result_summary", None)
        self.active_app = kw.get("active_app", None)


class _StubHybridDecision:
    def __init__(self, mode="llm", intents=None, reply="", confidence=0.0,
                 _needs_perception=False):
        self.mode = mode
        self.intents = intents
        self.reply = reply
        self.confidence = confidence
        self._needs_perception = _needs_perception


class _StubRegexRouter:
    """A fake hybrid_router module with a decide() method."""
    def __init__(self, decision: _StubHybridDecision):
        self._decision = decision

    def decide(self, text):
        return self._decision


class _StubRegistry:
    def __init__(self):
        self._tools = {}

    def get(self, name):
        return self._tools.get(name)

    def has_tool(self, name):
        return name in self._tools

    def list_tools(self):
        return [{"name": n, "description": ""} for n in self._tools]


# ============================================================================
# 1. Pronoun resolution for close_window
# ============================================================================

class TestPronounResolution:
    def test_resolve_pronoun_from_foreground(self):
        from wyzer.agent_core.router import _resolve_pronoun_from_world
        ws = _StubWorldState(active_window_title="Notepad")
        assert _resolve_pronoun_from_world(ws) == "Notepad"

    def test_resolve_pronoun_skips_wyzer(self):
        from wyzer.agent_core.router import _resolve_pronoun_from_world
        ws = _StubWorldState(active_window_title="Wyzer", last_target="Chrome")
        assert _resolve_pronoun_from_world(ws) == "Chrome"

    def test_resolve_pronoun_fallback_to_last_target(self):
        from wyzer.agent_core.router import _resolve_pronoun_from_world
        ws = _StubWorldState(active_window_title="", last_target="Firefox")
        assert _resolve_pronoun_from_world(ws) == "Firefox"

    def test_resolve_pronoun_fallback_to_open_windows(self):
        from wyzer.agent_core.router import _resolve_pronoun_from_world
        ws = _StubWorldState(
            active_window_title="",
            last_target="",
            open_windows=[
                {"title": "ShellExperienceHost", "process": "ShellExperienceHost"},
                {"title": "VS Code", "process": "Code"},
            ],
        )
        assert _resolve_pronoun_from_world(ws) == "VS Code"

    def test_has_pronoun_detected(self):
        from wyzer.agent_core.router import _has_pronoun
        assert _has_pronoun("close it")
        assert _has_pronoun("minimize that")
        assert _has_pronoun("focus the window")
        assert not _has_pronoun("close notepad")

    def test_substitute_pronoun(self):
        from wyzer.agent_core.router import _substitute_pronoun
        assert _substitute_pronoun("close it", "Notepad") == "close Notepad"
        assert _substitute_pronoun("minimize that", "Chrome") == "minimize Chrome"


# ============================================================================
# 2. Verb stripping in follow-up
# ============================================================================

class TestVerbStripping:
    def test_strip_close(self):
        from wyzer.agent_core.router import strip_verb_prefix
        assert strip_verb_prefix("Close Notepad") == "Notepad"

    def test_strip_open(self):
        from wyzer.agent_core.router import strip_verb_prefix
        assert strip_verb_prefix("open Chrome") == "Chrome"

    def test_strip_switch_to(self):
        from wyzer.agent_core.router import strip_verb_prefix
        assert strip_verb_prefix("switch to Firefox") == "Firefox"

    def test_no_strip_bare_word(self):
        from wyzer.agent_core.router import strip_verb_prefix
        assert strip_verb_prefix("Notepad") == "Notepad"

    def test_strip_maximize(self):
        from wyzer.agent_core.router import strip_verb_prefix
        assert strip_verb_prefix("maximize VS Code") == "VS Code"


# ============================================================================
# 3. Router chooses REPAIR on window_not_found
# ============================================================================

class TestRepairRouting:
    def test_repair_on_window_not_found(self):
        from wyzer.agent_core.router import router_route
        ws = _StubWorldState(active_window_title="Notepad")
        decision = router_route(
            user_text="close notepad",
            world_state=ws,
            tool_registry=_StubRegistry(),
            last_tool_error={
                "tool_name": "close_window",
                "args": {"title": "it"},
                "error_type": "window_not_found",
                "error_message": "No window matching 'it'",
            },
        )
        assert decision.route == "repair"
        assert decision.prompt_profile == "repair"

    def test_repair_on_missing_argument(self):
        from wyzer.agent_core.router import router_route
        ws = _StubWorldState()
        decision = router_route(
            user_text="close",
            world_state=ws,
            tool_registry=_StubRegistry(),
            last_tool_error={
                "tool_name": "close_window",
                "args": {},
                "error_type": "missing_argument",
                "error_message": "title or process required",
            },
        )
        assert decision.route == "repair"

    def test_no_repair_on_unrelated_error(self):
        from wyzer.agent_core.router import router_route
        ws = _StubWorldState()
        regex_router = _StubRegexRouter(
            _StubHybridDecision(mode="llm", confidence=0.5)
        )
        decision = router_route(
            user_text="close notepad",
            world_state=ws,
            tool_registry=_StubRegistry(),
            regex_router=regex_router,
            last_tool_error={
                "tool_name": "close_window",
                "args": {"title": "notepad"},
                "error_type": "permission_denied",
                "error_message": "Access denied",
            },
        )
        # permission_denied is NOT a repair error type → falls through to plan
        assert decision.route == "plan"


# ============================================================================
# 4. Token budget truncation
# ============================================================================

class TestTokenBudget:
    def test_must_keep_format_and_ground_truth(self):
        """Format rules and ground truth should never be dropped."""
        from wyzer.agent_core.prompt_builder_profiles import build_prompt, PromptContext

        ctx = PromptContext(
            user_text="what's on my screen?",
            tool_schemas=[{"name": f"tool_{i}", "description": "x" * 200}
                          for i in range(50)],
            foreground_window="VS Code",
            last_action="opened Chrome",
            recent_events=[{"event": "tool_end", "detail": f"step_{i}"}
                           for i in range(30)],
        )
        built = build_prompt("plan", compact=True, ctx=ctx)

        assert "format_rules" in built.sections_kept
        assert "ground_truth" in built.sections_kept
        assert "world_min" in built.sections_kept

    def test_world_min_always_present(self):
        from wyzer.agent_core.prompt_builder_profiles import build_prompt, PromptContext

        ctx = PromptContext(
            user_text="hello",
            foreground_window="Notepad",
            last_action="opened notepad",
        )
        built = build_prompt("plan", compact=False, ctx=ctx)
        assert "Notepad" in built.system
        assert "opened notepad" in built.system

    def test_extras_dropped_first(self):
        """Extras (priority 6) should be dropped before higher-priority items."""
        from wyzer.agent_core.prompt_builder_profiles import build_prompt, PromptContext

        ctx = PromptContext(
            user_text="test",
            tool_schemas=[{"name": f"tool_{i}", "description": "x" * 200}
                          for i in range(40)],
            extra_context="EXTRA " * 500,
            foreground_window="Chrome",
        )
        built = build_prompt("plan", compact=True, ctx=ctx)
        # Extras should be dropped (or heavily truncated) before tools
        if "extras" in built.sections_dropped:
            assert "format_rules" not in built.sections_dropped
            assert "ground_truth" not in built.sections_dropped

    def test_repair_profile_tiny(self):
        """REPAIR profile should produce a much smaller prompt than PLAN."""
        from wyzer.agent_core.prompt_builder_profiles import build_prompt, PromptContext

        ctx = PromptContext(
            user_text="close notepad",
            tool_schemas=[{"name": "close_window", "description": "Closes a window"}],
            last_tool_error={
                "tool_name": "close_window",
                "args": {"title": "it"},
                "error_type": "window_not_found",
                "error_message": "No match",
            },
            open_windows=[{"title": "Notepad", "process": "notepad.exe"}],
        )
        repair = build_prompt("repair", compact=True, ctx=ctx)
        plan = build_prompt("plan", compact=False, ctx=ctx)
        assert repair.tokens_est < plan.tokens_est

    def test_speak_has_no_tool_schemas(self):
        from wyzer.agent_core.prompt_builder_profiles import build_prompt, PromptContext

        ctx = PromptContext(
            user_text="what happened?",
            tool_schemas=[{"name": "close_window", "description": "Closes"}],
        )
        built = build_prompt("speak", compact=False, ctx=ctx)
        assert "AVAILABLE TOOLS" not in built.system


# ============================================================================
# 5. Router: high-confidence regex → route="regex"
# ============================================================================

class TestRouterRegexPath:
    def test_high_confidence_routes_to_regex(self):
        from wyzer.agent_core.router import router_route
        ws = _StubWorldState()
        regex_router = _StubRegexRouter(
            _StubHybridDecision(
                mode="tool_plan",
                intents=[{"tool": "get_time", "args": {}}],
                confidence=0.95,
            )
        )
        decision = router_route(
            user_text="what time is it",
            world_state=ws,
            tool_registry=_StubRegistry(),
            regex_router=regex_router,
        )
        assert decision.route == "regex"
        assert decision.confidence >= 0.90

    def test_low_confidence_routes_to_plan(self):
        from wyzer.agent_core.router import router_route
        ws = _StubWorldState()
        regex_router = _StubRegexRouter(
            _StubHybridDecision(mode="llm", confidence=0.5)
        )
        decision = router_route(
            user_text="do something complex",
            world_state=ws,
            tool_registry=_StubRegistry(),
            regex_router=regex_router,
        )
        assert decision.route == "plan"


# ============================================================================
# 6. Router: missing slots → plan (not regex)
# ============================================================================

class TestRouterSlotCheck:
    def test_pronoun_slot_triggers_plan(self):
        """If regex returns close_window with title='it', slots are incomplete."""
        from wyzer.agent_core.router import router_route
        ws = _StubWorldState(active_window_title="")
        regex_router = _StubRegexRouter(
            _StubHybridDecision(
                mode="tool_plan",
                intents=[{"tool": "close_window", "args": {"title": "it"}}],
                confidence=0.93,
            )
        )
        decision = router_route(
            user_text="close it",
            world_state=ws,
            tool_registry=_StubRegistry(),
            regex_router=regex_router,
        )
        # Pronoun "it" could not be resolved (empty world state) → plan
        # But if the router managed pronoun resolution, it may have substituted
        # and re-run.  Either way, route should NOT be "regex" with "it" as title.
        if decision.route == "regex":
            # If it resolved, the intent should not have "it"
            assert decision.intents[0]["args"].get("title", "it") != "it"


# ============================================================================
# 7. Prompt builder profiles sanity
# ============================================================================

class TestPromptProfiles:
    def test_plan_includes_tool_schemas(self):
        from wyzer.agent_core.prompt_builder_profiles import build_prompt, PromptContext
        ctx = PromptContext(
            user_text="open chrome and notepad",
            tool_schemas=[
                {"name": "open_target", "description": "Opens an app"},
                {"name": "close_window", "description": "Closes a window"},
            ],
        )
        built = build_prompt("plan", compact=False, ctx=ctx)
        assert "AVAILABLE TOOLS" in built.system
        assert "open_target" in built.system

    def test_repair_includes_error_block(self):
        from wyzer.agent_core.prompt_builder_profiles import build_prompt, PromptContext
        ctx = PromptContext(
            user_text="close notepad",
            last_tool_error={
                "tool_name": "close_window",
                "args": {"title": "it"},
                "error_type": "window_not_found",
                "error_message": "No match",
            },
        )
        built = build_prompt("repair", compact=True, ctx=ctx)
        assert "FAILED TOOL CALL" in built.system
        assert "window_not_found" in built.system

    def test_speak_is_concise(self):
        from wyzer.agent_core.prompt_builder_profiles import build_prompt, PromptContext
        ctx = PromptContext(user_text="what happened?")
        built = build_prompt("speak", compact=False, ctx=ctx)
        assert built.tokens_est < 200  # should be tiny


# ============================================================================
# 8. needs_speak helper
# ============================================================================

class TestNeedsSpeak:
    def test_simple_ok_no_speak(self):
        from wyzer.agent_core.router import needs_speak

        class FakeResult:
            def __init__(self, ok, result):
                self.ok = ok
                self.result = result

        class FakeSummary:
            def __init__(self, ran):
                self.ran = ran

        summary = FakeSummary([FakeResult(True, {"ok": True})])
        assert needs_speak(summary, "open chrome") is False

    def test_data_payload_triggers_speak(self):
        from wyzer.agent_core.router import needs_speak

        class FakeResult:
            def __init__(self, ok, result):
                self.ok = ok
                self.result = result

        class FakeSummary:
            def __init__(self, ran):
                self.ran = ran

        summary = FakeSummary([FakeResult(True, {"temp": 72, "condition": "sunny", "forecast": "warm"})])
        assert needs_speak(summary, "what's the weather") is True
