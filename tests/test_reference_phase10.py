"""Tests for Phase 10 - Continuity & Reference Resolver (Enhanced).

Tests the enhanced reference resolution features including:
- "close it" after "open Chrome"
- "move it to monitor 2"
- "the other one" toggle
- "do that again" repeats last intents
- Multi-intent with pronoun resolution
- Ambiguous resolution requires clarification

Run with: python -m pytest tests/test_reference_phase10.py -v
"""

import pytest
import time
from typing import Dict, Any, List, Optional

# Import the modules under test
from wyzer.context.world_state import (
    WorldState,
    get_world_state,
    update_world_state,
    update_from_tool_execution,
    clear_world_state,
    update_last_intents,
    update_last_active_window,
    set_last_llm_reply_only,
    get_last_intents,
    get_last_active_window,
    get_last_targets,
    update_after_tool,
    TargetRecord,
    LastAction,
)
from wyzer.core.reference_resolver import (
    resolve_references,
    resolve_pronoun_target,
    resolve_other_one,
    resolve_repeat_last,
    resolve_move_it_to_monitor,
    resolve_intent_args,
    is_replay_request,
    is_other_one_request,
    has_unresolved_pronoun,
    REPLAY_LAST_ACTION_SENTINEL,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def clean_world_state():
    """Ensure clean state before and after each test."""
    clear_world_state()
    yield
    clear_world_state()


@pytest.fixture
def chrome_opened_state():
    """State after user opened Chrome (Chrome is now focused)."""
    update_from_tool_execution(
        tool_name="open_target",
        tool_args={"query": "Chrome"},
        tool_result={
            "success": True,
            "resolved": {
                "type": "app",
                "matched_name": "Chrome",
                "path": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            },
        },
    )
    update_last_intents([{"tool": "open_target", "args": {"query": "Chrome"}}])
    # After opening Chrome, it should be the focused window
    update_last_active_window(app_name="Chrome", window_title="New Tab - Google Chrome")
    return get_world_state()


@pytest.fixture
def chrome_and_notepad_state():
    """State after opening Chrome, then Notepad (Notepad is now focused)."""
    # First, open Chrome
    update_from_tool_execution(
        tool_name="open_target",
        tool_args={"query": "Chrome"},
        tool_result={
            "success": True,
            "resolved": {
                "type": "app",
                "matched_name": "Chrome",
            },
        },
    )
    # Then, open Notepad
    update_from_tool_execution(
        tool_name="open_target",
        tool_args={"query": "Notepad"},
        tool_result={
            "success": True,
            "resolved": {
                "type": "app",
                "matched_name": "Notepad",
            },
        },
    )
    update_last_intents([{"tool": "open_target", "args": {"query": "Notepad"}}])
    # Notepad is the most recently opened, so it should be focused
    update_last_active_window(app_name="Notepad", window_title="Untitled - Notepad")
    return get_world_state()


@pytest.fixture
def focused_window_state():
    """State after focusing a window (Discord is now focused)."""
    update_from_tool_execution(
        tool_name="focus_window",
        tool_args={"query": "Discord"},
        tool_result={
            "success": True,
            "matched": {
                "title": "Discord",
                "process": "Discord.exe",
                "hwnd": 12345,
            },
        },
    )
    update_last_intents([{"tool": "focus_window", "args": {"query": "Discord"}}])
    # Discord is now the focused window
    update_last_active_window(app_name="Discord", window_title="Discord", pid=5678)
    return get_world_state()


# ============================================================================
# TEST: "close it" after "open Chrome"
# ============================================================================

class TestCloseItAfterOpenChrome:
    """Exit criterion 1: "Open Chrome" → "close it" closes Chrome."""
    
    def test_close_it_resolves_to_chrome(self, chrome_opened_state):
        """'close it' resolves to 'close Chrome' after opening Chrome."""
        result = resolve_references("close it")
        assert result == "close Chrome"
    
    def test_close_that_resolves_to_chrome(self, chrome_opened_state):
        """'close that' resolves to 'close Chrome' after opening Chrome."""
        result = resolve_references("close that")
        assert result == "close Chrome"
    
    def test_shut_it_resolves_to_chrome(self, chrome_opened_state):
        """'shut it' resolves to 'shut Chrome' after opening Chrome."""
        result = resolve_references("shut it")
        assert result == "shut Chrome"
    
    def test_close_it_no_context_unchanged(self):
        """'close it' unchanged when no context exists (mocked to prevent real window detection)."""
        from unittest.mock import patch
        # Mock get_foreground_window at its source module to return no window
        with patch('wyzer.vision.window_context.get_foreground_window', return_value={"app": None, "title": None, "pid": None}):
            result = resolve_references("close it")
            # Returns unchanged since there's no target to resolve to
            assert result == "close it"
    
    def test_pronoun_resolution_returns_chrome(self, chrome_opened_state):
        """resolve_pronoun_target returns Chrome for 'close it'."""
        target, reason = resolve_pronoun_target("close it", chrome_opened_state)
        assert target == "Chrome"
        assert reason is not None


# ============================================================================
# TEST: Deictic Resolution - Foreground Window Priority
# ============================================================================

class TestDeicticForegroundPriority:
    """
    Exit criterion: "close it/this/that" should close the CURRENT FOREGROUND window,
    not the previously opened/last_target app.
    
    Priority order: active_app > last_target > last_tool > ask
    """
    
    def test_foreground_notepad_overrides_last_target_chrome(self):
        """
        If Chrome was last opened (last_target=Chrome) but Notepad is now focused,
        'close it' should resolve to Notepad, not Chrome.
        """
        # Setup: User opened Chrome, then manually switched to Notepad (focused)
        update_from_tool_execution(
            tool_name="open_target",
            tool_args={"query": "Chrome"},
            tool_result={
                "success": True,
                "resolved": {"type": "app", "matched_name": "Chrome"},
            },
        )
        # Simulate Notepad being the current foreground window (via Window Watcher)
        update_last_active_window(
            app_name="Notepad",
            window_title="Untitled - Notepad",
            pid=9999,
        )
        
        # Now "close it" should resolve to Notepad (foreground), NOT Chrome (last_target)
        result = resolve_references("close it")
        assert result == "close Notepad", f"Expected 'close Notepad' but got '{result}'"
    
    def test_minimize_it_uses_foreground_over_last_target(self):
        """'minimize it' should minimize the foreground window, not last_target."""
        # Setup: Opened Discord, but VS Code is now focused
        update_from_tool_execution(
            tool_name="open_target",
            tool_args={"query": "Discord"},
            tool_result={"success": True, "resolved": {"type": "app", "matched_name": "Discord"}},
        )
        update_last_active_window(app_name="VS Code", window_title="main.py - VS Code")
        
        result = resolve_references("minimize it")
        assert "VS Code" in result, f"Expected VS Code but got '{result}'"
    
    def test_maximize_that_uses_foreground(self):
        """'maximize that' should maximize the foreground window."""
        update_last_active_window(app_name="Explorer", window_title="Documents")
        
        result = resolve_references("maximize that")
        assert "Explorer" in result, f"Expected Explorer but got '{result}'"
    
    def test_focus_it_uses_foreground(self):
        """'focus on it' should target the foreground window."""
        update_last_active_window(app_name="Firefox", window_title="Mozilla Firefox")
        
        result = resolve_references("focus on it")
        assert "Firefox" in result, f"Expected Firefox but got '{result}'"
    
    def test_fallback_to_last_target_when_no_foreground(self, chrome_opened_state):
        """
        When no foreground window info is available,
        fall back to last_target from recent tool execution.
        """
        from unittest.mock import patch
        
        # Clear foreground window info
        ws = get_world_state()
        ws.active_app = None
        ws.last_active_window = None
        ws.focused_window = None
        
        # Mock get_foreground_window to return no window (prevent real window detection)
        with patch('wyzer.vision.window_context.get_foreground_window', return_value={"app": None, "title": None, "pid": None}):
            # Should fall back to Chrome (from chrome_opened_state fixture's last_target)
            result = resolve_references("close it")
            assert result == "close Chrome", f"Expected 'close Chrome' (fallback) but got '{result}'"
    
    def test_fallback_returns_unchanged_when_no_context(self):
        """
        When neither foreground nor last_target is available,
        'close it' should remain unchanged (to trigger clarification).
        """
        from unittest.mock import patch
        
        # Mock get_foreground_window to return no window (prevent real window detection)
        with patch('wyzer.vision.window_context.get_foreground_window', return_value={"app": None, "title": None, "pid": None}):
            # Fresh state with no context
            result = resolve_references("close it")
            assert result == "close it", f"Expected unchanged 'close it' but got '{result}'"


class TestDeicticResolutionAPI:
    """Test the resolve_deictic_window_target API function."""
    
    def test_resolve_returns_foreground_target(self):
        """resolve_deictic_window_target returns foreground window first."""
        from wyzer.core.reference_resolver import resolve_deictic_window_target
        
        update_last_active_window(app_name="Notepad", window_title="Untitled - Notepad")
        
        result = resolve_deictic_window_target("close")
        assert result["resolved"] is True
        assert result["process"] == "Notepad"
        assert result["source"] == "active_app"
    
    def test_resolve_returns_last_target_fallback(self, chrome_opened_state):
        """resolve_deictic_window_target falls back to last_target."""
        from unittest.mock import patch
        from wyzer.core.reference_resolver import resolve_deictic_window_target
        
        # Clear foreground context
        ws = get_world_state()
        ws.active_app = None
        ws.last_active_window = None
        ws.focused_window = None
        
        # Mock get_foreground_window to prevent real window detection
        with patch('wyzer.vision.window_context.get_foreground_window', return_value={"app": None, "title": None, "pid": None}):
            result = resolve_deictic_window_target("close")
            assert result["resolved"] is True
            assert result["process"] == "Chrome"
            assert result["source"] == "last_target"
    
    def test_resolve_returns_clarification_when_unresolved(self):
        """resolve_deictic_window_target returns clarification when no context."""
        from unittest.mock import patch
        from wyzer.core.reference_resolver import resolve_deictic_window_target
        
        # Mock get_foreground_window to prevent real window detection
        with patch('wyzer.vision.window_context.get_foreground_window', return_value={"app": None, "title": None, "pid": None}):
            result = resolve_deictic_window_target("close")
            assert result["resolved"] is False
            assert "clarification" in result
            assert "close" in result["clarification"]


class TestMultiIntentDeicticResolution:
    """Test deictic resolution in multi-intent scenarios."""
    
    def test_open_chrome_then_close_it_when_chrome_focused(self):
        """
        'open chrome then close it' - if Chrome becomes focused after opening,
        'close it' should close Chrome.
        """
        # Simulate: "open chrome" executed, Chrome is now focused
        update_from_tool_execution(
            tool_name="open_target",
            tool_args={"query": "Chrome"},
            tool_result={"success": True, "resolved": {"type": "app", "matched_name": "Chrome"}},
        )
        # Chrome is now the active window
        update_last_active_window(app_name="Chrome", window_title="New Tab - Google Chrome")
        
        result = resolve_references("close it")
        assert result == "close Chrome"
    
    def test_open_chrome_then_close_it_when_different_window_focused(self):
        """
        'open chrome then close it' - if user switched to Notepad after Chrome opened,
        'close it' should close Notepad (current foreground), not Chrome.
        """
        # Simulate: "open chrome" executed
        update_from_tool_execution(
            tool_name="open_target",
            tool_args={"query": "Chrome"},
            tool_result={"success": True, "resolved": {"type": "app", "matched_name": "Chrome"}},
        )
        # But user clicked on Notepad, making it the active window
        update_last_active_window(app_name="Notepad", window_title="Untitled - Notepad")
        
        result = resolve_references("close it")
        # Should close Notepad (foreground), not Chrome (last_target)
        assert result == "close Notepad", f"Expected 'close Notepad' but got '{result}'"


# ============================================================================
# TEST: "move it to monitor 2"
# ============================================================================

class TestMoveItToMonitor:
    """Exit criterion 2: 'Move it to monitor 2' targets last window."""
    
    def test_move_it_resolves_to_window(self, focused_window_state):
        """'move it to monitor 2' resolves to Discord."""
        args, reason = resolve_move_it_to_monitor("move it to monitor 2", focused_window_state)
        assert args is not None
        assert args.get("process") == "Discord"
        assert args.get("monitor") == 2
    
    def test_move_it_monitor_1(self, focused_window_state):
        """'move it to monitor 1' resolves with monitor 1."""
        args, reason = resolve_move_it_to_monitor("move it to monitor 1", focused_window_state)
        assert args is not None
        assert args.get("monitor") == 1
    
    def test_move_it_monitor_word_form(self, focused_window_state):
        """'move it to monitor two' (word form) resolves to monitor 2."""
        args, reason = resolve_move_it_to_monitor("move it to monitor two", focused_window_state)
        assert args is not None
        assert args.get("monitor") == 2, f"Expected monitor 2 but got {args.get('monitor')}"
        
        # Also test "one" and "three"
        args1, _ = resolve_move_it_to_monitor("move it to monitor one", focused_window_state)
        assert args1.get("monitor") == 1
        
        args3, _ = resolve_move_it_to_monitor("move it to monitor three", focused_window_state)
        assert args3.get("monitor") == 3
    
    def test_move_it_ordinal_form(self, focused_window_state):
        """'move it to the second monitor' (ordinal) resolves correctly."""
        # "second monitor" pattern
        args, reason = resolve_move_it_to_monitor("move it to the second monitor", focused_window_state)
        assert args is not None
        assert args.get("monitor") == 2, f"Expected monitor 2 but got {args.get('monitor')}"
        
        # "first monitor" 
        args1, _ = resolve_move_it_to_monitor("move it to the first monitor", focused_window_state)
        assert args1 is not None
        assert args1.get("monitor") == 1
        
        # "2nd monitor"
        args2nd, _ = resolve_move_it_to_monitor("move it to 2nd monitor", focused_window_state)
        assert args2nd is not None
        assert args2nd.get("monitor") == 2
    
    def test_move_it_positional(self, focused_window_state):
        """'move it to the left monitor' uses positional."""
        args, reason = resolve_move_it_to_monitor("move it to the left monitor", focused_window_state)
        assert args is not None
        assert args.get("monitor") == "left"
        
        args_right, _ = resolve_move_it_to_monitor("move it to the right monitor", focused_window_state)
        assert args_right is not None
        assert args_right.get("monitor") == "right"
    
    def test_move_that_to_monitor(self, focused_window_state):
        """'move that to monitor 2' pattern detected."""
        args, reason = resolve_move_it_to_monitor("move that to monitor 2", focused_window_state)
        assert args is not None
        assert args.get("monitor") == 2
    
    def test_move_it_no_window_asks_clarification(self):
        """Returns clarification when no window context."""
        ws = get_world_state()
        args, reason = resolve_move_it_to_monitor("move it to monitor 2", ws)
        # Should ask which window since there's no context
        assert args is None or reason is not None


# ============================================================================
# TEST: "the other one" toggle
# ============================================================================

class TestOtherOneToggle:
    """Exit criterion 5: 'the other one' toggles between last two targets."""
    
    def test_other_one_returns_previous_target(self, chrome_and_notepad_state):
        """'the other one' returns Chrome when Notepad was most recent."""
        target, reason = resolve_other_one("the other one", chrome_and_notepad_state)
        assert target == "Chrome"  # Toggle to the "other" one
    
    def test_other_one_with_switch_prefix(self, chrome_and_notepad_state):
        """'switch to the other one' is recognized."""
        assert is_other_one_request("switch to the other one")
        target, reason = resolve_other_one("switch to the other one", chrome_and_notepad_state)
        assert target == "Chrome"
    
    def test_other_one_only_one_target_asks_clarification(self, chrome_opened_state):
        """Asks clarification when only one target exists."""
        target, clarification = resolve_other_one("the other one", chrome_opened_state)
        assert target is None
        assert clarification is not None
        assert "Chrome" in clarification or "only" in clarification.lower()
    
    def test_other_one_no_targets_asks_clarification(self):
        """Asks clarification when no targets exist."""
        ws = get_world_state()
        target, clarification = resolve_other_one("the other one", ws)
        assert target is None
        assert clarification is not None
    
    def test_is_other_one_patterns(self):
        """Various 'other one' patterns are detected."""
        assert is_other_one_request("the other one")
        assert is_other_one_request("switch to the other one")
        assert is_other_one_request("focus on the other one")
        assert is_other_one_request("close the other one")
        assert is_other_one_request("use the other window")
        assert not is_other_one_request("open chrome")
        assert not is_other_one_request("close it")


# ============================================================================
# TEST: "do that again" repeats last intents
# ============================================================================

class TestDoThatAgainRepeatsIntents:
    """Exit criterion 3: 'Do that again' repeats last successful intents."""
    
    def test_do_that_again_returns_last_intents(self, chrome_opened_state):
        """'do that again' returns the last intent list."""
        intents, reason = resolve_repeat_last("do that again", chrome_opened_state)
        assert intents is not None
        assert len(intents) == 1
        assert intents[0]["tool"] == "open_target"
        assert intents[0]["args"]["query"] == "Chrome"
    
    def test_repeat_that_returns_intents(self, chrome_opened_state):
        """'repeat that' works the same."""
        intents, reason = resolve_repeat_last("repeat that", chrome_opened_state)
        assert intents is not None
        assert intents[0]["tool"] == "open_target"
    
    def test_is_replay_request_patterns(self):
        """Various replay patterns are detected."""
        assert is_replay_request("do that again")
        assert is_replay_request("repeat that")
        assert is_replay_request("do it again")
        assert is_replay_request("again")
        assert is_replay_request("same thing")
        assert is_replay_request("one more time")
        assert not is_replay_request("open chrome")
        assert not is_replay_request("close it")
    
    def test_do_that_again_no_intents_asks(self):
        """Returns question when no intents to repeat."""
        ws = get_world_state()
        intents, question = resolve_repeat_last("do that again", ws)
        assert intents is None
        assert question is not None
        assert "repeat" in question.lower() or "what" in question.lower()
    
    def test_do_that_again_after_chat_only(self, chrome_opened_state):
        """Doesn't repeat if last response was chat-only."""
        set_last_llm_reply_only(True)
        intents, reason = resolve_repeat_last("do that again", chrome_opened_state)
        assert intents is None
        assert "action" in reason.lower() or "chat" in reason.lower()


# ============================================================================
# TEST: Multi-intent follow-ups
# ============================================================================

class TestMultiIntentFollowups:
    """Exit criterion 6: Multi-intent with resolution works."""
    
    def test_has_unresolved_pronoun_detected(self):
        """Detects unresolved pronouns in intent args."""
        intent_with_it = {"tool": "close_window", "args": {"query": "it"}}
        assert has_unresolved_pronoun(intent_with_it)
        
        intent_with_that = {"tool": "close_window", "args": {"app": "that"}}
        assert has_unresolved_pronoun(intent_with_that)
        
        intent_resolved = {"tool": "close_window", "args": {"query": "Chrome"}}
        assert not has_unresolved_pronoun(intent_resolved)
    
    def test_resolve_intent_args_resolves_it(self, chrome_opened_state):
        """resolve_intent_args resolves 'it' to Chrome."""
        intent = {"tool": "close_window", "args": {"query": "it"}}
        resolved_args, clarification = resolve_intent_args(intent, chrome_opened_state)
        assert resolved_args.get("query") == "Chrome"
        assert clarification is None
    
    def test_multi_intent_close_it_and_open_spotify(self, chrome_opened_state):
        """'close it and open spotify' resolves 'it' to Chrome."""
        # Simulate what the orchestrator would do with a multi-intent plan
        intents = [
            {"tool": "close_window", "args": {"query": "it"}},
            {"tool": "open_target", "args": {"query": "spotify"}},
        ]
        
        # Resolve each intent
        resolved = []
        for intent in intents:
            if has_unresolved_pronoun(intent):
                args, _ = resolve_intent_args(intent, chrome_opened_state)
            else:
                args = intent.get("args", {})
            resolved.append({"tool": intent["tool"], "args": args})
        
        assert resolved[0]["args"]["query"] == "Chrome"
        assert resolved[1]["args"]["query"] == "spotify"


# ============================================================================
# TEST: Ambiguous resolution requires clarification
# ============================================================================

class TestAmbiguousRequiresClarification:
    """Exit criterion 7: Ambiguous resolution asks clarification."""
    
    def test_no_context_asks_clarification(self):
        """When no context exists, resolution asks clarification."""
        from unittest.mock import patch
        
        # Mock get_foreground_window to prevent real window detection
        with patch('wyzer.vision.window_context.get_foreground_window', return_value={"app": None, "title": None, "pid": None}):
            ws = get_world_state()
            intent = {"tool": "close_window", "args": {"query": "it"}}
            _, clarification = resolve_intent_args(intent, ws)
            # Should ask since there's no target to resolve to
            assert clarification is not None
            assert "which" in clarification.lower() or "mean" in clarification.lower()


# ============================================================================
# TEST: WorldState new fields
# ============================================================================

class TestWorldStateNewFields:
    """Test the new WorldState fields for Phase 10."""
    
    def test_last_targets_ring_buffer(self, chrome_and_notepad_state):
        """last_targets ring buffer stores recent targets."""
        targets = get_last_targets(2)
        assert len(targets) == 2
        assert targets[0].name == "Notepad"  # Most recent
        assert targets[1].name == "Chrome"   # Previous
    
    def test_last_targets_max_size(self):
        """Ring buffer doesn't exceed max size."""
        # Add more than 5 targets
        for i in range(7):
            update_from_tool_execution(
                tool_name="open_target",
                tool_args={"query": f"App{i}"},
                tool_result={
                    "success": True,
                    "resolved": {"type": "app", "matched_name": f"App{i}"},
                },
            )
        
        targets = get_last_targets(10)
        assert len(targets) <= 5  # Max 5 targets
    
    def test_update_last_intents(self):
        """update_last_intents stores intents for replay."""
        intents = [
            {"tool": "open_target", "args": {"query": "Chrome"}},
            {"tool": "volume_control", "args": {"action": "set", "level": 50}},
        ]
        update_last_intents(intents)
        
        retrieved = get_last_intents()
        assert retrieved is not None
        assert len(retrieved) == 2
        assert retrieved[0]["tool"] == "open_target"
        assert retrieved[1]["tool"] == "volume_control"
    
    def test_update_last_active_window(self):
        """update_last_active_window stores window info."""
        update_last_active_window(
            app_name="Chrome",
            window_title="Google - Chrome",
            pid=1234,
            hwnd=5678,
        )
        
        window = get_last_active_window()
        assert window is not None
        assert window["app_name"] == "Chrome"
        assert window["window_title"] == "Google - Chrome"
        assert window["pid"] == 1234
        assert window["hwnd"] == 5678
    
    def test_set_last_llm_reply_only(self):
        """set_last_llm_reply_only controls replay behavior."""
        update_last_intents([{"tool": "open_target", "args": {"query": "Chrome"}}])
        
        # Should return intents
        assert get_last_intents() is not None
        
        # Mark as reply-only
        set_last_llm_reply_only(True)
        
        # Should now return None
        assert get_last_intents() is None
    
    def test_update_after_tool_contract(self):
        """update_after_tool is the canonical update function."""
        update_after_tool(
            tool_name="focus_window",
            args={"query": "Discord"},
            result={"success": True, "matched": {"process": "Discord.exe"}},
            success=True,
        )
        
        ws = get_world_state()
        assert ws.last_tool == "focus_window"
        assert ws.last_target == "Discord"
    
    def test_update_after_tool_ignores_failures(self):
        """update_after_tool doesn't update on failure."""
        # First, set some state
        update_from_tool_execution(
            "open_target",
            {"query": "Chrome"},
            {"success": True, "resolved": {"type": "app", "matched_name": "Chrome"}},
        )
        
        # Attempt failed update
        update_after_tool(
            tool_name="open_target",
            args={"query": "BadApp"},
            result={"error": {"type": "not_found"}},
            success=False,
        )
        
        ws = get_world_state()
        # State should be unchanged
        assert ws.last_target == "Chrome"


# ============================================================================
# TEST: TargetRecord
# ============================================================================

class TestTargetRecord:
    """Test the TargetRecord dataclass."""
    
    def test_target_record_matches_same_window(self):
        """Two window records with same hwnd match."""
        r1 = TargetRecord(type="window", name="Chrome", hwnd=1234)
        r2 = TargetRecord(type="window", name="Chrome", hwnd=1234)
        assert r1.matches(r2)
    
    def test_target_record_matches_same_app(self):
        """Two app records with same name match."""
        r1 = TargetRecord(type="app", name="Chrome")
        r2 = TargetRecord(type="app", name="chrome")  # Case insensitive
        assert r1.matches(r2)
    
    def test_target_record_different_types_dont_match(self):
        """Different types don't match."""
        r1 = TargetRecord(type="app", name="Chrome")
        r2 = TargetRecord(type="window", name="Chrome")
        assert not r1.matches(r2)
    
    def test_target_record_to_dict(self):
        """to_dict returns all fields."""
        r = TargetRecord(
            type="window",
            name="Chrome",
            app_name="chrome.exe",
            window_title="Google - Chrome",
            hwnd=1234,
        )
        d = r.to_dict()
        assert d["type"] == "window"
        assert d["name"] == "Chrome"
        assert d["app_name"] == "chrome.exe"
        assert d["window_title"] == "Google - Chrome"
        assert d["hwnd"] == 1234


# ============================================================================
# TEST: Streaming TTS bypass for reference resolution patterns
# ============================================================================

class TestStreamingTTSBypass:
    """Ensure reference patterns don't go to streaming TTS."""
    
    def test_move_it_to_monitor_bypasses_streaming(self):
        """'move it to monitor 2' should NOT use streaming TTS."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        # This pattern must be handled deterministically, not streamed
        assert should_use_streaming_tts("move it to monitor 2") is False
        assert should_use_streaming_tts("move that to monitor 1") is False
        assert should_use_streaming_tts("move the window to screen 2") is False
    
    def test_other_one_bypasses_streaming(self):
        """'the other one' patterns should NOT use streaming TTS."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        assert should_use_streaming_tts("the other one") is False
        assert should_use_streaming_tts("switch to the other one") is False
    
    def test_pronoun_actions_bypass_streaming(self):
        """'close it', 'minimize that', etc. should NOT use streaming TTS."""
        from wyzer.core.orchestrator import should_use_streaming_tts
        # These need tool execution, not LLM streaming
        assert should_use_streaming_tts("close it") is False
        assert should_use_streaming_tts("close that") is False
        assert should_use_streaming_tts("Close that.") is False
        assert should_use_streaming_tts("minimize it") is False
        assert should_use_streaming_tts("maximize that") is False
        assert should_use_streaming_tts("shut it") is False
        assert should_use_streaming_tts("kill that") is False
        assert should_use_streaming_tts("focus on it") is False


# ============================================================================
# TEST: Chained command pronoun resolution - last_action.resolved priority
# ============================================================================

class TestChainedCommandResolution:
    """
    Exit criterion: Chained commands like "switch to Spotify and move it to monitor 1"
    should resolve "it" to the LAST ACTION's target (Spotify), not active_app.
    
    This tests the fix for the bug where "move it" resolved to WindowsTerminal
    instead of Spotify after switch_app succeeded.
    """
    
    def test_switch_app_then_move_it_resolves_to_switched_app(self):
        """
        After switch_app("Spotify") succeeds, "move it" resolves to Spotify.
        This is the core bug fix scenario.
        """
        # Simulate switch_app execution with result containing to_app
        update_from_tool_execution(
            tool_name="switch_app",
            tool_args={"mode": "named", "app": "Spotify"},
            tool_result={
                "status": "switched",
                "from_app": "WindowsTerminal",
                "to_app": "Spotify",
                "hwnd": 12345,
                "message": "Switched to Spotify.",
                "spoken": "Switching to Spotify.",
            },
        )
        
        # Simulate that active_app might still show the old window due to timing
        # This simulates the race condition where the window watcher hasn't updated yet
        ws = get_world_state()
        ws.active_app = "WindowsTerminal"  # Stale value
        ws.last_active_window = {"app_name": "WindowsTerminal"}  # Stale value
        
        # "move it" should resolve to Spotify (from last_action.resolved), not WindowsTerminal
        result = resolve_references("move it to monitor 1")
        # The result should contain Spotify, not WindowsTerminal
        # Note: resolve_references may return unchanged for "move it to monitor X"
        # so we also test resolve_pronoun_target directly
        target, reason = resolve_pronoun_target("move it")
        assert target == "Spotify", f"Expected Spotify but got {target} (reason: {reason})"
        assert "last_action" in reason.lower() or "last action" in reason.lower()
    
    def test_switch_app_then_close_it_resolves_to_switched_app(self):
        """
        After switch_app("Discord") succeeds, "close it" resolves to Discord.
        """
        update_from_tool_execution(
            tool_name="switch_app",
            tool_args={"mode": "named", "app": "Discord"},
            tool_result={
                "status": "switched",
                "from_app": "Chrome",
                "to_app": "Discord",
                "hwnd": 54321,
            },
        )
        
        # Stale active_app
        ws = get_world_state()
        ws.active_app = "Chrome"
        ws.last_active_window = {"app_name": "Chrome"}
        
        result = resolve_references("close it")
        assert result == "close Discord", f"Expected 'close Discord' but got '{result}'"
    
    def test_open_target_then_close_it_resolves_to_opened_app(self):
        """
        After open_target("Chrome") succeeds, "close it" resolves to Chrome.
        """
        update_from_tool_execution(
            tool_name="open_target",
            tool_args={"query": "Chrome"},
            tool_result={
                "success": True,
                "resolved": {
                    "type": "app",
                    "matched_name": "Chrome",
                    "path": "C:\\Program Files\\Google\\Chrome\\chrome.exe",
                },
            },
        )
        
        # Simulate stale active_app (still shows Notepad)
        ws = get_world_state()
        ws.active_app = "Notepad"
        ws.last_active_window = {"app_name": "Notepad"}
        
        target, reason = resolve_pronoun_target("close it")
        assert target == "Chrome", f"Expected Chrome but got {target}"
    
    def test_fallback_to_active_app_when_no_last_action(self):
        """
        When no last_action.resolved exists, fall back to active_app.
        """
        # Clear last_action
        ws = get_world_state()
        ws.last_action = None
        ws.last_tool = None
        ws.last_target = None
        
        # Set active_app
        update_last_active_window(app_name="Firefox", window_title="Mozilla Firefox")
        
        result = resolve_references("close it")
        assert result == "close Firefox", f"Expected 'close Firefox' but got '{result}'"
    
    def test_stale_last_action_falls_back_to_active_app(self):
        """
        When last_action is too old (> 30 seconds), fall back to active_app.
        """
        import time
        from wyzer.context.world_state import LastAction
        
        # Set up old last_action (60 seconds ago)
        ws = get_world_state()
        ws.last_action = LastAction(
            tool="switch_app",
            args={"mode": "named", "app": "OldApp"},
            resolved={"to_app": "OldApp"},
            ts=time.time() - 60,  # 60 seconds ago
        )
        
        # Set current active_app
        update_last_active_window(app_name="CurrentApp", window_title="Current Window")
        
        target, reason = resolve_pronoun_target("close it")
        # Should use active_app since last_action is stale
        assert target == "CurrentApp", f"Expected CurrentApp but got {target} (last_action was stale)"
    
    def test_minimize_it_after_switch_app(self):
        """'minimize it' after switch_app resolves to switched app."""
        update_from_tool_execution(
            tool_name="switch_app",
            tool_args={"mode": "named", "app": "Teams"},
            tool_result={
                "status": "switched",
                "from_app": "Slack",
                "to_app": "Teams",
            },
        )
        
        ws = get_world_state()
        ws.active_app = "Slack"  # Stale
        
        result = resolve_references("minimize it")
        assert "Teams" in result, f"Expected Teams in result but got '{result}'"
    
    def test_focus_it_after_switch_app(self):
        """'focus on it' after switch_app resolves to switched app."""
        update_from_tool_execution(
            tool_name="switch_app",
            tool_args={"mode": "named", "app": "Zoom"},
            tool_result={
                "status": "switched",
                "to_app": "Zoom",
            },
        )
        
        ws = get_world_state()
        ws.active_app = "OldApp"  # Stale
        
        result = resolve_references("focus on it")
        assert "Zoom" in result, f"Expected Zoom in result but got '{result}'"
    
    def test_existing_close_it_patterns_still_work(self):
        """Existing 'close it' patterns should still work with active_app."""
        # This tests that when last_action matches active_app, behavior is unchanged
        update_from_tool_execution(
            tool_name="open_target",
            tool_args={"query": "Notepad"},
            tool_result={
                "success": True,
                "resolved": {"type": "app", "matched_name": "Notepad"},
            },
        )
        update_last_active_window(app_name="Notepad", window_title="Untitled - Notepad")
        
        result = resolve_references("close it")
        assert result == "close Notepad"
        
        result = resolve_references("close that")
        assert result == "close Notepad"
        
        result = resolve_references("shut it")
        assert result == "shut Notepad"


class TestMultiIntentRouting:
    """Test that multi-intent detection works with various separators."""
    
    def test_comma_separated_detected_as_multi_intent(self):
        """'Switch to X, move it' should be detected as multi-intent."""
        from wyzer.core.hybrid_router import looks_multi_intent
        
        # Both "and" and comma should be detected as multi-intent
        assert looks_multi_intent("Switch to Spotify and move it to monitor 1")
        assert looks_multi_intent("Switch to Spotify, move it to monitor 1")
        assert looks_multi_intent("Focus Chrome, close it")
        assert looks_multi_intent("go to spotify, minimize it")
    
    def test_comma_separated_routes_correctly(self):
        """'Switch to X, move it' should route to switch_app + leftover."""
        from wyzer.core.hybrid_router import decide
        
        result = decide("Switch to Spotify, move it to monitor 1")
        assert result.mode == "tool_plan"
        assert result.intents is not None
        assert len(result.intents) == 1
        assert result.intents[0]["tool"] == "switch_app"
        assert result.intents[0]["args"]["app"] == "Spotify"
        # The leftover should be marked for processing after switch_app succeeds
        assert "__LEFTOVER__" in result.reply
        assert "move it" in result.reply
    
    def test_period_separated_routes_correctly(self):
        """'Switch to X. Move it to screen 1.' should route correctly."""
        from wyzer.core.hybrid_router import decide
        
        result = decide("Switch to Spotify. Move it to screen 1.")
        assert result.mode == "tool_plan"
        assert result.intents is not None
        assert len(result.intents) == 1
        assert result.intents[0]["tool"] == "switch_app"
        assert result.intents[0]["args"]["app"] == "Spotify"
        # The leftover should be marked for processing after switch_app succeeds
        assert "__LEFTOVER__" in result.reply
        assert "move it" in result.reply.lower() or "Move it" in result.reply
    
    def test_open_and_move_it_routes_correctly(self):
        """'Open X and move it to monitor 2' should NOT infer 'open' for 'move it'."""
        from wyzer.core.hybrid_router import decide
        
        result = decide("Open Chrome and move it to monitor 2")
        assert result.mode == "tool_plan"
        assert result.intents is not None
        # Should only have ONE open_target intent, not two
        assert len(result.intents) == 1
        assert result.intents[0]["tool"] == "open_target"
        assert result.intents[0]["args"]["query"] == "Chrome"
        # The leftover "move it to monitor 2" should be handled after open succeeds
        assert "__LEFTOVER__" in result.reply
        assert "move it" in result.reply.lower()


class TestResolvedInfoExtraction:
    """Test that _extract_resolved_info_for_replay extracts the right info."""
    
    def test_switch_app_extracts_to_app(self):
        """switch_app result extracts to_app for reference resolution."""
        update_from_tool_execution(
            tool_name="switch_app",
            tool_args={"mode": "named", "app": "Spotify"},
            tool_result={
                "status": "switched",
                "from_app": "Terminal",
                "to_app": "Spotify",
                "hwnd": 123,
            },
        )
        
        ws = get_world_state()
        assert ws.last_action is not None
        assert ws.last_action.resolved is not None
        assert ws.last_action.resolved.get("to_app") == "Spotify"
    
    def test_switch_app_already_focused_extracts_app(self):
        """switch_app with already_focused status extracts app for reference resolution."""
        update_from_tool_execution(
            tool_name="switch_app",
            tool_args={"mode": "named", "app": "Spotify"},
            tool_result={
                "status": "already_focused",
                "app": "Spotify",
                "message": "Already on Spotify.",
            },
        )
        
        ws = get_world_state()
        assert ws.last_action is not None
        assert ws.last_action.resolved is not None
        # Should extract 'app' as 'to_app' for consistency
        assert ws.last_action.resolved.get("to_app") == "Spotify"
        
        # And pronoun resolution should work
        target, reason = resolve_pronoun_target("move it")
        assert target == "Spotify"
    
    def test_focus_window_extracts_process(self):
        """focus_window result extracts process for reference resolution."""
        update_from_tool_execution(
            tool_name="focus_window",
            tool_args={"query": "Discord"},
            tool_result={
                "success": True,
                "matched": {
                    "title": "Discord",
                    "process": "Discord.exe",
                    "hwnd": 456,
                },
            },
        )
        
        ws = get_world_state()
        assert ws.last_action is not None
        assert ws.last_action.resolved is not None
        assert ws.last_action.resolved.get("process") == "Discord.exe"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
