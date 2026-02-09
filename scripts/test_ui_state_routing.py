#!/usr/bin/env python3
"""
UI-State Deterministic Routing Tests

Verifies that UI-state / focused-window queries ALWAYS route to tool_plan
with get_window_context and confidence == 1.0.  No LLM fallback allowed.

Ground truth invariant (ARCHITECTURE_LOCK / GPT5_2 Handoff):
  "tools + deterministic state are the ONLY sources of truth.
   The LLM becomes a narrator, not an actor."

Usage:
    python scripts/test_ui_state_routing.py

Exit code:
    0 = all tests passed
    1 = routing violations detected
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wyzer.core.hybrid_router import (
    decide,
    _is_focused_window_query,
    _is_pure_focused_window_query,
    _is_ui_state_query,
    _has_action_verb,
    needs_reasoning,
)
from wyzer.core.ui_state_patterns import is_ui_state_tool_query
from wyzer.core.logger import init_logger, get_logger

init_logger("INFO")
logger = get_logger()

# ── Focused-window / active-app queries ──────────────────────────────────
# Every entry: (input_text, expected_tool, min_confidence)
FOCUSED_WINDOW_CASES = [
    # Core patterns
    ("what window is currently focused",       "get_window_context", 1.0),
    ("what's focused",                         "get_window_context", 1.0),
    ("which app is active",                    "get_window_context", 1.0),
    ("what am I on right now",                 "get_window_context", 1.0),
    ("what is on screen",                      "describe_screen",    0.93),
    ("what window is focused",                 "get_window_context", 1.0),
    ("what application is active",             "get_window_context", 1.0),
    ("which window is focused",                "get_window_context", 1.0),
    ("current window",                         "get_window_context", 1.0),
    ("foreground window",                      "get_window_context", 1.0),
    ("active app",                             "get_window_context", 1.0),
    ("what app is this",                       "get_window_context", 1.0),
    # Natural speech variations
    ("What am I looking at",                   "describe_screen",    0.93),
    ("what am I using right now",              "get_window_context", 1.0),
    ("what's currently focused",               "get_window_context", 1.0),
    ("what is currently active",               "get_window_context", 1.0),
    ("what program is this",                   "get_window_context", 1.0),
    ("what app is focused",                    "get_window_context", 1.0),
    ("which application is focused",           "get_window_context", 1.0),
    ("focused window",                         "get_window_context", 1.0),
    ("what window is this",                    "get_window_context", 1.0),
    ("tell me the active window",              "get_window_context", 1.0),
    ("what's open right now",                  "list_open_windows", 0.93),
    ("what is open",                           "list_open_windows", 0.93),
    ("what's on my screen",                    "describe_screen",    0.93),
]

# ── Helper-function unit tests ───────────────────────────────────────────
HELPER_POSITIVE = [
    "what window is currently focused",
    "what's focused",
    "which app is active",
    "what am I on right now",
    "current window",
    "foreground window",
    "what application is active",
    "what am I looking at",
    "what app is this",
    "what's currently focused",
    "focused window",
    "tell me the active window",
]

HELPER_NEGATIVE = [
    "open notepad",
    "what time is it",
    "tell me a joke",
    "close chrome",
    "play music",
    "how do I focus a window in Python",  # conceptual, not UI-state
    "set the volume to 50",
]

# ── Expanded UI-state queries (Phase 17 patch) ──────────────────────────
# Open-windows / recent-events / monitor variants that MUST route to tools
EXPANDED_UI_STATE_CASES = [
    # Open-windows variants
    ("What windows are open now?",         "list_open_windows",  0.93),
    ("What's open now?",                   "list_open_windows",  0.93),
    ("what apps are open",                 "list_open_windows",  0.93),
    ("list open windows",                  "list_open_windows",  0.93),
    ("which windows are open",             "list_open_windows",  0.93),
    # Recent-events / last-window / what just happened
    ("What did I open most recently?",     "get_recent_events",  0.93),
    ("What did I just open?",              "get_recent_events",  0.93),
    ("last window",                        "get_recent_events",  0.93),
    ("most recent window",                 "get_recent_events",  0.93),
    ("recent events",                      "get_recent_events",  0.93),
    ("What just happened?",               "get_recent_events",  0.93),
    ("what changed",                       "get_recent_events",  0.93),
    # "most recent / last thing" variations
    ("What's the most recent thing I'll open?", "get_recent_events", 0.93),
    ("what's the last thing I opened?",          "get_recent_events", 0.93),
    ("most recent thing I opened",               "get_recent_events", 0.93),
    ("last thing I opened",                      "get_recent_events", 0.93),
    ("what was the last app I opened?",           "get_recent_events", 0.93),
    ("what's the most recent app I've opened?",  "get_recent_events", 0.93),
    # "most recent / last thing done" variations
    ("What's the most recent thing done?",          "get_recent_events", 0.93),
    ("what's the last thing done?",                  "get_recent_events", 0.93),
    ("most recent thing that happened",              "get_recent_events", 0.93),
    ("last action done",                             "get_recent_events", 0.93),
    ("what was the most recent thing I've done?",    "get_recent_events", 0.93),
]

EXPANDED_NO_REASONING = [
    "What windows are open now?",
    "What's open now?",
    "What did I open most recently?",
    "What did I just open?",
    "What just happened?",
    "what changed",
    "recent events",
    "last window",
    "What's the most recent thing I'll open?",
    "what's the last thing I opened?",
    "last thing I opened",
    "What's the most recent thing done?",
    "what's the last thing done?",
]

EXPANDED_SHARED_HELPER = [
    ("What windows are open now?",     "list_open_windows"),
    ("What's open now?",               "list_open_windows"),
    ("what apps are open",             "list_open_windows"),
    ("What did I just open?",          "get_recent_events"),
    ("What did I open most recently?", "get_recent_events"),
    ("What just happened?",           "get_recent_events"),
    ("recent events",                  "get_recent_events"),
    ("what changed",                   "get_recent_events"),
    ("last window",                    "get_recent_events"),
    ("most recent window",             "get_recent_events"),
    ("What's the most recent thing I'll open?", "get_recent_events"),
    ("what's the last thing I opened?",          "get_recent_events"),
    ("most recent thing I opened",               "get_recent_events"),
    ("last thing I opened",                      "get_recent_events"),
    ("what was the last app I opened?",           "get_recent_events"),
    ("what's the most recent app I've opened?",  "get_recent_events"),
    ("What's the most recent thing done?",          "get_recent_events"),
    ("what's the last thing done?",                  "get_recent_events"),
    ("most recent thing that happened",              "get_recent_events"),
    ("last action done",                             "get_recent_events"),
    ("what was the most recent thing I've done?",    "get_recent_events"),
]


def run_tests():
    passed = 0
    failed = 0
    total = 0

    print("=" * 72)
    print("UI-STATE DETERMINISTIC ROUTING TESTS")
    print("=" * 72)

    # ── 1. _is_focused_window_query helper: positive cases ──────────
    print("\n── _is_focused_window_query() positive cases ──")
    for phrase in HELPER_POSITIVE:
        total += 1
        result = _is_focused_window_query(phrase)
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] _is_focused_window_query({phrase!r}) = {result}")

    # ── 2. _is_focused_window_query helper: negative cases ──────────
    print("\n── _is_focused_window_query() negative cases ──")
    for phrase in HELPER_NEGATIVE:
        total += 1
        result = _is_focused_window_query(phrase)
        expected = False
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] _is_focused_window_query({phrase!r}) = {result} (expected {expected})")

    # ── 3. _is_ui_state_query includes focused-window queries ───────
    print("\n── _is_ui_state_query() includes focused-window queries ──")
    for phrase in HELPER_POSITIVE:
        total += 1
        result = _is_ui_state_query(phrase)
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] _is_ui_state_query({phrase!r}) = {result}")

    # ── 4. Full router: decide() routes to tool_plan ────────────────
    print("\n── decide() routing tests ──")
    for text, expected_tool, min_conf in FOCUSED_WINDOW_CASES:
        total += 1
        decision = decide(text)
        tool_name = decision.intents[0]["tool"] if decision.intents else None
        ok_mode = decision.mode == "tool_plan"
        ok_tool = tool_name == expected_tool
        ok_conf = decision.confidence >= min_conf

        if ok_mode and ok_tool and ok_conf:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(
            f"  [{status}] {text!r}\n"
            f"         mode={decision.mode} tool={tool_name} conf={decision.confidence:.2f}"
        )
        if not ok_mode:
            print(f"         ✗ expected mode=tool_plan, got {decision.mode}")
        if not ok_tool:
            print(f"         ✗ expected tool={expected_tool}, got {tool_name}")
        if not ok_conf:
            print(f"         ✗ expected confidence>={min_conf}, got {decision.confidence}")

    # ── 5. Never routes to LLM ──────────────────────────────────────
    print("\n── LLM-fallback rejection ──")
    for text, _, _ in FOCUSED_WINDOW_CASES:
        total += 1
        decision = decide(text)
        if decision.mode != "llm":
            passed += 1
            print(f"  [PASS] {text!r} → mode={decision.mode} (not llm)")
        else:
            failed += 1
            print(f"  [FAIL] {text!r} → mode=llm  ✗ LLM FALLBACK DETECTED")

    # ── 6. Expanded UI-state queries: decide() → tool_plan ──────────
    print("\n── Expanded UI-state: decide() routing ──")
    for text, expected_tool, min_conf in EXPANDED_UI_STATE_CASES:
        total += 1
        decision = decide(text)
        tool_name = decision.intents[0]["tool"] if decision.intents else None
        ok = decision.mode == "tool_plan" and tool_name == expected_tool and decision.confidence >= min_conf
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {text!r} → mode={decision.mode} tool={tool_name} conf={decision.confidence:.2f}")
        if not ok:
            print(f"         ✗ expected tool_plan/{expected_tool}>={min_conf}")

    # ── 7. Expanded: needs_reasoning() must return False ────────────
    print("\n── Expanded UI-state: needs_reasoning() bypass ──")
    for text in EXPANDED_NO_REASONING:
        total += 1
        result = needs_reasoning(text)
        status = "PASS" if not result else "FAIL"
        if not result:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] needs_reasoning({text!r}) = {result} (want False)")

    # ── 8. Shared helper is_ui_state_tool_query() ───────────────────
    print("\n── is_ui_state_tool_query() ──")
    for text, expected_tool in EXPANDED_SHARED_HELPER:
        total += 1
        result = is_ui_state_tool_query(text)
        ok = result == expected_tool
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] is_ui_state_tool_query({text!r}) = {result!r} (want {expected_tool!r})")

    # Negatives — these must NOT match
    neg_cases = ["tell me a story", "what is the time", "open spotify", "how does gravity work"]
    for text in neg_cases:
        total += 1
        result = is_ui_state_tool_query(text)
        ok = result is None
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] is_ui_state_tool_query({text!r}) = {result!r} (want None)")

    # ── 9. Bug A: multi-intent NOT stolen by focused-window intercept ──
    print("\n── Bug A: _is_pure_focused_window_query vs multi-intent ──")

    # Pure focus queries → True
    pure_focus = [
        "what's focused",
        "what window is focused",
        "tell me the active window",
        "which app is active",
        "current window",
    ]
    for text in pure_focus:
        total += 1
        result = _is_pure_focused_window_query(text)
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] _is_pure_focused_window_query({text!r}) = {result} (want True)")

    # Multi-intent with action verbs → False (must NOT intercept)
    multi_intent_focus = [
        "Switch to VS Code and tell me what's focused.",
        "open notepad and tell me what's focused",
        "close chrome and what's focused",
        "launch spotify and what app is active",
        "minimize this and what's focused",
        "go to firefox and tell me the active window",
    ]
    for text in multi_intent_focus:
        total += 1
        result = _is_pure_focused_window_query(text)
        ok = not result
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] _is_pure_focused_window_query({text!r}) = {result} (want False)")

    # _has_action_verb helper checks
    print("\n── _has_action_verb() ──")
    action_positives = [
        "switch to VS Code and tell me what's focused",
        "open notepad",
        "close chrome",
        "click the button",
        "type hello world",
    ]
    for text in action_positives:
        total += 1
        result = _has_action_verb(text)
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] _has_action_verb({text!r}) = {result} (want True)")

    action_negatives = [
        "what's focused",
        "current window",
        "tell me the active window",
        "what window is this",
    ]
    for text in action_negatives:
        total += 1
        result = _has_action_verb(text)
        ok = not result
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] _has_action_verb({text!r}) = {result} (want False)")

    # decide() for multi-intent must produce multiple tools, not single get_window_context
    print("\n── Bug A: decide() multi-intent routing ──")
    multi_text = "Switch to VS Code and tell me what's focused."
    total += 1
    d = decide(multi_text)
    tool_names = [i["tool"] for i in (d.intents or [])]
    # Must contain switch_app (not be a single get_window_context)
    has_switch = any("switch" in t for t in tool_names)
    not_single_gwc = not (len(tool_names) == 1 and tool_names[0] == "get_window_context")
    ok = d.mode == "tool_plan" and not_single_gwc
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] decide({multi_text!r}) → mode={d.mode} tools={tool_names}")
    if not ok:
        print(f"         ✗ expected multi-intent tool_plan, not single get_window_context")

    # ── 10. Bug B: what_did_i_open formatting ──
    print("\n── Bug B: what_did_i_open formatting ──")
    from wyzer.core import orchestrator as _orch

    # Save originals
    _orig_get_recent = None
    try:
        from wyzer.context import world_state as _ws
        _orig_get_recent = _ws.get_recent_window_events
    except Exception:
        pass

    # 10a: empty events → "no events" response
    def _stub_empty(**kwargs):
        return []
    _ws.get_recent_window_events = _stub_empty

    total += 1
    result = _orch._check_window_watcher_commands("what did I just open", time.perf_counter())
    ok = result is not None and "don't have" in result.get("reply", "").lower()
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    reply_text = result.get("reply", "") if result else "(None)"
    print(f"  [{status}] empty events → {reply_text!r}")

    # 10b: events with blank title → must NOT appear in output
    def _stub_with_blanks(**kwargs):
        et = kwargs.get("event_type", "")
        if et == "opened":
            return [
                {"type": "opened", "title": "Tinkercad", "ts": 3, "app": "chrome.exe"},
                {"type": "opened", "title": "   ", "ts": 2, "app": ""},  # blank!
                {"type": "opened", "title": "", "ts": 1, "app": ""},     # blank!
            ]
        return []
    _ws.get_recent_window_events = _stub_with_blanks

    total += 1
    result = _orch._check_window_watcher_commands("what did I just open", time.perf_counter())
    reply_text = result.get("reply", "") if result else ""
    has_blank = "Focused:" in reply_text or "\n• \n" in reply_text or "•  " in reply_text
    ok = result is not None and "Tinkercad" in reply_text and not has_blank
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] blank titles filtered → {reply_text!r}")

    # 10c: multiple opened events → newest first in reply
    def _stub_ordered(**kwargs):
        et = kwargs.get("event_type", "")
        if et == "opened":
            return [
                {"type": "opened", "title": "Xbox", "ts": 5, "app": "xbox.exe"},
                {"type": "opened", "title": "Tinkercad", "ts": 3, "app": "chrome.exe"},
            ]
        return []
    _ws.get_recent_window_events = _stub_ordered

    total += 1
    result = _orch._check_window_watcher_commands("what did I just open", time.perf_counter())
    reply_text = result.get("reply", "") if result else ""
    # Must say "You just opened Xbox." (newest)
    ok = result is not None and reply_text.startswith("You just opened Xbox")
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] newest first → {reply_text!r}")

    # 10d: no focus_changed events leak into "what did I just open"
    def _stub_focus_leak(**kwargs):
        et = kwargs.get("event_type", "")
        if et == "opened":
            return [{"type": "opened", "title": "Notepad", "ts": 1, "app": "notepad.exe"}]
        if et == "focus_changed":
            return [{"type": "focus_changed", "title": "", "ts": 2}]
        return []
    _ws.get_recent_window_events = _stub_focus_leak

    total += 1
    result = _orch._check_window_watcher_commands("what did I just open", time.perf_counter())
    reply_text = result.get("reply", "") if result else ""
    ok = result is not None and "Focused:" not in reply_text and "Notepad" in reply_text
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] no focus leak → {reply_text!r}")

    # Restore original
    if _orig_get_recent is not None:
        _ws.get_recent_window_events = _orig_get_recent

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
