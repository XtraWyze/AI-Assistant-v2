"""
Tests for the 4-issue fix: per-monitor routing, UI-content guard,
install_succeeded_check correlation, event_log wiring.
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wyzer.core.hybrid_router import decide, needs_reasoning
from wyzer.core.ui_state_patterns import (
    PER_MONITOR_WINDOWS_QUERY_RE,
    is_ui_state_tool_query,
    is_ui_content_query,
)


def run_tests():
    passed = 0
    failed = 0
    total = 0

    def check(label, condition, detail=""):
        nonlocal passed, failed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        suffix = f" → {detail}" if detail else ""
        print(f"  [{status}] {label}{suffix}")

    print("=" * 72)
    print("FOUR-FIX TESTS")
    print("=" * 72)

    # ═══════════════════════════════════════════════════════════════════
    # Issue 1: Per-monitor routing
    # ═══════════════════════════════════════════════════════════════════
    print("\n── Issue 1: Per-monitor routing regex ──")
    per_monitor_positive = [
        "What windows are opened on each monitor?",
        "What windows are open on each monitor?",
        "What's open on each monitor?",
        "What apps are on every monitor?",
        "show me the windows on each screen",
        "list windows on each monitor",
        "windows per each monitor",
        "what's on every screen?",
        "what's on each display?",
        "What windows are showing on all my monitors?",
    ]
    for phrase in per_monitor_positive:
        result = PER_MONITOR_WINDOWS_QUERY_RE.match(phrase)
        check(f"PER_MONITOR_RE('{phrase}')", result is not None, f"match={bool(result)}")

    per_monitor_negative = [
        "What's on monitor 2?",       # single monitor → _WHATS_ON_MONITOR_RE
        "open notepad",
        "what's focused?",
        "tell me a joke",
    ]
    for phrase in per_monitor_negative:
        result = PER_MONITOR_WINDOWS_QUERY_RE.match(phrase)
        check(f"PER_MONITOR_RE('{phrase}') = None", result is None, f"match={bool(result)}")

    print("\n── Issue 1: shared helper routing ──")
    for phrase in per_monitor_positive:
        tool = is_ui_state_tool_query(phrase)
        check(f"is_ui_state_tool_query('{phrase[:50]}...')", tool == "list_open_windows", f"got={tool}")

    print("\n── Issue 1: decide() routing ──")
    for phrase in per_monitor_positive[:5]:
        d = decide(phrase)
        matched_tool = d.intents[0]["tool"] if d.intents else None
        check(
            f"decide('{phrase[:50]}...')",
            d.mode == "tool_plan" and matched_tool == "list_open_windows",
            f"mode={d.mode} tool={matched_tool}",
        )

    print("\n── Issue 1: needs_reasoning() bypass ──")
    for phrase in per_monitor_positive[:5]:
        nr = needs_reasoning(phrase)
        check(f"needs_reasoning('{phrase[:50]}...')", nr is False, f"got={nr}")

    # ═══════════════════════════════════════════════════════════════════
    # Issue 2: UI-content query guard
    # ═══════════════════════════════════════════════════════════════════
    print("\n── Issue 2: is_ui_content_query() positive ──")
    ui_content_positive = [
        "Is there a button that says install on the current window?",
        "Do you see a button called Play?",
        "Can you find a text that says error?",
        "What buttons are on the screen?",
        "Read the dialog",
        "What does the error message say?",
        "Is there an error on the screen?",
        "Can you see a progress bar?",
        "check the window",
        "Is there a button that says OK?",
    ]
    for phrase in ui_content_positive:
        result = is_ui_content_query(phrase)
        check(f"is_ui_content_query('{phrase[:55]}')", result is True, f"got={result}")

    print("\n── Issue 2: is_ui_content_query() negative ──")
    ui_content_negative = [
        "what's focused?",
        "current window",
        "open notepad",
        "what time is it",
        "which app is active",
        "tell me a joke",
    ]
    for phrase in ui_content_negative:
        result = is_ui_content_query(phrase)
        check(f"is_ui_content_query('{phrase}') = False", result is False, f"got={result}")

    print("\n── Issue 2: button queries route to ui_find_text ──")
    button_queries = [
        "Is there a button that says install?",
        "do you see a button called Play?",
        "can you find a button that says OK?",
        "is there an install button?",
    ]
    for phrase in button_queries:
        d = decide(phrase)
        if d.intents:
            tool = d.intents[0]["tool"]
            check(f"decide('{phrase[:45]}') → ui_find_text", tool == "ui_find_text", f"got={tool}")
        else:
            check(f"decide('{phrase[:45]}') → has intents", False, "no intents!")

    # ═══════════════════════════════════════════════════════════════════
    # Issue 3: install_succeeded_check correlation
    # ═══════════════════════════════════════════════════════════════════
    print("\n── Issue 3: install_succeeded_check without correlation ──")
    from wyzer.tools.desktop.assertions import install_succeeded_check, _find_install_correlation
    from wyzer.context.world_state import get_world_state, emit_event

    # Clear event_log
    ws = get_world_state()
    ws.event_log.clear()

    # No correlation → must return "unknown"
    result = install_succeeded_check()
    check(
        "No correlation → status='unknown'",
        result["status"] == "unknown",
        f"got={result['status']}",
    )
    check(
        "No correlation → correlated=False",
        result.get("details", {}).get("correlated") is False,
        f"got={result.get('details', {}).get('correlated')}",
    )

    print("\n── Issue 3: _find_install_correlation() ──")
    # No events → no correlation
    ws.event_log.clear()
    check("empty event_log → None", _find_install_correlation() is None)

    # Add an install event
    emit_event("tool_start", {"tool": "install_app", "args": {"name": "firefox"}})
    check("after tool_start(install_app) → correlated", _find_install_correlation() is not None)

    # Add a non-install event
    ws.event_log.clear()
    emit_event("tool_start", {"tool": "get_window_context", "args": {}})
    check("non-install tool → None", _find_install_correlation() is None)

    # Keyword match in args
    ws.event_log.clear()
    emit_event("tool_end", {"tool": "run_command", "args": {"cmd": "winget install firefox"}})
    check("keyword 'install' in args → correlated", _find_install_correlation() is not None)

    print("\n── Issue 3: generic UI elements ignored ──")
    # "Open conversation options" must NOT be in success indicators
    from wyzer.tools.desktop.assertions import _GENERIC_IGNORE
    check("'open conversation options' in GENERIC_IGNORE", "open conversation options" in _GENERIC_IGNORE)
    check("'open' in GENERIC_IGNORE", "open" in _GENERIC_IGNORE)
    check("'done' in GENERIC_IGNORE", "done" in _GENERIC_IGNORE)
    # "play" should NOT be in generic ignore (it's a real install indicator)
    check("'play' NOT in GENERIC_IGNORE", "play" not in _GENERIC_IGNORE)
    check("'installed' NOT in GENERIC_IGNORE", "installed" not in _GENERIC_IGNORE)

    # ═══════════════════════════════════════════════════════════════════
    # Issue 4: Event log wiring
    # ═══════════════════════════════════════════════════════════════════
    print("\n── Issue 4: emit_event + get_event_log ──")
    from wyzer.context.world_state import get_event_log

    ws.event_log.clear()
    emit_event("test_event", {"data": "hello"})
    events = get_event_log(limit=10)
    check("emit_event → get_event_log returns it", len(events) == 1 and events[0]["event"] == "test_event")

    print("\n── Issue 4: mirror window events into event_log ──")
    from wyzer.context.world_state import _mirror_window_events_to_log
    import wyzer.context.world_state as ws_mod

    ws.event_log.clear()
    ws_mod._last_mirrored_ts = 0.0  # Reset high-water mark

    fake_events = [
        {"type": "opened", "hwnd": 1234, "title": "Notepad", "process": "notepad.exe", "ts": 100.0},
        {"type": "focus_changed", "hwnd": 5678, "title": "Chrome", "process": "chrome.exe", "ts": 101.0},
    ]
    _mirror_window_events_to_log(fake_events)
    events = get_event_log(limit=10)
    check("2 window events mirrored", len(events) == 2, f"got={len(events)}")
    check("first is world_evt", events[0]["event"] == "world_evt", f"got={events[0].get('event')}")
    check("first type=opened", events[0].get("type") == "opened")
    check("second type=focus_changed", events[1].get("type") == "focus_changed")

    # Duplicate mirror should NOT add more events (high-water mark)
    _mirror_window_events_to_log(fake_events)
    events2 = get_event_log(limit=10)
    check("no duplicates on re-mirror", len(events2) == 2, f"got={len(events2)}")

    # New event with higher ts SHOULD be added
    _mirror_window_events_to_log([{"type": "opened", "hwnd": 9999, "title": "Firefox", "process": "firefox.exe", "ts": 200.0}])
    events3 = get_event_log(limit=10)
    check("new event added", len(events3) == 3, f"got={len(events3)}")

    print("\n── Issue 4: get_recent_events tool fallback ──")
    from wyzer.tools.desktop.get_recent_events import GetRecentEventsTool
    from wyzer.context.world_state import update_window_watcher_state

    ws.event_log.clear()
    ws_mod._last_mirrored_ts = 0.0

    # Simulate window events via update_window_watcher_state
    update_window_watcher_state(
        open_windows=[{"hwnd": 111, "title": "Test", "process": "test.exe"}],
        windows_by_monitor={1: [{"hwnd": 111, "title": "Test", "process": "test.exe"}]},
        focused_window={"hwnd": 111, "title": "Test", "process": "test.exe"},
        recent_events=[{"type": "opened", "hwnd": 111, "title": "Test", "process": "test.exe", "ts": time.time()}],
        detected_monitor_count=1,
    )

    tool = GetRecentEventsTool()
    result = tool.run(limit=10)
    check(
        "get_recent_events returns non-empty after window events",
        result["count"] > 0,
        f"count={result['count']}",
    )

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
