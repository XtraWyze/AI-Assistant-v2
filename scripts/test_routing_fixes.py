"""Tests for hybrid routing fixes + reference resolution improvements.

Covers:
1. Foreground/app context queries -> get_window_context
2. Verification/status queries -> get_recent_events (NOT open_target)
3. Question guard: interrogative sentences with "open" don't route to open_target
4. "close it" in multi-intent sequences resolves pronoun correctly

Run:
  python scripts/test_routing_fixes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PASS = 0
FAIL = 0


def _check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


def test_hybrid_router_decide():
    """Test that hybrid_router.decide() returns the correct tool plan."""
    from wyzer.core.hybrid_router import decide

    print("\n=== 1. Foreground / app context queries => get_window_context ===")
    foreground_queries = [
        "what app am I currently using",
        "What app am I currently using?",
        "what app am I in",
        "what app am I using",
        "what app am I on",
        "what am I in",
        "what am I using",
        "where am I",
        "where am I right now",
        "what window is focused",
        "which app is active",
        "what's the active window",
        "what is the current app",
        "current app",
        "active window",
    ]
    for q in foreground_queries:
        d = decide(q)
        tool = d.intents[0]["tool"] if d.intents else None
        _check(
            f"{q!r} => get_window_context",
            d.mode == "tool_plan" and tool == "get_window_context",
            f"got mode={d.mode} tool={tool}",
        )

    print("\n=== 2. Verification / status queries => get_recent_events ===")
    verification_queries = [
        "did the last thing you open actually open successfully",
        "Did the last thing you open actually open successfully?",
        "did that work",
        "did it open",
        "did it succeed",
        "did it open successfully",
        "did that open",
        "was that successful",
        "what just happened",
        "what happened",
        "did it fail",
    ]
    for q in verification_queries:
        d = decide(q)
        tool = d.intents[0]["tool"] if d.intents else None
        _check(
            f"{q!r} => get_recent_events",
            d.mode == "tool_plan" and tool == "get_recent_events",
            f"got mode={d.mode} tool={tool}",
        )

    print("\n=== 3. Question guard: questions with 'open' NOT routed to open_target ===")
    question_queries = [
        "did it open",
        "did that open successfully",
        "did the last thing you open actually open successfully",
        "was it opened",
        "is it open",
        "has it opened yet",
    ]
    for q in question_queries:
        d = decide(q)
        tool = d.intents[0]["tool"] if d.intents else None
        _check(
            f"{q!r} != open_target",
            tool != "open_target",
            f"got mode={d.mode} tool={tool}",
        )

    print("\n=== 4. Imperative 'open' still works correctly ===")
    imperative_queries = [
        ("open notepad", "open_target"),
        ("launch chrome", "open_target"),
        ("start spotify", "open_target"),
        ("open settings", "open_target"),
    ]
    for q, expected_tool in imperative_queries:
        d = decide(q)
        tool = d.intents[0]["tool"] if d.intents else None
        _check(
            f"{q!r} => {expected_tool}",
            d.mode == "tool_plan" and tool == expected_tool,
            f"got mode={d.mode} tool={tool}",
        )


def test_segment_matches_tool_question_guard():
    """Test that _segment_matches_tool doesn't match questions as open_target."""
    from wyzer.core.hybrid_router import _segment_matches_tool

    print("\n=== 5. _segment_matches_tool question guard ===")
    # These should NOT return open_target
    bad_segments = [
        "did it open",
        "did that open successfully",
        "was it opened correctly",
    ]
    for seg in bad_segments:
        result = _segment_matches_tool(seg)
        tool = result["tool"] if result else None
        _check(
            f"segment {seg!r} != open_target",
            tool != "open_target",
            f"got {tool}",
        )

    # These SHOULD return open_target
    good_segments = [
        "open notepad",
        "launch chrome",
        "start spotify",
    ]
    for seg in good_segments:
        result = _segment_matches_tool(seg)
        tool = result["tool"] if result else None
        _check(
            f"segment {seg!r} => open_target",
            tool == "open_target",
            f"got {tool}",
        )


def test_verification_query_regex():
    """Test the _VERIFICATION_QUERY_RE regex directly."""
    from wyzer.core.hybrid_router import _VERIFICATION_QUERY_RE

    print("\n=== 6. _VERIFICATION_QUERY_RE pattern matching ===")
    should_match = [
        "did the last thing you open actually open successfully",
        "Did the last thing you open actually open successfully?",
        "did that work",
        "did it open",
        "did it succeed",
        "did it open successfully",
        "did it fail",
        "did that open",
        "was that successful",
        "was it successful",
        "what just happened",
        "what happened",
        "did it actually open",
        "did this work",
    ]
    for q in should_match:
        m = _VERIFICATION_QUERY_RE.match(q.strip())
        _check(f"match {q!r}", m is not None, "NO MATCH")

    should_not_match = [
        "open notepad",
        "close chrome",
        "what time is it",
        "tell me a story",
    ]
    for q in should_not_match:
        m = _VERIFICATION_QUERY_RE.match(q.strip())
        _check(f"no-match {q!r}", m is None, "UNEXPECTED MATCH")


def test_window_context_regex():
    """Test the extended _WINDOW_CONTEXT_RE regex."""
    from wyzer.core.hybrid_router import _WINDOW_CONTEXT_RE

    print("\n=== 7. _WINDOW_CONTEXT_RE pattern matching ===")
    should_match = [
        "what app am I currently using",
        "What app am I currently using?",
        "what app am I in",
        "what app am I using",
        "what app am I on",
        "what am I in",
        "what am I using",
        "where am I",
        "where am I right now",
        "what window is focused",
        "which app is active",
        "what's the active window",
        "what is the current app",
        "current app",
        "active window",
        "what am I currently using",
        "what am I currently in",
    ]
    for q in should_match:
        m = _WINDOW_CONTEXT_RE.match(q.strip())
        _check(f"match {q!r}", m is not None, "NO MATCH")


def test_has_unresolved_pronoun_title_key():
    """Test that has_unresolved_pronoun checks 'title' key (for close_window)."""
    from wyzer.core.reference_resolver import has_unresolved_pronoun

    print("\n=== 8. has_unresolved_pronoun checks 'title' key ===")

    # close_window with title="it" should be detected
    intent_it = {"tool": "close_window", "args": {"title": "it"}}
    _check(
        "close_window(title='it') has pronoun",
        has_unresolved_pronoun(intent_it),
        "returned False",
    )

    intent_that = {"tool": "close_window", "args": {"title": "that"}}
    _check(
        "close_window(title='that') has pronoun",
        has_unresolved_pronoun(intent_that),
        "returned False",
    )

    # close_window with title="Notepad" should NOT be detected
    intent_notepad = {"tool": "close_window", "args": {"title": "Notepad"}}
    _check(
        "close_window(title='Notepad') no pronoun",
        not has_unresolved_pronoun(intent_notepad),
        "returned True",
    )


def test_resolve_intent_args_title_key():
    """Test that resolve_intent_args resolves 'title' key pronouns."""
    from wyzer.core.reference_resolver import resolve_intent_args
    from wyzer.context.world_state import WorldState, LastAction

    print("\n=== 9. resolve_intent_args resolves 'title' key ===")

    # Create a mock world state with last_action from open_target
    ws = WorldState()
    ws.last_tool = "open_target"
    ws.last_target = "Notepad"
    ws.active_app = "Notepad"
    ws.last_action = LastAction(
        tool="open_target",
        args={"query": "notepad"},
        resolved={"matched_name": "Notepad", "app_name": "Notepad"},
    )

    intent = {"tool": "close_window", "args": {"title": "it"}}
    resolved_args, clarification = resolve_intent_args(intent, ws)

    _check(
        "title='it' resolved to Notepad",
        clarification is None and resolved_args.get("title", "").lower() != "it",
        f"got args={resolved_args} clarification={clarification}",
    )


def main():
    print("=" * 60)
    print("ROUTING FIXES + REFERENCE RESOLUTION TESTS")
    print("=" * 60)

    test_hybrid_router_decide()
    test_segment_matches_tool_question_guard()
    test_verification_query_regex()
    test_window_context_regex()
    test_has_unresolved_pronoun_title_key()
    test_resolve_intent_args_title_key()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
