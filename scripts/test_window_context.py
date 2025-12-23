"""Test script for Phase 9: Screen Awareness - Window Context.

Tests:
- window_context.get_foreground_window() returns correct structure
- Keys always present (app, title, pid)
- Handles failure gracefully (no crashes)
- get_visual_context_block() returns formatted string
- Tool get_window_context returns JSON output

Run:
  python scripts/test_window_context.py
"""

import sys
import os

sys.path.insert(0, ".")

from wyzer.vision.window_context import get_foreground_window, get_visual_context_block
from wyzer.tools.get_window_context import GetWindowContextTool
from wyzer.core.hybrid_router import decide


def test_get_foreground_window():
    """Test get_foreground_window() returns valid structure."""
    print("=" * 60)
    print("TEST: get_foreground_window()")
    print("=" * 60)
    
    result = get_foreground_window()
    
    # Check return type
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    print("  ✓ Returns dict")
    
    # Check required keys always present
    assert "app" in result, "Missing 'app' key"
    assert "title" in result, "Missing 'title' key"
    assert "pid" in result, "Missing 'pid' key"
    print("  ✓ All required keys present (app, title, pid)")
    
    # Check types (can be None or proper type)
    app = result["app"]
    title = result["title"]
    pid = result["pid"]
    
    assert app is None or isinstance(app, str), f"app should be str or None, got {type(app)}"
    assert title is None or isinstance(title, str), f"title should be str or None, got {type(title)}"
    assert pid is None or isinstance(pid, int), f"pid should be int or None, got {type(pid)}"
    print("  ✓ Types are correct (str|None for app/title, int|None for pid)")
    
    # On Windows, we should get actual data
    if os.name == "nt":
        # At least one of app or title should be non-None
        has_data = app is not None or title is not None
        if has_data:
            print(f"  ✓ Got actual data: app={app}, title={title[:50] if title else None}...")
        else:
            print("  ⚠ Warning: No app or title detected (may be running headless)")
    else:
        print("  ⚠ Non-Windows platform - expected None values")
    
    print()
    return True


def test_get_visual_context_block():
    """Test get_visual_context_block() returns formatted string."""
    print("=" * 60)
    print("TEST: get_visual_context_block()")
    print("=" * 60)
    
    result = get_visual_context_block()
    
    # Check return type
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    print("  ✓ Returns string")
    
    # If we got data, check format
    if result.strip():
        assert "Visual Context" in result, "Missing 'Visual Context' header"
        assert "Foreground app:" in result, "Missing 'Foreground app:' line"
        assert "Window title:" in result, "Missing 'Window title:' line"
        print("  ✓ Block is properly formatted")
        print(f"\n  Block contents:\n{result}")
    else:
        print("  ⚠ Empty block (no window detected or running headless)")
    
    print()
    return True


def test_tool_get_window_context():
    """Test GetWindowContextTool returns valid JSON output."""
    print("=" * 60)
    print("TEST: GetWindowContextTool.run()")
    print("=" * 60)
    
    tool = GetWindowContextTool()
    
    # Check tool metadata
    assert tool.name == "get_window_context", f"Wrong tool name: {tool.name}"
    print(f"  ✓ Tool name: {tool.name}")
    
    assert "foreground window" in tool.description.lower() or "window" in tool.description.lower(), \
        f"Description missing window info: {tool.description}"
    print(f"  ✓ Description: {tool.description}")
    
    # Run the tool
    result = tool.run()
    
    # Check result structure
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "app" in result, "Missing 'app' key"
    assert "title" in result, "Missing 'title' key"
    assert "pid" in result, "Missing 'pid' key"
    assert "latency_ms" in result, "Missing 'latency_ms' key"
    print("  ✓ All required keys present (app, title, pid, latency_ms)")
    
    # Check latency is reasonable
    latency = result.get("latency_ms", 0)
    assert isinstance(latency, int), f"latency_ms should be int, got {type(latency)}"
    assert latency >= 0, f"latency_ms should be >= 0, got {latency}"
    print(f"  ✓ Latency: {latency}ms")
    
    # Print result
    print(f"\n  Tool output: {result}")
    print()
    return True


def test_hybrid_router_patterns():
    """Test hybrid router recognizes window context patterns."""
    print("=" * 60)
    print("TEST: Hybrid Router Patterns")
    print("=" * 60)
    
    test_cases = [
        ("what am I looking at", "get_window_context", 0.9),
        ("what window is this", "get_window_context", 0.9),
        ("what app is active", "get_window_context", 0.9),
        ("what's the active window", "get_window_context", 0.9),
        ("what's the current app", "get_window_context", 0.9),
        ("which app is active", "get_window_context", 0.9),
        ("what application am I in", "get_window_context", 0.9),
        ("current window", "get_window_context", 0.9),
        ("active app", "get_window_context", 0.9),
    ]
    
    passed = 0
    failed = 0
    
    for phrase, expected_tool, min_confidence in test_cases:
        decision = decide(phrase)
        
        if decision.mode == "tool_plan":
            if decision.intents and len(decision.intents) > 0:
                tool_name = decision.intents[0].get("tool")
                confidence = decision.confidence
                
                if tool_name == expected_tool and confidence >= min_confidence:
                    print(f"  ✓ '{phrase}' -> {tool_name} (conf={confidence:.2f})")
                    passed += 1
                else:
                    print(f"  ✗ '{phrase}' -> {tool_name} (conf={confidence:.2f}) - Expected {expected_tool} with conf >= {min_confidence}")
                    failed += 1
            else:
                print(f"  ✗ '{phrase}' -> tool_plan but no intents")
                failed += 1
        else:
            print(f"  ✗ '{phrase}' -> {decision.mode} - Expected tool_plan")
            failed += 1
    
    print(f"\n  Results: {passed} passed, {failed} failed")
    print()
    return failed == 0


def test_graceful_failure():
    """Test that failures are handled gracefully (no crashes)."""
    print("=" * 60)
    print("TEST: Graceful Failure Handling")
    print("=" * 60)
    
    # Test get_foreground_window never crashes
    try:
        result = get_foreground_window()
        assert result is not None, "Should return dict, not None"
        assert isinstance(result, dict), f"Should return dict, got {type(result)}"
        print("  ✓ get_foreground_window() doesn't crash")
    except Exception as e:
        print(f"  ✗ get_foreground_window() crashed: {e}")
        return False
    
    # Test get_visual_context_block never crashes
    try:
        result = get_visual_context_block()
        assert isinstance(result, str), f"Should return str, got {type(result)}"
        print("  ✓ get_visual_context_block() doesn't crash")
    except Exception as e:
        print(f"  ✗ get_visual_context_block() crashed: {e}")
        return False
    
    # Test tool never crashes
    try:
        tool = GetWindowContextTool()
        result = tool.run()
        assert isinstance(result, dict), f"Should return dict, got {type(result)}"
        print("  ✓ GetWindowContextTool.run() doesn't crash")
    except Exception as e:
        print(f"  ✗ GetWindowContextTool.run() crashed: {e}")
        return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE 9: SCREEN AWARENESS TESTS")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    try:
        if not test_get_foreground_window():
            all_passed = False
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    
    try:
        if not test_get_visual_context_block():
            all_passed = False
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    
    try:
        if not test_tool_get_window_context():
            all_passed = False
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    
    try:
        if not test_hybrid_router_patterns():
            all_passed = False
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    
    try:
        if not test_graceful_failure():
            all_passed = False
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
