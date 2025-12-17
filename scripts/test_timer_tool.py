"""Test script for TimerTool.

Tests:
- Start timer
- Timer status
- Cancel timer
- Invalid arguments
- Hybrid router integration
"""

import sys
import time

sys.path.insert(0, ".")

from wyzer.tools.timer_tool import TimerTool
from wyzer.core.hybrid_router import decide


def test_tool_direct():
    """Test tool execution directly."""
    print("=" * 60)
    print("DIRECT TOOL TESTS")
    print("=" * 60)
    
    tool = TimerTool()
    
    # Test 1: Start a timer
    print("\n[TEST] Start timer for 5 seconds...")
    result = tool.run(action="start", duration_seconds=5)
    print(f"  Result: {result}")
    assert result.get("status") == "running", "Expected status='running'"
    assert result.get("duration") == 5, "Expected duration=5"
    print("  ✓ Start timer works")
    
    # Test 2: Check status
    print("\n[TEST] Check timer status...")
    result = tool.run(action="status")
    print(f"  Result: {result}")
    assert result.get("status") == "running", "Expected status='running'"
    assert "remaining_seconds" in result, "Expected remaining_seconds in result"
    print("  ✓ Status check works")
    
    # Test 3: Cancel timer
    print("\n[TEST] Cancel timer...")
    result = tool.run(action="cancel")
    print(f"  Result: {result}")
    assert result.get("status") == "cancelled", "Expected status='cancelled'"
    print("  ✓ Cancel timer works")
    
    # Test 4: Status when no timer
    print("\n[TEST] Check status when no timer...")
    result = tool.run(action="status")
    print(f"  Result: {result}")
    assert result.get("status") == "idle", "Expected status='idle'"
    print("  ✓ Idle status works")
    
    # Test 5: Cancel when no timer
    print("\n[TEST] Cancel when no timer...")
    result = tool.run(action="cancel")
    print(f"  Result: {result}")
    assert "error" in result, "Expected error"
    assert result["error"]["type"] == "no_timer", "Expected no_timer error"
    print("  ✓ No-timer error works")
    
    # Test 6: Start without duration
    print("\n[TEST] Start without duration_seconds...")
    result = tool.run(action="start")
    print(f"  Result: {result}")
    assert "error" in result, "Expected error"
    assert result["error"]["type"] == "missing_argument", "Expected missing_argument error"
    print("  ✓ Missing argument error works")
    
    # Test 7: Invalid duration
    print("\n[TEST] Start with invalid duration...")
    result = tool.run(action="start", duration_seconds=-5)
    print(f"  Result: {result}")
    assert "error" in result, "Expected error"
    assert result["error"]["type"] == "invalid_duration", "Expected invalid_duration error"
    print("  ✓ Invalid duration error works")
    
    # Test 8: Missing action
    print("\n[TEST] Call without action...")
    result = tool.run()
    print(f"  Result: {result}")
    assert "error" in result, "Expected error"
    print("  ✓ Missing action error works")
    
    # Test 9: Timer completion (short timer)
    print("\n[TEST] Timer completion (2 second timer)...")
    result = tool.run(action="start", duration_seconds=2)
    assert result.get("status") == "running"
    print("  Timer started, waiting for completion...")
    time.sleep(2.5)  # Wait for timer to complete
    result = tool.run(action="status")
    print(f"  Result: {result}")
    assert result.get("status") in {"finished", "idle"}, "Expected finished or idle status"
    print("  ✓ Timer completion works")
    
    print("\n" + "=" * 60)
    print("ALL DIRECT TOOL TESTS PASSED ✓")
    print("=" * 60)


def test_hybrid_router():
    """Test that hybrid router correctly routes to the timer tool."""
    print("\n" + "=" * 60)
    print("HYBRID ROUTER TESTS")
    print("=" * 60)
    
    # Test start patterns
    start_phrases = [
        ("set a timer for 5 minutes", 300),
        ("start a timer for 30 seconds", 30),
        ("set timer for 2 hours", 7200),
        ("create a timer for 10 mins", 600),
        ("set a timer for 1 minute", 60),
        ("start timer for 90 seconds", 90),
    ]
    
    print("\n[TEST] Timer start patterns:")
    for phrase, expected_seconds in start_phrases:
        decision = decide(phrase)
        print(f"  '{phrase}'")
        print(f"    mode={decision.mode}, confidence={decision.confidence:.2f}")
        
        if decision.mode == "tool_plan" and decision.intents:
            intent = decision.intents[0]
            print(f"    tool={intent['tool']}, args={intent['args']}")
            assert intent["tool"] == "timer", f"Expected tool='timer', got '{intent['tool']}'"
            assert intent["args"]["action"] == "start", "Expected action='start'"
            assert intent["args"]["duration_seconds"] == expected_seconds, \
                f"Expected duration={expected_seconds}, got {intent['args']['duration_seconds']}"
            assert decision.confidence >= 0.9, f"Expected confidence >= 0.9, got {decision.confidence}"
            print("    ✓ Correct routing")
        else:
            print(f"    ✗ FAILED: Expected tool_plan, got {decision.mode}")
            assert False, f"Expected tool_plan for '{phrase}'"
    
    # Test cancel patterns
    cancel_phrases = [
        "cancel the timer",
        "stop the timer",
        "clear timer",
        "cancel my timer",
        "end the timer",
    ]
    
    print("\n[TEST] Timer cancel patterns:")
    for phrase in cancel_phrases:
        decision = decide(phrase)
        print(f"  '{phrase}'")
        print(f"    mode={decision.mode}, confidence={decision.confidence:.2f}")
        
        if decision.mode == "tool_plan" and decision.intents:
            intent = decision.intents[0]
            print(f"    tool={intent['tool']}, args={intent['args']}")
            assert intent["tool"] == "timer", f"Expected tool='timer'"
            assert intent["args"]["action"] == "cancel", "Expected action='cancel'"
            assert decision.confidence >= 0.9, f"Expected confidence >= 0.9"
            print("    ✓ Correct routing")
        else:
            print(f"    ✗ FAILED: Expected tool_plan, got {decision.mode}")
            assert False, f"Expected tool_plan for '{phrase}'"
    
    # Test status patterns
    status_phrases = [
        "how much time is left",
        "timer status",
        "check the timer",
        "what's my timer at",
        "how long is left on the timer",
        "time remaining on timer",
        "show timer",
    ]
    
    print("\n[TEST] Timer status patterns:")
    for phrase in status_phrases:
        decision = decide(phrase)
        print(f"  '{phrase}'")
        print(f"    mode={decision.mode}, confidence={decision.confidence:.2f}")
        
        if decision.mode == "tool_plan" and decision.intents:
            intent = decision.intents[0]
            print(f"    tool={intent['tool']}, args={intent['args']}")
            assert intent["tool"] == "timer", f"Expected tool='timer'"
            assert intent["args"]["action"] == "status", "Expected action='status'"
            assert decision.confidence >= 0.9, f"Expected confidence >= 0.9"
            print("    ✓ Correct routing")
        else:
            print(f"    ✗ FAILED: Expected tool_plan, got {decision.mode}")
            assert False, f"Expected tool_plan for '{phrase}'"
    
    print("\n" + "=" * 60)
    print("ALL HYBRID ROUTER TESTS PASSED ✓")
    print("=" * 60)


def test_non_blocking():
    """Test that timer operations don't block."""
    print("\n" + "=" * 60)
    print("NON-BLOCKING TEST")
    print("=" * 60)
    
    tool = TimerTool()
    
    # Start a long timer
    start_time = time.time()
    result = tool.run(action="start", duration_seconds=60)  # 1 minute timer
    elapsed = time.time() - start_time
    
    print(f"\n[TEST] Start 60-second timer...")
    print(f"  Call completed in {elapsed:.4f} seconds")
    assert elapsed < 0.1, "Timer start should be nearly instant (non-blocking)"
    print("  ✓ Timer start is non-blocking")
    
    # Check status
    start_time = time.time()
    result = tool.run(action="status")
    elapsed = time.time() - start_time
    
    print(f"\n[TEST] Check status...")
    print(f"  Call completed in {elapsed:.4f} seconds")
    assert elapsed < 0.1, "Status check should be nearly instant"
    print("  ✓ Status check is non-blocking")
    
    # Cancel
    start_time = time.time()
    result = tool.run(action="cancel")
    elapsed = time.time() - start_time
    
    print(f"\n[TEST] Cancel timer...")
    print(f"  Call completed in {elapsed:.4f} seconds")
    assert elapsed < 0.1, "Cancel should be nearly instant"
    print("  ✓ Cancel is non-blocking")
    
    print("\n" + "=" * 60)
    print("NON-BLOCKING TEST PASSED ✓")
    print("=" * 60)


def test_json_returns():
    """Test that all returns are proper JSON-serializable dicts."""
    import json
    
    print("\n" + "=" * 60)
    print("JSON SERIALIZATION TEST")
    print("=" * 60)
    
    tool = TimerTool()
    
    operations = [
        ("start", {"action": "start", "duration_seconds": 10}),
        ("status", {"action": "status"}),
        ("cancel", {"action": "cancel"}),
        ("status_idle", {"action": "status"}),
        ("cancel_no_timer", {"action": "cancel"}),
        ("missing_action", {}),
        ("invalid_action", {"action": "unknown"}),
    ]
    
    for name, args in operations:
        result = tool.run(**args)
        print(f"\n[TEST] {name}: {args}")
        
        # Verify it's a dict
        assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
        
        # Verify it's JSON-serializable
        try:
            json_str = json.dumps(result)
            print(f"  JSON: {json_str}")
            print("  ✓ Valid JSON")
        except (TypeError, ValueError) as e:
            print(f"  ✗ FAILED: Not JSON-serializable: {e}")
            assert False, f"Result not JSON-serializable for {name}"
    
    print("\n" + "=" * 60)
    print("JSON SERIALIZATION TEST PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TIMER TOOL TEST SUITE")
    print("=" * 60)
    
    test_tool_direct()
    test_hybrid_router()
    test_non_blocking()
    test_json_returns()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60 + "\n")
