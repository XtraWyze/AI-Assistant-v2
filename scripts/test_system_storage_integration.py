"""
Integration test to verify system_storage tools work with the orchestrator.
"""

import json
from wyzer.core.hybrid_router import decide
from wyzer.tools.registry import build_default_registry


def test_hybrid_router_patterns():
    """Test that hybrid router routes system storage commands correctly."""
    print("\n=== test_hybrid_router_patterns ===")
    
    test_cases = [
        ("system scan", "system_storage_scan"),
        ("scan my drives", "system_storage_scan"),
        ("refresh drive index", "system_storage_scan"),
        ("list drives", "system_storage_list"),
        ("how much space do i have", "system_storage_list"),
        ("storage summary", "system_storage_list"),
        ("how much space does d drive have", "system_storage_list"),
        ("space on c drive", "system_storage_list"),
        ("open c:", "system_storage_open"),
        ("open c drive", "system_storage_open"),
    ]
    
    for phrase, expected_tool in test_cases:
        decision = decide(phrase)
        assert decision.mode == "tool_plan", f"'{phrase}' didn't trigger tool_plan mode"
        assert decision.intents, f"'{phrase}' has no intents"
        
        actual_tool = decision.intents[0]["tool"]
        assert actual_tool == expected_tool, f"'{phrase}' routed to {actual_tool}, expected {expected_tool}"
        
        print(f"✓ '{phrase}' -> {expected_tool}")


def test_registry_contains_tools():
    """Test that registry contains all three system storage tools."""
    print("\n=== test_registry_contains_tools ===")
    
    registry = build_default_registry()
    
    tools = ["system_storage_scan", "system_storage_list", "system_storage_open"]
    for tool_name in tools:
        assert registry.has_tool(tool_name), f"Tool {tool_name} not in registry"
        tool = registry.get(tool_name)
        assert tool is not None
        assert tool.description
        print(f"✓ {tool_name} in registry")


def test_tool_execution():
    """Test that tools can be executed through registry."""
    print("\n=== test_tool_execution ===")
    
    registry = build_default_registry()
    
    # Test system_storage_scan
    tool = registry.get("system_storage_scan")
    result = tool.run(refresh=False)
    assert "status" in result or "error" in result, f"Invalid result: {result}"
    json.dumps(result)  # Verify JSON-serializable
    print(f"✓ system_storage_scan executed: status={result.get('status', 'error')}")
    
    # Test system_storage_list
    tool = registry.get("system_storage_list")
    result = tool.run(refresh=False)
    assert "status" in result or "error" in result, f"Invalid result: {result}"
    json.dumps(result)  # Verify JSON-serializable
    print(f"✓ system_storage_list executed: status={result.get('status', 'error')}")
    
    # Test system_storage_open with invalid drive (should fail gracefully)
    tool = registry.get("system_storage_open")
    result = tool.run(drive="zzz_invalid_drive")
    assert "error" in result, f"Invalid drive should return error: {result}"
    json.dumps(result)  # Verify JSON-serializable
    print(f"✓ system_storage_open with invalid drive returned error")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("System Storage Integration Test Suite")
    print("=" * 60)
    
    try:
        test_hybrid_router_patterns()
        test_registry_contains_tools()
        test_tool_execution()
        
        print("\n" + "=" * 60)
        print("All integration tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
