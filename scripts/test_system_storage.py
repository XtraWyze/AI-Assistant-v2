"""
Test suite for system_storage tools.

Verifies:
- Drive token normalization
- Tool outputs are JSON-serializable
- Error format consistency
- Tool registration
"""

import json
import platform
from wyzer.tools.system_storage import (
    normalize_drive_token,
    SystemStorageScanTool,
    SystemStorageListTool,
    SystemStorageOpenTool,
)
from wyzer.tools.registry import build_default_registry


def test_normalize_drive_token():
    """Test drive token normalization."""
    print("\n=== test_normalize_drive_token ===")
    
    # Test normalization doesn't crash and returns expected types
    # These should always be None/invalid
    invalid_cases = [
        ("", type(None)),
        (None, type(None)),
        ("zzz", type(None)),
    ]
    
    for input_val, expected_type in invalid_cases:
        result = normalize_drive_token(input_val)
        assert isinstance(result, expected_type), f"Failed: {input_val} should return {expected_type}, got {type(result)}"
        if isinstance(result, str):
            print(f"✓ normalize_drive_token({repr(input_val)}) -> {repr(result)}")
        else:
            print(f"✓ normalize_drive_token({repr(input_val)}) -> None")
    
    # Test that it doesn't crash on drive-like input (even if not present)
    if platform.system() == "Windows":
        result = normalize_drive_token("c")  # C: should exist on Windows
        # Just verify no exception is thrown
        print(f"✓ normalize_drive_token('c') returned: {result}")
        
        result = normalize_drive_token("c:")
        print(f"✓ normalize_drive_token('c:') returned: {result}")
        
        result = normalize_drive_token("c drive")
        print(f"✓ normalize_drive_token('c drive') returned: {result}")


def test_tool_outputs_json_serializable():
    """Test that tool outputs are JSON-serializable."""
    print("\n=== test_tool_outputs_json_serializable ===")
    
    tools = [
        SystemStorageScanTool(),
        SystemStorageListTool(),
    ]
    
    for tool in tools:
        result = tool.run()
        try:
            json_str = json.dumps(result)
            print(f"✓ {tool.name} output is JSON-serializable")
        except Exception as e:
            print(f"✗ {tool.name} output failed: {e}")
            raise


def test_error_format_consistency():
    """Test that errors follow the standard format."""
    print("\n=== test_error_format_consistency ===")
    
    # SystemStorageListTool with invalid drive
    tool = SystemStorageListTool()
    result = tool.run(drive="zzz")
    
    assert "error" in result, f"Missing 'error' key: {result}"
    assert isinstance(result["error"], dict), f"error should be dict: {result['error']}"
    assert "type" in result["error"], f"Missing 'type' in error: {result['error']}"
    assert "message" in result["error"], f"Missing 'message' in error: {result['error']}"
    print("✓ Invalid drive error has correct format")
    
    # SystemStorageOpenTool with missing drive arg
    tool = SystemStorageOpenTool()
    result = tool.run()
    
    assert "error" in result, f"Missing 'error' key: {result}"
    assert result["error"]["type"] == "invalid_argument"
    print("✓ Missing required arg error has correct format")


def test_tools_registered():
    """Test that all three tools are registered."""
    print("\n=== test_tools_registered ===")
    
    registry = build_default_registry()
    
    tools = [
        "system_storage_scan",
        "system_storage_list",
        "system_storage_open",
    ]
    
    for tool_name in tools:
        assert registry.has_tool(tool_name), f"Tool '{tool_name}' not registered"
        tool = registry.get(tool_name)
        assert tool is not None
        assert tool.name == tool_name
        assert tool.description, f"Tool {tool_name} missing description"
        assert tool.args_schema, f"Tool {tool_name} missing args_schema"
        print(f"✓ {tool_name} registered with description and schema")


def test_tool_schema_validity():
    """Test that tool schemas are valid."""
    print("\n=== test_tool_schema_validity ===")
    
    tools = [
        SystemStorageScanTool(),
        SystemStorageListTool(),
        SystemStorageOpenTool(),
    ]
    
    for tool in tools:
        schema = tool.args_schema
        assert isinstance(schema, dict), f"Schema for {tool.name} is not dict"
        assert "type" in schema or "properties" in schema, f"Schema for {tool.name} missing required keys"
        print(f"✓ {tool.name} has valid schema")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("System Storage Tools Test Suite")
    print("=" * 60)
    
    try:
        test_normalize_drive_token()
        test_tool_outputs_json_serializable()
        test_error_format_consistency()
        test_tools_registered()
        test_tool_schema_validity()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
