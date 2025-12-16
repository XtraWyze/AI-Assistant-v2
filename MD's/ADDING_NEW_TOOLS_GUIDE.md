# Adding Deterministic Tools to Wyzer AI Assistant

## Complete Developer Guide for LLM-Bypass Tools

This document provides a comprehensive guide for adding **deterministic tools** that bypass the LLM entirely using regex pattern matching in the hybrid router. For LLM-based tools, see [ADDING_LLM_TOOLS_GUIDE.md](ADDING_LLM_TOOLS_GUIDE.md).

---

## What Are Deterministic Tools?

Deterministic tools are tools that:
- **Bypass the LLM completely** - No AI inference required
- **Use regex patterns** - Match user input directly
- **Execute immediately** - Faster response times
- **Have predictable behavior** - Same input = same output
- **Require confidence ≥ 0.75** - To qualify for LLM bypass

Examples: `get_time`, `volume_control`, `open_target`, `system_storage_scan`

---

## Table of Contents

1. [Overview](#overview)
2. [Tool Architecture](#tool-architecture)
3. [Step 1: Create the Tool Class](#step-1-create-the-tool-class)
4. [Step 2: Register the Tool](#step-2-register-the-tool)
5. [Step 3: Add Hybrid Router Bypass (Regex Patterns)](#step-3-add-hybrid-router-bypass-regex-patterns)
6. [Step 4: Testing Your Tool](#step-4-testing-your-tool)
7. [Confidence Levels & Scoring](#confidence-levels--scoring)
8. [Complete Example: Adding a New Tool](#complete-example-adding-a-new-tool)
9. [Checklist](#checklist)

---

## Overview

The deterministic tool system has three main components:

| Component | File | Purpose |
|-----------|------|---------|
| **Tool Base** | `wyzer/tools/tool_base.py` | Abstract base class all tools inherit from |
| **Tool Registry** | `wyzer/tools/registry.py` | Registers and manages available tools |
| **Hybrid Router** | `wyzer/core/hybrid_router.py` | **KEY:** Regex-based routing to bypass LLM |

### Deterministic Flow (This Guide)

```
User Input
    │
    ▼
┌─────────────────────────┐
│   Hybrid Router         │
│   (Regex Pattern Match) │
└───────────┬─────────────┘
            │
    [confidence ≥ 0.75]
            │
            ▼
    ┌───────────────┐
    │ Direct Tool   │
    │ Execution     │
    │ (NO LLM!)     │
    └───────────────┘
```

### When to Use Deterministic Tools

| Use Deterministic | Use LLM-Based |
|-------------------|---------------|
| Simple, predictable commands | Complex, context-dependent requests |
| "open spotify" | "find me some good music to play" |
| "volume 50" | "make it a bit quieter" (ambiguous) |
| "what time is it" | "how long until my meeting" |
| "close chrome" | "close whatever I'm not using" |
| Single action, clear target | Multi-step or reasoning required |

---

## Tool Architecture

### ToolBase Class

Every tool must inherit from `ToolBase` and implement:

```python
from wyzer.tools.tool_base import ToolBase

class MyNewTool(ToolBase):
    def __init__(self):
        super().__init__()
        self._name = "my_tool_name"           # Unique identifier (snake_case)
        self._description = "What this tool does"
        self._args_schema = {                  # JSON Schema for arguments
            "type": "object",
            "properties": { ... },
            "required": [ ... ],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        # Execute tool logic
        # Return JSON-serializable dict
        pass
```

### Key Requirements

1. **Name**: Must be unique, use `snake_case`
2. **Description**: Clear, concise description for LLM context
3. **Args Schema**: Valid JSON Schema defining expected parameters
4. **Return Value**: Always return a `Dict[str, Any]` (JSON-serializable)

---

## Step 1: Create the Tool Class

Create a new file in `wyzer/tools/` (e.g., `my_new_tool.py`):

```python
"""
My New Tool - Brief description.

Detailed documentation about what this tool does.
"""

from typing import Dict, Any
from wyzer.tools.tool_base import ToolBase


class MyNewTool(ToolBase):
    """Tool to perform a specific action."""
    
    def __init__(self):
        """Initialize the tool with metadata."""
        super().__init__()
        self._name = "my_new_tool"
        self._description = "Performs a specific action with optional parameters"
        self._args_schema = {
            "type": "object",
            "properties": {
                "required_param": {
                    "type": "string",
                    "minLength": 1,
                    "description": "A required parameter description",
                },
                "optional_param": {
                    "type": "integer",
                    "description": "An optional parameter with default",
                },
            },
            "required": ["required_param"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            required_param: Description of the param
            optional_param: Description with default behavior
            
        Returns:
            Dict with results or error
        """
        try:
            # Extract arguments
            required_param = kwargs.get("required_param", "")
            optional_param = kwargs.get("optional_param", 10)  # Default value
            
            # Validate inputs
            if not required_param:
                return {
                    "error": {
                        "type": "validation_error",
                        "message": "required_param is required"
                    }
                }
            
            # Tool logic here
            result = self._do_something(required_param, optional_param)
            
            # Return success response
            return {
                "status": "ok",
                "data": result,
                "message": f"Successfully processed {required_param}"
            }
            
        except Exception as e:
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                }
            }
    
    def _do_something(self, param1: str, param2: int) -> Dict[str, Any]:
        """Internal helper method."""
        # Implementation
        return {"processed": param1, "value": param2}
```

### Best Practices

1. **Always handle exceptions** - Return error dicts, don't raise
2. **Validate inputs** - Check required parameters and types
3. **Use helper methods** - Keep `run()` clean and readable
4. **Document everything** - Docstrings for LLM context

---

## Step 2: Register the Tool

Edit `wyzer/tools/registry.py`:

### 2.1 Add Import

At the top of `build_default_registry()`:

```python
def build_default_registry() -> ToolRegistry:
    # ... existing imports ...
    
    # Add your new tool import
    from wyzer.tools.my_new_tool import MyNewTool
```

### 2.2 Register the Tool

In the same function, add registration:

```python
    registry = ToolRegistry()
    
    # ... existing registrations ...
    
    # Register your new tool
    registry.register(MyNewTool())
    
    return registry
```

### Complete Registry Example

```python
def build_default_registry() -> ToolRegistry:
    from wyzer.tools.get_time import GetTimeTool
    from wyzer.tools.get_system_info import GetSystemInfoTool
    # ... other imports ...
    from wyzer.tools.my_new_tool import MyNewTool  # <-- ADD THIS
    
    registry = ToolRegistry()
    
    registry.register(GetTimeTool())
    registry.register(GetSystemInfoTool())
    # ... other registrations ...
    registry.register(MyNewTool())  # <-- ADD THIS
    
    return registry
```

---

## Step 3: Add Hybrid Router Bypass (Regex Patterns)

This is the **critical step** for achieving ≥0.75 confidence to bypass the LLM.

Edit `wyzer/core/hybrid_router.py`:

### 3.1 Create Your Regex Pattern

Add a compiled regex pattern near the top of the file (after line ~140):

```python
# My New Tool patterns: match obvious invocations
_MY_TOOL_RE = re.compile(
    r"^(?:"
    r"(?:run|execute|do)\s+my\s+(?:new\s+)?tool|"           # "run my tool"
    r"(?:perform|make)\s+(?:a\s+)?specific\s+action|"       # "perform specific action"
    r"my\s+tool\s+(?:on|for)\s+(.+)|"                       # "my tool on X"
    r"do\s+the\s+thing"                                      # "do the thing"
    r").*$",
    re.IGNORECASE,
)
```

### Regex Pattern Best Practices

| Pattern Element | Purpose | Example |
|-----------------|---------|---------|
| `^...$` | Anchor to full input | Prevents partial matches |
| `(?:...)` | Non-capturing group | Groups alternatives without capturing |
| `\b` | Word boundary | Prevents "time" matching "sometimes" |
| `\s+` | One or more spaces | Handles variable spacing |
| `\s*` | Zero or more spaces | Optional spacing |
| `(?:word1\|word2)` | Alternatives | Match either word |
| `(.+)` | Capture group | Extract parameters |
| `(?:..)?` | Optional group | Make a phrase optional |

### 3.2 Add Handler in `_decide_single_clause()`

Find the `_decide_single_clause()` function (around line 243) and add your handler. **Order matters** - more specific patterns should come first.

```python
def _decide_single_clause(text: str) -> HybridDecision:
    clause = (text or "").strip()
    if not clause:
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.0)

    # ... existing handlers (time, weather, etc.) ...

    # My New Tool detection (ADD THIS BLOCK)
    if _MY_TOOL_RE.match(clause):
        # Extract parameter if captured
        m = _MY_TOOL_RE.match(clause)
        param = None
        if m and m.groups():
            param = (m.group(1) or "").strip()
            if param and param.lower() in {"it", "this", "that"}:
                param = None
        
        # Build arguments
        args = {}
        if param:
            args["required_param"] = param
        
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "my_new_tool",
                    "args": args,
                    "continue_on_error": False,
                }
            ],
            reply="",
            confidence=0.90,  # Must be ≥ 0.75 to bypass LLM!
        )

    # ... rest of existing handlers ...
```

### 3.3 Confidence Scoring Guidelines

| Confidence | When to Use | Result |
|------------|-------------|--------|
| **0.95** | Exact match, no ambiguity | Direct execution, no LLM |
| **0.90-0.94** | High confidence, clear intent | Direct execution, no LLM |
| **0.85-0.89** | Good match, minor ambiguity possible | Direct execution, no LLM |
| **0.75-0.84** | Reasonable match, some context needed | Direct execution, no LLM |
| **0.70-0.74** | Borderline - might go to LLM | May use LLM |
| **< 0.70** | Low confidence | Uses LLM |

**To bypass LLM, you need confidence ≥ 0.75**

### 3.4 Parameter Extraction Patterns

#### Simple Extraction (Single Value)
```python
# Pattern: "my tool on <target>"
m = re.search(r"\b(?:on|for)\s+(.+?)(?:\?|$)", clause, re.IGNORECASE)
if m:
    target = (m.group(1) or "").strip().rstrip("?").strip()
```

#### Multiple Parameters
```python
# Pattern: "move <target> to monitor <number>"
m = re.match(r"^move\s+(.+?)\s+to\s+monitor\s+(\d+)$", clause, re.IGNORECASE)
if m:
    target = m.group(1).strip()
    monitor = m.group(2).strip()
```

#### Numeric Extraction
```python
# Pattern: "set volume to 35" or "volume 35%"
m = re.search(r"\b(\d{1,3})\s*%?\b", clause)
if m:
    value = int(m.group(1))
    if 0 <= value <= 100:  # Validate range
        # Use value
```

### 3.5 Common Filtering Patterns

Always filter out ambiguous targets:

```python
# Filter ambiguous targets
if target.lower() in {"it", "this", "that", "something", "anything"}:
    return HybridDecision(mode="llm", intents=None, reply="", confidence=0.4)
```

---

## Step 4: Testing Your Tool

### 4.1 Add Test Cases

Edit `scripts/test_hybrid_router.py` and add test cases:

```python
cases = [
    # ... existing cases ...
    
    # My New Tool tests
    {
        "text": "run my tool",
        "expect_route": "tool_plan",
        "expect_tool": "my_new_tool",
        "expect_llm_calls": 0,
        "expect_args": {},
    },
    {
        "text": "my tool on target_value",
        "expect_route": "tool_plan",
        "expect_tool": "my_new_tool",
        "expect_llm_calls": 0,
        "expect_args": {"required_param": "target_value"},
    },
    {
        "text": "do something ambiguous",  # Should fall back to LLM
        "expect_route": "llm",
        "expect_tool": None,
        "expect_llm_calls": 1,
    },
]
```

### 4.2 Run Tests

```bash
python scripts/test_hybrid_router.py
```

### 4.3 Smoke Test

```bash
python scripts/test_tool_pool_smoke.py
```

---

## Confidence Levels & Scoring

### Existing Tool Confidence Reference

| Tool | Pattern | Confidence |
|------|---------|------------|
| `get_time` | "what time is it" | 0.95 |
| `local_library_refresh` | "refresh library" | 0.93 |
| `volume_control` (mute) | "mute" | 0.93 |
| `get_weather_forecast` | "what's the weather" | 0.92 |
| `system_storage_list` | "list drives" | 0.92 |
| `system_storage_scan` | "scan drives" | 0.95 |
| `open_target` | "open X" | 0.90 |
| `get_location` | "where am I" | 0.90 |
| `volume_control` (change) | "volume up" | 0.88 |
| `close_window` | "close X" | 0.85 |
| `minimize_window` | "minimize X" | 0.85 |
| `media_next` | "next track" | 0.85 |

### How to Choose Confidence

1. **Start high (0.90+)** for exact, unambiguous patterns
2. **Lower slightly (0.85-0.89)** if pattern could match unintended inputs
3. **Use 0.75-0.84** for broader patterns that should still bypass LLM
4. **Use < 0.70** (or just return `mode="llm"`) for patterns that need LLM interpretation

---

## Complete Example: Adding a New Tool

Let's add a hypothetical "brightness_control" tool:

### File: `wyzer/tools/brightness_control.py`

```python
"""
Brightness control tool for adjusting screen brightness.
"""

import platform
from typing import Dict, Any
from wyzer.tools.tool_base import ToolBase


class BrightnessControlTool(ToolBase):
    """Tool to control screen brightness."""
    
    def __init__(self):
        super().__init__()
        self._name = "brightness_control"
        self._description = "Adjust screen brightness level (0-100)"
        self._args_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "increase", "decrease"],
                    "description": "Action to perform",
                },
                "level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Brightness level (0-100) for 'set' action",
                },
                "delta": {
                    "type": "integer",
                    "description": "Amount to change for increase/decrease",
                },
            },
            "required": ["action"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        try:
            action = kwargs.get("action", "get")
            level = kwargs.get("level")
            delta = kwargs.get("delta", 10)
            
            if action == "get":
                current = self._get_brightness()
                return {"status": "ok", "brightness": current}
            
            elif action == "set":
                if level is None:
                    return {"error": {"type": "validation_error", "message": "level required for set"}}
                self._set_brightness(level)
                return {"status": "ok", "brightness": level}
            
            elif action == "increase":
                current = self._get_brightness()
                new_level = min(100, current + delta)
                self._set_brightness(new_level)
                return {"status": "ok", "brightness": new_level}
            
            elif action == "decrease":
                current = self._get_brightness()
                new_level = max(0, current - delta)
                self._set_brightness(new_level)
                return {"status": "ok", "brightness": new_level}
            
            return {"error": {"type": "invalid_action", "message": f"Unknown action: {action}"}}
            
        except Exception as e:
            return {"error": {"type": "execution_error", "message": str(e)}}
    
    def _get_brightness(self) -> int:
        # Platform-specific implementation
        return 50  # Placeholder
    
    def _set_brightness(self, level: int) -> None:
        # Platform-specific implementation
        pass
```

### Registry Update: `wyzer/tools/registry.py`

```python
from wyzer.tools.brightness_control import BrightnessControlTool

# In build_default_registry():
registry.register(BrightnessControlTool())
```

### Hybrid Router Update: `wyzer/core/hybrid_router.py`

```python
# Add regex pattern (near other patterns, ~line 180)
_BRIGHTNESS_RE = re.compile(
    r"\b(?:"
    r"brightness|"
    r"screen\s+(?:brightness|light)|"
    r"(?:dim|brighten)\s+(?:the\s+)?screen|"
    r"(?:how\s+)?bright\s+is\s+(?:my\s+)?screen"
    r")\b",
    re.IGNORECASE,
)

# Add handler in _decide_single_clause() (after volume controls, ~line 650)
    # Brightness control
    if _BRIGHTNESS_RE.search(clause):
        tl = clause.lower()
        
        # Get brightness
        if re.search(r"\b(?:get|check|show|what|how\s+bright)\b", tl):
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "brightness_control", "args": {"action": "get"}, "continue_on_error": False}],
                reply="",
                confidence=0.90,
            )
        
        # Set brightness to specific level
        m = re.search(r"\b(\d{1,3})\s*%?\b", tl)
        if m:
            level = int(m.group(1))
            if 0 <= level <= 100:
                return HybridDecision(
                    mode="tool_plan",
                    intents=[{"tool": "brightness_control", "args": {"action": "set", "level": level}, "continue_on_error": False}],
                    reply="",
                    confidence=0.88,
                )
        
        # Increase brightness
        if re.search(r"\b(?:brighter|increase|raise|up)\b", tl):
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "brightness_control", "args": {"action": "increase"}, "continue_on_error": False}],
                reply="",
                confidence=0.85,
            )
        
        # Decrease brightness
        if re.search(r"\b(?:dim|dimmer|decrease|lower|down)\b", tl):
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "brightness_control", "args": {"action": "decrease"}, "continue_on_error": False}],
                reply="",
                confidence=0.85,
            )
```

### Test Cases: `scripts/test_hybrid_router.py`

```python
{
    "text": "what's my screen brightness",
    "expect_route": "tool_plan",
    "expect_tool": "brightness_control",
    "expect_llm_calls": 0,
    "expect_args": {"action": "get"},
},
{
    "text": "set brightness to 75",
    "expect_route": "tool_plan",
    "expect_tool": "brightness_control",
    "expect_llm_calls": 0,
    "expect_args": {"action": "set", "level": 75},
},
{
    "text": "dim the screen",
    "expect_route": "tool_plan",
    "expect_tool": "brightness_control",
    "expect_llm_calls": 0,
    "expect_args": {"action": "decrease"},
},
```

---

## Checklist

Use this checklist when adding a new tool:

### Tool Implementation
- [ ] Created new file in `wyzer/tools/`
- [ ] Inherits from `ToolBase`
- [ ] Set unique `_name` (snake_case)
- [ ] Set descriptive `_description`
- [ ] Defined complete `_args_schema` (JSON Schema)
- [ ] Implemented `run()` method
- [ ] Returns `Dict[str, Any]` (JSON-serializable)
- [ ] Handles exceptions (returns error dict)
- [ ] Validates input parameters

### Tool Registration
- [ ] Added import in `wyzer/tools/registry.py`
- [ ] Added `registry.register(MyTool())` call

### Hybrid Router (LLM Bypass)
- [ ] Created regex pattern (`_MY_TOOL_RE`)
- [ ] Pattern uses `re.IGNORECASE`
- [ ] Pattern anchored with `^...$` or uses `\b` word boundaries
- [ ] Added handler in `_decide_single_clause()`
- [ ] Handler placed in correct order (before catch-all)
- [ ] Confidence ≥ 0.75 for LLM bypass
- [ ] Extracts parameters from input
- [ ] Filters ambiguous inputs ("it", "this", "that")
- [ ] Returns `HybridDecision(mode="tool_plan", ...)`

### Testing
- [ ] Added test cases in `scripts/test_hybrid_router.py`
- [ ] Tests cover success cases (expect `tool_plan`)
- [ ] Tests cover failure cases (expect `llm`)
- [ ] Tests verify correct arguments
- [ ] All tests pass

### Documentation
- [ ] Added tool to `TOOLS_GUIDE.md`
- [ ] Documented user-friendly examples
- [ ] Listed all supported voice commands

---

## Troubleshooting

### Tool Not Being Called
1. Check registration in `registry.py`
2. Verify tool name matches between registry and hybrid router
3. Ensure regex pattern matches your test input

### LLM Still Being Used (Not Bypassing)
1. Check confidence level (must be ≥ 0.75)
2. Test regex pattern matches with Python:
   ```python
   import re
   pattern = re.compile(r"your pattern", re.IGNORECASE)
   print(pattern.match("your test input"))  # Should not be None
   ```
3. Verify handler is before the catch-all return statement

### Wrong Arguments Extracted
1. Debug with print statements in handler
2. Check capture group indices (`m.group(1)`, etc.)
3. Verify argument names match schema

### Tests Failing
1. Check exact tool name spelling
2. Verify expected arguments match handler output
3. Ensure `expect_llm_calls: 0` for hybrid routes

---

## Summary

Adding a new tool requires:

1. **Create tool class** → `wyzer/tools/my_tool.py`
2. **Register tool** → `wyzer/tools/registry.py`
3. **Add hybrid routing** → `wyzer/core/hybrid_router.py`
   - Add regex pattern (anchored, case-insensitive)
   - Add handler with confidence ≥ 0.75
   - Extract and validate parameters
4. **Test thoroughly** → `scripts/test_hybrid_router.py`

The key to LLM bypass is a **well-crafted regex pattern** with **confidence ≥ 0.75**.

---

## See Also

- [ADDING_LLM_TOOLS_GUIDE.md](ADDING_LLM_TOOLS_GUIDE.md) - For tools that require LLM interpretation
- [TOOLS_GUIDE.md](../TOOLS_GUIDE.md) - User-facing tool documentation
- [HYBRID_ROUTER_BYPASS_IMPLEMENTATION.md](HYBRID_ROUTER_BYPASS_IMPLEMENTATION.md) - Hybrid router deep dive
