# Complete Guide: Adding New Tools to Wyzer AI Assistant

> *Last Updated: December 2025*

## Overview

This guide walks you through **every step** required to add a new tool to Wyzer, from creating the tool class to ensuring proper routing and user-friendly responses.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Step 1: Create the Tool Class](#step-1-create-the-tool-class)
3. [Step 2: Register the Tool](#step-2-register-the-tool)
4. [Step 3: Add Hybrid Router Bypass (Deterministic Routing)](#step-3-add-hybrid-router-bypass-deterministic-routing)
5. [Step 4: Add Human-Readable Reply Formatting](#step-4-add-human-readable-reply-formatting)
6. [Step 5: Add Error-to-Speech Mapping](#step-5-add-error-to-speech-mapping)
7. [Step 6: Testing Your Tool](#step-6-testing-your-tool)
8. [Complete Checklist](#complete-checklist)
9. [Complete Example: Battery Status Tool](#complete-example-battery-status-tool)
10. [Advanced: Stateful Tools (Timer Pattern)](#advanced-stateful-tools-timer-pattern)

---

## Architecture Overview

When a user speaks a command, the system processes it through these stages:

```
User Speech
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CORE PROCESS                             │
│   (assistant.py - hotword detection, audio, state machine)      │
└─────────────────────────────┬───────────────────────────────────┘
                              │ IPC Queue
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BRAIN PROCESS                             │
│   (brain_worker.py - STT, LLM, orchestrator, TTS)              │
│                              │                                  │
│   ┌──────────────────────────▼──────────────────────────────┐  │
│   │              Hybrid Router                               │  │
│   │         (hybrid_router.py)                               │  │
│   └──────────────────────────┬──────────────────────────────┘  │
│                              │                                  │
│              ┌───────────────┴───────────────┐                 │
│              ▼                               ▼                 │
│       [Regex Match]                    [No Match]              │
│       Confidence≥0.75                        │                 │
│              │                               ▼                 │
│              │                    ┌─────────────────┐          │
│              │                    │  LLM Routing    │          │
│              │                    │ (llama.cpp/     │          │
│              │                    │  Ollama)        │          │
│              │                    └────────┬────────┘          │
│              └───────────────┬─────────────┘                   │
│                              ▼                                  │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │                   TOOL WORKER POOL                        │ │
│   │              (tool_worker_pool.py)                        │ │
│   │   ┌─────────────┐  ┌─────────────┐                       │ │
│   │   │  Worker 1   │  │  Worker 2   │  ← Tools run HERE     │ │
│   │   │  (Process)  │  │  (Process)  │                       │ │
│   │   └─────────────┘  └─────────────┘                       │ │
│   └──────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │              Response Formatting                          │ │
│   │         (_format_fastpath_reply)                          │ │
│   └──────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│                         TTS Output                              │
└─────────────────────────────────────────────────────────────────┘
```

### ⚠️ CRITICAL: Multiprocess Architecture

**Tools execute in separate WORKER PROCESSES**, not in the Brain process!

This means:
- **In-memory state is NOT shared** between tool executions
- Each tool call may run in a different worker process
- Module-level variables (like `threading.Timer`) will NOT work across calls
- For persistent state, use **file-based storage** (see Timer Pattern below)

### Key Files

| File | Purpose |
|------|---------|
| `wyzer/tools/tool_base.py` | Base class all tools inherit from |
| `wyzer/tools/registry.py` | Registers tools so the system knows they exist |
| `wyzer/core/hybrid_router.py` | Regex patterns for deterministic (fast) routing |
| `wyzer/core/orchestrator.py` | Executes tools and formats responses |
| `wyzer/core/tool_worker_pool.py` | Pool of worker processes that execute tools |
| `wyzer/core/brain_worker.py` | Brain process (can poll for async events) |

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
        
        # REQUIRED: Unique snake_case identifier
        self._name = "my_new_tool"
        
        # REQUIRED: Clear description (used by LLM for tool selection)
        self._description = "Performs a specific action with optional parameters"
        
        # REQUIRED: JSON Schema for arguments
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
        
        IMPORTANT: Always return a Dict[str, Any] - never raise exceptions!
        
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
                        "type": "missing_argument",
                        "message": "required_param is required"
                    }
                }
            
            # ═══════════════════════════════════════════════
            # YOUR TOOL LOGIC HERE
            # ═══════════════════════════════════════════════
            result = self._do_something(required_param, optional_param)
            
            # Return success response with data
            # Include fields that _format_fastpath_reply will use!
            return {
                "status": "ok",
                "value": result,
                "message": f"Processed {required_param}"
            }
            
        except Exception as e:
            # NEVER raise - always return error dict
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                }
            }
    
    def _do_something(self, param1: str, param2: int) -> Any:
        """Internal helper method."""
        # Implementation
        return {"processed": param1, "value": param2}
```

### Key Requirements

| Requirement | Description |
|-------------|-------------|
| `_name` | Unique `snake_case` identifier |
| `_description` | Clear description for LLM context |
| `_args_schema` | Valid JSON Schema for parameters |
| `run()` returns `Dict` | **ALWAYS** return dict, never raise |
| Error format | Return `{"error": {"type": "...", "message": "..."}}` |

---

## Step 2: Register the Tool

Edit `wyzer/tools/registry.py`:

### 2.1 Add Import

Find the `build_default_registry()` function and add your import:

```python
def build_default_registry() -> ToolRegistry:
    # ... existing imports ...
    
    # ADD YOUR IMPORT HERE
    from wyzer.tools.my_new_tool import MyNewTool
```

### 2.2 Register the Tool

In the same function, register your tool:

```python
    registry = ToolRegistry()
    
    # ... existing registrations ...
    
    # ADD YOUR REGISTRATION HERE
    registry.register(MyNewTool())
    
    return registry
```

### Complete Example

```python
def build_default_registry() -> ToolRegistry:
    from wyzer.tools.get_time import GetTimeTool
    from wyzer.tools.get_system_info import GetSystemInfoTool
    # ... other imports ...
    from wyzer.tools.my_new_tool import MyNewTool  # ← ADD THIS
    
    registry = ToolRegistry()
    
    registry.register(GetTimeTool())
    registry.register(GetSystemInfoTool())
    # ... other registrations ...
    registry.register(MyNewTool())  # ← ADD THIS
    
    return registry
```

---

## Step 3: Add Hybrid Router Bypass (Deterministic Routing)

**This step is CRITICAL for fast response times!**

Without this, every command goes through the LLM, adding latency. With proper regex patterns, obvious commands bypass the LLM entirely.

Edit `wyzer/core/hybrid_router.py`:

### 3.1 Add Your Regex Pattern

Near the top of the file (around line 150-250), add a compiled regex:

```python
# My New Tool patterns: match obvious invocations
_MY_TOOL_RE = re.compile(
    r"^(?:"
    r"(?:check|get|show)\s+(?:my\s+)?(?:new\s+)?tool(?:\s+status)?|"  # "check my tool", "get tool status"
    r"(?:what|what's)\s+(?:is\s+)?(?:my\s+)?tool\s+(?:status|level)|"  # "what's my tool status"
    r"tool\s+info"                                                       # "tool info"
    r").*$",
    re.IGNORECASE,
)
```

### 3.2 Add Decision Logic

Find the `_decide_single_clause()` function and add your matching logic:

```python
def _decide_single_clause(text: str) -> HybridDecision:
    clause = (text or "").strip()
    if not clause:
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.0)
    
    # ... existing patterns ...
    
    # ═══════════════════════════════════════════════════════════════
    # MY NEW TOOL: Add your pattern matching here
    # ═══════════════════════════════════════════════════════════════
    if _MY_TOOL_RE.match(clause):
        # Extract any parameters from the regex if needed
        # For simple tools with no args:
        return HybridDecision(
            mode="tool_plan",
            intents=[{
                "tool": "my_new_tool",
                "args": {"required_param": "default_value"},
                "continue_on_error": False
            }],
            reply="",
            confidence=0.92,  # Must be ≥ 0.75 to bypass LLM
        )
    
    # ... rest of function ...
```

### Regex Pattern Best Practices

| Pattern | Purpose | Example |
|---------|---------|---------|
| `^...$` | Anchor to full input | Prevents partial matches |
| `(?:...)` | Non-capturing group | Groups alternatives |
| `\b` | Word boundary | Prevents "time" in "sometimes" |
| `\s+` | One or more spaces | Variable spacing |
| `(?:word1\|word2)` | Alternatives | Match either word |
| `(.+)` | Capture group | Extract parameters |

### Extracting Parameters Example

```python
# Volume control with level extraction
_VOLUME_SET_RE = re.compile(
    r"^(?:set\s+)?volume\s+(?:to\s+)?(\d{1,3})\s*%?$",
    re.IGNORECASE,
)

# In _decide_single_clause:
m = _VOLUME_SET_RE.match(clause)
if m:
    level = int(m.group(1))
    return HybridDecision(
        mode="tool_plan",
        intents=[{
            "tool": "volume_control",
            "args": {"action": "set", "level": level},
            "continue_on_error": False
        }],
        reply="",
        confidence=0.95,
    )
```

---

## Step 4: Add Human-Readable Reply Formatting

**⚠️ THIS IS CRITICAL: Without this, users hear raw JSON data!**

Edit `wyzer/core/orchestrator.py` and find the `_format_fastpath_reply()` function (around line 1373).

### 4.1 Add Your Tool's Response Formatter

Inside the `format_info()` nested function, add a handler for your tool:

```python
def _format_fastpath_reply(user_text: str, intents: List[Intent], execution_summary: ExecutionSummary) -> str:
    def format_info(tool: str, args: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        # ... existing handlers ...
        
        # ═══════════════════════════════════════════════════════════════
        # MY NEW TOOL: Add human-readable response formatting
        # ═══════════════════════════════════════════════════════════════
        if tool == "my_new_tool":
            value = result.get("value")
            status = result.get("status")
            
            # Format the response for speech
            if value is not None:
                return f"Your tool value is {value}."
            if status == "ok":
                return "Tool executed successfully."
            return "Here's your tool information."
        
        return None
```

### Response Formatting Guidelines

| Guideline | Good | Bad |
|-----------|------|-----|
| Speak naturally | "It's 3:45 PM" | "time: 15:45:00" |
| Be concise | "You have 50 gigs free" | "The free space on drive C is 50.234 gigabytes" |
| Use pronouns | "Set volume to 50%" | "The volume_control tool set level to 50" |
| Round numbers | "About 3 hours" | "2.847293 hours remaining" |
| Skip technical details | "Chrome is open" | "Process chrome.exe with PID 1234 is running" |

### Example: Volume Control Response

```python
if tool == "volume_control":
    action = str(args.get("action") or "").lower()
    level = result.get("level")
    new_level = result.get("new_level")
    muted = result.get("muted")
    
    if action == "get":
        if isinstance(level, int):
            return f"Volume is at {level}%."
        return "OK."
    
    if action == "set":
        if isinstance(new_level, int):
            return f"Set volume to {new_level}%."
        return "OK."
    
    if action in {"mute", "unmute", "toggle_mute"}:
        if isinstance(muted, bool):
            return "Muted." if muted else "Unmuted."
        return "OK."
```

---

## Step 5: Add Error-to-Speech Mapping

When your tool fails, users should hear a helpful message, not technical errors.

Edit `wyzer/core/orchestrator.py` and find the `_tool_error_to_speech()` function (around line 170).

### 5.1 Add Your Error Handlers

```python
def _tool_error_to_speech(
    error: Any,
    tool: str,
    args: Optional[Dict[str, Any]] = None
) -> str:
    args = args or {}
    
    # Parse error
    error_type = ""
    error_msg = ""
    if isinstance(error, dict):
        error_type = str(error.get("type", "")).lower()
        error_msg = str(error.get("message", ""))
    elif isinstance(error, str):
        error_msg = error
    
    # ... existing handlers ...
    
    # ═══════════════════════════════════════════════════════════════
    # MY NEW TOOL: Add error message mappings
    # ═══════════════════════════════════════════════════════════════
    if tool == "my_new_tool":
        if error_type == "not_available":
            return "That feature isn't available right now."
        if error_type == "invalid_input":
            param = args.get("required_param", "")
            return f"I didn't understand '{param}'. Can you try again?"
        if error_type == "permission_denied":
            return "I don't have permission to do that."
    
    # ... rest of function ...
```

### Error Type Conventions

| Error Type | When to Use | User Message Example |
|------------|-------------|---------------------|
| `not_found` | Target doesn't exist | "I can't find Spotify. Is it open?" |
| `permission_denied` | Access blocked | "Try running as administrator." |
| `invalid_args` | Bad parameters | "I didn't understand that command." |
| `execution_error` | General failure | "Something went wrong. Try again." |
| `unsupported_platform` | OS doesn't support | "That's not supported on this system." |

---

## Step 6: Testing Your Tool

### 6.1 Create a Test Script

Create `scripts/test_my_new_tool.py`:

```python
"""Test script for MyNewTool."""

import sys
sys.path.insert(0, ".")

from wyzer.tools.my_new_tool import MyNewTool
from wyzer.core.hybrid_router import decide


def test_tool_direct():
    """Test tool execution directly."""
    tool = MyNewTool()
    
    # Test with valid args
    result = tool.run(required_param="test_value")
    print(f"Success result: {result}")
    assert "error" not in result
    
    # Test with missing args
    result = tool.run()
    print(f"Missing args result: {result}")
    assert "error" in result


def test_hybrid_router():
    """Test that hybrid router correctly routes to the tool."""
    test_phrases = [
        "check my tool",
        "get tool status",
        "what's my tool status",
        "tool info",
    ]
    
    for phrase in test_phrases:
        decision = decide(phrase)
        print(f"'{phrase}' -> mode={decision.mode}, confidence={decision.confidence:.2f}")
        
        if decision.mode == "tool_plan":
            print(f"  intents: {decision.intents}")
            assert decision.intents[0]["tool"] == "my_new_tool"
            assert decision.confidence >= 0.75
        else:
            print(f"  WARNING: Expected tool_plan, got {decision.mode}")


if __name__ == "__main__":
    print("=== Direct Tool Test ===")
    test_tool_direct()
    
    print("\n=== Hybrid Router Test ===")
    test_hybrid_router()
    
    print("\n✓ All tests passed!")
```

### 6.2 Run Tests

```bash
# Activate virtual environment
.\venv_new\Scripts\Activate.ps1

# Run your test
python scripts/test_my_new_tool.py
```

### 6.3 Test End-to-End

Run the assistant and test voice commands:

```bash
python run.py
```

Then speak your test phrases and verify:
1. The tool executes (check logs)
2. The response is human-readable (not JSON)
3. Errors produce friendly messages

---

## Complete Checklist

Use this checklist when adding a new tool:

### Tool Creation
- [ ] Created `wyzer/tools/my_tool.py` with `ToolBase` subclass
- [ ] Set unique `_name` (snake_case)
- [ ] Set clear `_description`
- [ ] Defined `_args_schema` with JSON Schema
- [ ] Implemented `run()` method returning `Dict[str, Any]`
- [ ] Returns `{"error": {"type": "...", "message": "..."}}` on failure
- [ ] Never raises exceptions from `run()`

### Registration
- [ ] Added import in `wyzer/tools/registry.py`
- [ ] Added `registry.register(MyTool())` call

### Hybrid Router (Deterministic Bypass)
- [ ] Added `_MY_TOOL_RE` regex pattern in `hybrid_router.py`
- [ ] Added matching logic in `_decide_single_clause()`
- [ ] Confidence is ≥ 0.75 for bypass
- [ ] Tested regex patterns match expected phrases

### Response Formatting (NO RAW JSON TO USER!)
- [ ] Added handler in `format_info()` within `_format_fastpath_reply()`
- [ ] Response is natural speech (not technical)
- [ ] Numbers are rounded appropriately
- [ ] Technical details are hidden

### Error Handling
- [ ] Added error handlers in `_tool_error_to_speech()`
- [ ] Each error type has a friendly message
- [ ] Target/parameter info is included when available

### Testing
- [ ] Created test script in `scripts/`
- [ ] Direct tool execution works
- [ ] Hybrid router matches expected phrases
- [ ] End-to-end voice test passes
- [ ] Error cases produce friendly messages

---

## Complete Example: Battery Status Tool

Here's a complete example following all the steps:

### 1. Tool Class (`wyzer/tools/battery_status.py`)

```python
"""Battery status tool - Get laptop battery information."""

import platform
from typing import Dict, Any
from wyzer.tools.tool_base import ToolBase


class BatteryStatusTool(ToolBase):
    """Tool to get battery status and charge level."""
    
    def __init__(self):
        super().__init__()
        self._name = "battery_status"
        self._description = "Get battery level and charging status"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        try:
            if platform.system() != "Windows":
                return {
                    "error": {
                        "type": "unsupported_platform",
                        "message": "Battery status only works on Windows"
                    }
                }
            
            import psutil
            battery = psutil.sensors_battery()
            
            if battery is None:
                return {
                    "error": {
                        "type": "no_battery",
                        "message": "No battery detected"
                    }
                }
            
            return {
                "percent": int(battery.percent),
                "plugged_in": battery.power_plugged,
                "time_left_minutes": battery.secsleft // 60 if battery.secsleft > 0 else None
            }
            
        except Exception as e:
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                }
            }
```

### 2. Registry (`wyzer/tools/registry.py`)

```python
# Add import
from wyzer.tools.battery_status import BatteryStatusTool

# Add registration
registry.register(BatteryStatusTool())
```

### 3. Hybrid Router (`wyzer/core/hybrid_router.py`)

```python
# Add regex pattern (near other patterns)
_BATTERY_RE = re.compile(
    r"^(?:"
    r"(?:what|what's|whats)\s+(?:is\s+)?(?:my\s+)?battery(?:\s+(?:level|status|percent(?:age)?|charge))?|"
    r"(?:check|get|show)\s+(?:my\s+)?battery(?:\s+(?:level|status|percent(?:age)?|charge))?|"
    r"battery\s+(?:level|status|percent(?:age)?|charge)|"
    r"how\s+much\s+battery(?:\s+(?:do\s+i\s+have|left|remaining))?"
    r")\??$",
    re.IGNORECASE,
)

# Add in _decide_single_clause():
if _BATTERY_RE.match(clause):
    return HybridDecision(
        mode="tool_plan",
        intents=[{"tool": "battery_status", "args": {}, "continue_on_error": False}],
        reply="",
        confidence=0.94,
    )
```

### 4. Response Formatting (`wyzer/core/orchestrator.py`)

```python
# In format_info() within _format_fastpath_reply():
if tool == "battery_status":
    percent = result.get("percent")
    plugged = result.get("plugged_in")
    time_left = result.get("time_left_minutes")
    
    if percent is not None:
        status = "charging" if plugged else "on battery"
        response = f"Battery is at {percent}%, {status}."
        
        if time_left and not plugged:
            hours = time_left // 60
            mins = time_left % 60
            if hours > 0:
                response += f" About {hours} hours {mins} minutes remaining."
            else:
                response += f" About {mins} minutes remaining."
        
        return response
    return "Battery status retrieved."
```

### 5. Error Handling (`wyzer/core/orchestrator.py`)

```python
# In _tool_error_to_speech():
if tool == "battery_status":
    if error_type == "no_battery":
        return "This device doesn't have a battery."
    if error_type == "unsupported_platform":
        return "Battery status isn't available on this system."
```

### 6. Test Results

```
User: "What's my battery?"
Wyzer: "Battery is at 73%, on battery. About 2 hours 15 minutes remaining."

User: "Check battery" (on desktop)
Wyzer: "This device doesn't have a battery."
```

---

## Quick Reference

| Step | File | What to Add |
|------|------|-------------|
| 1 | `wyzer/tools/my_tool.py` | Tool class with `run()` |
| 2 | `wyzer/tools/registry.py` | Import + `registry.register()` |
| 3 | `wyzer/core/hybrid_router.py` | Regex + `_decide_single_clause()` logic |
| 4 | `wyzer/core/orchestrator.py` | `format_info()` handler |
| 5 | `wyzer/core/orchestrator.py` | `_tool_error_to_speech()` handlers |
| 6 | `scripts/test_*.py` | Test script |

---

## Common Mistakes to Avoid

1. **Returning raw JSON to user** - Always add `format_info()` handler
2. **Forgetting to register** - Tool won't be found
3. **Regex confidence < 0.75** - Won't bypass LLM
4. **Raising exceptions** - Always return error dict
5. **Technical error messages** - Users should hear friendly language
6. **Not testing regex patterns** - Phrases won't route correctly
7. **Using in-memory state** - Tools run in worker processes; use file-based state
8. **Using threading.Timer directly** - Worker process exits; timer dies with it

---

## Advanced: Stateful Tools (Timer Pattern)

### The Problem

Tools run in a **ToolWorkerPool** with separate worker processes. This means:

- `threading.Timer` won't work (worker process may exit)
- Module-level variables don't persist between calls
- Each tool invocation may run in a different process

### The Solution: File-Based State

For tools that need persistent state (like a timer), use a JSON file:

```python
"""
Timer tool - Example of stateful tool using file-based persistence.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Any

from wyzer.tools.tool_base import ToolBase


# File-based state (works across processes)
_STATE_FILE = Path(__file__).parent.parent / "data" / "my_tool_state.json"
_file_lock = threading.Lock()


def _load_state() -> Dict[str, Any]:
    """Load state from file."""
    try:
        if _STATE_FILE.exists():
            with open(_STATE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"active": False}  # Default state


def _save_state(state: Dict[str, Any]) -> None:
    """Save state to file."""
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


class MyStatefulTool(ToolBase):
    """Tool with persistent state across process boundaries."""
    
    def __init__(self):
        super().__init__()
        self._name = "my_stateful_tool"
        self._description = "A tool that maintains state"
        self._args_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "stop", "status"]}
            },
            "required": ["action"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        action = kwargs.get("action", "").lower()
        
        if action == "start":
            # Save state with timestamp
            _save_state({
                "active": True,
                "start_time": time.time(),
                "end_time": time.time() + 60  # 1 minute from now
            })
            return {"status": "started"}
        
        elif action == "stop":
            _save_state({"active": False})
            return {"status": "stopped"}
        
        elif action == "status":
            state = _load_state()
            return state
        
        return {"error": {"type": "invalid_action", "message": "Unknown action"}}
```

### Async Events: Brain Worker Polling

For tools that need to trigger events later (like timer alarms), the **Brain Worker** polls for completion:

#### 1. Add a check function to your tool:

```python
# In your tool file (e.g., timer_tool.py)

def check_my_event() -> bool:
    """
    Check if an async event has occurred.
    Called by brain worker every ~100ms.
    Returns True once when event triggers.
    """
    with _file_lock:
        state = _load_state()
        if state.get("active") and time.time() >= state.get("end_time", 0):
            # Event triggered! Clear state and return True
            state["active"] = False
            _save_state(state)
            return True
        return False
```

#### 2. Import and call from brain_worker.py:

```python
# In wyzer/core/brain_worker.py

from wyzer.tools.my_tool import check_my_event

# In the main loop (uses queue timeout for polling):
while True:
    # Check for async events
    if check_my_event():
        logger.info("[MY_TOOL] Event triggered!")
        tts_controller.enqueue("Your event has occurred.")
    
    # Use timeout so we poll every 100ms
    try:
        msg = core_to_brain_q.get(timeout=0.1)
    except queue.Empty:
        continue
    
    # ... handle messages ...
```

### Complete Timer Example

See `wyzer/tools/timer_tool.py` for a complete implementation that:
- Uses file-based state (`wyzer/data/timer_state.json`)
- Records `end_time` when timer starts
- Brain worker polls `check_timer_finished()` every 100ms
- Triggers TTS when `time.time() >= end_time`

### Key Points for Stateful Tools

| Issue | Solution |
|-------|----------|
| State doesn't persist | Use JSON file in `wyzer/data/` |
| Timer callback doesn't fire | Don't use `threading.Timer`; store `end_time` and poll |
| Need async notification | Add check function, poll from brain_worker |
| File access races | Use `threading.Lock()` for file operations |
| Tool hangs | Never block in `run()`; return immediately |
