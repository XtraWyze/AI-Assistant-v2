# Adding LLM-Based Tools to Wyzer AI Assistant

## Complete Developer Guide for LLM-Routed Tools

This document covers adding tools that **require LLM interpretation** to determine when and how to call them. For deterministic tools that bypass the LLM, see [ADDING_NEW_TOOLS_GUIDE.md](ADDING_NEW_TOOLS_GUIDE.md).

---

## What Are LLM-Based Tools?

LLM-based tools are tools that:
- **Require AI interpretation** - LLM decides when/how to use them
- **Handle complex queries** - Ambiguous or context-dependent requests
- **Accept natural language parameters** - LLM extracts and formats args
- **Support reasoning** - Multi-step or conditional logic

Examples: `open_website` (URLs), complex searches, context-aware actions

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Types of LLM Tools](#types-of-llm-tools)
3. [Step 1: Create the Tool Class](#step-1-create-the-tool-class)
4. [Step 2: Register the Tool](#step-2-register-the-tool)
5. [Step 3: Configure LLM Awareness](#step-3-configure-llm-awareness)
6. [Hybrid Tools: Partial Deterministic + LLM Fallback](#hybrid-tools-partial-deterministic--llm-fallback)
7. [Best Practices for LLM Tool Design](#best-practices-for-llm-tool-design)
8. [Testing LLM Tools](#testing-llm-tools)
9. [Complete Examples](#complete-examples)
10. [Checklist](#checklist)

---

## Overview & Architecture

### LLM Tool Flow

```
User Input
    │
    ▼
┌─────────────────────────┐
│   Hybrid Router         │
│   (No regex match OR    │
│    confidence < 0.75)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   LLM Processing        │
│   (Ollama/OpenAI)       │
│                         │
│   - Understands intent  │
│   - Selects tool(s)     │
│   - Extracts parameters │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Tool Execution        │
│   (With LLM-provided    │
│    arguments)           │
└─────────────────────────┘
```

### Key Differences from Deterministic Tools

| Aspect | Deterministic | LLM-Based |
|--------|---------------|-----------|
| Routing | Regex patterns in `hybrid_router.py` | LLM selects from tool registry |
| Speed | Fast (no AI inference) | Slower (requires LLM call) |
| Flexibility | Fixed patterns only | Handles natural language variations |
| Parameters | Regex extraction | LLM extracts and validates |
| Confidence | Must be ≥ 0.75 | N/A (LLM decides) |

---

## Types of LLM Tools

### 1. Pure LLM Tools
Tools that **only** work through LLM routing (no hybrid router patterns).

```
User: "Find me a good Italian restaurant nearby"
       → LLM interprets → search_restaurants tool
```

### 2. Hybrid Tools (Recommended)
Tools with **both** deterministic patterns AND LLM fallback.

```
User: "open spotify"        → Hybrid router (0.90 confidence) → Direct
User: "launch my music app" → LLM interprets → open_target tool
```

### 3. LLM-Only Parameter Tools
Deterministic trigger, but LLM extracts complex parameters.

```
User: "set a reminder for tomorrow at 3pm about the meeting"
       → Hybrid detects "reminder" → LLM extracts time/message
```

---

## Step 1: Create the Tool Class

LLM tools use the same `ToolBase` class, but with **richer descriptions** for LLM context.

### File: `wyzer/tools/my_llm_tool.py`

```python
"""
My LLM Tool - Handles complex queries requiring interpretation.

The LLM uses the description and args_schema to understand when
and how to call this tool.
"""

from typing import Dict, Any, Optional, List
from wyzer.tools.tool_base import ToolBase


class MyLLMTool(ToolBase):
    """Tool that requires LLM interpretation to use effectively."""
    
    def __init__(self):
        super().__init__()
        self._name = "my_llm_tool"
        
        # CRITICAL: Rich description helps LLM understand when to use this tool
        self._description = (
            "Performs [specific action] when the user wants to [use case]. "
            "Use this tool when the user asks about [topic] or wants to [action]. "
            "Do NOT use for [common confusion case]."
        )
        
        # Detailed schema with descriptions for each parameter
        self._args_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "The user's search query or request. "
                        "Extract the core intent, removing filler words."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["fast", "detailed", "comprehensive"],
                    "description": (
                        "Processing mode. Use 'fast' for quick lookups, "
                        "'detailed' for thorough results, "
                        "'comprehensive' when user wants everything."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum results to return. Default 10.",
                },
            },
            "required": ["query"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with LLM-provided arguments."""
        try:
            query = kwargs.get("query", "")
            mode = kwargs.get("mode", "fast")
            limit = kwargs.get("limit", 10)
            
            if not query:
                return {
                    "error": {
                        "type": "validation_error",
                        "message": "query is required"
                    }
                }
            
            # Tool implementation
            results = self._process_query(query, mode, limit)
            
            return {
                "status": "ok",
                "results": results,
                "count": len(results),
                "mode": mode
            }
            
        except Exception as e:
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                }
            }
    
    def _process_query(self, query: str, mode: str, limit: int) -> List[Dict]:
        """Internal processing logic."""
        # Implementation here
        return []
```

### Key Differences for LLM Tools

| Element | Deterministic Tool | LLM Tool |
|---------|-------------------|----------|
| `_description` | Brief, technical | **Rich, contextual, with examples** |
| `args_schema` descriptions | Optional | **Essential for LLM understanding** |
| Parameter types | Simple | Can be complex (LLM handles extraction) |
| Validation | Strict | Can be more flexible (LLM pre-validates) |

---

## Step 2: Register the Tool

Same as deterministic tools - edit `wyzer/tools/registry.py`:

```python
def build_default_registry() -> ToolRegistry:
    # ... existing imports ...
    from wyzer.tools.my_llm_tool import MyLLMTool
    
    registry = ToolRegistry()
    
    # ... existing registrations ...
    registry.register(MyLLMTool())
    
    return registry
```

**That's it!** Once registered, the LLM automatically sees the tool and can use it.

---

## Step 3: Configure LLM Awareness

The LLM learns about your tool from:

1. **Tool Name** - Should be descriptive (`search_files` not `sf`)
2. **Description** - Most important! Tell the LLM when to use it
3. **Args Schema** - Parameter descriptions guide extraction

### Writing Effective Descriptions

#### Bad Description ❌
```python
self._description = "Searches for things"
```

#### Good Description ✅
```python
self._description = (
    "Search for files on the user's computer by name, extension, or content. "
    "Use when the user asks to find, locate, or search for files, documents, "
    "or specific file types (e.g., 'find my resume', 'where are my PDFs'). "
    "Do NOT use for web searches or application launching."
)
```

### Writing Effective Parameter Descriptions

#### Bad Parameter ❌
```python
"pattern": {
    "type": "string",
    "description": "The pattern"
}
```

#### Good Parameter ✅
```python
"pattern": {
    "type": "string",
    "description": (
        "File name pattern to search for. Can include wildcards (* for any characters). "
        "Examples: '*.pdf' for all PDFs, 'report*' for files starting with 'report', "
        "'budget.xlsx' for exact match. Extract from user's request."
    )
}
```

---

## Hybrid Tools: Partial Deterministic + LLM Fallback

The best approach is often **hybrid**: catch obvious patterns deterministically, fall back to LLM for complex cases.

### Pattern in `hybrid_router.py`

```python
# Hybrid router catches simple cases
_SEARCH_FILES_RE = re.compile(
    r"^(?:find|search\s+for|locate)\s+(?:my\s+)?(.+?)\s+(?:files?|documents?)$",
    re.IGNORECASE,
)

def _decide_single_clause(text: str) -> HybridDecision:
    # ... other handlers ...
    
    # Simple file search patterns - bypass LLM
    m = _SEARCH_FILES_RE.match(clause)
    if m:
        pattern = m.group(1).strip()
        if pattern and pattern.lower() not in {"it", "this", "that", "some"}:
            return HybridDecision(
                mode="tool_plan",
                intents=[{
                    "tool": "search_files",
                    "args": {"pattern": pattern},
                    "continue_on_error": False,
                }],
                reply="",
                confidence=0.85,
            )
    
    # Complex cases fall through to LLM
    # "find files I edited last week" → LLM handles
```

### Decision Matrix

| User Input | Route | Confidence |
|------------|-------|------------|
| "find my pdf files" | Deterministic | 0.85 |
| "search for report.docx" | Deterministic | 0.85 |
| "find files I worked on yesterday" | LLM | N/A |
| "where did I put that thing" | LLM | N/A |
| "locate documents about the project" | LLM | N/A |

---

## Best Practices for LLM Tool Design

### 1. Descriptive Tool Names

```python
# Good
self._name = "search_local_files"
self._name = "set_reminder"
self._name = "translate_text"

# Bad
self._name = "search"      # Too generic
self._name = "util1"       # Meaningless
self._name = "do_thing"    # Uninformative
```

### 2. Explicit Use Cases in Description

```python
self._description = (
    "Translate text between languages. "
    "USE when user asks to translate, convert language, or says "
    "'how do you say X in Y language'. "
    "DO NOT use for definitions or explanations."
)
```

### 3. Enum Parameters for Constrained Choices

```python
"language": {
    "type": "string",
    "enum": ["english", "spanish", "french", "german", "japanese", "chinese"],
    "description": "Target language for translation"
}
```

### 4. Provide Defaults in Schema

```python
"limit": {
    "type": "integer",
    "default": 10,  # LLM knows to use this if user doesn't specify
    "description": "Maximum results. Defaults to 10."
}
```

### 5. Handle Missing/Invalid Args Gracefully

```python
def run(self, **kwargs) -> Dict[str, Any]:
    query = kwargs.get("query", "").strip()
    
    # Provide helpful error for LLM to retry
    if not query:
        return {
            "error": {
                "type": "missing_argument",
                "message": "The 'query' argument is required. Please extract the search term from the user's request.",
                "hint": "Look for what the user wants to find or search for."
            }
        }
```

---

## Testing LLM Tools

### Manual Testing

Since LLM tools require actual LLM calls, test them manually:

```bash
# Start the assistant
python run.py

# Try natural language variations
> "find my documents"
> "search for pdf files"  
> "where are my photos"
> "locate the budget spreadsheet"
```

### Test Cases in `scripts/test_hybrid_router.py`

For hybrid tools, test that complex cases go to LLM:

```python
{
    "text": "find files I edited last week",
    "expect_route": "llm",  # Should NOT match deterministic pattern
    "expect_tool": None,
    "expect_llm_calls": 1,
},
{
    "text": "find my pdf files",  # Simple pattern
    "expect_route": "tool_plan",
    "expect_tool": "search_files",
    "expect_llm_calls": 0,
},
```

### Prompt Testing

Test your tool description helps LLM understand:

```python
# In Python REPL or test script
from wyzer.tools.my_llm_tool import MyLLMTool

tool = MyLLMTool()
print(f"Name: {tool.name}")
print(f"Description: {tool.description}")
print(f"Schema: {tool.args_schema}")

# Ask: Would an LLM know when to use this from the description?
```

---

## Complete Examples

### Example 1: Pure LLM Tool (Web Search)

```python
"""
Web search tool - requires LLM to understand query intent.
"""

from typing import Dict, Any, List
from wyzer.tools.tool_base import ToolBase


class WebSearchTool(ToolBase):
    """Search the web for information."""
    
    def __init__(self):
        super().__init__()
        self._name = "web_search"
        self._description = (
            "Search the internet for current information, news, or answers. "
            "Use when user asks about recent events, wants to look something up, "
            "or needs information not available locally. "
            "Examples: 'search for Python tutorials', 'look up today's news', "
            "'find information about climate change'. "
            "Do NOT use for local file searches or app launching."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "The search query. Extract the core topic or question "
                        "the user wants to search for."
                    ),
                },
                "num_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Number of results to return. Default 5.",
                },
            },
            "required": ["query"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        query = kwargs.get("query", "")
        num_results = kwargs.get("num_results", 5)
        
        if not query:
            return {"error": {"type": "validation_error", "message": "query required"}}
        
        # Implementation would use a search API
        results = self._search(query, num_results)
        
        return {
            "status": "ok",
            "query": query,
            "results": results,
            "count": len(results)
        }
    
    def _search(self, query: str, limit: int) -> List[Dict]:
        # Actual search implementation
        return []
```

### Example 2: Hybrid Tool (Reminders)

```python
"""
Reminder tool with hybrid routing.
- Simple: "remind me at 5pm" → Deterministic
- Complex: "remind me tomorrow about the meeting" → LLM
"""

from typing import Dict, Any, Optional
from wyzer.tools.tool_base import ToolBase


class SetReminderTool(ToolBase):
    """Set reminders and alarms."""
    
    def __init__(self):
        super().__init__()
        self._name = "set_reminder"
        self._description = (
            "Set a reminder or alarm for the user. "
            "Use when user wants to be reminded about something at a specific time. "
            "Examples: 'remind me at 3pm', 'set an alarm for 7am', "
            "'remind me to call mom tomorrow'. "
            "Can handle relative times (in 30 minutes) or absolute times (at 5pm)."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "time": {
                    "type": "string",
                    "description": (
                        "When to trigger the reminder. Can be: "
                        "- Absolute: '3:00 PM', '15:00', 'tomorrow at 9am' "
                        "- Relative: 'in 30 minutes', 'in 2 hours' "
                        "Extract from user's request."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": (
                        "What to remind the user about. "
                        "Extract the reminder content from the user's request. "
                        "Can be empty for simple alarms."
                    ),
                },
                "repeat": {
                    "type": "string",
                    "enum": ["once", "daily", "weekly", "weekdays"],
                    "default": "once",
                    "description": "Repeat schedule. Default 'once'.",
                },
            },
            "required": ["time"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        time_str = kwargs.get("time", "")
        message = kwargs.get("message", "")
        repeat = kwargs.get("repeat", "once")
        
        if not time_str:
            return {
                "error": {
                    "type": "validation_error",
                    "message": "time is required - when should the reminder trigger?"
                }
            }
        
        # Parse and set reminder
        parsed_time = self._parse_time(time_str)
        if not parsed_time:
            return {
                "error": {
                    "type": "parse_error",
                    "message": f"Could not understand time: {time_str}"
                }
            }
        
        # Store reminder (implementation)
        reminder_id = self._create_reminder(parsed_time, message, repeat)
        
        return {
            "status": "ok",
            "reminder_id": reminder_id,
            "time": str(parsed_time),
            "message": message or "(no message)",
            "repeat": repeat
        }
    
    def _parse_time(self, time_str: str):
        # Time parsing implementation
        pass
    
    def _create_reminder(self, time, message: str, repeat: str) -> str:
        # Storage implementation
        return "reminder_123"
```

#### Hybrid Router for Reminders

```python
# In hybrid_router.py

_REMINDER_SIMPLE_RE = re.compile(
    r"^(?:set\s+)?(?:a\s+)?(?:reminder|alarm)\s+(?:for\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*$",
    re.IGNORECASE,
)

def _decide_single_clause(text: str) -> HybridDecision:
    # Simple time-only reminders
    m = _REMINDER_SIMPLE_RE.match(clause)
    if m:
        time_str = m.group(1).strip()
        return HybridDecision(
            mode="tool_plan",
            intents=[{
                "tool": "set_reminder",
                "args": {"time": time_str},
                "continue_on_error": False,
            }],
            reply="",
            confidence=0.85,
        )
    
    # Complex reminders ("remind me tomorrow about...") fall through to LLM
```

---

## Checklist

### Tool Implementation
- [ ] Created file in `wyzer/tools/`
- [ ] Inherits from `ToolBase`
- [ ] **Rich `_description`** with use cases and examples
- [ ] **Detailed `args_schema`** with parameter descriptions
- [ ] Handles missing/invalid arguments gracefully
- [ ] Returns helpful error messages

### Tool Registration
- [ ] Added import in `wyzer/tools/registry.py`
- [ ] Added `registry.register(MyTool())` call

### For Hybrid Tools (Optional but Recommended)
- [ ] Added simple patterns to `hybrid_router.py`
- [ ] Patterns have confidence ≥ 0.75
- [ ] Complex cases fall through to LLM

### Testing
- [ ] Tested with various natural language inputs
- [ ] Verified LLM correctly selects tool
- [ ] Verified LLM correctly extracts parameters
- [ ] Tested error cases

### Documentation
- [ ] Added to `TOOLS_GUIDE.md`
- [ ] Documented example phrases

---

## Troubleshooting

### LLM Not Using Your Tool

1. **Check description** - Is it clear when to use the tool?
2. **Add explicit examples** - "Use when user says X, Y, Z"
3. **Check registration** - Is the tool in the registry?

### LLM Extracting Wrong Parameters

1. **Improve parameter descriptions** - Be specific about format
2. **Add examples in description** - Show expected values
3. **Use enums** - Constrain choices where possible

### LLM Using Tool When It Shouldn't

1. **Add "DO NOT use" clauses** - Explicitly exclude cases
2. **Make description more specific** - Narrow the use case
3. **Create separate tools** - Split overlapping functionality

---

## Summary

Adding LLM tools:

1. **Create tool class** with rich descriptions
2. **Register** in `wyzer/tools/registry.py`
3. **Optionally** add hybrid patterns for simple cases
4. **Test** with natural language variations

The key is **detailed descriptions** that help the LLM understand when and how to use your tool.

---

## See Also

- [ADDING_NEW_TOOLS_GUIDE.md](ADDING_NEW_TOOLS_GUIDE.md) - Deterministic tools (LLM bypass)
- [TOOLS_GUIDE.md](../TOOLS_GUIDE.md) - User-facing tool documentation
