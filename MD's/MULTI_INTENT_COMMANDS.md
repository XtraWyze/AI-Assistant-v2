# Multi-Intent Command Support - Complete Implementation

> *Last Updated: December 2025*

## Summary
Implemented full support for multi-intent commands combining time, weather, and window management operations. Users can now compose multiple independent operations in a single query using natural separators.

## Status: 15/15 Tests Passing ✓

All combinations of time, weather, and window commands work correctly in multi-intent queries.

## Architecture

### Multi-Intent Parser
Located in `wyzer/core/multi_intent_parser.py`, the parser:

1. **Detects multi-intent markers** - Recognizes separators in user input
2. **Splits queries into clauses** - Breaks apart combined commands
3. **Parses each clause deterministically** - Uses `_decide_single_clause()` from hybrid_router
4. **Validates confidence levels** - Ensures all clauses meet minimum threshold (0.7 per-clause, 0.75 combined)
5. **Returns composed intents** - Returns list of tool calls for orchestrator

### Supported Separators

Multi-intent commands are split by these separators (in precedence order):

| Separator | Mode | Examples |
|-----------|------|----------|
| `and then` | sequential | "open spotify and then play music" |
| `then` | sequential | "open spotify then minimize it" |
| `and` | parallel | "open spotify and close chrome" |
| `;` | sequential | "open spotify; close chrome" |
| `,` | parallel | "open spotify, close chrome" |

## Supported Command Combinations

### Time + Weather
- "what time is it and what's the weather"
- "what time is it and weather in new york"
- "what time is it then weather"
- **Tools:** get_time → get_weather_forecast

### Weather + Window
- "what's the weather and close chrome"
- "weather in london and minimize spotify"
- "what's the weather and fullscreen chrome"
- **Tools:** get_weather_forecast → close/minimize/maximize_window

### Time + Window
- "what time is it and open spotify"
- "what time is it and minimize chrome"
- **Tools:** get_time → open/close/minimize/maximize_window

### Window + Window
- "open spotify and close chrome"
- "minimize chrome and maximize notepad"
- "close discord and move chrome to monitor 2"
- **Tools:** open_target/close/minimize/maximize_window → move_window_to_monitor

### Three+ Commands
- "what time is it, weather in paris, and open spotify"
- "open chrome, minimize spotify, and move notepad to monitor 2"
- **Tools:** 3+ commands in sequence

### Sequential Execution
- "what time is it then weather"
- "open spotify then maximize it"
- **Execution mode:** Sequential (left-to-right)

## Test Coverage

### Test Cases Added: 16 new multi-intent tests

Test cases cover:
- Time + Weather combinations
- Weather + Window combinations
- Time + Window combinations
- Window + Window combinations
- Three or more commands
- Sequential vs parallel execution
- Verb inference for abbreviated commands
- Location preservation in weather queries

### Updated Existing Tests: 2 corrections

Fixed expectations for pre-existing tests that are now correctly handled by multi-intent parser:
- `"open spotify and play lofi"` - Now correctly routes to tool_plan (was incorrectly expecting LLM)
- `"what time is it and open spotify"` - Now correctly routes to tool_plan (was incorrectly expecting LLM)

## Confidence Scoring

Multi-intent queries use composite confidence calculation:

```
Composite Confidence = min(clause_confidences) * 0.95
```

- Minimum per-clause confidence: 0.7 (for it to parse)
- Minimum composite confidence: 0.75 (to execute without LLM fallback)
- Typical multi-intent confidence: 0.81 - 0.87

Example:
- `"open spotify and close chrome"`: min(0.90, 0.85) * 0.95 = 0.8075 ✓
- `"what time is it and weather in paris"`: min(0.95, 0.92) * 0.95 = 0.874 ✓

## Key Features

### Parameter Preservation
- Location extraction in weather queries works across multi-intent
- Window titles extracted correctly for all window operations
- Monitor destinations preserved in move_window_to_monitor

### Verb Inference
- "open spotify then maximize it" → Infers "maximize it" = "maximize spotify"
- Applies previous verb to subsequent noun-only clauses
- Prevents unnecessary LLM fallback for abbreviated commands

### Safety Filters
- Rejects ambiguous targets ("it", "this", "that")
- Detects conflicting action verbs and delegates to LLM
- Handles up to 7 commands per query (enforced limit)

### Execution Semantics
- **Parallel mode** ('and', ',') - Independent execution of each command
- **Sequential mode** ('then', ';') - Execution in order
- Both modes use same tool composition mechanism

## Integration Points

### With Hybrid Router
- Multi-intent parser leverages `_decide_single_clause()` for each clause
- Each clause must return confidence ≥ 0.7 to proceed
- All new time/weather/window patterns automatically supported

### With Orchestrator
- Multi-intent results returned as list of intents
- Orchestrator handles multi-intent routes as "tool_plan" with confidence scores
- Executes all composed intents in sequence

### Files Modified
- `wyzer/core/multi_intent_parser.py` - No changes (already supported)
- `wyzer/core/hybrid_router.py` - Added time, weather, and window patterns
- `scripts/test_hybrid_router.py` - Added 16 new test cases, fixed 2 existing tests

## Example Flows

### Example 1: "what time is it and weather in london"
```
Input: "what time is it and weather in london"
  ↓
Multi-intent detection: "and" separator found
  ↓
Split into clauses: ["what time is it", "weather in london"]
  ↓
Parse clause 1: get_time (confidence: 0.95)
Parse clause 2: get_weather_forecast with location="london" (confidence: 0.92)
  ↓
All ≥ 0.7 and composite ≥ 0.75? YES
  ↓
Return: [
  {"tool": "get_time", "args": {}, "continue_on_error": false},
  {"tool": "get_weather_forecast", "args": {"location": "london"}, "continue_on_error": false}
]
Confidence: 0.92 * 0.95 = 0.874
  ↓
Orchestrator executes both tools
```

### Example 2: "open chrome, minimize spotify, and move notepad to monitor 2"
```
Input: "open chrome, minimize spotify, and move notepad to monitor 2"
  ↓
Multi-intent detection: "and" with comma-separated list
  ↓
Split into clauses: ["open chrome", "minimize spotify", "move notepad to monitor 2"]
  ↓
Parse clause 1: open_target with query="chrome" (confidence: 0.90)
Parse clause 2: minimize_window with title="spotify" (confidence: 0.85)
Parse clause 3: move_window_to_monitor with title="notepad", monitor="2" (confidence: 0.85)
  ↓
All ≥ 0.7 and composite ≥ 0.75? YES
  ↓
Return: [
  {"tool": "open_target", "args": {"query": "chrome"}, "continue_on_error": false},
  {"tool": "minimize_window", "args": {"title": "spotify"}, "continue_on_error": false},
  {"tool": "move_window_to_monitor", "args": {"title": "notepad", "monitor": "2"}, "continue_on_error": false}
]
Confidence: 0.85 * 0.95 = 0.8075
  ↓
Orchestrator executes all three tools in sequence
```

## Performance Impact

- **Multi-intent detection:** O(n) where n = query length (regex search)
- **Clause parsing:** O(m) where m = number of clauses (typically 2-7)
- **Overall:** Negligible overhead compared to LLM inference (500-2000ms)
- **No LLM inference required** for multi-intent commands with valid patterns

## Implicit Multi-Intent (No Separators)

Commands without explicit separators are now detected via verb boundary detection:

```
Input: "close chrome open spotify"
  ↓
Verb boundary detection: 2 action verbs found at positions 0, 14
  ↓
Split by verb boundaries: ["close chrome", "open spotify"]
  ↓
Parse clause 1: close_window with title="chrome" (confidence: 0.85)
Parse clause 2: open_target with query="spotify" (confidence: 0.90)
  ↓
Return: [
  {"tool": "close_window", "args": {"title": "chrome"}, "continue_on_error": false},
  {"tool": "open_target", "args": {"query": "spotify"}, "continue_on_error": false}
]
Confidence: 0.85 * 0.95 = 0.8075
```

### Supported Action Verbs for Implicit Detection
`open`, `launch`, `start`, `close`, `quit`, `exit`, `minimize`, `shrink`, `maximize`, `fullscreen`, `expand`, `move`, `send`, `play`, `pause`, `resume`, `mute`, `unmute`, `scan`

## Backward Compatibility

- ✓ Single-intent commands unchanged
- ✓ All existing hybrid routing patterns preserved
- ✓ LLM fallback for unparseable combinations
- ✓ No breaking changes to tool interfaces
- ✓ Implicit multi-intent adds support, doesn't break existing
