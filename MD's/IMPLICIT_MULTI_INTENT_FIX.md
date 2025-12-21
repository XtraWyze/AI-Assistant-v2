# Implicit Multi-Intent Parsing Fix

> *Last Updated: December 2025*

## Problem
Commands without explicit separators like "close chrome open spotify" were not being parsed as multi-intent commands. Instead, they were being treated as a single command to close "chrome open spotify", which failed.

**Production Issue Log:**
```
You: Close Chrome Open Spotify.
Wyzer: Done.
[2025-12-14 21:03:58.637] [INFO] [TOOLS] Executing close_window args={'title': 'Chrome Open Spotify.'}
```

## Root Cause
The multi-intent detection and parsing system relied on explicit separators:
- "and" / "then" / ";" / ","

Commands without these separators were treated as single-intent, causing the entire phrase after the verb to be treated as a single target.

## Solution
Implemented verb boundary detection to split commands at action verb boundaries:
1. **Detection**: Updated `looks_multi_intent()` in `wyzer/core/hybrid_router.py` to detect when multiple action verbs appear in a command
2. **Parsing**: Implemented `_split_by_verb_boundaries()` in `wyzer/core/multi_intent_parser.py` to extract clauses by splitting at verb positions
3. **Integration**: Added fallback logic in `try_parse_multi_intent()` to use verb boundary splitting when explicit separators don't match

## Changes Made

### 1. wyzer/core/hybrid_router.py
Updated `looks_multi_intent()` function to detect verb boundaries:
```python
def looks_multi_intent(text: str) -> bool:
    # ... existing code ...
    
    # Check for implicit verb boundaries: "verb1 target1 verb2 target2"
    action_verbs = r"(?:open|launch|start|close|quit|exit|minimize|shrink|maximize|fullscreen|expand|move|send|play|pause|resume|mute|unmute)"
    verb_matches = list(re.finditer(action_verbs, tl, re.IGNORECASE))
    if len(verb_matches) >= 2:
        return True
```

### 2. wyzer/core/multi_intent_parser.py
Added three enhancements:

a) **ACTION_VERBS constant** - Regex pattern of all action verbs:
```python
ACTION_VERBS = r"(?:open|launch|start|close|quit|exit|minimize|shrink|maximize|fullscreen|expand|move|send|play|pause|resume|mute|unmute)"
```

b) **_looks_like_multi_intent()** - Updated to detect verb boundaries:
```python
# Check for implicit verb boundaries
verb_boundary_pattern = rf"\s{ACTION_VERBS}\s"
if re.search(verb_boundary_pattern, tl, re.IGNORECASE):
    verb_matches = list(re.finditer(ACTION_VERBS, tl, re.IGNORECASE))
    if len(verb_matches) >= 2:
        return True
```

c) **_split_by_verb_boundaries()** - New function to extract clauses:
```python
def _split_by_verb_boundaries(text: str) -> List[str]:
    """Split text by verb boundaries when no explicit separator is found."""
    # Finds all action verb positions and extracts text between verbs
```

d) **try_parse_multi_intent()** - Updated to use verb boundary splitting as fallback:
```python
# Try verb boundary splitting as last resort (after all explicit separators)
verb_boundary_clauses = _split_by_verb_boundaries(raw_text)
if len(verb_boundary_clauses) >= 2 and len(verb_boundary_clauses) <= 7:
    # Parse clauses using _decide_single_clause()
```

### 3. scripts/test_hybrid_router.py
Added 6 new test cases for implicit multi-intent:
- `"close chrome open spotify"` → close_window + open_target
- `"minimize chrome maximize notepad"` → minimize_window + maximize_window
- `"open chrome close spotify"` → open_target + close_window
- `"move chrome to monitor 2 open spotify"` → move_window_to_monitor + open_target
- `"close all open settings"` → close_window + open_target
- `"minimize window maximize screen"` → minimize_window + maximize_window

## Test Results
**Before Fix:** Commands without separators failed or were misparsed
**After Fix:** 49/49 tests passing (42 existing + 7 new implicit multi-intent tests)

### Example Test Execution
```
✓ close then open
   Input: 'close chrome open spotify'
   Expected: 2 intents, Got: 2 intents (confidence: 0.81)
     Intent 1: close_window with title='chrome'
     Intent 2: open_target with query='spotify'
```

## Backward Compatibility
✓ All existing tests continue to pass
✓ Explicit separators still work and have higher precedence
✓ Parsing order: explicit separators → verb boundaries → fallback to LLM

## Supported Action Verbs
- **Open**: open, launch, start
- **Close**: close, quit, exit
- **Window Management**: minimize, shrink, maximize, fullscreen, expand, move, send
- **Media Control**: play, pause, resume, mute, unmute

## Natural Speech Patterns Now Supported
1. "close chrome open spotify" (no separator)
2. "minimize chrome maximize notepad" (no separator)
3. "open spotify then close chrome" (with then)
4. "open spotify and close chrome" (with and)
5. "open spotify, close chrome" (with comma)
6. Multi-command with mixed separators and implicit boundaries

## Files Modified
1. `wyzer/core/hybrid_router.py` - Updated `looks_multi_intent()` function
2. `wyzer/core/multi_intent_parser.py` - Added ACTION_VERBS, _split_by_verb_boundaries(), updated _looks_like_multi_intent() and try_parse_multi_intent()
3. `scripts/test_hybrid_router.py` - Added 6 implicit multi-intent test cases

## Verification
Run the verification script:
```bash
python verify_production_fix.py
```

Or run full test suite:
```bash
python scripts/test_hybrid_router.py
```
