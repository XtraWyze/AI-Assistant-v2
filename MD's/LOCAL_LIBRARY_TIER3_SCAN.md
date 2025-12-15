# Local Library Tier 3 Scan Implementation

## Overview
Updated the local library system to support tier 3 (full file system) scanning as the default behavior for "scan files" and "scan apps" commands, with tier 1 (normal) as the default for generic "refresh library" and "rebuild library" commands.

## Changes Made

### 1. wyzer/tools/local_library_refresh.py
**Updated tool schema and run method to accept mode parameter**

- Added `mode` parameter to `_args_schema` with enum values: `["normal", "full", "tier3"]`
- Modified `run()` method to extract and pass `mode` parameter to `refresh_index()`
- Defaults to "normal" mode if no mode is specified

```python
"properties": {
    "mode": {
        "type": "string",
        "enum": ["normal", "full", "tier3"],
        "description": "Scan mode: 'normal' (Start Menu only), 'full' (includes Tier 2 EXE scanning), 'tier3' (includes full file system scan)"
    }
}
```

### 2. wyzer/core/hybrid_router.py
**Added detection for library scan commands in the hybrid decision router**

- Added pattern matching for "scan files", "scan my files", "scan apps", "scan my apps" → tier 3 mode
- Added pattern matching for "refresh library", "rebuild library", "rescan library" → normal mode
- Confidence scores: 0.92 for tier 3 scans, 0.93 for normal refresh
- Patterns inserted after weather detection, before system storage commands

```python
# "scan files", "scan my files", "scan apps", "scan my apps" -> tier 3 (full file system scan)
if re.match(r"^scan\s+(?:my\s+)?(?:files|apps?)$", tl_norm):
    return HybridDecision(
        mode="tool_plan",
        intents=[{"tool": "local_library_refresh", "args": {"mode": "tier3"}, ...}],
        confidence=0.92,
    )

# "refresh library", "rebuild library" -> normal mode
if re.match(r"^(?:refresh|rebuild|rescan)\s+library$", tl_norm):
    return HybridDecision(
        mode="tool_plan",
        intents=[{"tool": "local_library_refresh", "args": {}, ...}],
        confidence=0.93,
    )
```

### 3. wyzer/core/orchestrator.py
**Added fastpath detection and tier 3 routing for scan commands**

- Added "scan" token to `_FASTPATH_COMMAND_TOKEN_RE` regex for fastpath recognition
- Added "scan files"/"scan apps" detection with tier 3 mode in `_fastpath_parse_clause()`
- Maintains backward compatibility with explicit "refresh library"/"rebuild library" commands

```python
# "scan files", "scan apps" -> tier 3 (full file system + apps scan)
if "scan files" in c_lower or "scan apps" in c_lower:
    return [Intent(tool="local_library_refresh", args={"mode": "tier3"})]
```

## Tier Levels Explained

**Tier 1 (Normal)** - Start Menu scanning
- Indexes applications from Windows Start Menu shortcuts
- Fastest option, suitable for regular refreshes
- Default for generic "refresh library" commands

**Tier 2 (Full)** - EXE scanning
- Includes Tier 1 plus application executable files from:
  - Program Files directories
  - User install locations
  - Common application folders
- Slower but more comprehensive

**Tier 3 (Full File System)** - Complete scan
- Includes Tier 2 plus entire C: drive file system scan
- Indexes all files with common extensions (exe, zip, doc, etc.)
- Slowest option but most comprehensive
- Default for "scan files" and "scan apps" commands

## Command Examples

### Tier 3 Scan (Full File System)
```
User: "Scan files"
User: "Scan my files"
User: "Scan apps"
User: "Scan my apps"
→ Executes with mode="tier3"
```

### Normal Refresh (Start Menu Only)
```
User: "Refresh library"
User: "Rebuild library"
User: "Rescan library"
→ Executes with mode="normal"
```

## Implementation Details

### Detection Priority
1. **Hybrid Router** (highest priority): Pattern matching for tier 3 scans
2. **Fastpath**: Fallback detection using command token regex
3. **LLM**: Final fallback if neither deterministic path matches

### Confidence Scores
- Tier 3 scan detection: 0.92 (high confidence, deterministic)
- Normal refresh detection: 0.93 (very high confidence, deterministic)
- Both bypass LLM entirely when matched

### Arguments Structure
- **Tier 3 scan**: `{"mode": "tier3"}`
- **Normal refresh**: `{}` (empty args, defaults to "normal" mode)

## Testing

Run the test suite to verify functionality:
```bash
python scripts/test_library_scan.py
```

Verify no regressions in existing tests:
```bash
python scripts/test_hybrid_router.py
```

## Files Modified
1. `wyzer/tools/local_library_refresh.py` - Tool schema and implementation
2. `wyzer/core/hybrid_router.py` - Hybrid decision routing
3. `wyzer/core/orchestrator.py` - Fastpath detection and fastpath clause parsing
4. `scripts/test_library_scan.py` - New test suite (created)

## Backward Compatibility
✓ All existing commands continue to work
✓ Generic "refresh library" commands unaffected
✓ New tier 3 scanning is opt-in via new command patterns
✓ All existing tests pass without modification
