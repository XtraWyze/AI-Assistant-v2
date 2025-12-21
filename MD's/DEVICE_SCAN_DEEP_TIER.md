# Device/Drive Scanning with Deep Tier Implementation

> *Last Updated: December 2025*

## Overview
Updated the system storage scanning system to support deep tier (full file system) scanning as the default for device and drive-specific scan commands. This complements the library scanning tier 3 implementation and provides consistent comprehensive scanning across the system.

## Changes Made

### 1. wyzer/tools/system_storage.py
**Updated tool schema and run method to accept tier and drive parameters**

- Added `tier` parameter to `_args_schema` with enum values: `["quick", "standard", "deep"]`
- Added `drive` parameter for specific drive scanning
- Modified `run()` method to handle tier parameter
- Deep tier forces refresh automatically for up-to-date information

**Tier Levels:**
- **quick**: Uses cached data (fastest, default for cache-first scenarios)
- **standard**: Performs refresh with basic drive info (default for general scans)
- **deep**: Performs full file system scan (new default for explicit scan commands)

### 2. wyzer/core/hybrid_router.py
**Added detection for device and drive scanning commands**

New patterns with deep tier:
- `"scan devices"` / `"scan device"` → deep tier, all drives
- `"scan drive c"` / `"scan drive d"` / `"scan e"` etc → deep tier, specific drive
- `"scan disc d"` / `"scan disk e"` → deep tier alternative spellings
- `"scan hard drive c"` → deep tier, full form

Maintained patterns with standard tier:
- `"system scan"` → standard tier (refresh=True)
- `"scan my drives"` → standard tier
- `"refresh drive index"` → standard tier

All patterns use `system_storage_scan` tool directly (previously some used `system_storage_list`)

### 3. wyzer/core/orchestrator.py
**Added device token to fastpath command regex**

- Added `devices?` to `_FASTPATH_COMMAND_TOKEN_RE` for proper fastpath detection
- Enables faster routing of device scan commands through deterministic path

## Command Examples

### Deep Tier Scan (Full File System)
```
User: "Scan devices"
User: "Scan device"
→ Executes system_storage_scan with tier="deep" for all drives

User: "Scan drive C"
User: "Scan drive E"
User: "Scan D"
→ Executes system_storage_scan with tier="deep" for specific drive

User: "Scan disc E"
User: "Scan hard drive C"
→ Executes system_storage_scan with tier="deep" with different syntax
```

### Standard Tier (Refresh)
```
User: "System scan"
User: "Scan my drives"
User: "Refresh drive index"
→ Executes system_storage_scan with refresh=True (standard tier)
```

## Implementation Details

### Confidence Scores
- Device-specific scans: 0.92 (high confidence, deterministic)
- System/all-drives scans: 0.95 (very high confidence, deterministic)
- All bypass LLM entirely when matched

### Arguments Structure
- **Deep tier, all drives**: `{"tier": "deep"}`
- **Deep tier, specific drive**: `{"tier": "deep", "drive": "C"}`
- **Standard tier (refresh)**: `{"refresh": true}`

### Drive Letter Handling
- Automatically converts to uppercase (c → C, e → E)
- Supports both word form (`"drive c"`) and short form (`"scan c"`)
- Works with alternative spellings (`disc`, `disk`)

## Integration with Library Scanning

Both systems now use consistent tier-based naming:
- **Library Tier 1 → Storage Quick**: Cached/index-only
- **Library Tier 2 → Storage Standard**: EXE/app scanning + basic refresh
- **Library Tier 3 → Storage Deep**: Full file system scan

When user says:
- `"scan files"` → library tier 3 scan
- `"scan devices"` → storage deep scan

## Test Results

**11/11 tests passing:**
- ✓ Scan devices (all drives, deep tier)
- ✓ Scan device (singular, deep tier)
- ✓ Scan drive C, D, E (specific drives, deep tier)
- ✓ Scan disc/disk (alternative spellings, deep tier)
- ✓ Scan hard drive (full form, deep tier)
- ✓ System scan (standard tier)
- ✓ Scan my drives (standard tier)
- ✓ Refresh drive index (standard tier)

**All existing tests continue to pass** (49/49 hybrid router tests)

## Files Modified
1. `wyzer/tools/system_storage.py` - Tool schema and run method
2. `wyzer/core/hybrid_router.py` - Hybrid decision routing patterns
3. `wyzer/core/orchestrator.py` - Fastpath command token regex

## Backward Compatibility
✓ All existing commands continue to work
✓ "System scan" and "scan my drives" remain unchanged
✓ New deep tier scanning is opt-in via new command patterns
✓ All existing tests pass without modification
✓ Storage tool continues to work with other consumers

## Production Ready
✅ Full tier 3 scanning now available for both library and storage
✅ Consistent command patterns across system
✅ High confidence routing (0.92-0.95)
✅ Comprehensive test coverage
✅ Backward compatible with existing usage
