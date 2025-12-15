# Wyzer Tool Chain

This document describes all available tools in the Wyzer AI Assistant toolchain. Each tool is a stateless function that returns JSON-serializable results.

**Related Documents:**
- [REPO_STRUCTURE.md](REPO_STRUCTURE.md) - Complete repository tree and module descriptions
- [SYSTEM_FLOW.md](SYSTEM_FLOW.md) - System flow diagrams and state machine
- [ARCHITECTURE_LOCK.md](ARCHITECTURE_LOCK.md) - Design constraints and principles

---

## Tool Architecture

All tools inherit from `ToolBase` and implement a `run(**kwargs)` method that:
- Takes keyword arguments specific to the tool
- Returns a JSON-serializable dictionary with results or errors
- Provides metadata via `name`, `description`, and `args_schema` properties

---

## System Information Tools

### get_time
**Module**: `wyzer/tools/get_time.py`  
**Purpose**: Get the current local time and date  
**Arguments**: None  
**Returns**: 
- `time`: Current time (HH:MM:SS)
- `date`: Current date (YYYY-MM-DD)
- `timezone`: Timezone identifier

**Use Cases**: "What time is it?", "Tell me the current time"

---

### get_system_info
**Module**: `wyzer/tools/get_system_info.py`  
**Purpose**: Get basic system information  
**Arguments**: None  
**Returns**:
- `os`: Operating system name
- `os_version`: Full OS version string
- `architecture`: CPU architecture (x86_64, ARM, etc.)
- `cpu_cores`: Number of CPU cores
- `ram_gb`: Total RAM in gigabytes
- `ram_available_gb`: Available RAM in gigabytes

**Use Cases**: "What are my computer specs?", "Check system information"

---

### get_location
**Module**: `wyzer/tools/get_location.py`  
**Purpose**: Get approximate current location via IP geolocation  
**Arguments**: None  
**Returns**:
- `ip`: Current IP address
- `city`: City name
- `region`: State/region name
- `country`: Country name
- `postal`: Postal/ZIP code
- `timezone`: Timezone identifier
- `latitude`: Approximate latitude
- `longitude`: Approximate longitude

**Requirements**: Internet connection  
**Accuracy**: IP-based geolocation is approximate  
**Use Cases**: "Where am I?", "What's my location?", "What city am I in?"

---

### monitor_info
**Module**: `wyzer/tools/monitor_info.py`  
**Purpose**: Get information about connected monitors  
**Arguments**: None  
**Returns**:
- `monitor_count`: Number of connected monitors
- `monitors`: List of monitor objects with:
  - `index`: Monitor number
  - `width`: Screen width in pixels
  - `height`: Screen height in pixels
  - `x`: X position
  - `y`: Y position
  - `work_width`: Available work area width
  - `work_height`: Available work area height
  - `is_primary`: Whether it's the primary monitor

**Use Cases**: "How many monitors do I have?", "What's my screen resolution?"

---

## Weather Tools

### get_weather_forecast
**Module**: `wyzer/tools/get_weather_forecast.py`  
**Purpose**: Get current weather and forecast using Open-Meteo API  
**Arguments**:
- `location` (optional): City name or coordinates (falls back to IP-based location if not provided)

**Returns**:
- `current_weather`: Current conditions with temperature, wind, precipitation, etc.
- `forecast`: 7-day forecast with hourly/daily data
- `location`: Detected location info
- `units`: Temperature/wind/precipitation units used

**Requirements**: Internet connection  
**API**: Uses Open-Meteo (no API key required)  
**Use Cases**: "What's the weather?", "Will it rain tomorrow?", "Weather forecast for New York"

---

## System Storage Tools

### system_storage_scan
**Module**: `wyzer/tools/system_storage.py`  
**Purpose**: Scan all mounted drives and create a cached index with detailed storage information  
**Arguments**:
- `refresh` (bool, optional): Force refresh (default False to use cache)

**Returns**:
- `status`: "ok"
- `refreshed`: Whether cache was refreshed
- `index_path`: Path to cache file
- `drives`: List of drive objects with:
  - `name`: Drive name (e.g., "C:", "/dev/sda1")
  - `mountpoint`: Mount path (e.g., "C:\\", "/mnt/storage")
  - `fstype`: Filesystem type (e.g., "NTFS", "ext4")
  - `total_gb`: Total capacity in gigabytes
  - `used_gb`: Used space in gigabytes
  - `free_gb`: Free space in gigabytes
  - `percent_used`: Usage percentage (0-100)
  - `is_removable`: Whether drive is removable (null if unknown)
  - `label`: Drive label/name (null if unknown)
- `latency_ms`: Time taken to scan

**Caching**: Results are cached in `wyzer/data/system_storage_index.json`  
**Platform**: Windows, Linux, macOS (with graceful fallbacks)  
**Dependencies**: psutil (optional; falls back to standard library)  
**Use Cases**: "System scan", "Scan my drives", "Refresh drive index"

---

### system_storage_list
**Module**: `wyzer/tools/system_storage.py`  
**Purpose**: Quickly list all mounted drives with free/total space  
**Arguments**:
- `refresh` (bool, optional): Force refresh (default False to use cache)
- `drive` (str, optional): Filter to specific drive (e.g., "D", "D:", "/mnt/storage")

**Returns**:
- `status`: "ok"
- `drives`: List of drive objects with:
  - `name`: Drive name (e.g., "C:")
  - `mountpoint`: Mount path (e.g., "C:\\")
  - `free_gb`: Free space in gigabytes
  - `total_gb`: Total capacity in gigabytes
  - `percent_used`: Usage percentage (0-100)

**Features**: Uses cache by default for fast responses; supports filtering by drive letter  
**Platform**: Windows, Linux, macOS  
**Use Cases**: "List drives", "How much space do I have?", "How much space does D drive have?", "Storage summary"

---

### system_storage_open
**Module**: `wyzer/tools/system_storage.py`  
**Purpose**: Open a drive or folder in the system file manager  
**Arguments**:
- `drive` (str, required): Drive or path to open (e.g., "D", "D:", "D:\\", "/mnt/storage")

**Returns**:
- `status`: "ok"
- `opened`: The mountpoint that was opened

**Platform Support**:
- Windows: Uses `os.startfile()`
- macOS: Uses `open` command
- Linux: Uses `xdg-open` command

**Error Handling**:
- `invalid_argument`: If drive token cannot be parsed
- `not_found`: If drive/path doesn't exist
- `platform_error`: If platform is unsupported

**Use Cases**: "Open D drive", "Open d:", "Open storage folder"

---

## Media & Audio Control Tools

### media_play_pause
**Module**: `wyzer/tools/media_controls.py`  
**Purpose**: Toggle play/pause for currently playing media  
**Arguments**: None  
**Returns**: Status of the operation

**Platform**: Windows  
**Use Cases**: "Play the music", "Pause the video"

---

### media_next_track
**Module**: `wyzer/tools/media_controls.py`  
**Purpose**: Skip to next media track  
**Arguments**: None  
**Returns**: Status of the operation

**Platform**: Windows  
**Use Cases**: "Skip to the next song", "Next track"

---

### media_previous_track
**Module**: `wyzer/tools/media_controls.py`  
**Purpose**: Go to previous media track  
**Arguments**: None  
**Returns**: Status of the operation

**Platform**: Windows  
**Use Cases**: "Go back to the previous song", "Last track"

---

### volume_control
**Module**: `wyzer/tools/volume_control.py`  
**Purpose**: Master and per-application volume control  
**Arguments**:
- `action`: "get_master", "set_master", "change_master", "mute_master", "unmute_master", "list", "get_app", "set_app", "change_app", "mute_app", "unmute_app"
- `volume` (optional): Target volume (0-100)
- `delta` (optional): Change amount for "change_*" actions
- `app` (optional): Application name for per-app volume (supports fuzzy matching)

**Returns**:
- For volume actions: `volume` (0-100), `is_muted` (boolean)
- For list action: List of audio sessions with volumes
- Supports fuzzy matching for app names (e.g., "spotify", "chrome", "discord")

**Platform**: Windows  
**Dependencies**: pycaw for audio control  
**Features**:
- Master volume control
- Per-application volume control
- Mute/unmute functionality
- Fuzzy matching for app names

**Use Cases**: 
- "Set volume to 50%"
- "Mute Spotify"
- "Turn up the volume"
- "Lower Discord volume by 10%"

---

### audio_output_device
**Module**: `wyzer/tools/audio_output_device.py`  
**Purpose**: List and switch audio output devices (speakers, headphones, etc.)  
**Arguments**:
- `action`: "list" (list devices) or "set" (change device)
- `device` (optional): Device name for "set" action (supports fuzzy matching)

**Returns**:
- For list action: List of available audio devices
- For set action: Confirmation of device switch
- Supports fuzzy matching for device names

**Platform**: Windows  
**Dependencies**: pycaw and Windows PolicyConfig COM interface  
**Use Cases**:
- "List audio devices"
- "Switch to headphones"
- "Change output to speakers"

---

## Window & Application Management Tools

### window_manager
**Module**: `wyzer/tools/window_manager.py`  
**Purpose**: Comprehensive window management operations  
**Arguments**:
- `action`: "list", "find", "focus", "minimize", "maximize", "restore", "close", "move", "resize"
- Various position/size parameters depending on action

**Returns**: List of windows or status of operation  

**Platform**: Windows  
**Features**:
- List all open windows
- Find windows by title or process
- Focus/activate windows
- Minimize/maximize/restore windows
- Close windows
- Move and resize windows
- Fuzzy matching for window titles
- Handle caching for reliable matching

**Use Cases**:
- "List all open windows"
- "Focus on the Notepad window"
- "Minimize all windows"
- "Close the browser"

---

### open_target
**Module**: `wyzer/tools/open_target.py`  
**Purpose**: Open folders, files, applications, or URLs based on natural language  
**Arguments**:
- `query`: Natural language query (e.g., "downloads", "chrome", "pictures")
- `open_mode` (optional): Force specific mode ("default", "folder", "file", "app")

**Returns**: Status and information about what was opened

**Features**:
- Smart resolution of natural language to targets
- Integration with local library (apps, folders, shortcuts)
- Fuzzy matching
- Multiple open modes

**Use Cases**:
- "Open downloads"
- "Open my documents"
- "Launch Chrome"
- "Open this file"

---

### open_website
**Module**: `wyzer/tools/open_website.py`  
**Purpose**: Open websites in the default browser  
**Arguments**:
- `url`: Website name or full URL

**Returns**: Status of the operation

**Features**:
- Automatic URL normalization (adds https:// if needed)
- Handles website names ("youtube") and full URLs ("https://example.com")

**Use Cases**:
- "Open YouTube"
- "Go to github.com"
- "Open Facebook"

---

## Library Management Tools

### local_library_refresh
**Module**: `wyzer/tools/local_library_refresh.py`  
**Purpose**: Refresh and rebuild the index of local apps, folders, and shortcuts  
**Arguments**: None  
**Returns**:
- `status`: Status of refresh operation
- `app_count`: Number of apps indexed
- `folder_count`: Number of folders indexed
- `shortcut_count`: Number of shortcuts indexed
- `latency_ms`: Time taken to refresh

**Use Cases**: 
- "Refresh the library"
- "Update app list"
- When new apps are installed and not yet indexed

---

## Tool Registry

**Location**: `wyzer/tools/registry.py`  
**Function**: `build_default_registry()`  
Returns a registry of all available tools with their configurations.

Tools are registered with:
- Unique name identifier
- Human-readable description
- JSON schema for arguments
- Implementation class

---

## Tool Execution Flow

1. LLM analyzes user request
2. Hybrid router determines if a tool is needed
3. Tool is selected from the registry
4. Arguments are validated against `args_schema`
5. Tool's `run()` method is executed
6. Results are returned as JSON
7. Response is formatted for user

---

## Error Handling

All tools follow a consistent error format:
```json
{
  "error": {
    "type": "error_type",
    "message": "Human-readable error message"
  }
}
```

Common error types:
- `invalid_argument`: Required argument missing or invalid
- `execution_error`: Error during tool execution
- `platform_error`: Platform-specific error (e.g., Windows-only tool on non-Windows)
- `network_error`: Network connectivity issue
- `permission_error`: Insufficient permissions

---

## Dependencies

### Core
- Python 3.x
- Standard library (urllib, json, subprocess, etc.)

### Optional/Per-Tool
- **pycaw**: Audio control (volume, device switching)
- **pywin32**: Enhanced Windows API access (optional, falls back to ctypes)
- **psutil**: System information (RAM details)

### External Services
- **Open-Meteo**: Weather and geocoding (free, no API key)

---

## Platform Support

- **Windows**: Full support for all tools
- **Linux/macOS**: Partial support (system info, time, location, web tools only)
  - Media controls, window management, audio output device tools are Windows-only

---

## Performance Notes

- **Window Handle Cache**: 300-second TTL for window handle caching
- **Location**: IP-based lookup with ~6-second timeout
- **Weather**: API call with ~6-second timeout
- **Volume Control**: Fuzzy matching optimized for responsiveness
