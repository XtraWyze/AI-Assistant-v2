# Wyzer Tools Guide

Complete reference for all available tools in Wyzer AI Assistant with user-end examples.

---

## Table of Contents

1. [System Information Tools](#system-information-tools)
2. [Time & Location Tools](#time--location-tools)
3. [Weather Tools](#weather-tools)
4. [Web & Navigation Tools](#web--navigation-tools)
5. [Media Control Tools](#media-control-tools)
6. [Volume & Audio Tools](#volume--audio-tools)
7. [Window Management Tools](#window-management-tools)
8. [Monitor Tools](#monitor-tools)
9. [System Storage Tools](#system-storage-tools)
10. [Library Management Tools](#library-management-tools)

---

## System Information Tools

### Get System Info

**Tool Name:** `get_system_info`

**Description:** Get basic system information about your computer (operating system, CPU cores, RAM, architecture)

**Parameters:** None

**Returns:**
- `os`: Operating system name (Windows, Linux, macOS)
- `os_version`: Full OS version string
- `architecture`: CPU architecture (x86_64, ARM, etc.)
- `cpu_cores`: Number of CPU cores
- `ram_gb`: Total RAM in gigabytes (if available)
- `ram_available_gb`: Available RAM in gigabytes (if available)

**User Examples:**
- "What are my system specs?"
- "How much RAM do I have?"
- "Tell me about my computer"
- "What OS am I running?"
- "How many cores does my CPU have?"

---

## Time & Location Tools

### Get Time

**Tool Name:** `get_time`

**Description:** Get the current local time and date

**Parameters:** None

**Returns:**
- `time`: Current time in HH:MM:SS format
- `date`: Current date in YYYY-MM-DD format
- `timezone`: Timezone identifier (local)

**User Examples:**
- "What time is it?"
- "What's the current time?"
- "Tell me the date"
- "What's today's date?"

### Get Location

**Tool Name:** `get_location`

**Description:** Get approximate geographical location based on your IP address. This is an approximate location and requires internet connection.

**Parameters:** None

**Returns:**
- `ip`: Your public IP address
- `city`: City name
- `region`: State/region name
- `country`: Country name
- `postal`: Postal/ZIP code
- `timezone`: Timezone identifier
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `approximate`: Boolean indicating accuracy is approximate
- `source`: Which geolocation provider was used

**User Examples:**
- "Where am I located?"
- "What's my approximate location?"
- "What city am I in?"
- "What's my timezone?"
- "What's my IP address?"

---

## Weather Tools

### Get Weather Forecast

**Tool Name:** `get_weather_forecast`

**Description:** Get current weather and 7-day forecast for a location. Uses Open-Meteo API (no API key needed). Can accept location name or use your current IP location.

**Parameters:**
- `location` (optional): Location name (e.g., "New York", "London", "Paris") - if not provided, uses your IP location
- `days` (optional): Number of forecast days to include (1-7, default 7)

**Returns:**
- `location`: Location info (name, country, timezone)
- `current`: Current weather conditions
  - `temperature`: Temperature in Celsius
  - `weather`: Weather description
  - `humidity`: Humidity percentage
  - `wind_speed`: Wind speed in km/h
  - `weather_code`: WMO code for weather type
- `forecast`: Daily forecast array with:
  - `date`: Date in YYYY-MM-DD format
  - `temperature_max`: High temperature
  - `temperature_min`: Low temperature
  - `weather`: Weather description
  - `precipitation`: Expected precipitation in mm
  - `humidity`: Humidity percentage

**User Examples:**
- "What's the weather like?"
- "What's the weather in New York?"
- "What's the forecast for London?"
- "Will it rain tomorrow in Paris?"
- "Show me the 7-day forecast"
- "What's the weather like in Tokyo?"
- "Is it going to be hot in Miami?"

---

## Web & Navigation Tools

### Open Website

**Tool Name:** `open_website`

**Description:** Open a website in your default web browser. Accepts website names, domains, or full URLs.

**Parameters:**
- `url` (required): Website name or full URL
  - Simple names: "facebook", "youtube", "amazon"
  - Domains: "youtube.com", "reddit.com", "github.com"
  - Full URLs: "https://example.com", "https://example.com/path"

**Returns:**
- `status`: "opened" if successful
- `url`: The full URL that was opened
- `error`: Error details if operation failed

**User Examples:**
- "Open YouTube"
- "Open Facebook"
- "Go to Reddit"
- "Open GitHub"
- "Open google.com"
- "Open https://stackoverflow.com"
- "Open Amazon"
- "Open Wikipedia"
- "Go to Netflix"

### Open Target

**Tool Name:** `open_target`

**Description:** Open a folder, file, app, or URL based on natural language query. Uses the local library to resolve targets intelligently.

**Parameters:**
- `query` (required): Natural language description of what to open
  - Common folders: "Downloads", "Documents", "Pictures", "Desktop"
  - Applications: "Chrome", "Notepad", "Spotify", "Discord"
  - Phrases: "My documents", "Downloads folder", "Program files"
- `open_mode` (optional): Force specific open mode - "default", "folder", "file", or "app"

**Returns:**
- `status`: "opened" if successful
- `resolved`: Resolved target info (path, type, confidence)
- `error`: Error details if operation failed
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Open Downloads"
- "Open my documents"
- "Open Notepad"
- "Open Spotify"
- "Open Chrome"
- "Show me my Pictures"
- "Open Desktop"
- "Open Discord"
- "Open Program Files"
- "Open Visual Studio Code"

---

## Media Control Tools

### Media Play/Pause

**Tool Name:** `media_play_pause`

**Description:** Toggle play/pause for currently playing media (music, video, etc.)

**Parameters:** None

**Returns:**
- `status`: "ok" if successful
- `action`: "play_pause"
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Play the music"
- "Pause the music"
- "Toggle play"
- "Play/pause"
- "Stop the video"
- "Resume playback"

### Media Next Track

**Tool Name:** `media_next`

**Description:** Skip to the next track in currently playing media

**Parameters:** None

**Returns:**
- `status`: "ok" if successful
- `action`: "next"
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Next track"
- "Skip"
- "Play next song"
- "Next video"
- "Skip this song"

### Media Previous Track

**Tool Name:** `media_previous`

**Description:** Go back to the previous track in currently playing media

**Parameters:** None

**Returns:**
- `status`: "ok" if successful
- `action`: "previous"
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Previous track"
- "Go back"
- "Play previous song"
- "Last track"
- "Rewind to previous"

---

## Volume & Audio Tools

### Volume Control (Advanced)

**Tool Name:** `volume_control`

**Description:** Advanced volume control for system master volume or specific applications. Supports fuzzy matching for app names.

**Parameters:**
- `action` (required): "get", "set", "change", "mute", or "unmute"
- `volume` (optional): Volume level 0-100 (required for "set" action)
- `change_amount` (optional): Amount to increase/decrease by (required for "change" action)
- `app` (optional): Target application name for per-app control. Omit for master volume.
  - Examples: "Spotify", "Chrome", "Discord", "Discord Web", "YouTube"

**Returns:**
- `status`: "ok" if successful
- `action`: The performed action
- `volume`: Current volume level
- `is_muted`: Boolean indicating mute state
- `app`: Target app if per-app control
- `available_apps`: List of available apps for per-app control

**User Examples:**
- "Set volume to 50"
- "Increase volume"
- "Decrease volume by 20"
- "Mute the system"
- "Unmute"
- "Set Spotify volume to 75"
- "Mute Chrome"
- "What's the current volume?"
- "Make Discord louder"
- "Reduce Discord volume by 10"
- "Get the volume level"

### Volume Up

**Tool Name:** `volume_up`

**Description:** Increase system master volume by one step

**Parameters:** None

**Returns:**
- `status`: "ok" if successful
- `action`: "volume_up"
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Turn up the volume"
- "Volume up"
- "Louder"
- "Increase volume"

### Volume Down

**Tool Name:** `volume_down`

**Description:** Decrease system master volume by one step

**Parameters:** None

**Returns:**
- `status`: "ok" if successful
- `action`: "volume_down"
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Turn down the volume"
- "Volume down"
- "Quieter"
- "Decrease volume"

### Volume Mute Toggle

**Tool Name:** `volume_mute_toggle`

**Description:** Toggle system audio mute on/off

**Parameters:** None

**Returns:**
- `status`: "ok" if successful
- `action`: "mute_toggle"
- `is_muted`: Current mute state
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Mute"
- "Unmute"
- "Toggle mute"
- "Silence the system"
- "Unmute the audio"

### Audio Output Device

**Tool Name:** `audio_output_device`

**Description:** Switch system default audio output device (speakers, headphones, etc.). Uses fuzzy matching for device names.

**Parameters:**
- `action` (required): "list" or "switch"
- `device` (optional): Device name to switch to (required for "switch" action)
  - Examples: "Speakers", "Headphones", "USB Headset", "Monitor"

**Returns:**
- For "list" action:
  - `devices`: List of available audio devices
  - `current_device`: Currently selected device
- For "switch" action:
  - `status`: "ok" if successful
  - `device`: Device switched to
  - `previous_device`: Previously selected device

**User Examples:**
- "List audio devices"
- "What audio devices are available?"
- "Switch to headphones"
- "Switch to speakers"
- "Use USB headset"
- "Change to monitor speakers"
- "Which audio device is active?"

---

## Window Management Tools

### Focus Window

**Tool Name:** `focus_window`

**Description:** Bring a window to focus by application name or window title

**Parameters:**
- `target` (required): Application name or window title
  - Examples: "Chrome", "Notepad", "Spotify", "Discord"

**Returns:**
- `status`: "ok" if successful
- `target`: The focused window
- `hwnd`: Window handle
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Focus Chrome"
- "Switch to Spotify"
- "Bring Discord to front"
- "Focus Notepad"
- "Switch to Firefox"
- "Open Discord window"

### Minimize Window

**Tool Name:** `minimize_window`

**Description:** Minimize a window by application name or window title

**Parameters:**
- `target` (optional): Application name or window title. If omitted, minimizes the active window.

**Returns:**
- `status`: "ok" if successful
- `target`: The minimized window
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Minimize Chrome"
- "Minimize this window"
- "Hide Spotify"
- "Minimize the current window"
- "Put Discord away"

### Maximize Window

**Tool Name:** `maximize_window`

**Description:** Maximize a window by application name or window title

**Parameters:**
- `target` (optional): Application name or window title. If omitted, maximizes the active window.

**Returns:**
- `status`: "ok" if successful
- `target`: The maximized window
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Maximize Chrome"
- "Make this window bigger"
- "Expand Firefox"
- "Maximize the current window"

### Close Window

**Tool Name:** `close_window`

**Description:** Close a window by application name or window title

**Parameters:**
- `target` (required): Application name or window title

**Returns:**
- `status`: "ok" if successful
- `target`: The closed window
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Close Chrome"
- "Close this window"
- "Close Notepad"
- "Quit Discord"
- "Exit Firefox"
- "Close the current window"

### Move Window to Monitor

**Tool Name:** `move_window_to_monitor`

**Description:** Move a window to a specific monitor (for multi-monitor setups)

**Parameters:**
- `target` (required): Application name or window title
- `monitor` (optional): Monitor index (0 for first, 1 for second, etc.) or description (e.g., "left", "right")

**Returns:**
- `status`: "ok" if successful
- `target`: The moved window
- `monitor_index`: Target monitor index
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "Move Chrome to second monitor"
- "Move Spotify to the left monitor"
- "Send Discord to monitor 2"
- "Move this window to monitor 1"
- "Move Firefox to the right"

---

## Monitor Tools

### Monitor Info

**Tool Name:** `monitor_info`

**Description:** Get information about all connected monitors (count, resolution, position)

**Parameters:** None

**Returns:**
- `monitors`: Array of monitor objects with:
  - `index`: Monitor number
  - `x`: X coordinate (pixel position)
  - `y`: Y coordinate (pixel position)
  - `width`: Monitor width in pixels
  - `height`: Monitor height in pixels
  - `primary`: Boolean indicating if primary monitor
- `count`: Number of connected monitors
- `latency_ms`: Execution time in milliseconds

**User Examples:**
- "How many monitors do I have?"
- "Tell me about my monitors"
- "What's my screen resolution?"
- "List my displays"
- "How many displays are connected?"
- "What are my monitor specs?"

---

## System Storage Tools

### System Storage Scan (Deep)

**Tool Name:** `system_storage_scan`

**Description:** Deep scan of system drives with large folder detection and caching. Results are cached to avoid repeated scanning.

**Parameters:**
- `drive` (optional): Specific drive to scan (e.g., "C", "D", "C:\\", "D:\\")
  - If omitted, scans all drives
- `include_details` (optional): Include detailed folder breakdown (default true)
- `skip_cache` (optional): Force fresh scan, don't use cache (default false)

**Returns:**
- `drives`: Array of drive information:
  - `mountpoint`: Drive letter or mount path
  - `total_bytes`: Total drive capacity in bytes
  - `used_bytes`: Used space in bytes
  - `free_bytes`: Free space in bytes
  - `percent_used`: Percentage of drive used
  - `large_folders`: Array of largest folders with size and path
- `scan_time_ms`: Scan duration in milliseconds
- `cached`: Boolean indicating if results were from cache

**User Examples:**
- "How much disk space do I have?"
- "Show me my storage"
- "What's using the most space on my computer?"
- "Scan my C drive"
- "How full is my D drive?"
- "Show me the largest folders on my computer"
- "Do I have enough disk space?"
- "What's taking up space on my drives?"

### System Storage List (Quick)

**Tool Name:** `system_storage_list`

**Description:** Quick listing of all system drives without deep scanning

**Parameters:** None

**Returns:**
- `drives`: Array of drive information:
  - `mountpoint`: Drive letter or mount path
  - `total_bytes`: Total capacity in bytes
  - `used_bytes`: Used space in bytes
  - `free_bytes`: Free space in bytes
  - `percent_used`: Percentage used

**User Examples:**
- "List my drives"
- "Show all my drives"
- "What drives do I have?"

### System Storage Open

**Tool Name:** `system_storage_open`

**Description:** Open a drive in the file manager

**Parameters:**
- `drive` (required): Drive to open ("C", "D", "C:\\", etc.)

**Returns:**
- `status`: "opened" if successful
- `drive`: The drive that was opened
- `path`: The full path opened

**User Examples:**
- "Open C drive"
- "Open D drive"
- "Show me the D drive"
- "Open my files on C"

---

## Library Management Tools

### Local Library Refresh

**Tool Name:** `local_library_refresh`

**Description:** Refresh and rebuild the index of local folders, applications, and shortcuts. Useful when you want to update the search index to find new apps or recently installed programs.

**Parameters:**
- `mode` (optional): Scanning depth level
  - `"normal"`: Fast scan of Start Menu only
  - `"full"`: Medium scan including EXE locations
  - `"tier3"`: Deep scan of entire file system (slow but comprehensive)
  - Default: "normal"

**Returns:**
- `status`: "completed" if successful
- `indexed_count`: Number of items indexed
- `scan_time_ms`: Time taken to refresh
- `error`: Error details if operation failed

**User Examples:**
- "Refresh my library"
- "Update the library"
- "Rebuild the index"
- "Do a full library scan"
- "Deep scan the library"
- "Update the app list"
- "Refresh with a deep scan"

---

## Summary

All tools are designed to be controlled using natural language commands. The assistant will interpret your voice or text input and invoke the appropriate tool with the correct parameters.

### Quick Reference: Most Common Commands

**System & Info:**
- "What time is it?"
- "What's the weather?"
- "Tell me about my computer"

**Navigation & Web:**
- "Open Chrome"
- "Go to YouTube"
- "Open my Documents"

**Media & Volume:**
- "Play the music"
- "Skip to the next track"
- "Set volume to 50"

**Windows & Displays:**
- "Focus Spotify"
- "Maximize this window"
- "How many monitors do I have?"

**Storage:**
- "How much disk space do I have?"
- "What's using the most space?"
- "Open the D drive"
