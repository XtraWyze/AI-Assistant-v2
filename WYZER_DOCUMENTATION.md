# Wyzer AI Assistant - Complete Documentation

> **A fully local, privacy-focused voice assistant for Windows**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [How to Use](#how-to-use)
5. [Configuration Options](#configuration-options)
6. [State Machine](#state-machine)
7. [Tools Reference](#tools-reference)
8. [Local Library System](#local-library-system)
9. [Voice Components](#voice-components)
10. [LLM Integration](#llm-integration)
11. [Multi-Intent Commands](#multi-intent-commands)
12. [Hybrid Router](#hybrid-router)
13. [Troubleshooting](#troubleshooting)

---

## Overview

Wyzer is a **local voice assistant** that runs entirely on your Windows machine. It provides:

- **Hotword Detection**: Wake the assistant with "Hey Wyzer" or just "Wyzer"
- **Speech-to-Text (STT)**: Uses Whisper for accurate transcription
- **Local LLM Processing**: Ollama integration for natural language understanding
- **Text-to-Speech (TTS)**: Piper TTS for natural voice responses
- **Tool Execution**: Control your system, manage windows, play media, and more
- **Multi-Intent Commands**: Execute multiple actions in a single request

### Key Principles

- **Privacy First**: All processing happens locally - no cloud services required
- **Modular Design**: Each component (STT, LLM, TTS, Tools) is independently configurable
- **Multiprocess Architecture**: Responsive UI with heavy processing offloaded to worker processes

---

## Architecture

Wyzer uses a **multiprocess architecture** (Phase 6) for optimal responsiveness:

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESS A (Realtime Core)                │
│  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Mic      │→→│  VAD    │→→│ Hotword  │→→│   State     │  │
│  │ Stream   │  │Detector │  │ Detector │  │  Machine    │  │
│  └──────────┘  └─────────┘  └──────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓ IPC Queues ↓
┌─────────────────────────────────────────────────────────────┐
│                  PROCESS B (Brain Worker)                   │
│  ┌─────────┐  ┌────────────┐  ┌───────┐  ┌─────────────┐   │
│  │   STT   │→→│Orchestrator│→→│  LLM  │→→│    TTS      │   │
│  │(Whisper)│  │   + Tools  │  │(Ollama│  │  (Piper)    │   │
│  └─────────┘  └────────────┘  └───────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   TOOL WORKER POOL                          │
│     [Worker 1] [Worker 2] [Worker 3] ... [Worker N]        │
│     (Isolated processes for parallel tool execution)        │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **Assistant** | `wyzer/core/assistant.py` | Audio + state management only |
| **Orchestrator** | `wyzer/core/orchestrator.py` | Intent routing + tool execution |
| **Brain Worker** | `wyzer/core/brain_worker.py` | STT, LLM, Tools, TTS processing |
| **Hybrid Router** | `wyzer/core/hybrid_router.py` | Fast-path deterministic command routing |
| **Tools** | `wyzer/tools/` | Stateless, JSON-only tool implementations |
| **Local Library** | `wyzer/local_library/` | App/folder indexing + resolution |

---

## Installation & Setup

### Prerequisites

1. **Python 3.10+** (tested with Python 3.10-3.11)
2. **Ollama** (for LLM features) - [Download Ollama](https://ollama.ai)
3. **FFmpeg** (for audio processing) - Add to PATH
4. **Piper TTS** executable and voice model

### Step 1: Clone/Setup the Project

```bash
cd C:\Users\kolyw\Desktop\AI-Assistant-v2
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv_new
.\venv_new\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama Model

```bash
ollama pull llama3.1:latest
```

### Step 5: Verify Piper Setup

Ensure these files exist:
- `wyzer/assets/piper/piper.exe`
- `wyzer/assets/piper/en_US-lessac-medium.onnx`

### Step 6: Run Wyzer

```bash
python run.py
```

Or use the batch file:
```bash
run.bat
```

---

## How to Use

### Basic Usage

1. **Start the assistant**: `python run.py`
2. **Say the wake word**: "Hey Wyzer" or "Wyzer"
3. **Wait for the listening prompt** (the assistant is now listening)
4. **Speak your command** (e.g., "What time is it?")
5. **Listen to the response**

### Command Examples

| Category | Example Commands |
|----------|------------------|
| **Time/Date** | "What time is it?", "What's the date?" |
| **Weather** | "What's the weather?", "Is it going to rain?" |
| **Apps** | "Open Chrome", "Launch Spotify", "Close Notepad" |
| **Folders** | "Open Downloads", "Show my Documents" |
| **Windows** | "Minimize this", "Maximize Chrome", "Focus Discord" |
| **Media** | "Pause music", "Next track", "Volume up" |
| **Volume** | "Set volume to 50%", "Mute Spotify", "Turn down Chrome" |
| **Timers** | "Set a timer for 5 minutes", "Cancel the timer" |
| **System** | "What's my system info?", "Show monitor info" |
| **Multi-Intent** | "Open Chrome and then pause music" |

### Follow-Up Mode

After Wyzer responds, you have a ~3 second window to ask a follow-up question **without** saying the wake word again. Just speak naturally!

To end follow-up mode, say:
- "No"
- "That's all"
- "Stop"
- "Never mind"

### Barge-In (Interruption)

You can interrupt Wyzer while it's speaking by saying the wake word. This is called "barge-in" and immediately stops the current response.

---

## Configuration Options

### Command Line Arguments

```bash
python run.py [OPTIONS]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--no-hotword` | Disable hotword (immediate listening) | Off |
| `--model` | Whisper model size (tiny/base/small/medium/large) | `small` |
| `--whisper-device` | Device for Whisper (cpu/cuda) | `cpu` |
| `--device` | Audio input device index | Auto |
| `--list-devices` | List audio devices and exit | - |
| `--no-ollama` | Run without Ollama (tools-only mode) | Off |
| `--llm` | LLM mode (ollama/off) | `ollama` |
| `--ollama-model` | Ollama model name | `llama3.1:latest` |
| `--ollama-url` | Ollama API URL | `http://127.0.0.1:11434` |
| `--llm-timeout` | LLM request timeout (seconds) | `30` |
| `--tts` | Enable TTS (on/off) | `on` |
| `--tts-device` | Audio output device for TTS | Auto |
| `--no-speak-interrupt` | Disable barge-in | Off |
| `--quiet` | Hide debug info (cleaner output) | Off |
| `--single-process` | Run in single-process mode (debugging) | Off |
| `--log-level` | Logging level (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `--profile` | Performance profile (low_end/normal) | `normal` |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WYZER_NO_OLLAMA` | Disable Ollama entirely | `false` |
| `WYZER_QUIET_MODE` | Enable quiet mode | `false` |
| `WYZER_SAMPLE_RATE` | Audio sample rate | `16000` |
| `WYZER_VAD_THRESHOLD` | VAD sensitivity (0.0-1.0) | `0.5` |
| `WYZER_VAD_SILENCE_TIMEOUT` | Silence timeout (seconds) | `0.8` |
| `WYZER_MAX_RECORD_SECONDS` | Max recording duration | `10.0` |
| `WYZER_HOTWORD_THRESHOLD` | Hotword sensitivity | `0.5` |
| `WYZER_HOTWORD_COOLDOWN_SEC` | Cooldown between hotwords | `1.5` |
| `WYZER_WHISPER_MODEL` | Default Whisper model | `small` |
| `WYZER_OLLAMA_MODEL` | Default Ollama model | `llama3.1:latest` |
| `WYZER_OLLAMA_URL` | Ollama API URL | `http://127.0.0.1:11434` |
| `WYZER_OLLAMA_STREAM` | Enable streaming responses | `true` |
| `WYZER_OLLAMA_TEMPERATURE` | LLM temperature | `0.4` |
| `WYZER_TTS_ENABLED` | Enable TTS | `true` |
| `WYZER_FOLLOWUP_ENABLED` | Enable follow-up mode | `true` |
| `WYZER_FOLLOWUP_TIMEOUT_SEC` | Follow-up timeout | `3.0` |

---

## State Machine

Wyzer operates as a state machine with the following states:

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│   ┌──────┐  hotword   ┌───────────┐  speech   ┌────────┐ │
│   │ IDLE │──────────→│ LISTENING │─────────→│TRANSCRIBING│
│   └──────┘            └───────────┘          └────────┘  │
│      ↑                     ↑                      │      │
│      │                     │                      ↓      │
│      │    ┌──────────┐     │              ┌──────────┐   │
│      │    │ FOLLOWUP │←────┴──────────────│ THINKING │   │
│      │    └──────────┘                    └──────────┘   │
│      │         │                                │        │
│      │         ↓ timeout                        ↓        │
│      │    ┌──────────┐                   ┌──────────┐    │
│      └────│          │←──────────────────│ SPEAKING │    │
│           └──────────┘                   └──────────┘    │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

| State | Description |
|-------|-------------|
| **IDLE** | Waiting for hotword |
| **HOTWORD_DETECTED** | Hotword just triggered |
| **LISTENING** | Recording user speech |
| **TRANSCRIBING** | Converting speech to text |
| **THINKING** | Processing with LLM/tools |
| **SPEAKING** | Playing TTS response |
| **FOLLOWUP** | Listening for follow-up (no hotword needed) |

---

## Tools Reference

Wyzer includes a comprehensive set of tools for system control:

### Time & Date

| Tool | Description | Example |
|------|-------------|---------|
| `get_time` | Get current local time | "What time is it?" |

### Location & Weather

| Tool | Description | Example |
|------|-------------|---------|
| `get_location` | Get approximate location (IP-based) | "Where am I?" |
| `get_weather_forecast` | Get weather for a location | "What's the weather in Seattle?" |

### System Information

| Tool | Description | Example |
|------|-------------|---------|
| `get_system_info` | Get CPU, RAM, disk info | "What's my system info?" |
| `monitor_info` | Get display/monitor details | "Show monitor info" |

### App & File Opening

| Tool | Description | Example |
|------|-------------|---------|
| `open_target` | Open apps, folders, files, URLs | "Open Chrome", "Open Downloads" |
| `open_website` | Open a website in browser | "Open youtube.com" |
| `local_library_refresh` | Re-scan installed apps | "Refresh the library" |

### Window Management

| Tool | Description | Example |
|------|-------------|---------|
| `focus_window` | Bring window to front | "Focus Discord" |
| `minimize_window` | Minimize a window | "Minimize Chrome" |
| `maximize_window` | Maximize a window | "Maximize Spotify" |
| `close_window` | Close a window | "Close Notepad" |
| `move_window_to_monitor` | Move window to another display | "Move Chrome to monitor 2" |
| `get_window_monitor` | Get which monitor a window is on | "What monitor is Chrome on?" |

### Media Controls

| Tool | Description | Example |
|------|-------------|---------|
| `media_play_pause` | Toggle play/pause | "Pause music", "Resume" |
| `media_next` | Skip to next track | "Next track" |
| `media_previous` | Go to previous track | "Previous song" |
| `get_now_playing` | Get current media info | "What's playing?" |

### Volume Control

| Tool | Description | Example |
|------|-------------|---------|
| `volume_control` | Master/app volume control | "Volume 50%", "Mute Spotify" |
| `volume_up` | Increase volume | "Volume up" |
| `volume_down` | Decrease volume | "Turn it down" |
| `volume_mute_toggle` | Toggle mute | "Mute", "Unmute" |

### Audio Devices

| Tool | Description | Example |
|------|-------------|---------|
| `set_audio_output_device` | Switch audio output | "Switch audio to headphones" |

### Storage & Drives

| Tool | Description | Example |
|------|-------------|---------|
| `system_storage_scan` | Scan drives for info | "Scan my drives" |
| `system_storage_list` | List available drives | "List my drives" |
| `system_storage_open` | Open a drive in Explorer | "Open D drive" |

### Timers

| Tool | Description | Example |
|------|-------------|---------|
| `timer` | Set/cancel/check timer | "Set timer for 5 minutes", "Cancel timer" |

---

## Local Library System

The Local Library indexes your system to enable natural language app/folder opening.

### What Gets Indexed

1. **Common Folders**: Desktop, Downloads, Documents, Pictures, Videos, Music
2. **Installed Apps**: Start Menu shortcuts
3. **User Aliases**: Custom shortcuts defined in `aliases.json`
4. **Tier 2 Apps** (full scan): EXEs from Program Files
5. **Tier 3 Files** (deep scan): Full file system scan
6. **Games**: Steam, Epic Games, Xbox games

### Location

- **Index File**: `wyzer/local_library/library.json`
- **Aliases File**: `wyzer/local_library/aliases.json`

### Custom Aliases

Create `wyzer/local_library/aliases.json` to define custom shortcuts:

```json
{
  "vscode": "C:\\Users\\YourName\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe",
  "work folder": "D:\\Work\\Projects",
  "my game": "C:\\Games\\MyGame\\game.exe"
}
```

### Refresh Commands

- **Normal refresh**: "Refresh the library"
- **Full scan (Tier 2)**: Includes Program Files EXEs
- **Deep scan (Tier 3)**: Full file system scan (slow)

---

## Voice Components

### Speech-to-Text (STT)

Uses **Faster-Whisper** for speech recognition.

| Model | Speed | Accuracy | VRAM/RAM |
|-------|-------|----------|----------|
| `tiny` | Fastest | Lower | ~1GB |
| `base` | Fast | Medium | ~1GB |
| `small` | Medium | Good | ~2GB |
| `medium` | Slow | Better | ~5GB |
| `large` | Slowest | Best | ~10GB |

### Voice Activity Detection (VAD)

Uses **Silero VAD** for detecting when speech starts/stops.

- Threshold: 0.5 (configurable)
- Silence timeout: 0.8 seconds
- Falls back to energy-based VAD if Silero unavailable

### Hotword Detection

Uses **OpenWakeWord** with custom ONNX models.

Default wake words:
- "Hey Wyzer" (threshold: 0.5)
- "Wyzer" (threshold: 0.75)

Models located in: `openwakemodels/`

### Text-to-Speech (TTS)

Uses **Piper TTS** for natural voice synthesis.

Default voice: `en_US-lessac-medium`

Files:
- `wyzer/assets/piper/piper.exe`
- `wyzer/assets/piper/en_US-lessac-medium.onnx`

---

## LLM Integration

### Ollama Setup

Wyzer uses Ollama for local LLM inference.

1. **Install Ollama**: https://ollama.ai
2. **Pull a model**: `ollama pull llama3.1:latest`
3. **Start Ollama**: `ollama serve` (usually runs automatically)

### Supported Models

Any Ollama-compatible model works. Recommended:
- `llama3.1:latest` (default, balanced)
- `llama3.1:8b` (faster, less capable)
- `llama3.2:latest` (newer, efficient)
- `mistral:latest` (alternative)

### No-Ollama Mode

Run without LLM for deterministic tool-only mode:

```bash
python run.py --no-ollama
```

In this mode:
- Only tool commands work (time, weather, media, etc.)
- Conversational queries return fallback messages
- Faster response times

---

## Multi-Intent Commands

Wyzer supports executing multiple commands in sequence:

### Syntax Examples

```
"Open Chrome and then play music"
"Pause Spotify, then set volume to 50%"
"Minimize Discord and focus Chrome"
"Open Downloads and show monitor info"
```

### Supported Conjunctions

- `and then`
- `then`
- `and`
- `after that`
- `;` (semicolon)
- `,` (comma with clear intent)

### How It Works

1. **Hybrid Router** checks for multi-intent markers
2. **Multi-Intent Parser** splits the command
3. Each intent is validated and executed in order
4. Results are combined into a single response

---

## Hybrid Router

The Hybrid Router provides **fast-path** deterministic command routing, bypassing the LLM for obvious commands.

### Fast-Path Commands (No LLM)

| Pattern | Action |
|---------|--------|
| "what time is it" | `get_time` tool |
| "open [app/folder]" | `open_target` tool |
| "pause/play music" | `media_play_pause` tool |
| "volume up/down/50%" | `volume_control` tool |
| "focus/minimize/close [window]" | Window management tools |
| "what's the weather" | `get_weather_forecast` tool |

### LLM-Required Queries

These patterns always route to the LLM:
- "Why..." questions
- "How do I..." questions
- "Explain..." requests
- "What is..." (general knowledge)
- Recommendations/opinions
- Complex reasoning

---

## Troubleshooting

### Common Issues

#### "Whisper model not loading"
```bash
pip install faster-whisper --force-reinstall
```

#### "Ollama connection failed"
1. Ensure Ollama is running: `ollama serve`
2. Check URL: `http://127.0.0.1:11434`
3. Test: `curl http://127.0.0.1:11434/api/tags`

#### "Hotword not detecting"
1. Check microphone permissions
2. List devices: `python run.py --list-devices`
3. Try specific device: `python run.py --device 1`
4. Lower threshold in config

#### "TTS not working"
1. Verify Piper exists: `wyzer/assets/piper/piper.exe`
2. Verify model exists: `wyzer/assets/piper/en_US-lessac-medium.onnx`
3. Check audio output device

#### "Tools not responding"
```bash
set WYZER_TOOLS_TEST=1
python run.py
```

### Debug Mode

For detailed logging:
```bash
python run.py --log-level DEBUG
```

### Single-Process Mode

For easier debugging:
```bash
python run.py --single-process
```

---

## Project Structure

```
AI-Assistant-v2/
├── run.py                      # Main entry point
├── run.bat                     # Windows batch launcher
├── requirements.txt            # Python dependencies
├── ARCHITECTURE_LOCK.md        # Architecture constraints
│
├── wyzer/
│   ├── __init__.py
│   │
│   ├── audio/                  # Audio processing
│   │   ├── mic_stream.py       # Microphone input
│   │   ├── vad.py              # Voice activity detection
│   │   ├── hotword.py          # Wake word detection
│   │   └── audio_utils.py      # Audio utilities
│   │
│   ├── stt/                    # Speech-to-Text
│   │   ├── stt_router.py       # STT routing
│   │   └── whisper_engine.py   # Faster-Whisper engine
│   │
│   ├── brain/                  # LLM & prompts
│   │   ├── llm_engine.py       # LLM interface
│   │   ├── ollama_client.py    # Ollama HTTP client
│   │   ├── prompt.py           # System prompts
│   │   └── prompt_compact.py   # Prompt optimization
│   │
│   ├── tts/                    # Text-to-Speech
│   │   ├── tts_router.py       # TTS routing
│   │   ├── piper_engine.py     # Piper TTS engine
│   │   └── audio_player.py     # Audio playback
│   │
│   ├── core/                   # Core logic
│   │   ├── assistant.py        # Main assistant classes
│   │   ├── orchestrator.py     # Intent & tool routing
│   │   ├── brain_worker.py     # Brain worker process
│   │   ├── hybrid_router.py    # Fast-path routing
│   │   ├── multi_intent_parser.py  # Multi-command parsing
│   │   ├── state.py            # State machine
│   │   ├── config.py           # Configuration
│   │   ├── followup_manager.py # Follow-up mode
│   │   ├── tool_worker_pool.py # Parallel tool execution
│   │   └── logger.py           # Logging
│   │
│   ├── tools/                  # Tool implementations
│   │   ├── tool_base.py        # Base tool class
│   │   ├── registry.py         # Tool registry
│   │   ├── get_time.py
│   │   ├── get_weather_forecast.py
│   │   ├── open_target.py
│   │   ├── volume_control.py
│   │   ├── window_manager.py
│   │   ├── media_controls.py
│   │   ├── timer_tool.py
│   │   ├── system_storage.py
│   │   └── ... (more tools)
│   │
│   ├── local_library/          # App/folder indexing
│   │   ├── indexer.py          # Index builder
│   │   ├── resolver.py         # Query resolver
│   │   ├── alias_manager.py    # Custom aliases
│   │   ├── game_indexer.py     # Game detection
│   │   └── library.json        # Generated index
│   │
│   ├── assets/                 # Binary assets
│   │   └── piper/              # Piper TTS files
│   │
│   └── data/                   # Runtime data
│       ├── timer_state.json
│       └── system_storage_index.json
│
├── openwakemodels/             # Custom hotword models
│   ├── hey_Wyzer.onnx
│   └── PEE_C_OH.onnx
│
├── scripts/                    # Test scripts
│   ├── test_hybrid_router.py
│   ├── test_timer_tool.py
│   └── ...
│
└── MD's/                       # Additional documentation
    ├── INSTALL.md
    ├── MULTI_INTENT_COMMANDS.md
    └── ...
```

---

## Quick Reference Card

### Starting Wyzer
```bash
python run.py                    # Normal mode
python run.py --no-hotword       # Immediate listening (no wake word)
python run.py --no-ollama        # Tools-only mode (no LLM)
python run.py --quiet            # Clean output
```

### Voice Commands
| Say | Does |
|-----|------|
| "Hey Wyzer" | Wake the assistant |
| "What time is it?" | Get current time |
| "Open Chrome" | Launch Chrome |
| "Volume 50%" | Set volume to 50% |
| "Pause" | Pause media |
| "Close Spotify" | Close Spotify window |
| "Set timer for 5 minutes" | Start a timer |

### Keyboard Controls
| Key | Action |
|-----|--------|
| `Ctrl+C` | Stop Wyzer |

---

*Wyzer AI Assistant - Phase 6*
*Local, Private, Powerful*
