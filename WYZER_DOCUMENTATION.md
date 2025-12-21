# Wyzer AI Assistant - Complete Documentation

> **A fully local, privacy-focused voice assistant for Windows**
> 
> *Last Updated: December 2025*

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
13. [Memory System](#memory-system)
14. [Interruption System](#interruption-system)
15. [Troubleshooting](#troubleshooting)

---

## Overview

Wyzer is a **local voice assistant** that runs entirely on your Windows machine. It provides:

- **Hotword Detection**: Wake the assistant with "Hey Wyzer" or just "Wyzer"
- **Speech-to-Text (STT)**: Uses Faster-Whisper for accurate transcription
- **Local LLM Processing**: Supports llama.cpp (embedded server) or Ollama for natural language understanding
- **Text-to-Speech (TTS)**: Piper TTS for natural voice responses
- **Tool Execution**: Control your system, manage windows, play media, and more
- **Multi-Intent Commands**: Execute multiple actions in a single request (with or without separators)
- **Memory System**: Remember facts about you with explicit "remember X" commands
- **Hybrid Routing**: Fast deterministic routing bypasses LLM for obvious commands
- **Interruption System**: Cancel any process with the wake word (barge-in)

### Key Principles

- **Privacy First**: All processing happens locally - no cloud services required
- **Modular Design**: Each component (STT, LLM, TTS, Tools) is independently configurable
- **Multiprocess Architecture**: Responsive UI with heavy processing offloaded to worker processes
- **Fast-Path Routing**: Deterministic commands bypass LLM for sub-100ms response times

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

### Step 4: Install Ollama Model (if using Ollama mode)

```bash
ollama pull llama3.1:latest
```

### Step 4b: Setup llama.cpp (if using llamacpp mode - default)

1. Download `llama-server.exe` from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
2. Place in `wyzer/llm_bin/`
3. Download a GGUF model (e.g., `llama-3.1-8b-instruct.Q4_K_M.gguf`)
4. Place in `wyzer/llm_models/` and rename to `model.gguf`

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
| **System** | "What's my system info?", "Show monitor info", "How many monitors?" |
| **Storage** | "List drives", "Scan devices", "Open D drive", "How much space on C?" |
| **Google** | "Google cats", "Search google for recipes" |
| **Multi-Intent** | "Open Chrome and then pause music", "Close Discord open Spotify" |

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
| `--no-ollama` | Run without LLM (tools-only mode) | Off |
| `--llm` | LLM mode (llamacpp/ollama/off) | `llamacpp` |
| `--ollama-model` | Ollama model name | `llama3.1:latest` |
| `--ollama-url` | Ollama API URL | `http://127.0.0.1:11434` |
| `--llamacpp-bin` | Path to llama-server executable | `./wyzer/llm_bin/llama-server.exe` |
| `--llamacpp-model` | Path to GGUF model file | `./wyzer/llm_models/model.gguf` |
| `--llamacpp-port` | HTTP port for llama.cpp server | `8081` |
| `--llamacpp-ctx` | Context window size | `2048` |
| `--llamacpp-threads` | Number of threads (0=auto) | `0` |
| `--llm-timeout` | LLM request timeout (seconds) | `30` |
| `--tts` | Enable TTS (on/off) | `on` |
| `--tts-device` | Audio output device for TTS | Auto |
| `--no-speak-interrupt` | Disable barge-in | Off |
| `--stream-tts` | Enable streaming TTS | Off |
| `--quiet` | Hide debug info (cleaner output) | Off |
| `--single-process` | Run in single-process mode (debugging) | Off |
| `--log-level` | Logging level (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `--profile` | Performance profile (low_end/normal) | `normal` |
| `--no-memories` | Disable long-term memory injection | Off |

### Environment Variables

All environment variables are prefixed with `WYZER_`.

| Variable | Description | Default |
|----------|-------------|---------|
| `WYZER_LLM_MODE` | LLM mode: `llamacpp`, `ollama`, or `off` | `llamacpp` |
| `WYZER_NO_OLLAMA` | Disable LLM entirely | `false` |
| `WYZER_SAMPLE_RATE` | Audio sample rate | `16000` |
| `WYZER_VAD_THRESHOLD` | VAD sensitivity (0.0-1.0) | `0.5` |
| `WYZER_VAD_SILENCE_TIMEOUT` | Silence timeout (seconds) | `1.2` |
| `WYZER_NO_SPEECH_START_TIMEOUT_SEC` | Abort if no speech within this window | `2.5` |
| `WYZER_MAX_RECORD_SECONDS` | Max recording duration | `10.0` |
| `WYZER_HOTWORD_THRESHOLD` | Hotword sensitivity | `0.5` |
| `WYZER_HOTWORD_COOLDOWN_SEC` | Cooldown between hotwords | `1.5` |
| `WYZER_HOTWORD_TRIGGER_STREAK` | Consecutive frames required | `3` |
| `WYZER_WHISPER_MODEL` | Default Whisper model | `small` |
| `WYZER_OLLAMA_MODEL` | Default Ollama model | `llama3.1:latest` |
| `WYZER_OLLAMA_URL` | Ollama API URL | `http://127.0.0.1:11434` |
| `WYZER_OLLAMA_STREAM` | Enable streaming responses | `true` |
| `WYZER_OLLAMA_TEMPERATURE` | LLM temperature | `0.4` |
| `WYZER_TTS_ENABLED` | Enable TTS | `true` |
| `WYZER_STREAM_TTS` | Enable streaming TTS | `false` |
| `WYZER_FOLLOWUP_ENABLED` | Enable follow-up mode | `true` |
| `WYZER_FOLLOWUP_TIMEOUT_SEC` | Follow-up timeout | `2.0` |
| `WYZER_QUIET_MODE` | Hide debug info | `false` |
| `WYZER_USE_MEMORIES` | Enable memory injection | `true` |
| `WYZER_TOOL_POOL_ENABLED` | Enable tool worker pool | `true` |
| `WYZER_TOOL_POOL_WORKERS` | Number of tool pool workers | `3` |
| `WYZER_VOICE_FAST` | Concise response mode | `auto` |

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
| `local_library_refresh` | Re-scan installed apps | "Refresh the library", "Scan files" |

### Window Management

| Tool | Description | Example |
|------|-------------|---------|
| `focus_window` | Bring window to front | "Focus Discord" |
| `minimize_window` | Minimize a window | "Minimize Chrome" |
| `maximize_window` | Maximize a window | "Maximize Spotify", "Fullscreen Chrome" |
| `close_window` | Close a window | "Close Notepad" |
| `move_window_to_monitor` | Move window to another display | "Move Chrome to monitor 2" |
| `get_window_monitor` | Get which monitor a window is on | "What monitor is Chrome on?" |

### Media Controls

| Tool | Description | Example |
|------|-------------|---------|
| `media_play_pause` | Toggle play/pause | "Pause music", "Resume", "Play" |
| `media_next` | Skip to next track | "Next track", "Skip" |
| `media_previous` | Go to previous track | "Previous song", "Go back" |
| `get_now_playing` | Get current media info | "What's playing?", "What song is this?" |

### Volume Control

| Tool | Description | Example |
|------|-------------|---------|
| `volume_control` | Master/app volume control | "Volume 50%", "Mute Spotify", "Turn Spotify down by 10%" |
| `volume_up` | Increase volume | "Volume up", "Louder" |
| `volume_down` | Decrease volume | "Turn it down", "Quieter" |
| `volume_mute_toggle` | Toggle mute | "Mute", "Unmute" |

### Audio Devices

| Tool | Description | Example |
|------|-------------|---------|
| `set_audio_output_device` | Switch audio output | "Switch audio to headphones" |

### Storage & Drives

| Tool | Description | Example |
|------|-------------|---------|
| `system_storage_scan` | Scan drives for info | "Scan my drives", "Scan devices", "Scan drive C" |
| `system_storage_list` | List available drives | "List drives", "How much space do I have?" |
| `system_storage_open` | Open a drive in Explorer | "Open D drive", "Open C:" |

### Timers

| Tool | Description | Example |
|------|-------------|---------|
| `timer` | Set/cancel/check timer | "Set timer for 5 minutes", "Cancel timer", "How much time left?" |

### Google Search

| Tool | Description | Example |
|------|-------------|---------|
| `google_search_open` | Open Google search in browser | "Google cats", "Search google for recipes" |

### Monitor Info

| Tool | Description | Example |
|------|-------------|---------|
| `monitor_info` | Get connected monitor details | "Monitor info", "How many monitors?" |

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

Wyzer supports multiple LLM backends for local inference:
- **llama.cpp** (default) - Embedded server for direct GGUF model loading with auto-optimization
- **Ollama** - Easy-to-use local LLM server

### LLM Mode Selection

Use the `--llm` flag to select the backend:
- `--llm llamacpp` (default) - Use embedded llama.cpp server
- `--llm ollama` - Use Ollama server
- `--llm off` or `--no-ollama` - Tools-only mode, no LLM

### llama.cpp Setup (Default - Recommended)

The embedded llama.cpp server provides the best performance with auto-optimization.

#### Step 1: Download llama.cpp

1. Download the latest `llama-server` from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
2. Place `llama-server.exe` (Windows) in `wyzer/llm_bin/`

#### Step 2: Download a GGUF Model

1. Download a GGUF model (e.g., from [HuggingFace](https://huggingface.co/models?search=gguf))
2. Place the `.gguf` file in `wyzer/llm_models/`
3. Rename to `model.gguf` or specify path with `--llamacpp-model`

Recommended models:
- `llama-3.1-8b-instruct.Q4_K_M.gguf` - Good balance of quality and speed
- `llama-3.2-3b-instruct.Q4_K_M.gguf` - Faster, smaller
- `mistral-7b-instruct-v0.2.Q4_K_M.gguf` - Alternative

#### Step 3: Run with llama.cpp

```bash
python run.py --llm llamacpp
```

Or just:
```bash
python run.py  # llamacpp is the default
```

#### llama.cpp CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--llamacpp-bin` | `./wyzer/llm_bin/llama-server.exe` | Path to llama-server executable |
| `--llamacpp-model` | `./wyzer/llm_models/model.gguf` | Path to GGUF model file |
| `--llamacpp-port` | `8081` | HTTP port for llama.cpp server |
| `--llamacpp-ctx` | `2048` | Context window size |
| `--llamacpp-threads` | `0` (auto) | Number of threads |

#### Auto-Optimization Features

When `WYZER_LLAMACPP_AUTO_OPTIMIZE=true` (default):
- Automatic GPU detection and layer offloading
- Optimal thread count based on CPU cores
- Flash attention enabled when available
- Batch size optimization

#### Voice-Fast Preset

When using llama.cpp, the "voice-fast" preset is enabled by default:
- Concise, snappy responses (max 64 tokens)
- Lower temperature (0.2) for more focused answers
- Story mode automatically uses higher limits (320 tokens)

#### Directory Structure

```
wyzer/
├── llm_bin/
│   └── llama-server.exe      # llama.cpp server binary
├── llm_models/
│   └── model.gguf            # Your GGUF model
└── logs/
    └── llamacpp_server.log   # Server output log
```

### Ollama Setup (Alternative)

### Ollama Setup (Alternative)

Wyzer also supports Ollama for local LLM inference.

1. **Install Ollama**: https://ollama.ai
2. **Pull a model**: `ollama pull llama3.1:latest`
3. **Start Ollama**: `ollama serve` (usually runs automatically)
4. **Run Wyzer with Ollama**: `python run.py --llm ollama`

#### Ollama CLI Options

```bash
python run.py --llm ollama --ollama-model llama3.1:latest --ollama-url http://127.0.0.1:11434
```

### Supported Ollama Models

Any Ollama-compatible model works. Recommended:
- `llama3.1:latest` (default, balanced)
- `llama3.1:8b` (faster, less capable)
- `llama3.2:latest` (newer, efficient)
- `mistral:latest` (alternative)

### llama.cpp Setup (Embedded Server)

For more control or to avoid Ollama, use the embedded llama.cpp server mode:

#### Step 1: Download llama.cpp

1. Download the latest `llama-server` from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
2. Place `llama-server.exe` (Windows) in `wyzer/llm_bin/`

#### Step 2: Download a GGUF Model

1. Download a GGUF model (e.g., from [HuggingFace](https://huggingface.co/models?search=gguf))
2. Place the `.gguf` file in `wyzer/llm_models/`
3. Rename to `model.gguf` or specify path with `--llamacpp-model`

Recommended models:
- `llama-3.1-8b-instruct.Q4_K_M.gguf` - Good balance of quality and speed
- `llama-3.2-3b-instruct.Q4_K_M.gguf` - Faster, smaller
- `mistral-7b-instruct-v0.2.Q4_K_M.gguf` - Alternative

#### Step 3: Run with llama.cpp

```bash
python run.py --llm llamacpp
```

#### llama.cpp CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--llamacpp-bin` | `./wyzer/llm_bin/llama-server.exe` | Path to llama-server executable |
| `--llamacpp-model` | `./wyzer/llm_models/model.gguf` | Path to GGUF model file |
| `--llamacpp-port` | `8081` | HTTP port for llama.cpp server |
| `--llamacpp-ctx` | `2048` | Context window size |
| `--llamacpp-threads` | `4` | Number of threads (0=auto) |

#### Example Full Command

```bash
python run.py --llm llamacpp --llamacpp-model ./wyzer/llm_models/llama-3.1-8b.Q4_K_M.gguf --llamacpp-ctx 4096 --llamacpp-threads 8
```

#### Directory Structure

```
wyzer/
├── llm_bin/
│   └── llama-server.exe      # llama.cpp server binary
├── llm_models/
│   └── model.gguf            # Your GGUF model
└── logs/
    └── llamacpp_server.log   # Server output log
```

### No-Ollama Mode (Tools Only)

Run without LLM for deterministic tool-only mode:

```bash
python run.py --no-ollama
# or
python run.py --llm off
```

In this mode:
- Only tool commands work (time, weather, media, etc.)
- Conversational queries return fallback messages
- Faster response times

---

## Multi-Intent Commands

Wyzer supports executing multiple commands in sequence, with or without explicit separators:

### Syntax Examples

**With separators:**
```
"Open Chrome and then play music"
"Pause Spotify, then set volume to 50%"
"Minimize Discord and focus Chrome"
"Open Downloads and show monitor info"
```

**Without separators (implicit multi-intent):**
```
"Close Chrome open Spotify"
"Minimize Discord maximize Notepad"
"Open settings close Discord"
```

### Supported Separators

- `and then` - Sequential execution
- `then` - Sequential execution
- `and` - Parallel execution
- `after that` - Sequential execution
- `;` (semicolon) - Sequential execution
- `,` (comma with clear intent) - Parallel execution

### Implicit Verb Boundary Detection

Commands without separators are automatically detected when multiple action verbs appear:
- `"close chrome open spotify"` → `close_window` + `open_target`
- `"minimize chrome maximize notepad"` → `minimize_window` + `maximize_window`

### Supported Action Verbs
`open`, `launch`, `start`, `close`, `quit`, `exit`, `minimize`, `shrink`, `maximize`, `fullscreen`, `expand`, `move`, `send`, `play`, `pause`, `resume`, `mute`, `unmute`, `scan`

### How It Works

1. **Hybrid Router** checks for multi-intent markers (explicit or implicit)
2. **Multi-Intent Parser** splits the command by separators or verb boundaries
3. Each intent is validated and executed in order
4. Results are combined into a single response

### Confidence Scoring

Multi-intent queries use composite confidence calculation:
```
Composite Confidence = min(clause_confidences) * 0.95
```

- Minimum per-clause confidence: 0.7
- Minimum composite confidence: 0.75 (to bypass LLM)

---

## Hybrid Router

The Hybrid Router provides **fast-path** deterministic command routing, bypassing the LLM for obvious commands.

### Fast-Path Commands (No LLM)

| Pattern | Tool | Confidence |
|---------|------|------------|
| "what time is it" | `get_time` | 0.95 |
| "what's the weather [in location]" | `get_weather_forecast` | 0.92 |
| "open [app/folder]" | `open_target` | 0.90 |
| "close [window]" | `close_window` | 0.85 |
| "minimize [window]" | `minimize_window` | 0.85 |
| "maximize/fullscreen [window]" | `maximize_window` | 0.85 |
| "move [window] to monitor [N]" | `move_window_to_monitor` | 0.85 |
| "what monitor is [app] on" | `get_window_monitor` | 0.85 |
| "pause/play music" | `media_play_pause` | 0.80 |
| "next/previous track" | `media_next/previous` | 0.85 |
| "volume up/down/50%" | `volume_control` | 0.85-0.93 |
| "mute/unmute [app]" | `volume_control` | 0.90 |
| "switch audio to [device]" | `set_audio_output_device` | 0.90 |
| "scan devices/drives" | `system_storage_scan` | 0.92-0.95 |
| "list drives" | `system_storage_list` | 0.92 |
| "open [drive] drive" | `system_storage_open` | 0.93 |
| "set timer for X minutes" | `timer` | 0.92 |
| "cancel timer" | `timer` | 0.93 |
| "google [query]" | `google_search_open` | 0.90 |
| "scan files/apps" | `local_library_refresh` | 0.92 |
| "refresh library" | `local_library_refresh` | 0.93 |
| "system info" | `get_system_info` | 0.90 |
| "monitor info" | `monitor_info` | 0.90 |
| "where am i" / "my location" | `get_location` | 0.90 |

### LLM-Required Queries

These patterns always route to the LLM:
- "Why..." questions
- "How do I..." questions
- "Explain..." requests
- "What is..." (general knowledge, not time/weather/volume)
- Recommendations/opinions
- Complex reasoning
- Creative content requests (stories, poems, jokes)

---

## Memory System

Wyzer has a **two-tier memory system** for personalization:

### Session Memory (RAM-only)
- Stores recent conversation turns (last 5-10 turns)
- Cleared on restart
- Provides context for follow-up questions

### Long-Term Memory (Disk-persisted)
- Explicitly saved facts via "remember X" commands
- Stored in `wyzer/data/memory.json`
- Persists across restarts

### Memory Commands

| Voice Command | Action |
|---------------|--------|
| "Remember that my name is John" | Save fact to long-term memory |
| "Remember my birthday is Sept 10" | Save fact to long-term memory |
| "What do you remember?" | List all saved memories |
| "Forget that" | Delete most recent memory |
| "Forget my birthday" | Delete specific memory |
| "Use memories" | Enable memory injection into LLM prompts |
| "Stop using memories" | Disable memory injection |

### Memory Injection

When enabled (default: **ON**), all long-term memories are injected into LLM prompts so Wyzer can answer questions about you:

- "What's my name?" → Uses remembered facts
- "When's my birthday?" → Uses remembered facts

**Control Methods:**
- **Voice:** "use memories" / "stop using memories"
- **CLI:** `--no-memories` flag disables injection
- **Env:** `WYZER_USE_MEMORIES=0` disables injection

**Priority Order:** CLI flag > env var > config default (true)

---

## Interruption System

Wyzer includes a clean interruption system that allows canceling any process without breaking the assistant.

### How It Works

- **Barge-In**: Say the wake word while Wyzer is speaking to interrupt immediately
- **State-Aware**: The system handles interruption based on the current state:
  - **SPEAKING**: Stops TTS immediately
  - **LISTENING**: Drains audio queue and returns to idle
  - **THINKING**: Sends interrupt to brain worker
  - **TRANSCRIBING**: Waits for transcription to complete

### Safety Features

1. **Non-Breaking**: All interruptions are clean and don't crash the system
2. **Thread-Safe**: Uses proper synchronization mechanisms
3. **Process-Safe**: Works across multiprocess architecture
4. **State Consistent**: Always transitions to IDLE after interruption

### Disabling Barge-In

To disable the ability to interrupt with the wake word:
```bash
python run.py --no-speak-interrupt
```

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
│   ├── memory/                 # Memory system
│   │   ├── memory_manager.py   # Session & long-term memory
│   │   └── command_detector.py # Memory command parsing
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
│       ├── memory.json         # Long-term memory storage
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
python run.py                    # Normal mode (uses llama.cpp)
python run.py --llm ollama       # Use Ollama instead
python run.py --no-hotword       # Immediate listening (no wake word)
python run.py --no-ollama        # Tools-only mode (no LLM)
python run.py --quiet            # Clean output
python run.py --stream-tts       # Streaming TTS (speak while generating)
```

### Voice Commands
| Say | Does |
|-----|------|
| "Hey Wyzer" | Wake the assistant |
| "What time is it?" | Get current time |
| "What's the weather?" | Get weather forecast |
| "Open Chrome" | Launch Chrome |
| "Close Discord" | Close window |
| "Minimize Spotify" | Minimize window |
| "Move Chrome to monitor 2" | Move window |
| "Volume 50%" | Set volume to 50% |
| "Mute Spotify" | Mute specific app |
| "Pause" / "Play" | Media controls |
| "Next track" | Skip to next |
| "Set timer for 5 minutes" | Start a timer |
| "Cancel timer" | Cancel the timer |
| "Google cats" | Search Google |
| "Scan devices" | Deep scan drives |
| "List drives" | Show drive info |
| "Open D drive" | Open in Explorer |
| "Remember my name is John" | Save fact to memory |
| "What do you remember?" | List memories |
| "Forget that" | Delete last memory |
| "Close Chrome open Spotify" | Multi-intent command |

### Keyboard Controls
| Key | Action |
|-----|--------|
| `Ctrl+C` | Stop Wyzer |

---

*Wyzer AI Assistant - December 2025*
*Local, Private, Powerful*
