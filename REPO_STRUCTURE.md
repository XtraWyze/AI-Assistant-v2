# Wyzer Repository Structure

## Overview
Complete directory tree of the Wyzer AI Assistant v2 project with descriptions of each module's purpose.

---

## Root Directory

```
AI-Assistant-v2/
├── ARCHITECTURE_LOCK.md          # Design constraints & architectural principles
├── README.md                     # Main project documentation
├── SYSTEM_FLOW.md                # Complete system flow diagrams & state machine
├── TOOL_CHAIN.md                 # Available tools & tool execution pipeline
├── THREE_PROCESS_ARCHITECTURE.md # Multiprocess design documentation
├── REPO_STRUCTURE.md             # This file - directory tree & module descriptions
├── requirements.txt              # Python dependencies
├── run.py                        # Main entry point
├── run.bat                       # Windows batch launcher

├── MD's/                         # Implementation documentation
│   ├── DEVICE_SCAN_DEEP_TIER.md
│   ├── IMPLICIT_MULTI_INTENT_FIX.md
│   ├── INTERRUPTION_SYSTEM.md
│   ├── INTERRUPTION_USAGE.md
│   ├── LOCAL_LIBRARY_TIER3_SCAN.md
│   ├── MULTI_INTENT_COMMANDS.md
│   ├── SYSTEM_STORAGE_IMPLEMENTATION.md
│   ├── WEATHER_HYBRID_IMPLEMENTATION.md
│   └── WINDOW_MANAGEMENT_HYBRID_ROUTING.md
│
├── scripts/                      # Testing & utility scripts
│   ├── test_followup_manager.py
│   ├── test_hybrid_router.py
│   ├── test_interruption.py
│   ├── test_system_storage_integration.py
│   ├── test_system_storage.py
│   └── volume_control_sanity.py
│
└── wyzer/                        # Main application package
    ├── __init__.py
    │
    ├── audio/                    # Audio input/hotword detection
    │   ├── __init__.py
    │   ├── audio_utils.py        # Audio utilities & buffer management
    │   ├── hotword.py            # Hotword detection engine
    │   ├── mic_stream.py         # Microphone input stream
    │   └── vad.py                # Voice Activity Detection (VAD)
    │
    ├── brain/                    # LLM & NLP processing
    │   ├── __init__.py
    │   ├── llm_engine.py         # LLM interface (Ollama)
    │   └── prompt.py             # System prompts & prompt engineering
    │
    ├── core/                     # Core orchestration & state
    │   ├── __init__.py
    │   ├── assistant.py          # Main assistant class
    │   ├── brain_worker.py       # Brain process worker
    │   ├── config.py             # Configuration management
    │   ├── followup_manager.py   # Follow-up window logic
    │   ├── hybrid_router.py      # Deterministic vs LLM routing
    │   ├── intent_plan.py        # Intent planning & parsing
    │   ├── ipc.py                # Inter-process communication
    │   ├── logger.py             # Logging utilities
    │   ├── multi_intent_parser.py# Multi-intent extraction
    │   ├── orchestrator.py       # Request orchestration
    │   ├── process_manager.py    # Process spawning & management
    │   └── state.py              # State machine & transitions
    │
    ├── data/                     # Data storage
    │   └── system_storage_index.json  # Cached drive/storage information
    │
    ├── llama-server/             # Local Ollama/Llama.cpp integration
    │
    ├── llm/                      # LLM providers (expandable)
    │   └── providers/            # LLM provider implementations
    │
    ├── llm_model/                # Local LLM model storage
    │
    ├── local_library/            # App/folder library management
    │   ├── __init__.py
    │   ├── alias_manager.py      # Alias resolution
    │   ├── aliases.json          # User-defined aliases
    │   ├── aliases.json.example  # Alias examples
    │   ├── game_indexer.py       # Windows Store games indexing
    │   ├── indexer.py            # Main library indexer
    │   ├── library.json          # Cached library data
    │   ├── resolver.py           # Library resolution engine
    │   └── uwp_indexer.py        # UWP (Windows app) indexing
    │
    ├── stt/                      # Speech-to-Text engines
    │   ├── __init__.py
    │   ├── stt_router.py         # STT engine router
    │   └── whisper_engine.py     # Whisper (OpenAI) engine
    │
    ├── tools/                    # Tool implementations
    │   ├── __init__.py
    │   ├── audio_output_device.py# Audio device switching
    │   ├── get_location.py       # IP-based geolocation
    │   ├── get_system_info.py    # System info retrieval
    │   ├── get_time.py           # Time/date tool
    │   ├── get_weather_forecast.py# Weather forecasting
    │   ├── local_library_refresh.py# Library refresh tool
    │   ├── media_controls.py     # Media playback control
    │   ├── monitor_info.py       # Monitor detection
    │   ├── open_target.py        # Smart target opening
    │   ├── open_website.py       # Website opening
    │   ├── registry.py           # Tool registry & registration
    │   ├── system_storage.py     # Storage management
    │   ├── tool_base.py          # Base class for all tools
    │   ├── validation.py         # Argument validation
    │   ├── volume_control.py     # Audio volume control
    │   └── window_manager.py     # Window management
    │
    ├── tts/                      # Text-to-Speech engines
    │   ├── __init__.py
    │   ├── audio_player.py       # TTS audio playback
    │   ├── piper_engine.py       # Piper TTS engine
    │   └── tts_router.py         # TTS engine router
    │
    └── assets/                   # Static assets & models
        └── piper/                # Piper TTS models & data
            ├── *.onnx            # Neural network models
            ├── *.onnx.json       # Model metadata
            ├── libtashkeel_model.ort  # Arabic diacritization
            ├── espeak-ng-data/    # eSpeakNG phonemes
            └── pkgconfig/        # Package config
```

---

## Core Module Descriptions

### `wyzer/core/` - Orchestration & State Management

The heart of the system, managing:
- **State machine**: IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING → FOLLOWUP
- **Process management**: Core process, Brain worker, process lifecycle
- **IPC**: Inter-process queues for async communication
- **Tool routing**: Deterministic vs LLM-based routing
- **Logging**: Centralized logging across processes

**Key Files:**
- `state.py`: State enum and transitions
- `assistant.py`: Main entry point class
- `hybrid_router.py`: Route to deterministic tools or LLM
- `followup_manager.py`: Follow-up conversation window
- `ipc.py`: Message queues and communication
- `orchestrator.py`: Request flow orchestration

### `wyzer/audio/` - Audio Input & Hotword Detection

Real-time audio processing:
- **Microphone streaming**: Continuous audio input at 16kHz
- **Hotword detection**: "Hey Jarvis" detection (local)
- **VAD (Voice Activity Detection)**: Speech/silence detection
- **Barge-in support**: Interrupt during playback

**Key Files:**
- `mic_stream.py`: Core microphone reader
- `hotword.py`: Hotword detection engine
- `vad.py`: Voice activity detection
- `audio_utils.py`: Buffer management & utilities

### `wyzer/brain/` - LLM & NLP

Language understanding & generation:
- **LLM Interface**: Ollama/Llama.cpp communication
- **Function calling**: Tool invocation via LLM
- **Prompt management**: System prompts & context
- **Response generation**: Natural language responses

**Key Files:**
- `llm_engine.py`: Ollama API interface
- `prompt.py`: Prompt templates and engineering

### `wyzer/stt/` - Speech-to-Text

Audio to text conversion:
- **Whisper engine**: Fast-Whisper for accurate transcription
- **STT routing**: Future extensibility for multiple engines
- **Error recovery**: Graceful fallbacks

**Key Files:**
- `whisper_engine.py`: Whisper implementation
- `stt_router.py`: Engine selection logic

### `wyzer/tts/` - Text-to-Speech

Text to audio conversion:
- **Piper TTS**: Local, fast, high-quality synthesis
- **Audio playback**: Device selection & playback
- **Streaming support**: Real-time playback

**Key Files:**
- `piper_engine.py`: Piper TTS implementation
- `audio_player.py`: Playback control
- `tts_router.py`: Engine selection logic

### `wyzer/tools/` - Tool Implementations

All executable tools:
- **System tools**: Time, location, system info
- **Media tools**: Volume, playback, device control
- **Window tools**: Window management
- **Web tools**: Website opening
- **Storage tools**: Drive scanning & opening
- **Weather**: Forecast retrieval

**Base Infrastructure:**
- `tool_base.py`: Abstract base class
- `registry.py`: Tool discovery & registration
- `validation.py`: Argument schema validation

### `wyzer/local_library/` - App/Folder Resolution

Smart app and folder discovery:
- **Game indexing**: Windows Store games
- **UWP indexing**: Modern Windows apps
- **Alias management**: User-defined shortcuts
- **Fuzzy matching**: Intelligent name resolution

**Key Files:**
- `indexer.py`: Main indexing engine
- `resolver.py`: Query resolution
- `game_indexer.py`: Game library scan
- `uwp_indexer.py`: Windows app indexing
- `alias_manager.py`: Alias handling

---

## Documentation Files (`MD's/`)

### System Implementation Details

- **MULTI_INTENT_COMMANDS.md**: Parsing "X, Y, and Z" commands
- **IMPLICIT_MULTI_INTENT_FIX.md**: Handling implicit intents
- **INTERRUPTION_SYSTEM.md**: Barge-in & interruption mechanism
- **INTERRUPTION_USAGE.md**: How to use interruption features
- **SYSTEM_STORAGE_IMPLEMENTATION.md**: Drive scanning & caching
- **LOCAL_LIBRARY_TIER3_SCAN.md**: App discovery deep-dive
- **DEVICE_SCAN_DEEP_TIER.md**: Device enumeration
- **WEATHER_HYBRID_IMPLEMENTATION.md**: Weather tool implementation
- **WINDOW_MANAGEMENT_HYBRID_ROUTING.md**: Window tool routing

---

## Dependencies

### Core Requirements
- **Python 3.10+**
- **faster-whisper**: Speech recognition
- **piper-tts**: Text-to-speech
- **ollama**: LLM backend
- **pydub**: Audio processing
- **pyaudio**: Microphone input
- **pycaw**: Windows audio control
- **pywin32**: Windows API

### Optional
- **psutil**: System monitoring
- **requests**: HTTP requests
- **numpy**: Array operations

See `requirements.txt` for complete list.

---

## Entry Points

### `run.py`
Main Python entry point:
```bash
python run.py
```
Starts Core and Brain processes.

### `run.bat`
Windows batch launcher:
```bash
run.bat
```
Activates virtual environment and runs `run.py`.

---

## Configuration

### Main Config: `wyzer/core/config.py`

Key settings:
- **Audio**: Sample rate (16kHz), chunk size, VAD sensitivity
- **Hotword**: Confidence thresholds, model path
- **LLM**: Model name, timeout, base URL
- **TTS**: Engine selection, audio device
- **Logging**: Log level, format
- **Features**: FOLLOWUP enabled, BARGE_IN enabled

### Library Config: `wyzer/local_library/aliases.json`

User-defined aliases for apps and folders:
```json
{
  "shortcuts": {
    "browser": "google chrome",
    "player": "vlc media player"
  }
}
```

---

## Data Storage

### `wyzer/data/system_storage_index.json`
Cached storage scan results with:
- Drive names and mount points
- Total/used/free space
- Filesystem type
- Timestamp of last scan

### `wyzer/local_library/library.json`
Indexed library data:
- Installed apps
- Game library
- User folders
- Aliases and shortcuts

---

## Testing

Test scripts in `scripts/`:
- `test_hybrid_router.py`: Route testing
- `test_followup_manager.py`: Follow-up logic
- `test_interruption.py`: Barge-in testing
- `test_system_storage_integration.py`: Storage tool integration
- `volume_control_sanity.py`: Volume control verification

Run with:
```bash
python scripts/test_hybrid_router.py
```

---

## Process Architecture

```
Main Process (run.py)
  ├─ Process A: Core (wyzer/core/assistant.py)
  │   ├─ Hotword detection (wyzer/audio/hotword.py)
  │   ├─ VAD monitoring (wyzer/audio/vad.py)
  │   ├─ State machine (wyzer/core/state.py)
  │   ├─ Barge-in handler (wyzer/audio/mic_stream.py)
  │   └─ Output playback (wyzer/tts/audio_player.py)
  │
  └─ Process B: Brain (wyzer/core/brain_worker.py)
      ├─ STT (wyzer/stt/whisper_engine.py)
      ├─ Hybrid Routing (wyzer/core/hybrid_router.py)
      ├─ LLM (wyzer/brain/llm_engine.py)
      ├─ Tool Execution (wyzer/tools/*.py)
      └─ TTS (wyzer/tts/piper_engine.py)

IPC Communication:
  └─ core_to_brain_q ←→ brain_to_core_q
```

---

## Asset Organization

### `/wyzer/assets/piper/`

**Language Models:**
- `en_US-lessac-medium.onnx`: English TTS model
- `en_US-lessac-medium.onnx.json`: Model metadata

**Phoneme Data:**
- `espeak-ng-data/`: Phoneme definitions for 40+ languages

**Library Info:**
- `pkgconfig/`: Library configuration files

---

## Execution Flow Summary

1. **Startup**: `run.py` → `assistant.py` → spawns Core & Brain
2. **Listening**: Core continuously monitors microphone
3. **Hotword**: Detected → transition to LISTENING state
4. **Transcription**: Brain receives audio → Whisper → transcript
5. **Routing**: Hybrid router → deterministic or LLM
6. **Execution**: Tools run, results collected
7. **Response**: TTS generates audio
8. **Follow-up**: Optionally listen for follow-up (3s window)
9. **Repeat**: Back to IDLE, listening for next hotword

---

## Quick Navigation

- **Architecture**: See [ARCHITECTURE_LOCK.md](ARCHITECTURE_LOCK.md) & [THREE_PROCESS_ARCHITECTURE.md](THREE_PROCESS_ARCHITECTURE.md)
- **System Flow**: See [SYSTEM_FLOW.md](SYSTEM_FLOW.md)
- **Tools**: See [TOOL_CHAIN.md](TOOL_CHAIN.md)
- **Configuration**: See [wyzer/core/config.py](wyzer/core/config.py)
- **Implementation Details**: See [MD's/](MD's/) folder
