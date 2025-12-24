# Wyzer AI Assistant - All Flags Reference

Complete reference for all command-line arguments and environment variable configurations.

> *Last Updated: December 2025*

---

## Command-Line Arguments

### General Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--single-process` | flag | `false` | Run everything in one process (legacy path; easier debugging) |
| `--profile` | choice | `normal` | Performance profile: `low_end`, `normal` |
| `--quiet` | flag | `false` | Enable quiet mode (hide debug info like heartbeats for cleaner output) |
| `--log-level` | choice | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--list-devices` | flag | - | List available audio devices and exit |

### Audio & Hotword

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--no-hotword` | flag | `false` | Disable hotword detection (immediate listening mode) |
| `--device` | string | `None` | Audio device index or name |

### Speech-to-Text (Whisper)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | choice | `small` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--whisper-device` | choice | `cpu` | Device for Whisper inference: `cpu`, `cuda` |

### LLM Configuration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--no-ollama` | flag | `false` | Run without LLM entirely (tools-only mode, LLM features disabled) |
| `--llm` | choice | `llamacpp` | LLM mode: `llamacpp`, `ollama`, `off` |
| `--ollama-model` | string | `llama3.1:latest` | Ollama model name |
| `--ollama-url` | string | `http://127.0.0.1:11434` | Ollama API base URL |
| `--llm-timeout` | int | `30` | LLM request timeout in seconds |

### llama.cpp Configuration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--llamacpp-bin` | string | `./wyzer/llm_bin/llama-server.exe` | Path to llama-server executable |
| `--llamacpp-model` | string | `./wyzer/llm_models/model.gguf` | Path to GGUF model file |
| `--llamacpp-port` | int | `8081` | HTTP port for llama.cpp server |
| `--llamacpp-ctx` | int | `2048` | Context window size |
| `--llamacpp-threads` | int | `0` (auto) | Number of threads (0=auto recommended) |

### Text-to-Speech (TTS)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--tts` | choice | `on` | Enable/disable TTS: `on`, `off` |
| `--tts-engine` | choice | `piper` | TTS engine to use: `piper` |
| `--piper-exe` | string | `./wyzer/assets/piper/piper.exe` | Path to Piper executable |
| `--piper-model` | string | `./wyzer/assets/piper/en_US-lessac-medium.onnx` | Path to Piper voice model |
| `--tts-device` | string | `None` | Audio output device index for TTS |
| `--no-speak-interrupt` | flag | `false` | Disable barge-in (hotword interrupt during speaking) |
| `--stream-tts` | flag | `false` | Enable streaming TTS (speak chunks as LLM generates them) |

### Memory Injection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--no-memories` | flag | `false` | Disable long-term memory injection into LLM prompts |

> Note: Memory injection is **ON by default**. Use `--no-memories` to disable it. Can also be controlled via `WYZER_USE_MEMORIES` env var or voice commands.

---

## Environment Variables

All environment variables are prefixed with `WYZER_`.

### Audio Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_SAMPLE_RATE` | int | `16000` | Audio sample rate in Hz |
| `WYZER_CHUNK_MS` | int | `20` | Audio chunk duration in milliseconds |
| `WYZER_AUDIO_QUEUE_MAX_SIZE` | int | `100` | Maximum size of audio queue |

### Recording Limits

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_MAX_RECORD_SECONDS` | float | `10.0` | Maximum recording duration in seconds |
| `WYZER_VAD_SILENCE_TIMEOUT` | float | `1.2` | Silence timeout before stopping recording (seconds) |
| `WYZER_NO_SPEECH_START_TIMEOUT_SEC` | float | `2.5` | Abort listening if VAD never detects speech start within this window |

### VAD (Voice Activity Detection)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_VAD_THRESHOLD` | float | `0.5` | VAD sensitivity threshold (0-1) |
| `WYZER_VAD_MIN_SPEECH_MS` | int | `250` | Minimum speech duration in milliseconds |

### Hotword Detection

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_HOTWORD_KEYWORDS` | string | `hey wyzer,wyzer` | Comma-separated list of hotword keywords |
| `WYZER_HOTWORD_THRESHOLD` | float | `0.5` | Hotword detection threshold (0-1) |
| `WYZER_HOTWORD_TRIGGER_STREAK` | int | `3` | Consecutive frames above threshold before triggering |
| `WYZER_HOTWORD_MODEL_PATH` | string | `hey_Wyzer.onnx` | Path to hotword model file |
| `WYZER_HOTWORD_COOLDOWN_SEC` | float | `1.5` | Cooldown period after hotword trigger (seconds) |
| `WYZER_POST_IDLE_DRAIN_SEC` | float | `0.5` | Post-idle drain duration (seconds) |

### Speech-to-Text (Whisper)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_WHISPER_MODEL` | string | `small` | Whisper model size |
| `WYZER_WHISPER_DEVICE` | string | `cpu` | Device for Whisper inference |
| `WYZER_WHISPER_COMPUTE_TYPE` | string | `int8` | Compute type for Whisper |
| `WYZER_MAX_TOKEN_REPEATS` | int | `6` | Token repeats above this threshold = garbage |
| `WYZER_MIN_TRANSCRIPT_LENGTH` | int | `2` | Minimum transcript length to accept |

### LLM / Ollama

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_LLM_MODE` | string | `llamacpp` | LLM mode: `llamacpp`, `ollama`, or `off` |
| `WYZER_NO_OLLAMA` | bool | `false` | Run without LLM entirely |
| `WYZER_OLLAMA_URL` | string | `http://127.0.0.1:11434` | Ollama API base URL |
| `WYZER_OLLAMA_MODEL` | string | `llama3.1:latest` | Ollama model name |
| `WYZER_LLM_TIMEOUT` | int | `30` | LLM request timeout in seconds |
| `WYZER_OLLAMA_STREAM` | bool | `true` | Enable streaming responses from Ollama |
| `WYZER_OLLAMA_TEMPERATURE` | float | `0.4` | Ollama temperature parameter |
| `WYZER_OLLAMA_TOP_P` | float | `0.9` | Ollama top_p parameter |
| `WYZER_OLLAMA_NUM_CTX` | int | `4096` | Ollama context window size |
| `WYZER_OLLAMA_NUM_PREDICT` | int | `120` | Ollama max tokens to predict |
| `WYZER_LLM_MAX_PROMPT_CHARS` | int | `8000` | Maximum prompt characters for LLM |

### llama.cpp Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_LLAMACPP_BIN` | string | `./wyzer/llm_bin/llama-server.exe` | Path to llama-server executable |
| `WYZER_LLAMACPP_MODEL` | string | `./wyzer/llm_models/model.gguf` | Path to GGUF model file |
| `WYZER_LLAMACPP_PORT` | int | `8081` | HTTP port for llama.cpp server |
| `WYZER_LLAMACPP_CTX` | int | `2048` | Context window size |
| `WYZER_LLAMACPP_THREADS` | int | `0` (auto) | Number of threads |
| `WYZER_LLAMACPP_AUTO_OPTIMIZE` | bool | `true` | Auto-detect GPU and optimize settings |
| `WYZER_LLAMACPP_GPU_LAYERS` | int | `-1` (all) | Number of layers to offload to GPU |
| `WYZER_LLAMACPP_BATCH_SIZE` | int | `512` | Batch size for inference |

### Voice-Fast Preset

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_VOICE_FAST` | string | `auto` | Enable voice-fast preset (true/false/auto) |
| `WYZER_VOICE_FAST_MAX_TOKENS` | int | `64` | Max tokens for concise responses |
| `WYZER_VOICE_FAST_TEMPERATURE` | float | `0.2` | Temperature for voice-fast mode |
| `WYZER_VOICE_FAST_TOP_P` | float | `0.9` | Top_p for voice-fast mode |
| `WYZER_VOICE_FAST_STORY_MAX_TOKENS` | int | `320` | Max tokens for creative content |

### Text-to-Speech (TTS)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_TTS_ENABLED` | bool | `true` | Enable TTS |
| `WYZER_STREAM_TTS` | bool | `false` | Enable streaming TTS (speak chunks as LLM generates them) |
| `WYZER_STREAM_TTS_BUFFER_CHARS` | int | `150` | Minimum buffer size before emitting a TTS segment |
| `WYZER_STREAM_TTS_FIRST_EMIT_CHARS` | int | `32` | First-emit optimization for lower perceived latency |
| `WYZER_TTS_ENGINE` | string | `piper` | TTS engine to use |
| `WYZER_PIPER_EXE_PATH` | string | `./assets/piper/piper.exe` | Path to Piper executable |
| `WYZER_PIPER_MODEL_PATH` | string | `./assets/piper/en_US-voice.onnx` | Path to Piper voice model |
| `WYZER_PIPER_SPEAKER_ID` | int | `None` | Piper speaker ID (optional) |
| `WYZER_TTS_RATE` | float | `1.0` | TTS speech rate multiplier |
| `WYZER_TTS_OUTPUT_DEVICE` | int | `None` | Audio output device for TTS |
| `WYZER_SPEAK_HOTWORD_INTERRUPT` | bool | `true` | Allow hotword to interrupt TTS (barge-in) |
| `WYZER_POST_SPEAK_DRAIN_SEC` | float | `0.35` | Post-speak drain duration (seconds) |
| `WYZER_SPEAK_START_COOLDOWN_SEC` | float | `1.8` | Speak start cooldown (seconds) |
| `WYZER_POST_BARGEIN_IGNORE_SEC` | float | `3.0` | Post barge-in ignore duration (seconds) |
| `WYZER_POST_BARGEIN_REQUIRE_SPEECH_START` | bool | `true` | Require speech start after barge-in |
| `WYZER_POST_BARGEIN_WAIT_FOR_SPEECH_SEC` | float | `2.0` | Wait for speech after barge-in (seconds) |

### Logging & Debug

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_LOG_LEVEL` | string | `INFO` | Logging level |
| `WYZER_QUIET_MODE` | bool | `false` | Hide debug info like heartbeats for cleaner output |
| `WYZER_VERIFY_MODE` | bool | `false` | Enable verification mode |

### Tool Safety Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_ENABLE_FORCE_CLOSE` | bool | `false` | Enable force close functionality |
| `WYZER_ALLOWED_APPS_TO_LAUNCH` | string | `notepad,calc,calculator,paint,explorer,chrome,firefox,edge,vscode,cmd,powershell` | Comma-separated list of allowed apps to launch |
| `WYZER_ALLOWED_PROCESSES_TO_CLOSE` | string | *(empty)* | Comma-separated list of allowed processes to close |
| `WYZER_REQUIRE_EXPLICIT_APP_MATCH` | bool | `true` | Require explicit app match for launching |

### LocalLibrary Auto-Alias

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_AUTO_ALIAS_ENABLED` | bool | `true` | Enable auto-alias learning for spoken phrases |
| `WYZER_AUTO_ALIAS_MIN_CONFIDENCE` | float | `0.85` | Minimum confidence for auto-alias |

### Follow-up System

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_FOLLOWUP_ENABLED` | bool | `true` | Enable follow-up listening window |
| `WYZER_FOLLOWUP_TIMEOUT_SEC` | float | `2.0` | Follow-up timeout in seconds |
| `WYZER_FOLLOWUP_MAX_CHAIN` | int | `3` | Maximum follow-up chain length |

### Tool Worker Pool

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_TOOL_POOL_ENABLED` | bool | `true` | Enable tool worker pool |
| `WYZER_TOOL_POOL_WORKERS` | int | `3` | Number of tool pool workers (1-5) |
| `WYZER_TOOL_POOL_TIMEOUT_SEC` | int | `15` | Tool pool timeout in seconds |

### Heartbeat & System

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_HEARTBEAT_INTERVAL_SEC` | float | `10.0` | Heartbeat interval in seconds |

### Memory Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_SESSION_MEMORY_TURNS` | int | `10` | Number of conversation turns to keep in session memory |
| `WYZER_MEMORY_FILE_PATH` | string | `wyzer/data/memory.json` | Path to memory JSON file |
| `WYZER_USE_MEMORIES` | bool | `true` | Enable long-term memory injection into LLM prompts |

### Memory Injection (Session Flags)

| Flag | Default | Voice Commands | Description |
|------|---------|----------------|-------------|
| `use_memories` | `true` | Enable: "use memories", "enable memories", "turn on memories" | When ON, all long-term memories are injected into LLM prompts |
| | | Disable: "stop using memories", "disable memories", "turn off memories" | Session-scoped only (resets on restart) |

**Priority Order:** CLI flag (`--no-memories`) > env var (`WYZER_USE_MEMORIES`) > config default (`true`)

### Autonomy Settings (Phase 11)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_AUTONOMY_DEFAULT` | string | `off` | Default autonomy mode: `off`, `low`, `normal`, `high` |
| `WYZER_AUTONOMY_CONFIRM_SENSITIVE` | bool | `true` | Always confirm high-risk actions even in high mode |
| `WYZER_AUTONOMY_CONFIRM_TIMEOUT_SEC` | float | `45.0` | Confirmation window timeout in seconds |
| `WYZER_CONFIRMATION_GRACE_MS` | int | `1500` | Grace period for responses during TTS playback |

### Window Watcher Settings (Phase 12)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WYZER_WINDOW_WATCHER_ENABLED` | bool | `true` | Enable/disable window watcher |
| `WYZER_WINDOW_WATCHER_POLL_MS` | int | `500` | Poll interval in milliseconds (min 100ms) |
| `WYZER_WINDOW_WATCHER_MAX_EVENTS` | int | `25` | Maximum recent events in ring buffer |
| `WYZER_WINDOW_WATCHER_IGNORE_PROCESSES` | string | *(empty)* | Comma-separated processes to ignore |
| `WYZER_WINDOW_WATCHER_IGNORE_TITLES` | string | *(empty)* | Comma-separated title substrings to ignore |
| `WYZER_WINDOW_WATCHER_MAX_BULK_CLOSE` | int | `10` | Max windows to close without confirmation |

---

## Usage Examples

### Basic Usage
```bash
python run.py                           # Normal mode with hotword (uses llama.cpp)
python run.py --llm ollama              # Use Ollama instead of llama.cpp
python run.py --no-hotword              # Immediate listening (no hotword)
python run.py --quiet                   # Clean output without debug info
```

### llama.cpp Configuration
```bash
python run.py --llm llamacpp            # Explicit llama.cpp mode (default)
python run.py --llamacpp-model ./wyzer/llm_models/custom.gguf  # Custom model
python run.py --llamacpp-ctx 4096       # Larger context window
python run.py --llamacpp-threads 8      # Specific thread count
```

### Autonomy Configuration
```bash
# Enable balanced autonomy mode (asks for confirmation on uncertain actions)
set WYZER_AUTONOMY_DEFAULT=normal
python run.py

# Enable high autonomy mode (executes most actions, confirms only risky)
set WYZER_AUTONOMY_DEFAULT=high
python run.py
```

### Memory Injection
```bash
python run.py                           # Memory injection ON by default
python run.py --no-memories             # Disable memory injection
set WYZER_USE_MEMORIES=0 && python run.py  # Disable via env var
```

### Model Configuration
```bash
python run.py --model medium            # Use Whisper medium model
python run.py --whisper-device cuda     # Use GPU for Whisper
python run.py --ollama-model mistral    # Use different Ollama model
```

### Tools-Only Mode (No LLM)
```bash
python run.py --no-ollama               # Disable Ollama completely
```

### TTS Configuration
```bash
python run.py --tts off                 # Disable TTS
python run.py --no-speak-interrupt      # Disable barge-in
```

### Debug & Development
```bash
python run.py --single-process          # Single process for debugging
python run.py --log-level DEBUG         # Verbose logging
python run.py --list-devices            # List audio devices
```

### Environment Variable Examples
```bash
# Windows CMD
set WYZER_QUIET_MODE=true
set WYZER_OLLAMA_MODEL=mistral
python run.py

# Windows PowerShell
$env:WYZER_QUIET_MODE = "true"
$env:WYZER_OLLAMA_MODEL = "mistral"
python run.py

# Linux/Mac
export WYZER_QUIET_MODE=true
export WYZER_OLLAMA_MODEL=mistral
python run.py
```

### Test Mode
```bash
# Windows
set WYZER_TOOLS_TEST=1 & python run.py
```

---

## Performance Profiles

| Profile | Description |
|---------|-------------|
| `normal` | Default settings for standard hardware |
| `low_end` | Optimized for lower-end hardware (uses int8 compute) |

Use with: `python run.py --profile low_end`
