# Wyzer AI Assistant - Phase 4

A voice-controlled AI assistant for Windows 10/11 with hotword detection, speech-to-text, and local LLM brain. This implementation covers Phase 1-4: project skeleton, audio pipeline with VAD and hotword detection, speech transcription, and local LLM integration via Ollama.

## Features

- ✅ **Hotword Detection**: Wake the assistant with "hey wyzer" or "wyzer"
- ✅ **Voice Activity Detection (VAD)**: Silero VAD with energy-based fallback
- ✅ **Speech-to-Text**: Fast, accurate transcription using faster-whisper
- ✅ **Local LLM Brain**: Conversational AI using Ollama (Phase 4)
- ✅ **State Machine**: Clean state transitions (IDLE → LISTENING → TRANSCRIBING → THINKING)
- ✅ **Cross-platform**: Windows-first, but compatible with Linux/macOS
- ✅ **Robust Audio**: 16kHz mono pipeline with proper buffering
- ✅ **Spam Filtering**: Automatically filters repetition spam and garbage output
- ✅ **Privacy-First**: Runs entirely offline on your device

## System Requirements

- **OS**: Windows 10/11 (primary), Linux/macOS (compatible)
- **Python**: 3.10-3.12 recommended (3.13+ has limited hotword support)
- **Microphone**: Working audio input device
- **RAM**: 4GB+ recommended (2GB for Whisper + 2GB for LLM)
- **CPU**: Modern CPU with AVX support recommended
- **Ollama**: Required for Phase 4 LLM features (install separately)

**Note on Python 3.13+**: Hotword detection (openWakeWord) requires `tflite-runtime` which is not available for Python 3.13+. For full hotword functionality, use Python 3.10-3.12. The `--no-hotword` mode works perfectly on all Python versions.

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install requirements
pip install -r requirements.txt
```
Install Ollama (Phase 4)

For full conversational AI capabilities:

1. **Download and install Ollama**:
   - Visit: https://ollama.ai/download
   - Download the Windows/macOS/Linux installer
   - Run the installer

2. **Start Ollama** (if not auto-started):
   ```bash
   ollama serve
   ```
Use different LLM model
python run.py --ollama-model llama3.2:3b

# Disable LLM (STT-only mode)
python run.py --llm off

# 
3. **Pull a model**:
   ```bash
   # Recommended: Fast, capable model
   ollama pull llama3.1:latest
   
   # Alternative: Smaller/faster model
   ollama pull llama3.2:3b
   
   # Alternative: Larger/more capable
   ollama pull llama3.1:70brespond with AI-generated answer

### Test Mode (no Hotword)
```bash
python run.py --no-hotword
```
1. Assistant immediately starts listening
2. Speak your request
3. Pauses when you stop speaking (1.2s silence timeout)
4. Transcribes, generates response, Assistant

```bash
# Normal mode with hotword detection
python run.py

# Test mode (no hotword, immediate listening)
pyUse different LLM model
python run.py --ollama-model llama3.2:3b

# STT-only mode (no LLM)
python run.py --llm off

# Enable debug logging
python run.py --log-level DEBUG

# Use GPU for Whisper (if CUDA available)
python run.py --whisper-device cuda

# Custom Ollama URL (e.g., remote server)
python run.py --ollama-url http://192.168.1.100:11434
# List available audio devices
python run.py --list-devices

# Use specific audio device
python run.py --device 1
```

## Usage Examples

### Normal Mode (with Hotword)
```bash
python run.py
```
1. Wait for "Ready. Listening for hotword..."
2. Say "hey wyzer" or "wyzer"
3. Speak your request after the beep/prompt
4. Assistant will transcribe and display your speech

### Test Mode (no Hotword)
```bash
python run.py --no-hotword
```
1. Assistant immediately starts listening
2. Speak your request
3. Pauses when you stop speaking (1.2s silence timeout)
4. Transcribes and exits

### Custom Configuration
```bash
# Use medium model for better accuracy
python run.py --model medium

# Enable debug logging
python run.py --log-level DEBUG

# Use GPU for Whisper (if CUDA available)
python run.py --whisper-device cuda
```
├── stt/
│   ├── whisper_engine.py # Whisper STT engine
│   └── stt_router.py     # STT routing (extensible)
└── brain/                # Phase 4: LLM integration
    ├── llm_engine.py     # Ollama LLM client
    └── prompt.py         # System prompts
```
wyzer/
├── core/
│   ├── config.py         # Central configuration
│   ├── logger.py         # Logging with rich formatting
│   ├── state.py          # State machine definitions
│   └── assistant.py      # Main coordinator
├── audio/
│   ├── mic_stream.py     # Microphone capture
│   ├── vad.py            # Voice activity detection
│   ├── hotword.py        # Wake word detection
│   └── audio_utils.py    # Audio utilities
└── stt/
    ├── whisper_engine.py # Whisper STT engine
    └── stt_router.py     # STT routing (extensible)
run.py                    # Entry point
requirements.txt          # Dependencies
README.md                 # This file
```

## Configuration

Configuration can be customized via environment variables:

```bash
# Audio settings
set WYZER_SAMPLE_RATE=16000
set WYZER_CHUNK_MS=20


# LLM settings (Phase 4)
set WYZER_LLM_MODE=ollama
set WYZER_OLLAMA_URL=http://127.0.0.1:11434
set WYZER_OLLAMA_MODEL=llama3.1:latest
set WYZER_LLM_TIMEOUT=30
# Recording limits
set WYZER_MAX_RECORD_SECONDS=12.0
set WYZER_VAD_SILENCE_TIMEOUT=1.2

# VAD settings
set WYZER_VAD_THRESHOLD=0.5

# Hotword settings
set WYZER_HOTWORD_KEYWORDS=hey wyzer,wyzer
set WYZER_HOTWORD_THRESHOLD=0.5

# Whisper settings
set WYZER_WHISPER_MODEL=small
set WYZER_WHISPER_DEVICE=cpu

# Spam filter
set WYZER_MAX_TOKEN_REPEATS=6
set WYZER_MIN_TRANSCRIPT_LENGTH=2
```

Or modify defaults in [wyzer/core/config.py](wyzer/core/config.py).

## Troubleshooting

### Ollama / LLM Issues

#### Ollama Not Running
```
Error: I couldn't reach the local model. Is Ollama running?
```

**Solutions**:
1. Start Ollama server:
   ```bash
   ollama serve
   ```

2. Verify Ollama is running:
   ```bash
   # Windows PowerShell
   Test-NetConnection -ComputerName localhost -Port 11434
   
   # Or check process
   Get-Process ollama
   ```

3. Test Ollama directly:
   ```bash
   ollama list
   curl http://localhost:11434/api/tags
   ```

#### Model Not Found
```
Error: Model 'llama3.1:latest' not found. Try: ollama pull llama3.1:latest
```

**Solutions**:
1. Pull the model:
   ```bash
   ollama pull llama3.1:latest
   ```

2. List available models:
   ```bash
   ollama list
   ```

3. Use a different model:
   ```bash
   python run.py --ollama-model llama3.2:3b
   ```

#### LLM Timeout
If responses are slow or timing out:

1. **Increase timeout**:
   ```bash
   python run.py --llm-timeout 60
   ```

2. **Use smaller model**:
   ```bash
   python run.py --ollama-model llama3.2:3b
   ```

3. **Check system resources**:
   - Close other applications
   - Monitor RAM usage
   - Check CPU usage during inference

#### Firewall Blocking Ollama
If Ollama is running but can't connect:

1. **Check Windows Firewall**:
   - Allow Ollama through firewall
   - Or disable temporarily for testing

2. **Test connection**:
   ```bash
   curl http://127.0.0.1:11434/api/tags
   ```

3. **Use alternative URL**:
   ```bash
   python run.py --ollama-url http://localhost:11434
   ```

#### Disable LLM (STT-only mode)
To test without LLM:
```bash
python run.py --llm off
```

### No Audio Input / Microphone Not Working

1. **Check device permissions** (Windows):
   - Go to Settings → Privacy → Microphone
   - Ensure "Allow apps to access your microphone" is ON
   - Ensure Python is allowed

2. **List and test devices**:
   ```bash
   python run.py --list-devices
   ```
   Find your microphone's index and use it:
   ```bash
   python run.py --device 1
   ```

3. **Test microphone separately**:
   ```bash
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```

### Hotword Not Detecting

1. **Check available models**:
   The assistant will log available openWakeWord models at startup with DEBUG level:
   ```bash
   python run.py --log-level DEBUG
   ```

2. **Adjust threshold**:
   ```bash
   set WYZER_HOTWORD_THRESHOLD=0.3
   python run.py
   ```

3. **Skip hotword for testing**:
   ```bash
   python run.py --no-hotword
   ```

### VAD/Silero Not Working

The assistant includes an energy-based VAD fallback. If you see:
```
Silero VAD not available. Using energy-based VAD fallback.
```

This is normal if `silero-vad` or `torch` failed to install. The fallback works but may be less accurate.

To install Silero VAD properly:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install silero-vad
```

### Audio Sample Rate Issues

If you see warnings about sample rate mismatches:
- The assistant expects 16kHz
- Most devices support this natively
- If not, basic resampling is attempted

To force device check:
```bash
python -c "import sounddevice as sd; print(sd.query_devices(1))"  # Replace 1 with your device
```

### Transcription is Empty or Filtered

The assistant filters:
1. Transcripts shorter than 2 characters
2. Repetition spam (token repeated >6 times)
3. Low alphabetic content (garbage)

If legitimate speech is filtered:
```bash
set WYZER_MAX_TOKEN_REPEATS=10
set WYZER_MIN_TRANSCRIPT_LENGTH=1
python run.py --log-level DEBUG
```

### Performance Issues / Slow Transcription

1. **Use smaller model**:
   ```bash
   python run.py --model tiny
   ```

2. **Use int8 compute** (default, fastest):
   Already enabled by default

3. **Upgrade to GPU** (if available):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   python run.py --whisper-device cuda
   ```

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Must be 3.10+

# Check if packages are installed
pip list | findstr "sounddevice openwakeword faster-whisper"
```

### Windows-Specific Issues

1. **Long path errors**: Enable long paths in Windows
   ```
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

2. **Antivirus blocking**: Add Python and the project folder to exclusions

3. **Virtual environment activation**:
   ```bash
   # PowerShell
   venv\Scripts\Activate.ps1
   
   # CMD
   venv\Scripts\activate.bat
   ```

## Development Notes

### State Machine

```
IDLE → (hotword detected) → LISTENING
LISTENING → (speech + silence timeout) → TRANSCRIBING
TRANSCRIBING → (done) → THINKING
THINKING → (LLM response) → IDLE
```

In `--no-hotword` mode:
```
IDLE → (immediate) → LISTENING → TRANSCRIBING → THINKING → EXIT
```

In `--llm off` mode:
```
IDLE → (hotword) → LISTENING → TRANSCRIBING → IDLE
```

### Audio Pipeline
4. **Thinking** (Phase 4): Transcript → Ollama LLM → Response → Display

### Non-Blocking Processing

Phase 4 uses background threads to prevent audio queue overflow:
- **Transcription thread**: Processes audio while main loop drains mic queue
- **Thinking thread**: LLM processing while main loop continues draining audio
- This ensures real-time audio capture never blocks on slow STT/LLM operations

1. **Capture**: sounddevice → 16kHz mono float32 → Queue
2. **Detection**: 
   - IDLE: hotword detector checks each frame
   - LISTENING: VAD checks each frame, buffers audio
3. **Transcription**: Concatenate buffer → Whisper → Filter → Display

### Thread Safety

- One audio callback thread (sounddevice)
- One main loop thread (queue consumer)
- Queue with maxsize (drops frames if full, logs warning)
- No complex locking needed

### Adding Custom Hotwords

openWakeWord supports custom models. To add your own:

1. Train or download a custom `.tflite` or `.onnx` model
2. Set path via environment:
   ```bash
   set WYZER_HOTWORD_MODEL_PATH=path/to/model.onnx
   ```conversation memory**: Each interaction is independent (stateless)
2. **No TTS**: No audio output yet (text-only responses)
3. **No tool calling**: LLM can't perform actions or access external data
4. **No function calls**: Pure conversational AI only
5. **Basic resampling**: Uses linear interpolation (scipy would be better)
6. **Energy VAD fallback**: Less accurate than Silero
7. **Single-turn conversations**: No context from previous exchanges

## Future Phases (Not Implemented)

- Phase 5: Tool calling system
- Phase 6: Conversation memory / context
- Phase 7: GUI/system tray
- Phase 8: Text-to-speech (TTS)
- Phase 9: Multi-turn conversationsses linear interpolation (scipy would be better)
6. **Energy VAD fallback**: Less accurate than Silero

## Future Phases (Not Implemented)

- Phase 4: LLM integration
- Phase 5: Tool calling system
- Phase 6: Conversation memory
- Phase 7: GUI/system tray

## License

[Your License Here]

## Support

For issues or questions:
1. Check this README's troubleshooting section
2. Enable debug logging: `python run.py --log-level DEBUG`
3. Check logs for specific error messages

- **Phase 4 LLM**: Uses Python stdlib `urllib` (no additional dependencies)
- **Ollama**: External dependency (install separately)
## Dependencies

Core libraries:
- `sounddevice`: Microphone capture
- `numpy`: Audio processing
- `openwakeword`: Hotword detection
- `silero-vad`: Voice activity detection (with fallback)
- `faster-whisper`: Speech-to-text
- `onnxruntime`: ONNX model support
- `rich`: Pretty console output (optional)

See [requirements.txt](requirements.txt) for complete list.
