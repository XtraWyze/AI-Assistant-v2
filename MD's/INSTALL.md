# Wyzer AI Assistant - Installation Guide

> *Last Updated: December 2025*

The `requirements.txt` file can cause dependency conflicts. Use this step-by-step guide instead.

## Quick Install (Recommended)

### 1. Create Virtual Environment
```bash
python -m venv venv_new
```

### 2. Activate Virtual Environment
**Windows (PowerShell):**
```powershell
.\venv_new\Scripts\Activate.ps1
```
**Windows (CMD):**
```cmd
venv_new\Scripts\activate.bat
```

### 3. Upgrade pip
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies (In Order)

Install packages in this specific order to avoid conflicts:

```bash
# Core numeric/scientific
pip install numpy==1.26.4
pip install scipy

# Audio
pip install sounddevice soundfile audioread librosa pycaw comtypes

# VAD & Hotword
pip install silero-vad openwakeword

# Speech-to-Text (upgrade ctranslate2 to avoid deprecation warning)
pip install faster-whisper
pip install --upgrade ctranslate2

# LLM & AI
pip install ollama transformers openai

# Web & API
pip install fastapi uvicorn requests httpx aiohttp websockets

# Data & NLP
pip install pandas spacy nltk beautifulsoup4 pydantic

# Utilities
pip install rich colorama coloredlogs python-dotenv psutil keyboard pynput

# Image Processing
pip install opencv-python-headless Pillow

# Windows Automation
pip install pywin32 pyautogui screeninfo

# Testing
pip install pytest
```

### 5. Download Hotword Models
```bash
python -c "from openwakeword.utils import download_models; download_models()"
```

### 6. Verify Installation
```bash
python -c "import wyzer; print('Wyzer imported successfully')"
python scripts/test_tool_pool_smoke.py
```

## Run the Assistant
```bash
python run.py
```
Or double-click `run.bat`

---

## Troubleshooting

### "faster-whisper not available"
```bash
pip install faster-whisper
```

### "No ONNX models found" (hotword)
```bash
python -c "from openwakeword.utils import download_models; download_models()"
```

### Dependency conflicts
Start fresh:
```bash
deactivate
rmdir /s /q venv_new
python -m venv venv_new
```
Then follow the install steps above.

### Wrong Python environment
Make sure you see `(venv_new)` in your terminal prompt before running commands.

---

## All 26 Tools Available

| Category | Tools |
|----------|-------|
| Time/Info | `get_time`, `get_system_info`, `get_location` |
| Weather | `get_weather_forecast` |
| Media | `media_play_pause`, `media_next`, `media_previous`, `get_now_playing` |
| Volume | `volume_control`, `volume_up`, `volume_down`, `volume_mute_toggle` |
| Audio | `set_audio_output_device` |
| Windows | `close_window`, `focus_window`, `maximize_window`, `minimize_window`, `move_window_to_monitor`, `get_window_monitor` |
| Monitor | `monitor_info` |
| Apps | `open_target`, `open_website` |
| Storage | `system_storage_list`, `system_storage_open`, `system_storage_scan` |
| Library | `local_library_refresh` |
| Timer | `timer` |
| Search | `google_search_open` |
