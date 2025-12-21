# Wyzer LLM Binaries Directory

Place your llama.cpp server binary here.

## Required Files

- `llama-server.exe` (Windows) or `llama-server` (Linux/Mac)

## Download llama-server

1. Go to [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
2. Download the appropriate build for your system:
   - Windows: `llama-*-bin-win-*.zip`
   - Linux: `llama-*-bin-ubuntu-*.zip`
   - macOS: `llama-*-bin-macos-*.zip`
3. Extract `llama-server` (or `llama-server.exe`) to this directory

## Usage

Once placed here, start Wyzer with:

```bash
python run.py --llm llamacpp
```

Or specify a custom path:

```bash
python run.py --llm llamacpp --llamacpp-bin /path/to/llama-server
```
