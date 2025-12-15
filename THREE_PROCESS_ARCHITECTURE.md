# Why Wyzer Runs 3 Python Processes

## Overview
Wyzer AI Assistant uses a **multiprocess architecture** with 3 separate Python processes to maximize performance and responsiveness. This design separates real-time audio handling from CPU-intensive language processing.

---

## The 3 Processes

### 1. **Main Process (Parent/Orchestrator)**
**Responsibility:** System coordination and entry point

- Initializes the application
- Spawns the Brain Worker process
- Manages lifecycle and graceful shutdown
- Handles configuration loading
- Acts as the IPC hub (if needed for orchestration)

**Key Files:** `run.py`, `wyzer/core/assistant.py` (WyzerAssistantMultiprocess)

---

### 2. **Core Process (Real-Time Audio Handler)**
**Responsibility:** Low-latency audio capture and hotword detection

This process handles everything that needs to respond **immediately** to user input:

- **Microphone Stream** - Captures audio in real-time with minimal latency
- **VAD (Voice Activity Detection)** - Detects when the user starts speaking
- **Hotword Detection** - Listens for activation keywords (e.g., "Wyzer")
- **State Machine** - Manages assistant states (IDLE → LISTENING → THINKING → SPEAKING)
- **Interruption Handling** - Allows users to interrupt responses with barge-in capability

**Why separate?** Audio capture cannot be delayed. If this runs in the same process as the LLM, heavy language processing would cause missed audio frames or unresponsive hotword detection.

**Key Files:** `wyzer/core/assistant.py` (main loop), `wyzer/audio/mic_stream.py`, `wyzer/audio/hotword.py`

---

### 3. **Brain Worker Process (Compute-Intensive Tasks)**
**Responsibility:** Speech-to-text, language understanding, and text-to-speech

This process handles CPU-heavy and I/O-heavy operations:

- **STT (Speech-to-Text)** - Whisper model for transcribing audio
- **Orchestrator** - LLM inference (Ollama/local models) + tool execution
- **Tool Execution** - Run system tasks (open files, check weather, get time, etc.)
- **TTS (Text-to-Speech)** - Piper for voice synthesis

**Why separate?** These operations block and consume significant CPU/memory. Running them in the same process as real-time audio would cause:
- Dropped audio frames during LLM inference
- Unresponsive hotword detection while processing
- Janky user experience

**Key Files:** `wyzer/core/brain_worker.py`, `wyzer/core/process_manager.py`

---

## Process Communication (IPC)

The Core and Brain Worker processes communicate via **queues**:

```
┌─────────────────────────────────────────────────────┐
│           Main Process (Parent)                      │
│  - Config loading                                   │
│  - Shutdown coordination                            │
└────────────────┬────────────────────────────────────┘
                 │
                 │ spawns
                 │
    ┌────────────▼─────────────────┐
    │   Core Process (Real-Time)    │
    │ ┌─────────────────────────┐   │
    │ │ Mic Stream              │   │
    │ │ VAD + Hotword Detection │   │
    │ │ State Machine           │   │
    │ └────────┬────────────────┘   │
    │          │ sends audio + text │
    └──────────┼────────────────────┘
               │ core_to_brain_q
               │ (Queue: audio/text requests)
               │
               ▼
    ┌──────────────────────────────┐
    │ Brain Worker (Compute)       │
    │ ┌──────────────────────────┐ │
    │ │ STT (Whisper)            │ │
    │ │ Orchestrator (LLM/Tools) │ │
    │ │ TTS (Piper)              │ │
    │ └──────────┬───────────────┘ │
    │            │ sends results   │
    └────────────┼────────────────┘
                 │ brain_to_core_q
                 │ (Queue: transcript/response/audio)
                 ▼
             Core Process
         (plays audio/updates state)
```

**Queue Details:**
- `core_to_brain_q`: Core → Brain Worker (audio chunks, requests)
- `brain_to_core_q`: Brain Worker → Core (results, transcripts, TTS audio)
- Both queues are **thread-safe** and **size-limited** (default 50 items max)

---

## Why Not Use Threads Instead?

Python has the **Global Interpreter Lock (GIL)**, which prevents true parallel execution of Python code in threads. With threads:

- ❌ Only one thread executes Python bytecode at a time
- ❌ Audio capture could be interrupted by LLM processing
- ❌ Hotword detection would lag during inference

**With Processes:**
- ✅ True parallelism across CPU cores
- ✅ Core process runs at native speed unblocked
- ✅ Brain Worker can use multiple cores for inference

---

## Process Lifecycle

```
┌─────────────────────────────────────────┐
│ User runs: python run.py                │
└──────────────┬──────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Main Process Starts   │
    └──────────┬───────────┘
               │
               │ (if --single-process flag not set)
               │ spawn Brain Worker
               ▼
    ┌──────────────────────────────┐
    │ Brain Worker Process Starts   │
    │ - Init STT/LLM/TTS           │
    │ - Wait on core_to_brain_q    │
    └──────────────────────────────┘
               ▲
               │
    ┌──────────┴──────────┐
    │ Core Process        │
    │ - Listen for audio  │
    │ - Send to Brain WKR │
    │ - Play responses    │
    └────────────────────┘

When user interrupts (Ctrl+C):
  Core Process → shutdown signal → Brain Worker → Clean exit
```

---

## Single-Process Mode (`--single-process`)

For debugging, you can run in single-process mode:

```bash
python run.py --single-process
```

This uses `WyzerAssistant` (non-multiprocess) instead of `WyzerAssistantMultiprocess`. Everything runs in one process, but:
- ⚠️ May drop audio during LLM processing
- ⚠️ Hotword detection lags during thinking
- ⚠️ Useful only for debugging and simple testing

---

## Summary

| Process | Latency Requirement | Work Type | Tools |
|---------|-------------------|-----------|-------|
| **Core** | Real-time (< 10ms) | Audio I/O, state mgmt | Mic stream, VAD, hotword |
| **Brain** | Batch processing (ok to wait) | CPU-intensive compute | STT, LLM, tools, TTS |
| **Main** | Startup/shutdown | Coordination | Config, spawn/monitor |

The 3-process design ensures Wyzer stays responsive to user voice input while allowing heavy language processing to run unblocked on separate CPU cores.
