# Wyzer Architecture Lock

> **Last Updated: December 2025**

## Core Architecture Principles

This document defines the architectural boundaries that **must not be violated** without explicit justification.

### Component Responsibilities

| Component | Location | Responsibility | Constraints |
|-----------|----------|----------------|-------------|
| **Assistant** | `wyzer/core/assistant.py` | Audio + state management only | NO tool logic, NO LLM calls |
| **Orchestrator** | `wyzer/core/orchestrator.py` | Intent routing + tool execution | NO audio, NO state management |
| **Brain Worker** | `wyzer/core/brain_worker.py` | STT, LLM, TTS processing | Runs in separate process |
| **Hybrid Router** | `wyzer/core/hybrid_router.py` | Fast-path deterministic routing | Regex-based, NO LLM calls |
| **Multi-Intent Parser** | `wyzer/core/multi_intent_parser.py` | Splitting multi-command queries | Uses hybrid router for clauses |
| **Tools** | `wyzer/tools/` | Stateless, JSON-only | NO I/O, NO global state |
| **Local Library** | `wyzer/local_library/` | Indexing + resolution only | NO tool logic |
| **Memory Manager** | `wyzer/memory/` | Session + long-term memory | File-based persistence |
| **Autonomy Policy** | `wyzer/policy/` | Risk assessment + confirmation | Deterministic, NO LLM |
| **Window Watcher** | `wyzer/world/` | Multi-monitor window tracking | NO OCR, NO screenshots |
| **World State** | `wyzer/context/` | In-RAM state tracking | Written only by tools |

### Multiprocess Architecture

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
│  │(Whisper)│  │   + Tools  │  │       │  │  (Piper)    │   │
│  └─────────┘  └────────────┘  └───────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   TOOL WORKER POOL                          │
│     [Worker 1] [Worker 2] [Worker 3] ... [Worker N]        │
│     (Isolated processes for parallel tool execution)        │
└─────────────────────────────────────────────────────────────┘
```

### Tool Rules

1. **Stateless**: Tools must not maintain state between calls
2. **JSON-Only**: Input and output must be JSON-serializable
3. **No I/O in Constructor**: All I/O happens in `run()` method
4. **Error Dict**: Never raise exceptions, always return `{"error": {...}}`
5. **Worker Pool Aware**: Tools run in separate worker processes

### Routing Rules

1. **Hybrid Router First**: All queries go through hybrid router
2. **Confidence Threshold**: Tool routes require confidence >= 0.75
3. **LLM Fallback**: Only route to LLM when hybrid router can't decide
4. **Creative Content**: Stories/poems always route to LLM (reply-only)

### Memory Rules

1. **Session Memory**: RAM-only, cleared on restart
2. **Long-Term Memory**: Disk-persisted in `wyzer/data/memory.json`
3. **Memory Injection**: Controlled by `use_memories` flag
4. **No Auto-Learn**: Memories only saved via explicit "remember X" commands

### Autonomy Rules (Phase 11)

1. **Mode OFF Default**: No confirmation prompts unless explicitly enabled
2. **Deterministic Policy**: All decisions based on confidence + risk, NO LLM
3. **Risk Classification**: Tools categorized as low/medium/high risk
4. **Confirmation Flow**: High-risk actions require yes/no voice confirmation
5. **Timeout Safety**: Pending confirmations expire after configurable timeout

### Window Watcher Rules (Phase 12)

1. **Metadata Only**: Track titles, processes, positions - NO content
2. **No OCR**: Never extract text from screen
3. **No Screenshots**: Never capture screen images
4. **Background Thread**: Runs in dedicated thread, not blocking audio
5. **World State Updates**: Push snapshots to WorldState for context

### World State Rules

1. **RAM-Only**: Never persisted to disk
2. **Read Anywhere**: Any component can read world state
3. **Write Deterministically**: Only tools and state updates can write
4. **Never LLM-Written**: LLM cannot modify world state directly

---

**Any change violating this architecture requires explicit justification and approval.**
