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

---

**Any change violating this architecture requires explicit justification and approval.**
