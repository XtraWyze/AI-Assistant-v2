"""
Wyzer Memory Subsystem - Phase 11

Two-layer memory architecture:
1. Session Memory (RAM-only): Recent conversation turns, cleared on restart
2. Long-Term Memory (disk): Structured records with explicit user-initiated saves via "remember X"

Phase 11 Structured Memory Model:
Each memory record contains: id, type, key, value, tags, created_at, updated_at, source

All memory operations are explicit and user-initiated - no automatic writes.
"""

from wyzer.memory.memory_manager import (
    MemoryManager,
    get_memory_manager,
    MEMORY_RECORD_TYPES,
    DEFAULT_RECORD_TYPE,
    DEFAULT_SOURCE,
)

__all__ = [
    "MemoryManager",
    "get_memory_manager",
    "MEMORY_RECORD_TYPES",
    "DEFAULT_RECORD_TYPE",
    "DEFAULT_SOURCE",
]
