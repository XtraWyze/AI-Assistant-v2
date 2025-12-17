"""
Wyzer Memory Subsystem - Phase 7

Two-layer memory architecture:
1. Session Memory (RAM-only): Recent conversation turns, cleared on restart
2. Long-Term Memory (disk): Explicit user-initiated saves via "remember X"

All memory operations are explicit and user-initiated - no automatic writes.
"""

from wyzer.memory.memory_manager import MemoryManager, get_memory_manager

__all__ = ["MemoryManager", "get_memory_manager"]
