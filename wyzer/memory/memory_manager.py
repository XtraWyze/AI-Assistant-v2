"""
Memory Manager for Wyzer AI Assistant - Phase 7

Provides:
1. Session Memory: In-memory conversation context (last N turns)
2. Long-Term Memory: Explicit disk storage via user commands

IMPORTANT: All disk writes are explicit and user-initiated.
Session memory is RAM-only and cleared on restart.
"""

import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

from wyzer.core.config import Config
from wyzer.core.logger import get_logger


def _get_memory_file_path() -> Path:
    """Get the path to the memory.json file."""
    # Use wyzer/data directory for consistency with existing data files
    base_dir = Path(__file__).parent.parent / "data"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "memory.json"


def _normalize_for_matching(text: str) -> str:
    """
    Normalize text for matching purposes.
    
    - lowercase
    - trim whitespace
    - strip trailing punctuation (. ! ?)
    - collapse multiple whitespace to single space
    
    Args:
        text: Original text
        
    Returns:
        Normalized text for matching
    """
    if not text:
        return ""
    
    # Lowercase and trim
    result = text.lower().strip()
    
    # Strip trailing punctuation
    result = re.sub(r'[.!?]+$', '', result)
    
    # Collapse whitespace
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()


class MemoryManager:
    """
    Manages session and long-term memory for Wyzer.
    
    Session Memory (RAM-only):
    - Stores last N turns of (user_text, assistant_text)
    - Cleared on restart
    - Used to provide context to LLM prompts
    
    Long-Term Memory (disk, explicit only):
    - JSON file with user's saved facts/notes
    - Only modified by explicit "remember X" / "forget X" commands
    - Persists across restarts
    """
    
    def __init__(self, max_session_turns: Optional[int] = None):
        """
        Initialize the memory manager.
        
        Args:
            max_session_turns: Maximum session turns to keep (default from Config)
        """
        self._lock = RLock()
        self._max_turns = max_session_turns or getattr(Config, 'SESSION_MEMORY_TURNS', 10)
        
        # Session memory: list of (user_text, assistant_text) tuples
        self._session_turns: List[Tuple[str, str]] = []
        
        # Session facts: short facts extracted or noted during session (not persisted)
        self._session_facts: List[str] = []
        
        # Phase 9: Promoted memory buffer (RAM-only, session-scoped)
        # These are user-approved memories for this conversation only
        self._promoted_memories: List[str] = []
        
        # Phase 9: Last recall result (for "use that" promotion)
        self._last_recall_result: Optional[str] = None
        
        # Phase 9: Recently forgotten texts (for LLM redaction)
        # This prevents LLM from using forgotten info from session context
        self._recently_forgotten: set = set()
        
        # Memory file path
        self._memory_file = _get_memory_file_path()
        
        # Session flag: inject all long-term memories into LLM prompts
        # Default from Config (which respects CLI > env var > config default)
        # Can be toggled via voice commands during session
        self._use_memories: bool = getattr(Config, 'USE_MEMORIES', True)
    
    # =========================================================================
    # Session Memory (RAM-only)
    # =========================================================================
    
    def add_session_turn(self, user_text: str, assistant_text: str, preserve_recall: bool = False) -> None:
        """
        Add a conversation turn to session memory.
        
        This is called after each interaction to track context.
        Session memory is never written to disk.
        
        Args:
            user_text: User's input text
            assistant_text: Assistant's response
            preserve_recall: If True, don't clear last_recall_result (used for RECALL commands)
        """
        with self._lock:
            self._session_turns.append((user_text.strip(), assistant_text.strip()))
            # Trim to max turns
            if len(self._session_turns) > self._max_turns:
                self._session_turns = self._session_turns[-self._max_turns:]
            
            # Phase 9 hardening: Clear last recall result after any NON-recall session turn
            # This prevents stale "use that" references across tool runs or other commands.
            # "use that" is only valid immediately after a RECALL command.
            # RECALL commands pass preserve_recall=True to keep the result for "use that".
            if not preserve_recall:
                self._last_recall_result = None
    
    def get_session_context(self, max_turns: Optional[int] = None) -> str:
        """
        Get formatted session context for LLM prompt injection.
        
        Returns a compact string with recent conversation history.
        
        Args:
            max_turns: Override max turns to include (default: all available up to config max)
            
        Returns:
            Formatted string of recent conversation, or empty string if none
        """
        with self._lock:
            if not self._session_turns:
                return ""
            
            limit = min(max_turns or self._max_turns, len(self._session_turns))
            recent = self._session_turns[-limit:]
            
            lines = []
            for user_text, assistant_text in recent:
                # Keep it compact - truncate long texts
                user_short = user_text[:100] + "..." if len(user_text) > 100 else user_text
                asst_short = assistant_text[:150] + "..." if len(assistant_text) > 150 else assistant_text
                lines.append(f"User: {user_short}")
                lines.append(f"Wyzer: {asst_short}")
            
            return "\n".join(lines)
    
    def get_session_turns_count(self) -> int:
        """Get number of turns in session memory."""
        with self._lock:
            return len(self._session_turns)
    
    def clear_session(self) -> None:
        """Clear session memory (useful for testing)."""
        with self._lock:
            self._session_turns.clear()
            self._session_facts.clear()
    
    # =========================================================================
    # Long-Term Memory (disk, explicit only)
    # =========================================================================
    
    def _load_memories(self) -> List[Dict[str, Any]]:
        """Load memories from disk. Returns empty list if file doesn't exist."""
        try:
            if self._memory_file.exists():
                with open(self._memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger = get_logger()
            logger.warning(f"[MEMORY] Failed to load memories: {e}")
        return []
    
    def _save_memories(self, memories: List[Dict[str, Any]]) -> bool:
        """
        Save memories to disk atomically.
        
        Uses write-to-temp-then-rename for atomic writes on Windows.
        
        Args:
            memories: List of memory entries to save
            
        Returns:
            True if save succeeded, False otherwise
        """
        logger = get_logger()
        try:
            # Ensure parent directory exists
            self._memory_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temp file first (atomic write pattern)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.json',
                prefix='memory_tmp_',
                dir=str(self._memory_file.parent)
            )
            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                    json.dump(memories, f, indent=2, ensure_ascii=False)
                
                # Atomic rename (Windows: need to remove target first)
                if os.name == 'nt' and self._memory_file.exists():
                    os.remove(self._memory_file)
                os.rename(temp_path, self._memory_file)
                return True
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                raise
        except (IOError, OSError) as e:
            logger.error(f"[MEMORY] Failed to save memories: {e}")
            return False
    
    def remember(self, text: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Save a memory to long-term storage (explicit user command).
        
        Deduplication (Phase 7 polish):
        - If an existing memory has the same index_text, replace it (update timestamp)
        - This prevents clutter from "my name is Levi" / "my name's Levi" / "I'm Levi"
        
        Args:
            text: The text to remember
            tags: Optional tags for categorization
            
        Returns:
            Dict with the saved entry (id, created_at, text, tags) or error
        """
        logger = get_logger()
        
        with self._lock:
            text = text.strip()
            if not text:
                return {"ok": False, "error": "Nothing to remember"}
            
            new_index_text = _normalize_for_matching(text)
            
            # Create new entry with normalized index_text for matching
            entry = {
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "text": text,  # Original text for display
                "index_text": new_index_text,  # Normalized for matching
                "tags": tags or []
            }
            
            # Load existing memories
            memories = self._load_memories()
            
            # Dedupe: remove any existing entry with the same index_text
            dedupe_removed = 0
            remaining = []
            for mem in memories:
                existing_index = mem.get("index_text") or _normalize_for_matching(mem.get("text", ""))
                if existing_index == new_index_text:
                    dedupe_removed += 1
                else:
                    remaining.append(mem)
            
            # Append new entry
            remaining.append(entry)
            
            if self._save_memories(remaining):
                if dedupe_removed > 0:
                    logger.info(f"[MEMORY] remember (updated): {text[:50]}{'...' if len(text) > 50 else ''}")
                else:
                    logger.info(f"[MEMORY] remember: {text[:50]}{'...' if len(text) > 50 else ''}")
                return {"ok": True, "entry": entry, "replaced": dedupe_removed > 0}
            else:
                return {"ok": False, "error": "Failed to save memory"}
    
    def forget(self, query: str) -> Dict[str, Any]:
        """
        Remove memories matching a query (explicit user command).
        
        Performs normalized substring matching:
        - Case-insensitive
        - Ignores trailing punctuation
        - Collapses whitespace
        
        Args:
            query: Text to search for in memories
            
        Returns:
            Dict with removed entries or error
        """
        logger = get_logger()
        
        with self._lock:
            query_normalized = _normalize_for_matching(query)
            if not query_normalized:
                return {"ok": False, "error": "Nothing to forget", "removed": []}
            
            memories = self._load_memories()
            if not memories:
                return {"ok": True, "removed": [], "message": "No memories found"}
            
            # Find and remove matching entries using normalized matching
            removed = []
            remaining = []
            for entry in memories:
                # Use index_text if available (new entries), else normalize on-the-fly (legacy)
                index_text = entry.get("index_text") or _normalize_for_matching(entry.get("text", ""))
                if query_normalized in index_text:
                    removed.append(entry)
                else:
                    remaining.append(entry)
            
            if removed:
                if self._save_memories(remaining):
                    logger.info(f"[MEMORY] forget: removed {len(removed)} entries matching '{query}'")
                    # Phase 9: Clear promoted memories and track for redaction
                    self.clear_promoted()
                    for entry in removed:
                        text = entry.get("text", "")
                        if text:
                            transformed = _transform_first_to_second_person(text)
                            if transformed:
                                self._recently_forgotten.add(transformed)
                            else:
                                self._recently_forgotten.add(text)
                    return {"ok": True, "removed": removed}
                else:
                    return {"ok": False, "error": "Failed to save changes", "removed": []}
            else:
                logger.debug(f"[MEMORY] forget: no matches for '{query}' (total memories: {len(memories)})")
                return {"ok": True, "removed": [], "message": f"No memories matched '{query}'"}
    
    def forget_last(self) -> Dict[str, Any]:
        """
        Remove the most recently saved memory (explicit user command).
        
        Used for "forget that" / "forget it" / "delete that" commands.
        
        Returns:
            Dict with removed entry or error
        """
        logger = get_logger()
        
        with self._lock:
            memories = self._load_memories()
            if not memories:
                return {"ok": True, "removed": None, "message": "No memories to forget"}
            
            # Remove the last (most recent) entry
            removed = memories.pop()
            
            if self._save_memories(memories):
                removed_text = removed.get("text", "")[:50]
                logger.info(f"[MEMORY] forget_last: removed '{removed_text}{'...' if len(removed.get('text', '')) > 50 else ''}'")
                # Phase 9: Clear promoted memories and track for redaction
                self.clear_promoted()
                text = removed.get("text", "")
                if text:
                    transformed = _transform_first_to_second_person(text)
                    if transformed:
                        self._recently_forgotten.add(transformed)
                    else:
                        self._recently_forgotten.add(text)
                return {"ok": True, "removed": removed}
            else:
                return {"ok": False, "error": "Failed to save changes", "removed": None}
    
    def list_memories(self) -> List[Dict[str, Any]]:
        """
        List all saved memories.
        
        Returns:
            List of memory entries
        """
        logger = get_logger()
        with self._lock:
            memories = self._load_memories()
            logger.info(f"[MEMORY] list: {len(memories)} entries")
            return memories
    
    def has_memories(self) -> bool:
        """Check if there are any saved memories."""
        with self._lock:
            return len(self._load_memories()) > 0

    # =========================================================================
    # Session Flag: use_memories (ALL long-term memories → LLM prompts)
    # =========================================================================
    
    def set_use_memories(self, enabled: bool, source: str = "unknown") -> bool:
        """
        Set the use_memories session flag.
        
        When enabled, ALL long-term memories are injected into LLM prompts.
        Session-scoped only (cleared on restart).
        
        Args:
            enabled: True to enable memory injection, False to disable
            source: Source of the change (voice_command, cli_flag, etc.)
            
        Returns:
            True if state changed, False if already in requested state
        """
        logger = get_logger()
        with self._lock:
            if self._use_memories == enabled:
                return False
            self._use_memories = enabled
            logger.info(f"[STATE] use_memories={enabled} (source={source})")
            return True
    
    def get_use_memories(self) -> bool:
        """Check if use_memories flag is enabled."""
        with self._lock:
            return self._use_memories
    
    def get_all_memories_for_injection(self, max_bullets: int = 30, max_chars: int = 1200) -> str:
        """
        Get ALL long-term memories formatted for LLM prompt injection.
        
        Only called when use_memories flag is True.
        Deduplicates and compresses into bullet points with hard caps.
        
        Args:
            max_bullets: Maximum number of bullet points (default 30)
            max_chars: Maximum total characters (default 1200)
            
        Returns:
            Formatted block with labeled memory section, or empty string
        """
        with self._lock:
            if not self._use_memories:
                return ""
            
            memories = self._load_memories()
            if not memories:
                return ""
            
            # Deduplicate by normalized text
            seen_normalized = set()
            unique_texts = []
            for mem in memories:
                text = mem.get("text", "").strip()
                if not text:
                    continue
                normalized = _normalize_for_matching(text)
                if normalized not in seen_normalized:
                    seen_normalized.add(normalized)
                    # Transform to second person for consistency
                    from wyzer.memory.command_detector import _transform_first_to_second_person
                    transformed = _transform_first_to_second_person(text)
                    unique_texts.append(transformed if transformed else text)
            
            if not unique_texts:
                return ""
            
            # Build bullet points with caps
            bullets = []
            total_chars = 0
            header = "[LONG-TERM MEMORY — facts about the user]\n"
            footer = "\nIMPORTANT: Use this information to answer questions about the user. If they ask about their name, preferences, or anything listed above, refer to this memory."
            reserved = len(header) + len(footer)
            
            for text in unique_texts[:max_bullets]:
                bullet = f"- {text}"
                if total_chars + len(bullet) + 1 > max_chars - reserved:
                    break
                bullets.append(bullet)
                total_chars += len(bullet) + 1  # +1 for newline
            
            if not bullets:
                return ""
            
            return header + "\n".join(bullets) + footer

    def recall(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories for a query (READ-ONLY, no disk writes).
        
        Phase 8: Explicit recall - deterministic search without LLM.
        
        Scoring (deterministic, priority order):
        1. Exact match on index_text: score 1000
        2. Substring match: score 100
        3. Word overlap: score = 10 * overlap_count
        
        Tie-break: newer memories first (by created_at timestamp, descending)
        
        Args:
            query: Search query text
            limit: Maximum number of results to return (default 5)
            
        Returns:
            List of matching memory entries (with original text), sorted by score/timestamp.
            Empty list if no matches. Does NOT write to disk.
        """
        logger = get_logger()
        
        with self._lock:
            query_normalized = _normalize_for_matching(query)
            if not query_normalized:
                return []
            
            memories = self._load_memories()
            if not memories:
                return []
            
            query_words = set(query_normalized.split())
            
            scored: List[Tuple[int, str, Dict[str, Any]]] = []
            
            for mem in memories:
                # Use index_text if available, else normalize on-the-fly
                index_text = mem.get("index_text") or _normalize_for_matching(mem.get("text", ""))
                if not index_text:
                    continue
                
                score = 0
                
                # Exact match (highest priority)
                if query_normalized == index_text:
                    score = 1000
                # Substring match (second priority)
                elif query_normalized in index_text:
                    score = 100
                else:
                    # Word overlap (third priority)
                    mem_words = set(index_text.split())
                    overlap = query_words & mem_words
                    if overlap:
                        score = 10 * len(overlap)
                
                if score > 0:
                    # Use created_at for tie-break (newer first = descending sort)
                    created_at = mem.get("created_at", "")
                    scored.append((score, created_at, mem))
            
            # Sort by score descending, then by timestamp descending
            # Tuple sort: (-score, -timestamp_str) - timestamps are ISO format, lexicographic works
            scored.sort(key=lambda x: (-x[0], x[1]), reverse=False)
            # Fix: we want descending timestamp for tie-break
            scored.sort(key=lambda x: (-x[0], x[1] if x[1] else ""), reverse=False)
            # Actually correct sort: primary by -score, secondary by timestamp desc
            scored.sort(key=lambda x: (-x[0], ""), reverse=False)
            # Simpler: just sort properly
            scored_final = sorted(scored, key=lambda x: (-x[0], x[1]), reverse=False)
            # For same score, we want newer (larger timestamp) first, so reverse=True for timestamp
            # Let's do it step by step:
            scored_final = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)
            
            results = [entry for (_, _, entry) in scored_final[:limit]]
            
            logger.debug(f"[MEMORY] recall '{query}': {len(results)} matches (searched {len(memories)})")
            
            return results

    # =========================================================================
    # Phase 9: Promoted Memory (RAM-only, session-scoped)
    # =========================================================================
    
    def promote(self, memory_text: str) -> bool:
        """
        Promote a memory for temporary use in the current session.
        
        Phase 9: User has explicitly authorized this memory for LLM use.
        
        IMPORTANT:
        - Stored only in RAM (not written to disk)
        - Cleared on restart
        - Cleared by explicit user command
        
        Args:
            memory_text: The memory text to promote
            
        Returns:
            True if promoted successfully
        """
        logger = get_logger()
        with self._lock:
            if not memory_text or not memory_text.strip():
                return False
            
            text = memory_text.strip()
            
            # Avoid duplicates
            normalized = _normalize_for_matching(text)
            for existing in self._promoted_memories:
                if _normalize_for_matching(existing) == normalized:
                    logger.debug(f"[MEMORY] promote: already promoted '{text[:50]}'")
                    return True  # Already promoted, still success
            
            self._promoted_memories.append(text)
            logger.info(f"[MEMORY] promote: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            return True
    
    def clear_promoted(self) -> int:
        """
        Clear all promoted memories (user command: "stop using that").
        
        Returns:
            Number of promoted memories cleared
        """
        logger = get_logger()
        with self._lock:
            count = len(self._promoted_memories)
            self._promoted_memories.clear()
            self._last_recall_result = None
            logger.info(f"[MEMORY] clear_promoted: cleared {count} promoted memories")
            return count
    
    def get_promoted_context(self) -> str:
        """
        Get formatted promoted memory context for LLM prompt injection.
        
        Phase 9: Only returns promoted memories (user-approved for this session).
        This is the ONLY way long-term memory reaches the LLM - after user consent.
        
        Returns:
            Formatted string like:
                User-approved memory for this conversation:
                - your name is Levi
            
            Empty string if no promoted memories.
        """
        with self._lock:
            if not self._promoted_memories:
                return ""
            
            lines = ["User-approved memory for this conversation:"]
            for mem in self._promoted_memories:
                # Use pronoun transformation for consistency
                transformed = _transform_first_to_second_person(mem)
                if transformed:
                    lines.append(f"- {transformed}")
                else:
                    lines.append(f"- {mem}")
            
            return "\n".join(lines)
    
    def get_promoted_count(self) -> int:
        """Get number of promoted memories."""
        with self._lock:
            return len(self._promoted_memories)
    
    def get_redaction_block(self) -> str:
        """
        Get formatted redaction block for LLM prompt injection.
        
        Phase 9: If user has forgotten certain facts, prevent LLM from using
        them even if they appear in session context.
        
        Returns:
            Formatted string like:
                The user has asked me to forget and not use these details:
                - your name is levi
            
            Empty string if nothing forgotten this session.
        """
        with self._lock:
            if not self._recently_forgotten:
                return ""
            
            lines = ["The user has asked me to forget and not use these details:"]
            for text in self._recently_forgotten:
                lines.append(f"- {text}")
            
            return "\n".join(lines)
    
    def has_redactions(self) -> bool:
        """Check if there are any recently forgotten items."""
        with self._lock:
            return len(self._recently_forgotten) > 0
    
    def clear_redactions(self) -> None:
        """Clear redactions (for testing or session reset)."""
        with self._lock:
            self._recently_forgotten.clear()
    
    def set_last_recall_result(self, memory_text: Optional[str]) -> None:
        """
        Store the most recent recall result for "use that" promotion.
        
        Args:
            memory_text: The recalled memory text (or None to clear)
        """
        with self._lock:
            self._last_recall_result = memory_text
    
    def get_last_recall_result(self) -> Optional[str]:
        """
        Get the most recent recall result.
        
        Returns:
            The last recalled memory text, or None if no recent recall
        """
        with self._lock:
            return self._last_recall_result
    
    def has_recent_recall(self) -> bool:
        """
        Check if there's a recent recall result available for promotion.
        
        Returns:
            True if there's a recall result that can be promoted
        """
        with self._lock:
            return self._last_recall_result is not None


def _transform_first_to_second_person(text: str) -> Optional[str]:
    """
    Transform first-person memory text to second-person for pronoun-safe context.
    
    Imported from command_detector for use in get_promoted_context.
    
    Args:
        text: The memory text to transform
        
    Returns:
        Transformed text in second person, or None if no confident transformation
    """
    from wyzer.memory.command_detector import _transform_first_to_second_person as transform_fn
    return transform_fn(text)


# Module-level singleton
_memory_manager: Optional[MemoryManager] = None
_manager_lock = RLock()


def get_memory_manager() -> MemoryManager:
    """Get or create the singleton MemoryManager instance."""
    global _memory_manager
    with _manager_lock:
        if _memory_manager is None:
            _memory_manager = MemoryManager()
        return _memory_manager


def reset_memory_manager() -> None:
    """Reset the singleton (for testing)."""
    global _memory_manager
    with _manager_lock:
        _memory_manager = None
