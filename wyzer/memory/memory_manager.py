"""
Memory Manager for Wyzer AI Assistant - Phase 11

Provides:
1. Session Memory: In-memory conversation context (last N turns)
2. Long-Term Memory: Structured records with explicit disk storage via user commands

Phase 11 Structured Memory Model:
Each memory record contains:
- id: unique identifier (UUID)
- type: one of ["fact", "preference", "skill", "history_marker"] (default "fact")
- key: short normalized key (optional, derived when possible)
- value: human readable string
- tags: list[str] (optional)
- created_at: ISO timestamp
- updated_at: ISO timestamp (optional)
- source: "explicit_user" (constant for now)

Backward Compatibility:
- Old list[str] or list[dict without type/value] formats are migrated on load

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


# Phase 11: Memory record types
MEMORY_RECORD_TYPES = ["fact", "preference", "skill", "history_marker"]
DEFAULT_RECORD_TYPE = "fact"
DEFAULT_SOURCE = "explicit_user"


def _derive_key(value: str) -> Optional[str]:
    """
    Derive a short normalized key from a memory value.
    
    Heuristics:
    - "my name is X" -> "name"
    - "my birthday is X" -> "birthday"
    - "my favorite X is Y" -> "favorite_X"
    - "I like X" -> "likes_X"
    
    Args:
        value: The memory value text
        
    Returns:
        Derived key or None if no confident derivation
    """
    if not value:
        return None
    
    text = value.lower().strip()
    
    # Pattern: "my name is X" or "my name's X"
    if re.match(r"^my\s+name(?:'s|\s+is)\s+", text):
        return "name"
    
    # Pattern: "my birthday is X"
    if re.match(r"^my\s+birthday\s+is\s+", text):
        return "birthday"
    
    # Pattern: "my age is X" or "I am X years old"
    if re.match(r"^my\s+age\s+is\s+", text) or re.match(r"^i(?:'m|\s+am)\s+\d+\s+years?\s+old", text):
        return "age"
    
    # Pattern: "my favorite X is Y"
    match = re.match(r"^my\s+fav(?:ou?rite)?\s+(\w+)\s+is\s+", text)
    if match:
        topic = match.group(1).strip()
        return f"favorite_{topic}"
    
    # Pattern: "I like X" / "I love X"
    match = re.match(r"^i\s+(?:like|love)\s+(.+?)(?:\.|$)", text)
    if match:
        topic = match.group(1).strip()[:20]  # Limit length
        topic_key = re.sub(r"\s+", "_", topic.lower())
        return f"likes_{topic_key}"
    
    # Pattern: "my X is Y" (generic possessive)
    match = re.match(r"^my\s+(\w+(?:\s+\w+)?)\s+is\s+", text)
    if match:
        what = match.group(1).strip()
        what_key = re.sub(r"\s+", "_", what.lower())
        return what_key[:30]  # Limit length
    
    return None


def _derive_type(value: str) -> str:
    """
    Derive the memory type from the value content.
    
    Heuristics:
    - "I like X" / "my favorite X" -> "preference"
    - "I can X" / "I know how to X" -> "skill"
    - Default -> "fact"
    
    Args:
        value: The memory value text
        
    Returns:
        One of MEMORY_RECORD_TYPES
    """
    if not value:
        return DEFAULT_RECORD_TYPE
    
    text = value.lower().strip()
    
    # Preference patterns
    if re.match(r"^(?:i\s+(?:like|love|prefer|enjoy|hate|dislike)|my\s+fav(?:ou?rite)?)", text):
        return "preference"
    
    # Skill patterns
    if re.match(r"^i\s+(?:can|know\s+how\s+to|am\s+able\s+to)", text):
        return "skill"
    
    return DEFAULT_RECORD_TYPE


def _migrate_legacy_entry(entry: Any) -> Dict[str, Any]:
    """
    Migrate a legacy memory entry to Phase 11+ structured format.
    
    Handles:
    - Plain strings: convert to {type, value, pinned, aliases, ...}
    - Old dict format with 'text' but no 'value': migrate
    - Already structured: ensure pinned/aliases exist
    
    Args:
        entry: Legacy entry (str or dict)
        
    Returns:
        Phase 11+ structured record with pinned and aliases fields
    """
    now = datetime.utcnow().isoformat() + "Z"
    
    # Plain string (very old format)
    if isinstance(entry, str):
        value = entry.strip()
        return {
            "id": str(uuid.uuid4()),
            "type": _derive_type(value),
            "key": _derive_key(value),
            "value": value,
            "tags": [],
            "pinned": False,
            "aliases": [],
            "created_at": now,
            "updated_at": None,
            "source": DEFAULT_SOURCE,
            # Keep for backward compat
            "text": value,
            "index_text": _normalize_for_matching(value),
        }
    
    # Dict format
    if isinstance(entry, dict):
        # Get value from 'value' or 'text'
        text = entry.get("value") or entry.get("text", "")
        text = text.strip() if text else ""
        
        # Build record ensuring all fields exist
        return {
            "id": entry.get("id", str(uuid.uuid4())),
            "type": entry.get("type", _derive_type(text)),
            "key": entry.get("key", _derive_key(text)),
            "value": text,
            "tags": entry.get("tags", []),
            "pinned": entry.get("pinned", False),
            "aliases": entry.get("aliases", []),
            "created_at": entry.get("created_at", now),
            "updated_at": entry.get("updated_at"),
            "source": entry.get("source", DEFAULT_SOURCE),
            # Keep for backward compat
            "text": text,
            "index_text": entry.get("index_text", _normalize_for_matching(text)),
        }
    
    # Unknown format - wrap as string
    value = str(entry)
    return {
        "id": str(uuid.uuid4()),
        "type": DEFAULT_RECORD_TYPE,
        "key": None,
        "value": value,
        "tags": [],
        "pinned": False,
        "aliases": [],
        "created_at": now,
        "updated_at": None,
        "source": DEFAULT_SOURCE,
        "text": value,
        "index_text": _normalize_for_matching(value),
    }


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
    
    def _load_memories(self, migrate: bool = True) -> List[Dict[str, Any]]:
        """
        Load memories from disk with optional Phase 11 migration.
        
        Args:
            migrate: If True, migrate legacy entries to Phase 11 format
        
        Returns:
            List of memory records (empty list if file doesn't exist)
        """
        try:
            if self._memory_file.exists():
                with open(self._memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        if migrate:
                            # Apply Phase 11 migration to each entry
                            return [_migrate_legacy_entry(entry) for entry in data]
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
            # All explicitly remembered memories are pinned by default (always injected)
            entry = {
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "text": text,  # Original text for display
                "index_text": new_index_text,  # Normalized for matching
                "tags": tags or [],
                "pinned": True,  # Always inject by default
                "aliases": []
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
    # Phase 11: Structured Memory API
    # =========================================================================
    
    def list_all(self) -> List[Dict[str, Any]]:
        """
        List all memory records (Phase 11 structured format).
        
        Returns:
            List of all memory records with full structured data
        """
        logger = get_logger()
        with self._lock:
            memories = self._load_memories()
            logger.info(f"[MEMORY] list_all: {len(memories)} records")
            return memories
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search memories by query (Phase 11).
        
        Simple case-insensitive substring match on value, key, and tags.
        
        Args:
            query: Search query text
            
        Returns:
            List of matching memory records
        """
        logger = get_logger()
        
        with self._lock:
            query_lower = query.lower().strip()
            if not query_lower:
                return []
            
            memories = self._load_memories()
            matches = []
            
            for mem in memories:
                # Search in value
                value = (mem.get("value") or mem.get("text") or "").lower()
                if query_lower in value:
                    matches.append(mem)
                    continue
                
                # Search in key
                key = (mem.get("key") or "").lower()
                if key and query_lower in key:
                    matches.append(mem)
                    continue
                
                # Search in tags
                tags = mem.get("tags", [])
                if any(query_lower in tag.lower() for tag in tags):
                    matches.append(mem)
                    continue
            
            logger.debug(f"[MEMORY] search '{query}': {len(matches)} matches")
            return matches
    
    def add_explicit(
        self,
        value: str,
        record_type: str = DEFAULT_RECORD_TYPE,
        key: Optional[str] = None,
        tags: Optional[List[str]] = None,
        pinned: bool = False,
        aliases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add a new memory record explicitly (Phase 11+).
        
        This is the structured API for adding memories with pinned/aliases support.
        Deduplicates by normalized value.
        
        Args:
            value: The memory content
            record_type: One of MEMORY_RECORD_TYPES (default "fact")
            key: Optional key (derived if not provided)
            tags: Optional list of tags
            pinned: If True, this memory is always injected when enabled (static)
            aliases: Optional list of alias strings for mention-triggered matching
            
        Returns:
            Dict with ok=True and the new record, or ok=False and error
        """
        logger = get_logger()
        
        with self._lock:
            value = value.strip()
            if not value:
                return {"ok": False, "error": "Nothing to remember"}
            
            # Validate type
            if record_type not in MEMORY_RECORD_TYPES:
                record_type = DEFAULT_RECORD_TYPE
            
            # Validate aliases
            validated_aliases = []
            if aliases:
                for alias in aliases[:8]:  # Max 8 aliases per record
                    alias = alias.strip()
                    if len(alias) >= 3:  # Min 3 chars for alias
                        validated_aliases.append(alias.lower())
            
            now = datetime.utcnow().isoformat() + "Z"
            normalized = _normalize_for_matching(value)
            
            # Create new record
            record = {
                "id": str(uuid.uuid4()),
                "type": record_type,
                "key": key or _derive_key(value),
                "value": value,
                "tags": tags or [],
                "pinned": pinned,
                "aliases": validated_aliases,
                "created_at": now,
                "updated_at": None,
                "source": DEFAULT_SOURCE,
                # Backward compat fields
                "text": value,
                "index_text": normalized,
            }
            
            # Load and deduplicate
            memories = self._load_memories()
            remaining = []
            replaced = False
            
            for mem in memories:
                existing_norm = mem.get("index_text") or _normalize_for_matching(
                    mem.get("value") or mem.get("text") or ""
                )
                if existing_norm == normalized:
                    replaced = True
                else:
                    remaining.append(mem)
            
            remaining.append(record)
            
            if self._save_memories(remaining):
                action = "updated" if replaced else "added"
                logger.info(f"[MEMORY] add_explicit ({action}): {value[:50]}...")
                return {"ok": True, "record": record, "replaced": replaced}
            else:
                return {"ok": False, "error": "Failed to save memory"}
    
    def delete_by_query(self, query: str) -> int:
        """
        Delete all memories matching the query (Phase 11).
        
        Uses the same search logic as search().
        
        Args:
            query: Search query for memories to delete
            
        Returns:
            Number of records deleted
        """
        logger = get_logger()
        
        with self._lock:
            query_lower = query.lower().strip()
            if not query_lower:
                return 0
            
            memories = self._load_memories()
            remaining = []
            deleted = []
            
            for mem in memories:
                matched = False
                
                # Check value
                value = (mem.get("value") or mem.get("text") or "").lower()
                if query_lower in value:
                    matched = True
                
                # Check key
                if not matched:
                    key = (mem.get("key") or "").lower()
                    if key and query_lower in key:
                        matched = True
                
                # Check tags
                if not matched:
                    tags = mem.get("tags", [])
                    if any(query_lower in tag.lower() for tag in tags):
                        matched = True
                
                if matched:
                    deleted.append(mem)
                    # Track for redaction
                    text = mem.get("value") or mem.get("text") or ""
                    if text:
                        transformed = _transform_first_to_second_person(text)
                        self._recently_forgotten.add(transformed if transformed else text)
                else:
                    remaining.append(mem)
            
            if deleted:
                if self._save_memories(remaining):
                    logger.info(f"[MEMORY] delete_by_query '{query}': deleted {len(deleted)} records")
                    self.clear_promoted()
                    return len(deleted)
                else:
                    logger.error(f"[MEMORY] delete_by_query: failed to save after deleting {len(deleted)} records")
                    return 0
            
            logger.debug(f"[MEMORY] delete_by_query '{query}': no matches")
            return 0
    
    def export_to(self, path: Optional[str] = None) -> str:
        """
        Export all memories to a JSON file (Phase 11).
        
        Args:
            path: Optional export path. If None, uses wyzer/data/memory_export_<timestamp>.json
            
        Returns:
            The path where the export was written
            
        Raises:
            ValueError: If path is outside wyzer/data/ (safety)
            IOError: If export fails
        """
        logger = get_logger()
        
        with self._lock:
            # Determine export path
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            if path:
                export_path = Path(path)
                # Safety: ensure path is inside wyzer/data/ or is an absolute path the user explicitly provided
                try:
                    export_path = export_path.resolve()
                    data_dir_resolved = data_dir.resolve()
                    # Allow if inside data dir
                    if not str(export_path).startswith(str(data_dir_resolved)):
                        # Also allow if in user's home or desktop (explicit user request)
                        home = Path.home()
                        if not (str(export_path).startswith(str(home)) or export_path.parent.exists()):
                            raise ValueError(f"Export path must be inside {data_dir} or user home directory")
                except Exception as e:
                    raise ValueError(f"Invalid export path: {e}")
            else:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                export_path = data_dir / f"memory_export_{timestamp}.json"
            
            # Load all memories
            memories = self._load_memories()
            
            # Write export
            try:
                export_path.parent.mkdir(parents=True, exist_ok=True)
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "version": "phase11",
                        "exported_at": datetime.utcnow().isoformat() + "Z",
                        "count": len(memories),
                        "memories": memories
                    }, f, indent=2, ensure_ascii=False)
                
                logger.info(f"[MEMORY] export_to: wrote {len(memories)} records to {export_path}")
                return str(export_path)
            except (IOError, OSError) as e:
                logger.error(f"[MEMORY] export_to failed: {e}")
                raise IOError(f"Failed to export memories: {e}")
    
    def import_from(self, path: str) -> int:
        """
        Import memories from a JSON file (Phase 11).
        
        IMPORTANT: Only called by explicit user command.
        Merges imported memories with existing, deduplicating by normalized value.
        
        Args:
            path: Path to the import file
            
        Returns:
            Number of new memories imported
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        logger = get_logger()
        
        with self._lock:
            import_path = Path(path)
            if not import_path.exists():
                raise FileNotFoundError(f"Import file not found: {path}")
            
            # Load import file
            try:
                with open(import_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in import file: {e}")
            
            # Handle different export formats
            if isinstance(data, dict) and "memories" in data:
                # Phase 11 export format
                import_memories = data["memories"]
            elif isinstance(data, list):
                # Direct list format
                import_memories = data
            else:
                raise ValueError("Invalid import file format: expected list or {memories: [...]}")
            
            if not isinstance(import_memories, list):
                raise ValueError("Invalid import file format: memories must be a list")
            
            # Load existing memories
            existing = self._load_memories()
            existing_normalized = {
                _normalize_for_matching(m.get("value") or m.get("text") or "")
                for m in existing
            }
            
            # Import new memories (deduplicate)
            imported_count = 0
            for entry in import_memories:
                migrated = _migrate_legacy_entry(entry)
                normalized = _normalize_for_matching(migrated.get("value") or "")
                
                if normalized and normalized not in existing_normalized:
                    existing.append(migrated)
                    existing_normalized.add(normalized)
                    imported_count += 1
            
            if imported_count > 0:
                if self._save_memories(existing):
                    logger.info(f"[MEMORY] import_from: imported {imported_count} new records from {path}")
                else:
                    logger.error(f"[MEMORY] import_from: failed to save after importing")
                    return 0
            else:
                logger.info(f"[MEMORY] import_from: no new records to import (all duplicates)")
            
            return imported_count
    
    def get_memories_grouped_by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all memories grouped by type (Phase 11).
        
        Returns:
            Dict mapping type -> list of records
        """
        with self._lock:
            memories = self._load_memories()
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            
            for mem in memories:
                mem_type = mem.get("type", DEFAULT_RECORD_TYPE)
                if mem_type not in grouped:
                    grouped[mem_type] = []
                grouped[mem_type].append(mem)
            
            return grouped
    
    # =========================================================================
    # Static/Pinned Memory Management
    # =========================================================================
    
    def set_pinned_by_query(self, query: str, pinned: bool) -> Dict[str, Any]:
        """
        Set pinned status on the first memory matching the query.
        
        Args:
            query: Search query (case-insensitive substring match on value/key)
            pinned: True to pin, False to unpin
            
        Returns:
            Dict with ok (bool), entry (dict if found), value (str)
        """
        logger = get_logger()
        
        with self._lock:
            query_lower = query.lower().strip()
            if not query_lower:
                return {"ok": False, "error": "empty_query"}
            
            memories = self._load_memories()
            now = datetime.utcnow().isoformat() + "Z"
            
            for mem in memories:
                # Search in value and key
                value = (mem.get("value") or mem.get("text") or "").lower()
                key = (mem.get("key") or "").lower()
                
                if query_lower in value or (key and query_lower in key):
                    # Found a match - update pinned status
                    mem["pinned"] = pinned
                    mem["updated_at"] = now
                    
                    if self._save_memories(memories):
                        action = "pinned" if pinned else "unpinned"
                        logger.info(f"[MEMORY] set_pinned_by_query: {action} record matching '{query}'")
                        return {
                            "ok": True,
                            "entry": mem,
                            "value": mem.get("value") or mem.get("text") or ""
                        }
                    else:
                        logger.error(f"[MEMORY] set_pinned_by_query: failed to save")
                        return {"ok": False, "error": "save_failed"}
            
            return {"ok": False, "error": "not_found"}
    
    def add_alias_by_query(self, query: str, alias: str) -> Dict[str, Any]:
        """
        Add an alias to the first memory matching the query.
        
        Args:
            query: Search query (case-insensitive substring match)
            alias: Alias to add (must be >= 3 chars)
            
        Returns:
            Dict with ok (bool), entry (dict if found)
        """
        logger = get_logger()
        
        with self._lock:
            query_lower = query.lower().strip()
            alias = alias.strip().lower()
            
            if not query_lower:
                return {"ok": False, "error": "empty_query"}
            
            if len(alias) < 2:
                return {"ok": False, "error": "alias_too_short"}
            
            memories = self._load_memories()
            now = datetime.utcnow().isoformat() + "Z"
            
            for mem in memories:
                value = (mem.get("value") or mem.get("text") or "").lower()
                key = (mem.get("key") or "").lower()
                
                if query_lower in value or (key and query_lower in key):
                    # Found a match - add alias if not already present
                    aliases = mem.get("aliases", [])
                    if len(aliases) >= 8:
                        logger.warning(f"[MEMORY] add_alias: max aliases reached for record")
                        return {"ok": False, "error": "max_aliases_reached"}
                    
                    if alias not in aliases:
                        aliases.append(alias)
                        mem["aliases"] = aliases
                        mem["updated_at"] = now
                        
                        if self._save_memories(memories):
                            logger.info(f"[MEMORY] add_alias: added '{alias}' to record matching '{query}'")
                            return {"ok": True, "entry": mem}
                        else:
                            return {"ok": False, "error": "save_failed"}
                    else:
                        # Alias already exists - still success
                        return {"ok": True, "entry": mem, "already_exists": True}
            
            return {"ok": False, "error": "not_found"}
    
    # =========================================================================
    # Deterministic Memory Selection for Injection
    # =========================================================================
    
    # Stopwords to ignore during mention matching
    STOPWORDS = frozenset([
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
        "because", "until", "while", "about", "against", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "it", "its",
        "me", "my", "myself", "we", "our", "ours", "you", "your", "yours",
        "he", "him", "his", "she", "her", "hers", "they", "them", "their",
        "i", "like", "love", "hate", "know", "think", "want", "tell", "remember",
    ])
    
    def _tokenize_for_matching(self, text: str) -> set:
        """
        Tokenize text for mention matching.
        
        - Lowercase
        - Remove punctuation
        - Split on whitespace
        - Remove stopwords and tokens < 3 chars
        
        Returns:
            Set of meaningful tokens
        """
        if not text:
            return set()
        
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and filter
        tokens = set()
        for token in text.split():
            if len(token) >= 3 and token not in self.STOPWORDS:
                tokens.add(token)
        
        return tokens
    
    def _is_mentioned(self, record: Dict[str, Any], user_text: str, user_tokens: set) -> bool:
        """
        Check if a memory record is mentioned in the user text.
        
        Deterministic matching:
        A) record.key token match (if key exists and appears in tokens)
        B) any alias matches a token or appears as substring in user_text
        C) (fallback) a meaningful token from record.value appears in user_text
        
        Args:
            record: Memory record
            user_text: Original user text (lowercased for substring matching)
            user_tokens: Pre-tokenized user text tokens
            
        Returns:
            True if mentioned, False otherwise
        """
        # A) Key match
        key = record.get("key")
        if key:
            key_lower = key.lower().replace("_", " ")
            key_tokens = self._tokenize_for_matching(key_lower)
            if key_tokens & user_tokens:
                return True
            # Also check if key appears as substring
            if key_lower in user_text or key.lower() in user_text:
                return True
        
        # B) Alias match
        aliases = record.get("aliases", [])
        for alias in aliases:
            alias_lower = alias.lower()
            # Token match
            if alias_lower in user_tokens:
                return True
            # Substring match
            if alias_lower in user_text:
                return True
        
        # C) Value token match (fallback - be more conservative)
        value = record.get("value") or record.get("text") or ""
        value_tokens = self._tokenize_for_matching(value)
        # Require at least one meaningful overlap
        overlap = value_tokens & user_tokens
        if overlap:
            # Filter out very common tokens that might cause false positives
            meaningful = [t for t in overlap if len(t) >= 4]
            if meaningful:
                return True
        
        return False
    
    def _score_record(self, record: Dict[str, Any], user_tokens: set) -> int:
        """
        Compute deterministic relevance score for a record.
        
        Scoring:
        - +10 if key match
        - +6 if alias match
        - +4 per token overlap between user tokens and record.value tokens
        - +1 recency bump (newer = higher)
        
        Args:
            record: Memory record
            user_tokens: Pre-tokenized user text tokens
            
        Returns:
            Integer score (0 if no relevance)
        """
        score = 0
        
        # Key match
        key = record.get("key")
        if key:
            key_tokens = self._tokenize_for_matching(key.lower().replace("_", " "))
            if key_tokens & user_tokens:
                score += 10
        
        # Alias match
        aliases = record.get("aliases", [])
        for alias in aliases:
            if alias.lower() in user_tokens:
                score += 6
                break  # Only count once
        
        # Value token overlap
        value = record.get("value") or record.get("text") or ""
        value_tokens = self._tokenize_for_matching(value)
        overlap = value_tokens & user_tokens
        meaningful_overlap = [t for t in overlap if len(t) >= 4]
        score += 4 * len(meaningful_overlap)
        
        return score
    
    def select_for_injection(
        self,
        user_text: str,
        k_total: int = 6,
        pinned_max: int = 4,
        mention_max: int = 4,
        max_chars: int = 1200
    ) -> str:
        """
        Select memories for injection using deterministic static + mention-triggered buckets.
        
        Selection order:
        1. PINNED/STATIC memories (always included up to pinned_max)
        2. MENTION-TRIGGERED memories (included if user mentions them, up to mention_max)
        3. TOP-K fallback (fill remaining slots with highest-scoring records)
        
        Args:
            user_text: The user's current input text
            k_total: Maximum total memories to include (default 6)
            pinned_max: Maximum pinned memories (default 4)
            mention_max: Maximum mention-triggered memories (default 4)
            max_chars: Maximum characters in output block (default 1200)
            
        Returns:
            Formatted injection block string, or "" if nothing selected or disabled
        """
        logger = get_logger()
        
        with self._lock:
            if not self._use_memories:
                return ""
            
            memories = self._load_memories()
            if not memories:
                return ""
            
            # Prepare user text for matching
            user_text_lower = user_text.lower() if user_text else ""
            user_tokens = self._tokenize_for_matching(user_text)
            
            selected = []
            selected_ids = set()
            
            # 1. PINNED/STATIC memories (always included when enabled)
            pinned_records = [m for m in memories if m.get("pinned", False)]
            # Sort by created_at descending for determinism
            pinned_records.sort(key=lambda m: m.get("created_at", ""), reverse=True)
            
            for record in pinned_records[:pinned_max]:
                selected.append(("static", record))
                selected_ids.add(record.get("id"))
            
            if selected:
                pinned_keys = [r.get("key") or r.get("text", "")[:30] for _, r in selected]
                logger.info(f"[MEMORY] Injecting {len(selected)} pinned: {pinned_keys}")
                logger.debug(f"[MEMORY] Pinned details: {[r.get('text', '')[:50] for _, r in selected]}")
            
            # 2. MENTION-TRIGGERED memories
            mention_count = 0
            mentioned_keys = []
            for record in memories:
                if record.get("id") in selected_ids:
                    continue
                if mention_count >= mention_max:
                    break
                if self._is_mentioned(record, user_text_lower, user_tokens):
                    selected.append(("mentioned", record))
                    selected_ids.add(record.get("id"))
                    mentioned_keys.append(record.get("key") or record.get("text", "")[:30])
                    mention_count += 1
            
            if mention_count > 0:
                logger.info(f"[MEMORY] Injecting {mention_count} mentioned: {mentioned_keys}")
            
            # 3. TOP-K FALLBACK (fill remaining slots)
            remaining_slots = k_total - len(selected)
            if remaining_slots > 0:
                # Score remaining records
                scored = []
                for record in memories:
                    if record.get("id") in selected_ids:
                        continue
                    score = self._score_record(record, user_tokens)
                    if score > 0:
                        scored.append((score, record.get("created_at", ""), record))
                
                # Sort by score desc, then by created_at desc
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                
                for score, _, record in scored[:remaining_slots]:
                    selected.append(("topk", record))
                    selected_ids.add(record.get("id"))
                
                logger.debug(f"[MEMORY] select_for_injection: {len(scored[:remaining_slots])} top-k fallback")
            
            if not selected:
                return ""
            
            # Format output block
            from wyzer.memory.command_detector import _transform_first_to_second_person
            
            header = "[LONG-TERM MEMORY — selected]\n"
            footer = "\nUse this information when relevant to the user's question."
            reserved = len(header) + len(footer)
            
            lines = []
            total_chars = 0
            
            for label, record in selected:
                value = record.get("value") or record.get("text") or ""
                key = record.get("key")
                
                # Transform to second person
                transformed = _transform_first_to_second_person(value)
                display_value = transformed if transformed else value
                
                # Format line
                if key:
                    line = f"- ({label}) {key}: {display_value}"
                else:
                    line = f"- ({label}) {display_value}"
                
                # Check character limit
                if total_chars + len(line) + 1 > max_chars - reserved:
                    break
                
                lines.append(line)
                total_chars += len(line) + 1
            
            if not lines:
                return ""
            
            result = header + "\n".join(lines) + footer
            logger.info(f"[MEMORY] Injection: {len(lines)} memories, {total_chars} chars → LLM")
            return result

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
