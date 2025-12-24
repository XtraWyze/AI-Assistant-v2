"""
Deterministic tool-then-text splitter for mixed utterances.

GOAL: Safe, narrow pre-split path for utterances like:
  "Pause music and what's a VPN?"
  "mute and tell me about cats"
  "volume down then what is python"

The split ONLY applies when:
1. The utterance starts with a HIGH-CONFIDENCE deterministic tool phrase
2. There is leftover text after a connector (" and ", " then ", etc.)
3. The tool phrase maps to an EXISTING tool in the registry

If any condition fails, the splitter returns None and normal routing continues.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from wyzer.tools.registry import ToolRegistry


# ═══════════════════════════════════════════════════════════════════════════
# HIGH-CONFIDENCE TOOL PHRASE MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════
# Only these phrases are considered safe for deterministic splitting.
# Each phrase maps to (tool_name, args_dict).
# We normalize and strip filler words before matching.

# Phrases are matched case-insensitively after normalization.
# Order matters: more specific patterns first.

_TOOL_PHRASE_MAP: List[Tuple[str, str, Dict[str, Any]]] = [
    # Media play/pause (most common)
    ("pause music", "media_play_pause", {}),
    ("pause media", "media_play_pause", {}),
    ("pause the music", "media_play_pause", {}),
    ("pause my music", "media_play_pause", {}),
    ("pause it", "media_play_pause", {}),
    ("pause", "media_play_pause", {}),
    ("play music", "media_play_pause", {}),
    ("play media", "media_play_pause", {}),
    ("play the music", "media_play_pause", {}),
    ("play my music", "media_play_pause", {}),
    ("resume music", "media_play_pause", {}),
    ("resume media", "media_play_pause", {}),
    ("resume the music", "media_play_pause", {}),
    ("resume", "media_play_pause", {}),
    ("unpause", "media_play_pause", {}),
    ("unpause music", "media_play_pause", {}),
    
    # Media skip/previous
    ("next track", "media_next", {}),
    ("next song", "media_next", {}),
    ("skip track", "media_next", {}),
    ("skip song", "media_next", {}),
    ("skip", "media_next", {}),
    ("previous track", "media_previous", {}),
    ("previous song", "media_previous", {}),
    ("prev track", "media_previous", {}),
    ("go back", "media_previous", {}),
    
    # Volume mute/unmute
    ("mute", "volume_control", {"action": "mute", "scope": "master"}),
    ("unmute", "volume_control", {"action": "unmute", "scope": "master"}),
    ("mute volume", "volume_control", {"action": "mute", "scope": "master"}),
    ("unmute volume", "volume_control", {"action": "unmute", "scope": "master"}),
    
    # Volume up/down (relative changes)
    ("volume up", "volume_control", {"action": "change", "delta": 10, "scope": "master"}),
    ("turn up volume", "volume_control", {"action": "change", "delta": 10, "scope": "master"}),
    ("turn up the volume", "volume_control", {"action": "change", "delta": 10, "scope": "master"}),
    ("turn it up", "volume_control", {"action": "change", "delta": 10, "scope": "master"}),
    ("louder", "volume_control", {"action": "change", "delta": 10, "scope": "master"}),
    ("volume down", "volume_control", {"action": "change", "delta": -10, "scope": "master"}),
    ("turn down volume", "volume_control", {"action": "change", "delta": -10, "scope": "master"}),
    ("turn down the volume", "volume_control", {"action": "change", "delta": -10, "scope": "master"}),
    ("turn it down", "volume_control", {"action": "change", "delta": -10, "scope": "master"}),
    ("quieter", "volume_control", {"action": "change", "delta": -10, "scope": "master"}),
]


# ═══════════════════════════════════════════════════════════════════════════
# CONNECTORS: words/punctuation that separate tool phrase from leftover text
# ═══════════════════════════════════════════════════════════════════════════
# These are matched case-insensitively, with surrounding whitespace.

_CONNECTORS = [
    " and then ",
    " then ",
    " and ",
    " also ",
    " plus ",
    "; ",
    ";",
    ". ",  # Sentence boundary: "pause. what's a VPN?"
    ", ",
    ",",
]


# ═══════════════════════════════════════════════════════════════════════════
# FILLER WORDS: stripped from the start of the tool phrase before matching
# ═══════════════════════════════════════════════════════════════════════════

_FILLER_PREFIXES = [
    "please ",
    "can you ",
    "could you ",
    "would you ",
    "hey wyzer ",
    "wyzer ",
    "okay ",
    "ok ",
]


@dataclass
class SplitResult:
    """Result of a successful tool+text split."""
    tool_intent: Dict[str, Any]  # {"tool": str, "args": dict, "continue_on_error": bool}
    leftover_text: str           # The remaining text to send to LLM reply-only


def _normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, collapse whitespace, strip."""
    result = (text or "").lower().strip()
    result = re.sub(r"\s+", " ", result)
    return result


def _strip_fillers(text: str) -> str:
    """Strip common filler prefixes from text."""
    normalized = text.lower()
    for filler in _FILLER_PREFIXES:
        if normalized.startswith(filler):
            text = text[len(filler):].strip()
            normalized = text.lower()
    return text


def _find_connector(text_lower: str) -> Optional[Tuple[int, int, str]]:
    """
    Find the first connector in text.
    
    Returns:
        (start_idx, end_idx, connector_str) if found, None otherwise.
    """
    earliest_match = None
    
    for connector in _CONNECTORS:
        idx = text_lower.find(connector.lower())
        if idx >= 0:
            end_idx = idx + len(connector)
            if earliest_match is None or idx < earliest_match[0]:
                earliest_match = (idx, end_idx, connector)
    
    return earliest_match


def _match_tool_phrase(text_lower: str, registry: "ToolRegistry") -> Optional[Tuple[str, Dict[str, Any], int]]:
    """
    Try to match a high-confidence tool phrase at the start of text.
    
    Args:
        text_lower: Lowercase, normalized text
        registry: Tool registry to verify tool existence
        
    Returns:
        (tool_name, args, phrase_length) if matched and tool exists, None otherwise.
    """
    for phrase, tool_name, args in _TOOL_PHRASE_MAP:
        phrase_lower = phrase.lower()
        
        # Check if text starts with this phrase
        if text_lower.startswith(phrase_lower):
            # Make sure it's a complete match (not just prefix of longer word)
            phrase_len = len(phrase_lower)
            if len(text_lower) == phrase_len or text_lower[phrase_len] in " \t,;":
                # Verify tool exists in registry
                if registry.has_tool(tool_name):
                    return (tool_name, args.copy(), phrase_len)
                else:
                    # Tool phrase matched but tool doesn't exist - don't split
                    return None
    
    return None


def split_tool_then_text(
    user_text: str,
    registry: "ToolRegistry"
) -> Optional[SplitResult]:
    """
    Attempt to split a mixed utterance into tool intent + leftover text.
    
    This is a CONSERVATIVE, DETERMINISTIC splitter. It only succeeds when:
    1. Text starts with a high-confidence tool phrase
    2. There is a clear connector (" and ", " then ", etc.)
    3. There is meaningful leftover text after the connector
    4. The tool exists in the registry
    
    Args:
        user_text: Raw user utterance
        registry: Tool registry for checking tool existence
        
    Returns:
        SplitResult if split is safe and confident, None otherwise.
    """
    logger = logging.getLogger("wyzer.deterministic_splitter")
    
    if not user_text or not registry:
        return None
    
    # Normalize and strip fillers
    original = user_text.strip()
    cleaned = _strip_fillers(original)
    text_lower = _normalize_text(cleaned)
    
    if not text_lower:
        logger.debug("[SPLIT] skipped reason=empty_text")
        return None
    
    # Find connector first
    connector_match = _find_connector(text_lower)
    if connector_match is None:
        logger.debug(f'[SPLIT] skipped reason=no_connector text="{original[:50]}"')
        return None
    
    conn_start, conn_end, connector = connector_match
    
    # Extract the part before the connector
    before_connector = text_lower[:conn_start].strip()
    
    if not before_connector:
        logger.debug(f'[SPLIT] skipped reason=empty_before_connector text="{original[:50]}"')
        return None
    
    # Try to match a tool phrase in the before-connector part
    tool_match = _match_tool_phrase(before_connector, registry)
    
    if tool_match is None:
        logger.debug(f'[SPLIT] skipped reason=tool_not_high_confidence before="{before_connector}"')
        return None
    
    tool_name, tool_args, phrase_len = tool_match
    
    # Check if the entire before-connector text is the tool phrase
    # (we don't want "pause music something something and what's a VPN")
    remaining_before = before_connector[phrase_len:].strip()
    if remaining_before:
        # There's extra text between tool phrase and connector - not a clean split
        logger.debug(f'[SPLIT] skipped reason=extra_text_before_connector remaining="{remaining_before}"')
        return None
    
    # Extract leftover text after connector
    # Use the original cleaned text (preserving case for the question part)
    leftover = cleaned[conn_end:].strip()
    
    if not leftover:
        # No leftover text - this is just a tool command, not a mixed utterance
        # Return None so normal routing handles it as pure tool
        logger.debug(f'[SPLIT] skipped reason=no_leftover (pure tool command)')
        return None
    
    # Clean up leftover: strip trailing punctuation that might be artifacts
    leftover = leftover.rstrip("?!.,;").strip()
    if leftover:
        # Add back question mark if it looks like a question
        leftover_lower = leftover.lower()
        if any(leftover_lower.startswith(q) for q in ["what", "who", "where", "when", "why", "how", "is ", "are ", "can ", "does ", "do ", "will "]):
            leftover = leftover + "?"
    
    # Success! Build the split result
    tool_intent = {
        "tool": tool_name,
        "args": tool_args,
        "continue_on_error": False,
    }
    
    # Log successful split at INFO level (important routing decision)
    logger.info(f'[SPLIT] matched_tool="{tool_name}" leftover="{leftover[:50]}"')
    
    return SplitResult(
        tool_intent=tool_intent,
        leftover_text=leftover,
    )


def get_split_intents(
    user_text: str,
    registry: "ToolRegistry"
) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """
    Convenience wrapper for orchestrator integration.
    
    Returns:
        (tool_intents_list, leftover_text) if split succeeds, None otherwise.
        
        The tool_intents_list contains a single tool intent dict.
        The leftover_text should be processed via reply-only LLM path.
    """
    result = split_tool_then_text(user_text, registry)
    if result is None:
        return None
    
    return ([result.tool_intent], result.leftover_text)
