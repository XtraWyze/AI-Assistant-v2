"""
Orchestrator for Wyzer AI Assistant - Phase 7/8
Coordinates LLM reasoning and tool execution.
Supports multi-intent commands (Phase 6 enhancement).
Phase 8: Added llamacpp mode support.
"""
import json
import time
import urllib.request
import urllib.error
import re
import socket
import shlex
import uuid
import random
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from wyzer.core.config import Config
from wyzer.core.logger import get_logger
from wyzer.core import hybrid_router
from wyzer.tools.registry import build_default_registry
from wyzer.tools.validation import validate_args
from wyzer.local_library import resolve_target
from wyzer.core.intent_plan import (
    normalize_plan,
    Intent,
    validate_intents,
    filter_unknown_tools,
    normalize_tool_aliases,
    ExecutionResult,
    ExecutionSummary
)


# Short varied responses for no-ollama mode when command isn't recognized
_NO_OLLAMA_FALLBACK_REPLIES = [
    "Not supported, try again.",
    "I didn't catch that, try again.",
    "That's not available right now.",
    "Can't do that one, try something else.",
    "Not recognized, please try again.",
    "Sorry, try a different command.",
]


# ============================================================================
# PHASE 11: AUTONOMY COMMAND PATTERNS (compiled early for should_use_streaming_tts)
# ============================================================================
# These patterns are used to detect autonomy commands before streaming TTS.
# Must be defined before should_use_streaming_tts() function.
# Patterns match natural variations while remaining deterministic.

# Pattern components for mode-setting:
# - Optional "set" prefix
# - Optional "my/your" possessive
# - "autonomy" with optional "level/mode/setting" suffix
# - Optional "to" before mode
# - The mode itself (off/low/normal/high)
_AUTONOMY_OFF_RE = re.compile(
    r"^(?:set\s+)?(?:(?:my|your)\s+)?autonomy(?:\s+(?:level|mode|setting))?(?:\s+to)?\s+off[.!]?$",
    re.IGNORECASE
)
_AUTONOMY_LOW_RE = re.compile(
    r"^(?:set\s+)?(?:(?:my|your)\s+)?autonomy(?:\s+(?:level|mode|setting))?(?:\s+to)?\s+low[.!]?$",
    re.IGNORECASE
)
_AUTONOMY_NORMAL_RE = re.compile(
    r"^(?:set\s+)?(?:(?:my|your)\s+)?autonomy(?:\s+(?:level|mode|setting))?(?:\s+to)?\s+normal[.!]?$",
    re.IGNORECASE
)
_AUTONOMY_HIGH_RE = re.compile(
    r"^(?:set\s+)?(?:(?:my|your)\s+)?autonomy(?:\s+(?:level|mode|setting))?(?:\s+to)?\s+high[.!]?$",
    re.IGNORECASE
)
_AUTONOMY_STATUS_RE = re.compile(
    r"^what(?:'s|\s+is)\s+(?:my|your)\s+autonomy(?:\s+(?:level|mode|setting))?(?:\s+(?:set\s+)?(?:at|to))?[.?]?$",
    re.IGNORECASE
)
_WHY_DID_YOU_RE = re.compile(
    r"^(?:why(?:'d|\s+did)?\s+you\s+(?:do\s+(?:that|this|it)|say\s+that|act\s+on\s+that|execute\s+that|ask(?:\s+me)?)|"
    r"explain\s+(?:that|your\s+(?:last\s+)?(?:decision|action))|"
    r"what\s+was\s+your\s+reason(?:ing)?|"
    r"why\s+(?:that|did\s+you))\.?\??$",
    re.IGNORECASE
)

# Confirmation response patterns (for pending confirmations)
# These patterns match if the response STARTS with a confirmation word
# to handle natural replies like "Yes, I did" or "No, don't do that"
_CONFIRM_YES_RE = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|ok|okay|confirm|proceed|do\s+it|go\s+ahead|absolutely|affirmative)(?:[.,!]|\s|$)",
    re.IGNORECASE
)
_CONFIRM_NO_RE = re.compile(
    r"^(?:no|nope|nah|cancel|stop|never\s+mind|nevermind|don't|do\s+not|abort|negative)(?:[.,!]|\s|$)",
    re.IGNORECASE
)

# ============================================================================
# PHASE 12: WINDOW WATCHER COMMAND PATTERNS (deterministic, no LLM)
# ============================================================================
# Commands for multi-monitor window awareness.

# "what's on monitor 2" / "whats on screen 1" / "what is on monitor 3"
# Note: Handle both straight (') and curly (') apostrophes from STT
# Robust pattern to handle STT transcription errors like:
# "what's on monitor 1", "whats on screen two", "it's on monitor 1" (misheard),
# "what is on my display 2", "what's monitor one", "what's on the second monitor",
# "what's on the secondary monitor", "what's on monitor to" (mishear of "two"), etc.
# Supports two forms:
#   1. "what's on monitor <number>" - number after monitor
#   2. "what's on the second/secondary monitor" - ordinal before monitor
_WHATS_ON_MONITOR_RE = re.compile(
    r"^(?:what(?:[''']?s|\s+is)|it(?:[''']?s|\s+is))\s+"  # "what's" or "it's" (common mishear)
    r"(?:on\s+)?(?:a\s+)?(?:my\s+)?"                       # optional: on/a/my
    r"(?:"
        # Form 1: [the] monitor/screen/display <number>
        r"(?:the\s+)?(?:monitor|screen|display)\s*"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|to|too|"  # "to/too" = mishear of "two"
        r"primary|secondary|main|other|left|right|first|second|third)"
        r"|"
        # Form 2: [the] <ordinal> monitor/screen/display
        r"(?:the\s+)?(first|second|third|primary|secondary|main|other|left|right)\s+"
        r"(?:monitor|screen|display)"
    r")[.?!]?$",
    re.IGNORECASE
)

# "close all on screen 1" / "close everything on monitor 2"
_CLOSE_ALL_ON_MONITOR_RE = re.compile(
    r"^close\s+(?:all|everything)\s+(?:on\s+)?(?:the\s+)?(?:monitor|screen|display)\s*"
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"primary|secondary|main|other|left|right)[.?]?$",
    re.IGNORECASE
)

# "what did I just open" / "what did I open" / "what was just opened"
_WHAT_DID_I_OPEN_RE = re.compile(
    r"^(?:what\s+did\s+(?:i|you)\s+(?:just\s+)?open|"
    r"what\s+(?:was|were)\s+(?:just\s+)?opened|"
    r"recently?\s+opened(?:\s+windows?)?)[.?]?$",
    re.IGNORECASE
)

# "where am I" / "which monitor am I on" / "what monitor is this"
_WHERE_AM_I_RE = re.compile(
    r"^(?:where\s+am\s+i|"
    r"which\s+(?:monitor|screen|display)\s+(?:am\s+i\s+on|is\s+(?:this|active|focused))|"
    r"what\s+(?:monitor|screen|display)\s+(?:am\s+i\s+on|is\s+(?:this|active|focused)))[.?]?$",
    re.IGNORECASE
)

# Word to number mapping for monitor references
_MONITOR_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "primary": 1, "main": 1, "secondary": 2, "other": 2,
    "left": 2, "right": 1,  # Primary is typically on right, secondary on left
    # Ordinal words
    "first": 1, "second": 2, "third": 3,
    # STT mishears of "two"
    "to": 2, "too": 2,
}


def _parse_monitor_number(text: str) -> int:
    """Parse a monitor reference (number or word) to integer."""
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    return _MONITOR_WORD_TO_NUM.get(text, 1)


def _get_no_ollama_reply() -> str:
    """Get a random short reply for unsupported commands in no-ollama mode."""
    return random.choice(_NO_OLLAMA_FALLBACK_REPLIES)


def _get_llm_client() -> Optional[Union["OllamaClient", "LlamaCppClient"]]:
    """
    Get the appropriate LLM client based on current Config.LLM_MODE.
    
    Returns:
        OllamaClient for ollama mode
        LlamaCppClient for llamacpp mode
        None if LLM is disabled
    """
    from wyzer.brain.ollama_client import OllamaClient
    from wyzer.brain.llamacpp_client import LlamaCppClient
    
    llm_mode = getattr(Config, "LLM_MODE", "ollama")
    
    if getattr(Config, "NO_OLLAMA", False) or llm_mode == "off":
        return None
    
    if llm_mode == "llamacpp":
        base_url = getattr(Config, "LLAMACPP_BASE_URL", "") or f"http://127.0.0.1:{Config.LLAMACPP_PORT}"
        return LlamaCppClient(base_url=base_url, timeout=Config.LLM_TIMEOUT)
    else:
        # Default: Ollama
        return OllamaClient(base_url=Config.OLLAMA_BASE_URL, timeout=Config.LLM_TIMEOUT)


def _get_llm_base_url() -> str:
    """Get the current LLM base URL based on mode."""
    llm_mode = getattr(Config, "LLM_MODE", "ollama")
    if llm_mode == "llamacpp":
        return getattr(Config, "LLAMACPP_BASE_URL", "") or f"http://127.0.0.1:{Config.LLAMACPP_PORT}"
    return Config.OLLAMA_BASE_URL.rstrip("/")


def _is_llm_available() -> bool:
    """Check if any LLM is available."""
    if getattr(Config, "NO_OLLAMA", False):
        return False
    llm_mode = getattr(Config, "LLM_MODE", "ollama")
    return llm_mode in ("ollama", "llamacpp")


# ============================================================================
# CONTINUATION PHRASE DETECTION & TOPIC TRACKING
# ============================================================================
# Deterministic pre-pass: detect vague follow-ups and rewrite them to include
# the last topic explicitly. This ensures continuity even if context is short.

# Continuation phrases that indicate user wants more on the previous topic
_CONTINUATION_PHRASES = [
    r"^tell me more[\.\?]?$",
    r"^can you tell me more[\.\?]?$",
    r"^more[\.\?]?$",
    r"^go on[\.\?]?$",
    r"^continue[\.\?]?$",
    r"^keep going[\.\?]?$",
    r"^what else[\.\?]?$",
    r"^and[\.\?]?$",
    r"^elaborate[\.\?]?$",
    r"^can you elaborate[\.\?]?$",
    r"^please elaborate[\.\?]?$",
    r"^explain more[\.\?]?$",
    r"^tell me more about (?:it|that|this)[\.\?]?$",
    r"^more (?:info|information|details?)[\.\?]?$",
    r"^give me more[\.\?]?$",
    r"^anything else[\.\?]?$",
    r"^what about it[\.\?]?$",
    r"^like what[\.\?]?$",
    r"^expand on that[\.\?]?$",
    r"^can you expand on that[\.\?]?$",
    r"^keep talking[\.\?]?$",
    r"^yes[\.\?]?$",
    r"^yeah[\.\?]?$",
    r"^sure[\.\?]?$",
    r"^okay[\.\?]?$",
    r"^ok[\.\?]?$",
    r"^go ahead[\.\?]?$",
    r"^i'm listening[\.\?]?$",
    r"^im listening[\.\?]?$",
]
_CONTINUATION_RE = re.compile("|".join(_CONTINUATION_PHRASES), re.IGNORECASE)

# "Tell me more about X" pattern - explicit topic continuation (reply-only, no tools)
_EXPLICIT_CONTINUATION_RE = re.compile(
    r"^(?:tell me more about|more about|explain more about|elaborate on|expand on|continue (?:explaining|about))\s+(.+?)[\.\?]?$",
    re.IGNORECASE
)

# Purely informational query patterns that should NEVER trigger tools
# These are conversational/knowledge queries, not action requests
# Note: what(?:'?s|\s+is) handles contractions like "what's" and STT outputs like "whats"
_INFORMATIONAL_QUERY_RE = re.compile(
    r"^(?:"
    r"tell me about|"
    r"what(?:'?s|\s+is|\s+are|\s+was|\s+were)|"  # Handle "what's", "whats", "what is"
    r"what'?s (?:a|an|some) (?:anime|show|movie|game|book|song|album|series|film) (?:like|similar to)|"  # "what's an anime like X"
    r"who (?:is|are|was|were)|"
    r"who'?s|"  # "who's" contraction
    r"when(?:'?s|\s+is)|"  # "when's my birthday"
    r"where (?:is|are|do)|"  # "where do I live"
    r"explain|"
    r"describe|"
    r"how does|"
    r"how do|"
    r"why (?:is|are|do|does)|"
    r"when (?:did|was|were)"
    r")\s+.+",
    re.IGNORECASE
)

# Patterns that need action-word checking before being marked informational
_RECOMMEND_SUGGEST_RE = re.compile(
    r"^(?:recommend|suggest)\s+.+",
    re.IGNORECASE
)
_WHAT_SHOULD_RE = re.compile(
    r"^what (?:should|would|could) (?:i|you)\s+.+",
    re.IGNORECASE
)
_ACTION_WORDS_RE = re.compile(
    r"\b(?:scan|open|set|change|timer|time|volume|weather|location|storage|monitor|window|close|minimize|maximize|focus|mute|unmute|play|pause|launch|start)\b",
    re.IGNORECASE
)

# Module-level last topic storage (RAM only, cleared on restart)
_last_topic: Optional[str] = None
_last_topic_lock = None  # Lazy init for threading
_continuation_hops: int = 0  # Tracks consecutive continuation requests
_MAX_CONTINUATION_HOPS: int = 3  # After this many, ask narrowing question

def _get_topic_lock():
    """Lazy init threading lock for topic access."""
    global _last_topic_lock
    if _last_topic_lock is None:
        import threading
        _last_topic_lock = threading.Lock()
    return _last_topic_lock


def set_last_topic(topic: str) -> None:
    """
    Update the last discussed topic (called after non-vague queries).
    
    Topic hygiene rules:
    - Only set from USER turns, never from assistant replies
    - Skip empty/garbage topics (STT glitches like "tell me about...")
    - Minimum 3 chars, must contain at least one letter
    """
    global _last_topic, _continuation_hops
    if not topic:
        return
    
    clean = topic.strip()
    
    # Hygiene: reject empty, too short, or letter-less garbage
    if len(clean) < 3:
        return
    if not any(c.isalpha() for c in clean):
        return
    # Reject if it's just filler words
    filler_only = {"the", "a", "an", "it", "that", "this", "them", "those"}
    if clean.lower() in filler_only:
        return
    
    with _get_topic_lock():
        _last_topic = clean
        _continuation_hops = 0  # Reset hops when new topic is set


def get_last_topic() -> Optional[str]:
    """Get the last discussed topic."""
    with _get_topic_lock():
        return _last_topic


def increment_continuation_hops() -> int:
    """Increment and return continuation hop count."""
    global _continuation_hops
    with _get_topic_lock():
        _continuation_hops += 1
        return _continuation_hops


def get_continuation_hops() -> int:
    """Get current continuation hop count."""
    with _get_topic_lock():
        return _continuation_hops


def reset_continuation_hops() -> None:
    """Reset continuation hop count (called on non-continuation queries)."""
    global _continuation_hops
    with _get_topic_lock():
        _continuation_hops = 0


def is_continuation_phrase(text: str) -> bool:
    """Check if text is a vague continuation phrase."""
    return bool(_CONTINUATION_RE.match(text.strip()))


def is_explicit_continuation(text: str) -> Optional[str]:
    """
    Check if text is an explicit continuation like "tell me more about X".
    
    Returns the topic X if matched, None otherwise.
    """
    match = _EXPLICIT_CONTINUATION_RE.match(text.strip())
    if match:
        topic = match.group(1).strip()
        # Clean up trailing punctuation
        topic = re.sub(r'[\.\?\!]+$', '', topic).strip()
        return topic if len(topic) > 2 else None
    return None


def is_informational_query(text: str) -> bool:
    """
    Check if text is a purely informational/knowledge query.
    
    These queries should use reply-only mode - no tools needed.
    Examples: "tell me about X", "what is Y", "who was Z"
    
    Special handling for "what should I..." - only informational
    if it doesn't contain action words like scan/open/set/etc.
    """
    stripped = text.strip()
    
    # Check main pattern first
    if _INFORMATIONAL_QUERY_RE.match(stripped):
        # But NOT if it contains action words (weather, timer, volume, etc.)
        if _ACTION_WORDS_RE.search(stripped):
            return False
        return True
    
    # Special handling for "recommend/suggest..." patterns
    # Only informational if no action words present
    if _RECOMMEND_SUGGEST_RE.match(stripped):
        if not _ACTION_WORDS_RE.search(stripped):
            return True
    
    # Special handling for "what should I..." patterns
    if _WHAT_SHOULD_RE.match(stripped):
        # Only informational if no action words present
        if not _ACTION_WORDS_RE.search(stripped):
            return True
    
    return False


def rewrite_continuation(text: str) -> str:
    """
    If text is a continuation phrase, rewrite it to include the last topic.
    
    Example:
        "Can you tell me more?" + last_topic="One Piece"
        -> "Continue explaining One Piece in more detail."
    
    Returns original text if not a continuation phrase or no topic available.
    """
    if not is_continuation_phrase(text):
        return text
    
    topic = get_last_topic()
    if not topic:
        return text  # No topic to continue, let LLM handle generically
    
    # Rewrite to be explicit
    return f"Continue explaining {topic} in more detail."


def _extract_topic_from_query(text: str) -> Optional[str]:
    """
    Extract the main topic from a user query for tracking.
    
    Simple heuristic: if query asks about something specific, extract it.
    Returns None for vague or tool-focused queries.
    """
    lower = text.lower().strip()
    
    # Skip if it's a continuation phrase
    if is_continuation_phrase(text):
        return None
    
    # Skip if it's clearly a command (open, close, volume, etc.)
    command_indicators = [
        "open ", "close ", "launch ", "start ", "run ", "execute ",
        "volume ", "mute", "pause", "play ", "stop ", "minimize",
        "maximize", "focus ", "switch to ", "set timer", "what time",
        "what's the time", "weather", "temperature", "refresh library",
    ]
    if any(lower.startswith(ind) or ind in lower for ind in command_indicators):
        return None
    
    # Extract topic from "tell me about X" or "what is X" patterns
    patterns = [
        r"(?:tell me about|explain|describe|what (?:is|are)|who (?:is|are)|talk about)\s+(.+?)[\.\?]?$",
        r"^(.+?)\s*\?$",  # Simple question like "One Piece?"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            # Clean up trailing punctuation
            topic = re.sub(r'[\.\?\!]+$', '', topic).strip()
            if len(topic) > 2:
                return topic
    
    # Fallback: if it's a conversational query (not a command), use the whole thing
    # but only if it's reasonably short (likely a topic)
    if len(text) < 100 and not any(c in lower for c in ["please", "can you", "could you", "would you"]):
        return text.strip()
    
    return None


_FASTPATH_SPLIT_RE = re.compile(r"\b(?:and then|then|and)\b", re.IGNORECASE)
_FASTPATH_COMMA_SPLIT_RE = re.compile(r"\s*,\s*")
_FASTPATH_SEMI_SPLIT_RE = re.compile(r"\s*;\s*")
_FASTPATH_COMMAND_TOKEN_RE = re.compile(
    r"\b(?:tool|run|execute|open|launch|start|close|exit|quit|focus|activate|switch\s+to|minimize|maximize|fullscreen|move|pause|play|resume|mute|unmute|volume|sound|audio|volume\s+up|volume\s+down|turn\s+up|turn\s+down|louder|quieter|set\s+audio|switch\s+audio|change\s+audio|refresh\s+library|rebuild\s+library|scan|devices?|weather|forecast|location|system\s+info|system\s+information|monitor\s+info|what\s+time\s+is\s+it|my\s+location|where\s+am\s+i|next\s+(?:track|song)|previous\s+(?:track|song)|prev\s+track)\b",
    re.IGNORECASE,
)

# Small allowlist of queries that are overwhelmingly likely to mean a website.
# Keep conservative to preserve the "only if unambiguous" rule.
_FASTPATH_COMMON_WEBSITES = {
    "youtube",
    "github",
    "google",
    "gmail",
    "wikipedia",
    "reddit",
}

_FASTPATH_EXPLICIT_TOOL_RE = re.compile(r"^(?:tool|run|execute)\s+(?P<tool>[a-zA-Z0-9_]+)(?:\s+(?P<rest>.*))?$", re.IGNORECASE)

# Tool worker pool singleton (initialized on demand in Brain process)
_tool_pool = None
_logger = None


def _user_explicitly_requested_library_refresh(user_text: str) -> bool:
    tl = (user_text or "").lower()
    has_noun = any(k in tl for k in ["library", "libary", "index", "apps", "app", "applications"])
    has_verb = any(k in tl for k in ["refresh", "rebuild", "rescan", "scan", "reindex", "re-index", "update"])
    return has_noun and has_verb


def _filter_spurious_intents(user_text: str, intents: List[Intent]) -> List[Intent]:
    """Drop obviously irrelevant tool intents.

    This protects against occasional LLM/tool-selection glitches.
    """
    logger = get_logger_instance()

    if not intents:
        return intents

    dropped = []
    filtered: List[Intent] = []
    for intent in intents:
        if intent.tool == "local_library_refresh" and not _user_explicitly_requested_library_refresh(user_text):
            dropped.append(intent.tool)
            continue
        filtered.append(intent)

    if dropped:
        logger.info(f"[INTENT FILTER] Dropped spurious intent(s): {', '.join(dropped)}")
    return filtered

# Module-level singleton registry and tool pool
_registry = None
_logger = None
_tool_pool = None


def get_logger_instance():
    """Get or create logger instance"""
    global _logger
    if _logger is None:
        _logger = get_logger()
    return _logger


def get_registry():
    """Get or create the tool registry singleton"""
    global _registry
    if _registry is None:
        _registry = build_default_registry()
    return _registry


def init_tool_pool():
    """Initialize the tool worker pool (call from Brain process on startup)"""
    global _tool_pool
    if _tool_pool is not None:
        return _tool_pool
    
    logger = get_logger_instance()
    if not Config.TOOL_POOL_ENABLED:
        logger.info("[POOL] Tool pool disabled via config")
        return None
    
    try:
        from wyzer.core.tool_worker_pool import ToolWorkerPool
        _tool_pool = ToolWorkerPool(num_workers=Config.TOOL_POOL_WORKERS)
        if _tool_pool.start():
            logger.info(f"[POOL] Tool pool initialized with {Config.TOOL_POOL_WORKERS} workers")
            return _tool_pool
        else:
            logger.warning("[POOL] Failed to start tool pool, will use in-process execution")
            _tool_pool = None
            return None
    except Exception as e:
        logger.warning(f"[POOL] Failed to initialize tool pool: {e}, will use in-process execution")
        _tool_pool = None
        return None


def shutdown_tool_pool():
    """Shutdown the tool worker pool"""
    global _tool_pool
    if _tool_pool is not None:
        logger = get_logger_instance()
        logger.info("[POOL] Shutting down tool pool")
        _tool_pool.shutdown()
        _tool_pool = None


def get_tool_pool_heartbeats() -> List[Dict[str, Any]]:
    """Get worker heartbeats from tool pool for Brain heartbeat logging"""
    global _tool_pool
    if _tool_pool is None:
        return []
    try:
        return _tool_pool.get_worker_heartbeats()
    except Exception:
        return []


def _tool_error_to_speech(
    error: Any,
    tool: str,
    args: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convert a tool error into a user-friendly speech reply.
    
    Args:
        error: The error returned by the tool (can be str, dict, or None)
        tool: The name of the tool that failed
        args: The arguments that were passed to the tool
        
    Returns:
        A friendly speech-safe error message
    """
    args = args or {}
    
    # Parse error type and message from dict or string
    error_type = ""
    error_msg = ""
    if isinstance(error, dict):
        error_type = str(error.get("type", "")).lower()
        error_msg = str(error.get("message", ""))
    elif isinstance(error, str):
        error_msg = error
        # Try to infer type from message
        if "not found" in error.lower():
            error_type = "not_found"
        elif "permission" in error.lower() or "denied" in error.lower() or "access" in error.lower():
            error_type = "permission_denied"
    
    # --- Window-related errors ---
    if error_type == "window_not_found":
        # Extract target from args
        target = args.get("title") or args.get("process") or args.get("query") or ""
        if target:
            return f"I can't find a {target} window. Is it open?"
        return "I can't find that window. Is it open?"
    
    if error_type == "invalid_monitor":
        return f"That monitor doesn't exist. {error_msg}"
    
    # --- Permission errors ---
    if error_type == "permission_denied":
        return "Windows blocked that action. Try running Wyzer as administrator."
    
    # --- Audio device errors ---
    if error_type == "no_devices":
        return "I couldn't find any audio devices."
    
    if error_type == "invalid_device_query":
        return "Please specify which audio device you want."
    
    if error_type == "device_not_found":
        device = args.get("device", "")
        if device:
            return f"I couldn't find an audio device matching '{device}'."
        return "I couldn't find that audio device."
    
    # --- Volume control errors ---
    if error_type == "not_found" and tool == "volume_control":
        process = args.get("process", "")
        if process:
            return f"I can't find {process} to adjust its volume. Is it running?"
        return "I can't find that app to adjust its volume."
    
    if error_type == "unsupported_platform":
        return "That's not supported on this system."
    
    # --- Timer errors ---
    if tool == "timer":
        if error_type == "no_timer":
            return "There is no active timer."
        if error_type == "invalid_duration":
            return "That timer duration isn't valid."
    
    # --- Argument errors ---
    if error_type in {"invalid_args", "missing_argument", "invalid_action"}:
        return f"I didn't understand that command. {error_msg}"
    
    # --- Generic execution errors ---
    if error_type == "execution_error":
        return "Something went wrong. Please try again."
    
    # --- Fallback: use message if available, else generic ---
    if error_msg:
        # Clean up technical messages for speech
        clean_msg = error_msg.rstrip(".")
        return f"I couldn't complete that: {clean_msg}."
    
    return "I couldn't complete that. Please try again."


def handle_user_text(text: str) -> Dict[str, Any]:
    """
    Handle user text input with optional multi-intent tool execution.
    
    Args:
        text: User's input text
        
    Returns:
        Dict with "reply", "latency_ms", and optional "execution_summary" keys
    """
    start_time = time.perf_counter()
    logger = get_logger_instance()
    
    # =========================================================================
    # PHASE 10: REFERENCE RESOLUTION (deterministic rewrite)
    # =========================================================================
    # Resolve vague follow-up phrases BEFORE any other processing.
    # "close it" → "close Chrome", "do that again" → repeat last action, etc.
    # This runs FIRST because it needs to transform the text before routing.
    from wyzer.core.reference_resolver import resolve_references, is_replay_sentinel, REPLAY_LAST_ACTION_SENTINEL
    from wyzer.context.world_state import get_world_state
    
    original_text = text
    resolved_text = resolve_references(text, get_world_state())
    if resolved_text != text:
        logger.info(f'[REF_RESOLVE] "{text}" → "{resolved_text}"')
        text = resolved_text
    
    # =========================================================================
    # PHASE 10.1: REPLAY_LAST_ACTION (deterministic replay)
    # =========================================================================
    # If reference resolver returned the replay sentinel, execute deterministic
    # replay of the last successful action without re-routing through LLM.
    if is_replay_sentinel(text):
        return _handle_replay_last_action(original_text, start_time)
    
    # =========================================================================
    # PHASE 10: "THE OTHER ONE" PATTERN (deterministic toggle)
    # =========================================================================
    # Handle "the other one" / "switch to the other one" patterns to toggle
    # between the last two distinct targets.
    from wyzer.core.reference_resolver import is_other_one_request, resolve_other_one, resolve_move_it_to_monitor, is_move_it_to_monitor_request
    if is_other_one_request(text):
        ws = get_world_state()
        resolved_target, reason = resolve_other_one(text, ws)
        if resolved_target:
            # Rewrite the command to use the resolved target
            # Determine the action from the text
            text_lower = text.lower()
            if "close" in text_lower:
                text = f"close {resolved_target}"
            elif "focus" in text_lower or "switch" in text_lower:
                text = f"focus {resolved_target}"
            elif "minimize" in text_lower:
                text = f"minimize {resolved_target}"
            elif "maximize" in text_lower:
                text = f"maximize {resolved_target}"
            elif "open" in text_lower:
                text = f"open {resolved_target}"
            else:
                text = f"focus {resolved_target}"  # Default action
            logger.info(f'[REF] "the other one" → "{text}" ({reason})')
        elif reason:
            # Ambiguous or no targets - return clarification
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            return {
                "reply": reason,
                "latency_ms": latency_ms,
                "meta": {"other_one": True, "clarification": True, "original_text": original_text},
            }
    
    # =========================================================================
    # PHASE 10: "MOVE IT TO MONITOR X" PATTERN (deterministic)
    # =========================================================================
    # Handle "move it to monitor 2" etc. - resolve the window and execute move_window.
    move_args, move_reason = resolve_move_it_to_monitor(text, get_world_state())
    if move_args is not None:
        target_name = move_args.get("_display_name") or move_args.get("process") or "window"
        logger.info(f'[REF] "move it to monitor" → process={move_args.get("process")} monitor={move_args.get("monitor")} ({move_reason})')
        # Execute move_window_to_monitor tool directly
        try:
            registry = get_registry()
            intent = {"tool": "move_window_to_monitor", "args": move_args}
            execution_summary, executed_intents = execute_tool_plan([intent], registry)
            
            # Format reply
            if execution_summary.ran and execution_summary.ran[0].ok:
                target = target_name
                monitor = move_args.get("monitor", 2)
                reply = f"Moved {target} to monitor {monitor}."
            else:
                error = execution_summary.ran[0].error if execution_summary.ran else None
                reply = _tool_error_to_speech(error, "move_window_to_monitor", move_args)
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            return {
                "reply": reply,
                "latency_ms": latency_ms,
                "execution_summary": {
                    "ran": [
                        {"tool": r.tool, "ok": r.ok, "result": r.result, "error": r.error}
                        for r in execution_summary.ran
                    ],
                    "stopped_early": execution_summary.stopped_early,
                },
                "meta": {"move_it_monitor": True, "original_text": original_text},
            }
        except Exception as e:
            logger.warning(f"[REF] move_window_to_monitor execution failed: {e}")
            # Fall through to normal processing
    elif move_reason:
        # Resolution failed - ask clarification
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        return {
            "reply": move_reason,
            "latency_ms": latency_ms,
            "meta": {"move_it_monitor": True, "clarification": True, "original_text": original_text},
        }
    
    # =========================================================================
    # PHASE 11: AUTONOMY COMMANDS (deterministic, no LLM)
    # =========================================================================
    # Check for autonomy-related voice commands before any other processing.
    # These are handled purely by regex matching and never touch the LLM.
    autonomy_result = _check_autonomy_commands(text, start_time)
    if autonomy_result is not None:
        return autonomy_result
    
    # =========================================================================
    # PHASE 12: WINDOW WATCHER COMMANDS (deterministic, no LLM)
    # =========================================================================
    # Check for window watcher commands: "what's on monitor 2", "close all on screen 1", etc.
    # These are handled purely by regex matching and never touch the LLM.
    window_watcher_result = _check_window_watcher_commands(text, start_time)
    if window_watcher_result is not None:
        return window_watcher_result
    
    # =========================================================================
    # PHASE 11: CONFIRMATION FLOW (deterministic)
    # =========================================================================
    # Check if user is responding to a pending confirmation (yes/no).
    # This must happen before routing to handle confirmation responses.
    confirmation_result = _check_confirmation_response(text, start_time)
    if confirmation_result is not None:
        return confirmation_result
    
    # =========================================================================
    # CONTINUATION PHRASE PRE-PASS (deterministic topic continuity)
    # =========================================================================
    # If user says "tell me more" etc., rewrite to include the last topic.
    # This ensures continuity even if session context is short/truncated.
    is_continuation = is_continuation_phrase(text)
    explicit_topic = is_explicit_continuation(text)  # "tell me more about X" -> X
    is_info_query = is_informational_query(text)  # "tell me about X", "what is Y"
    narrowing_question = None  # Set if continuation depth exceeded
    
    if is_continuation:
        # Track continuation hops to prevent infinite "tell me more" loops
        hops = increment_continuation_hops()
        if hops > _MAX_CONTINUATION_HOPS:
            # Too many consecutive continuations - ask narrowing question
            topic = get_last_topic() or "that"
            narrowing_question = f"I've covered the basics of {topic}. Would you like to know about specific characters, plot details, history, or something else?"
            logger.info(f"[CONTINUATION] Hop limit reached ({hops}), asking narrowing question")
        else:
            rewritten = rewrite_continuation(text)
            if rewritten != text:
                logger.info(f"[CONTINUATION] Rewriting '{text}' -> '{rewritten}' (hop {hops}/{_MAX_CONTINUATION_HOPS})")
                text = rewritten
    elif explicit_topic:
        # "Tell me more about One Piece" - update topic and mark as continuation
        set_last_topic(explicit_topic)
        logger.info(f"[CONTINUATION] Explicit topic: '{explicit_topic}'")
        is_continuation = True  # Treat as continuation (reply-only)
    else:
        # Not a continuation - extract and save topic for future follow-ups
        reset_continuation_hops()  # Reset hop counter on non-continuation
        topic = _extract_topic_from_query(text)
        if topic:
            set_last_topic(topic)
            logger.debug(f"[TOPIC] Tracking topic: {topic}")
    
    try:
        registry = get_registry()
        
        # =====================================================================
        # REPLY-ONLY FAST-PATH: Skip tools for conversational queries
        # =====================================================================
        # HARD INVARIANT (Phase 7+): When use_reply_only is True, we MUST NOT:
        #   - Run intent planning
        #   - Access tool registry for execution
        #   - Validate tool arguments
        #   - Execute any tools
        # This prevents tool hallucinations on conversational/knowledge queries.
        # Future refactors: DO NOT add "plan first" logic before this check.
        # =====================================================================
        use_reply_only = is_continuation or is_info_query
        if use_reply_only:
            reason = "continuation" if is_continuation else "informational query"
            logger.info(f"[REPLY-ONLY] Using reply-only mode ({reason})")
            
            # Handle narrowing question (continuation depth exceeded)
            if narrowing_question:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "reply": narrowing_question,
                    "latency_ms": latency_ms,
                    "meta": {"reply_only": True, "reason": "narrowing", "original_text": original_text},
                }
            
            # Early return if NO_OLLAMA mode is enabled
            if getattr(Config, "NO_OLLAMA", False):
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "reply": "I can't answer that without the language model.",
                    "latency_ms": latency_ms,
                    "meta": {"reply_only": True, "reason": reason, "no_ollama": True},
                }
            
            # Use reply-only LLM call (no tools)
            llm_response = _call_llm_reply_only(text)
            reply = llm_response.get("reply", "")
            
            # Phase 10: Mark as reply-only so "do that again" doesn't repeat chat
            from wyzer.context.world_state import set_last_llm_reply_only
            set_last_llm_reply_only(True)
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            return {
                "reply": reply,
                "latency_ms": latency_ms,
                "meta": {"reply_only": True, "reason": reason, "original_text": original_text},
            }

        # =====================================================================
        # DETERMINISTIC TOOL+TEXT SPLIT: Handle mixed utterances like
        # "Pause music and what's a VPN?" without relying on LLM intent planning.
        # This runs BEFORE the hybrid router to catch obvious tool phrases
        # followed by natural-language questions.
        # =====================================================================
        from wyzer.core.deterministic_splitter import get_split_intents
        
        split_result = get_split_intents(text, registry)
        if split_result is not None:
            tool_intents, leftover_text = split_result
            logger.info(f'[SPLIT] Executing tool+text split: tool={tool_intents[0]["tool"]} leftover="{leftover_text[:40]}..."')
            
            # ===============================================================
            # PHASE 11: Apply autonomy policy before execution (split path)
            # ===============================================================
            # Use high confidence for split path (deterministic match)
            autonomy_result = execute_tool_plan_with_autonomy(
                tool_intents,
                registry,
                0.95,  # Split path is high confidence
                start_time,
                original_text,
            )
            if autonomy_result is not None:
                # Autonomy policy blocked or deferred execution
                return autonomy_result
            # ===============================================================
            
            # Step 1: Execute the first tool intent(s)
            execution_summary, executed_intents = execute_tool_plan(tool_intents, registry)
            tool_reply = _format_fastpath_reply(text, executed_intents, execution_summary)
            all_executed_intents = list(executed_intents)
            
            # Step 2: Check if leftover text contains MORE tool intents (multi-intent)
            # This handles "Pause music, check weather, open chrome" where the splitter
            # only catches the first tool but leftover has more tools to execute.
            leftover_reply = ""
            leftover_tool_intents = []
            if leftover_text:
                leftover_decision = hybrid_router.decide(leftover_text)
                if leftover_decision.mode == "tool_plan" and leftover_decision.intents:
                    if _hybrid_tool_plan_is_registered(leftover_decision.intents, registry):
                        logger.info(f'[SPLIT] Leftover has {len(leftover_decision.intents)} more tool intents, executing...')
                        leftover_summary, leftover_executed = execute_tool_plan(leftover_decision.intents, registry)
                        leftover_reply = _format_fastpath_reply(leftover_text, leftover_executed, leftover_summary)
                        leftover_tool_intents = [i["tool"] for i in leftover_decision.intents]
                        all_executed_intents.extend(leftover_executed)
                        # Merge execution summaries
                        execution_summary.ran.extend(leftover_summary.ran)
                        execution_summary.stopped_early = execution_summary.stopped_early or leftover_summary.stopped_early
            
            # Step 3: If leftover wasn't handled as tools and LLM is available, get reply-only response
            llm_reply = ""
            if leftover_text and not leftover_reply and not getattr(Config, "NO_OLLAMA", False):
                llm_response = _call_llm_reply_only(leftover_text)
                llm_reply = llm_response.get("reply", "").strip()
            
            # Step 4: Combine replies (tool result + leftover tools or LLM answer)
            if leftover_reply:
                combined_reply = f"{tool_reply} {leftover_reply}"
            elif llm_reply:
                # Combine with natural flow
                combined_reply = f"{tool_reply} {llm_reply}"
            else:
                combined_reply = tool_reply
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            return {
                "reply": combined_reply,
                "latency_ms": latency_ms,
                "execution_summary": {
                    "ran": [
                        {
                            "tool": r.tool,
                            "ok": r.ok,
                            "result": r.result,
                            "error": r.error,
                        }
                        for r in execution_summary.ran
                    ],
                    "stopped_early": execution_summary.stopped_early,
                },
                "meta": {
                    "split_mode": True,
                    "tool_intents": [i["tool"] for i in tool_intents],
                    "leftover_text": leftover_text,
                    "leftover_tool_intents": leftover_tool_intents,
                    "had_llm_reply": bool(llm_reply),
                },
            }

        # Hybrid router FIRST: deterministic tool plans for obvious commands; LLM otherwise.
        hybrid_decision = hybrid_router.decide(text)
        if hybrid_decision.mode == "tool_plan" and hybrid_decision.intents:
            if _hybrid_tool_plan_is_registered(hybrid_decision.intents, registry):
                logger.info(f"[HYBRID] route=tool_plan confidence={hybrid_decision.confidence:.2f}")
                
                # ===============================================================
                # PHASE 11: Apply autonomy policy before execution
                # ===============================================================
                autonomy_result = execute_tool_plan_with_autonomy(
                    hybrid_decision.intents,
                    registry,
                    hybrid_decision.confidence,
                    start_time,
                    original_text,
                )
                if autonomy_result is not None:
                    # Autonomy policy blocked or deferred execution
                    return autonomy_result
                # ===============================================================
                
                execution_summary, executed_intents = execute_tool_plan(hybrid_decision.intents, registry)
                # Always use _format_fastpath_reply to properly handle tool execution failures.
                # The hybrid_decision.reply is just a prediction; we need to verify execution succeeded.
                reply = _format_fastpath_reply(
                    text, executed_intents, execution_summary
                )
                
                # Check for partial multi-intent with leftover text that needs LLM
                # Format: "__LEFTOVER__:<text>" in hybrid_decision.reply
                leftover_llm_reply = ""
                leftover_text = ""
                if hybrid_decision.reply and hybrid_decision.reply.startswith("__LEFTOVER__:"):
                    leftover_text = hybrid_decision.reply[len("__LEFTOVER__:"):]
                    if leftover_text and not getattr(Config, "NO_OLLAMA", False):
                        logger.info(f'[HYBRID] Partial multi-intent: executing LLM for leftover="{leftover_text[:40]}..."')
                        llm_response = _call_llm_reply_only(leftover_text)
                        leftover_llm_reply = llm_response.get("reply", "").strip()
                        if leftover_llm_reply:
                            reply = f"{reply} {leftover_llm_reply}"

                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "reply": reply,
                    "latency_ms": latency_ms,
                    "execution_summary": {
                        "ran": [
                            {
                                "tool": r.tool,
                                "ok": r.ok,
                                "result": r.result,
                                "error": r.error,
                            }
                            for r in execution_summary.ran
                        ],
                        "stopped_early": execution_summary.stopped_early,
                    },
                    "meta": {
                        "hybrid_route": "tool_plan",
                        "hybrid_confidence": float(hybrid_decision.confidence or 0.0),
                        "leftover_text": leftover_text if leftover_text else None,
                        "had_leftover_llm_reply": bool(leftover_llm_reply),
                    },
                }

            # Tool plan requested an unknown tool; fall back to LLM.
            logger.info(f"[HYBRID] route=llm confidence={hybrid_decision.confidence:.2f} (unregistered tool)")
        else:
            logger.info(f"[HYBRID] route=llm confidence={hybrid_decision.confidence:.2f}")

        # Hybrid explicit-tool handling (runs only after hybrid router chose LLM):
        # - If user explicitly names a tool and provides valid args -> run immediately (skip LLM).
        # - If the tool is explicit but args are ambiguous/missing -> ask the LLM to fill args for
        #   that specific tool (still avoiding full tool-selection/planning).
        explicit_req = _try_extract_explicit_tool_request(text, registry)
        if explicit_req is not None:
            tool_name, _rest = explicit_req
            parsed = _try_parse_explicit_tool_clause(text, registry)
            if parsed is not None:
                logger.info(f"[FASTPATH] Explicit tool invocation: {tool_name}")
                execution_summary = _execute_intents([parsed], registry)
                reply = _format_fastpath_reply(text, [parsed], execution_summary)

                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "reply": reply,
                    "latency_ms": latency_ms,
                    "execution_summary": {
                        "ran": [
                            {
                                "tool": r.tool,
                                "ok": r.ok,
                                "result": r.result,
                                "error": r.error,
                            }
                            for r in execution_summary.ran
                        ],
                        "stopped_early": execution_summary.stopped_early,
                    },
                    "meta": {
                        "hybrid_route": "llm",
                        "hybrid_confidence": float(hybrid_decision.confidence or 0.0),
                    },
                }

            logger.info(f"[HYBRID] Explicit tool '{tool_name}' needs LLM arg fill")
            llm_response = _call_llm_for_explicit_tool(text, tool_name, registry)
            intent_plan = normalize_plan(llm_response)

            # Constrain: only allow the explicitly requested tool.
            intent_plan.intents = [i for i in intent_plan.intents if (i.tool or "").strip().lower() == tool_name]
            if len(intent_plan.intents) > 1:
                intent_plan.intents = intent_plan.intents[:1]

            if intent_plan.intents:
                # Normalize tool aliases BEFORE unknown-tool filtering (safety net)
                intent_plan.intents, alias_logs = normalize_tool_aliases(intent_plan.intents, registry)
                for log_msg in alias_logs:
                    logger.debug(log_msg)
                
                # Filter unknown tools (though this path is for explicit tool requests)
                valid_intents, unknown_tools = filter_unknown_tools(intent_plan.intents, registry)
                if unknown_tools:
                    logger.warning(f"[UNKNOWN_TOOL] Explicit tool path had unknown tool(s): {', '.join(unknown_tools)}")
                
                if not valid_intents:
                    # Explicit tool request with unknown tool - fall back to reply
                    logger.info("[INTENT PLAN] Explicit tool was unknown, falling back to reply-only")
                    reply_only = _call_llm_reply_only(text)
                    end_time = time.perf_counter()
                    latency_ms = int((end_time - start_time) * 1000)
                    return {
                        "reply": reply_only.get("reply", "I don't have that capability."),
                        "latency_ms": latency_ms,
                        "meta": {
                            "hybrid_route": "llm",
                            "hybrid_confidence": float(hybrid_decision.confidence or 0.0),
                            "unknown_tools_fallback": True,
                        },
                    }
                
                intent_plan.intents = valid_intents
                
                try:
                    validate_intents(intent_plan.intents, registry)
                except ValueError as e:
                    end_time = time.perf_counter()
                    latency_ms = int((end_time - start_time) * 1000)
                    return {
                        "reply": f"I cannot execute that request: {str(e)}",
                        "latency_ms": latency_ms,
                        "meta": {
                            "hybrid_route": "llm",
                            "hybrid_confidence": float(hybrid_decision.confidence or 0.0),
                        },
                    }

                execution_summary = _execute_intents(intent_plan.intents, registry)
                final_response = _call_llm_with_execution_summary(text, execution_summary, registry)

                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "reply": final_response.get("reply", "I executed the action."),
                    "latency_ms": latency_ms,
                    "execution_summary": {
                        "ran": [
                            {
                                "tool": r.tool,
                                "ok": r.ok,
                                "result": r.result,
                                "error": r.error,
                            }
                            for r in execution_summary.ran
                        ],
                        "stopped_early": execution_summary.stopped_early,
                    },
                    "meta": {
                        "hybrid_route": "llm",
                        "hybrid_confidence": float(hybrid_decision.confidence or 0.0),
                    },
                }

            # LLM decided it couldn't safely infer args; return its reply.
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            return {
                "reply": intent_plan.reply or llm_response.get("reply", ""),
                "latency_ms": latency_ms,
                "meta": {
                    "hybrid_route": "llm",
                    "hybrid_confidence": float(hybrid_decision.confidence or 0.0),
                },
            }
        
        # Early return if NO_OLLAMA mode is enabled and we need LLM
        if getattr(Config, "NO_OLLAMA", False):
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            logger.info("[NO_OLLAMA] Request requires LLM but Ollama is disabled")
            return {
                "reply": _get_no_ollama_reply(),
                "latency_ms": latency_ms,
                "meta": {
                    "hybrid_route": "no_ollama_fallback",
                    "hybrid_confidence": 0.0,
                },
            }
        
        # First LLM call: interpret user intent(s)
        llm_response = _call_llm(text, registry)
        
        # Normalize LLM response to standard IntentPlan format
        intent_plan = normalize_plan(llm_response)

        # Heuristic rewrite: fix common LLM confusion where a game/app name
        # gets turned into an open_website URL (e.g., "Rocket League" -> rocketleague.com)
        _rewrite_open_website_intents(text, intent_plan.intents)

        # Heuristic filter: prevent obviously irrelevant tool calls (e.g. story requests triggering library refresh).
        intent_plan.intents = _filter_spurious_intents(text, intent_plan.intents)

        # If we dropped all intents and we have no usable reply, force a reply-only call.
        if not intent_plan.intents and not (intent_plan.reply or llm_response.get("reply", "")).strip():
            reply_only = _call_llm_reply_only(text)
            intent_plan.reply = reply_only.get("reply", "")
        
        # Check if there are any intents to execute
        if intent_plan.intents:
            # Log parsed plan (tool names only)
            tool_names = [intent.tool for intent in intent_plan.intents]
            logger.info(f"[INTENT PLAN] Executing {len(intent_plan.intents)} intent(s): {', '.join(tool_names)}")
            
            # Normalize tool aliases BEFORE unknown-tool filtering (safety net)
            intent_plan.intents, alias_logs = normalize_tool_aliases(intent_plan.intents, registry)
            for log_msg in alias_logs:
                logger.debug(log_msg)
            
            # Filter out unknown tools (graceful degradation instead of hard failure)
            valid_intents, unknown_tools = filter_unknown_tools(intent_plan.intents, registry)
            
            if unknown_tools:
                # Log unknown tools for debugging (single line, includes tool names)
                logger.warning(f"[UNKNOWN_TOOL] Skipping unknown tool(s): {', '.join(unknown_tools)}")
            
            # If ALL intents were unknown, fall back to reply-only
            if not valid_intents:
                logger.info("[INTENT PLAN] All intents had unknown tools, falling back to reply-only")
                reply_only = _call_llm_reply_only(text)
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "reply": reply_only.get("reply", "I'm not sure how to help with that."),
                    "latency_ms": latency_ms,
                    "meta": {
                        "unknown_tools_fallback": True,
                        "unknown_tools": unknown_tools,
                    },
                }
            
            # Update intents to only valid ones
            intent_plan.intents = valid_intents
            
            # Validate remaining intents (max count, args format, etc.)
            try:
                validate_intents(intent_plan.intents, registry)
            except ValueError as e:
                # Validation failed - return error
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                return {
                    "reply": f"I cannot execute that request: {str(e)}",
                    "latency_ms": latency_ms
                }
            
            # Execute intents sequentially
            execution_summary = _execute_intents(intent_plan.intents, registry)
            
            # Second LLM call: generate final reply with execution results
            final_response = _call_llm_with_execution_summary(
                text, execution_summary, registry
            )
            
            # Calculate latency
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "reply": final_response.get("reply", "I executed the action."),
                "latency_ms": latency_ms,
                "execution_summary": {
                    "ran": [
                        {
                            "tool": r.tool,
                            "ok": r.ok,
                            "result": r.result,
                            "error": r.error
                        }
                        for r in execution_summary.ran
                    ],
                    "stopped_early": execution_summary.stopped_early
                },
                "meta": {
                    "hybrid_route": "llm",
                    "hybrid_confidence": float(hybrid_decision.confidence or 0.0),
                },
            }
        else:
            # No intents needed, return direct reply
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "reply": intent_plan.reply or llm_response.get("reply", ""),
                "latency_ms": latency_ms,
                "meta": {
                    "hybrid_route": "llm",
                    "hybrid_confidence": float(hybrid_decision.confidence or 0.0),
                },
            }
            
    except Exception as e:
        # Safe fallback on any error
        logger.error(f"[ORCHESTRATOR ERROR] {str(e)}")
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        return {
            "reply": f"I encountered an error: {str(e)}",
            "latency_ms": latency_ms,
            "meta": {
                "hybrid_route": "llm",
                "hybrid_confidence": 0.0,
            },
        }


def should_use_streaming_tts(text: str) -> bool:
    """
    Determine if streaming TTS should be used for this request.
    
    Streaming TTS is ONLY for normal LLM chat responses, NOT for:
    - Tool/action commands (deterministic routing)
    - Hybrid router tool plans
    - Autonomy commands ("why did you do that", "autonomy off", etc.)
    
    This is a PRE-CHECK before calling handle_user_text_streaming.
    
    Args:
        text: User's input text
        
    Returns:
        True if streaming TTS should be used, False otherwise
    """
    # Check global config flag first
    if not getattr(Config, "OLLAMA_STREAM_TTS", False):
        return False
    
    # Check if Ollama is disabled
    if getattr(Config, "NO_OLLAMA", False):
        return False
    
    # Phase 11: Check for autonomy commands BEFORE hybrid router
    # These must be handled deterministically, never streamed to LLM
    text_stripped = text.strip()
    if (_WHY_DID_YOU_RE.match(text_stripped) or
        _AUTONOMY_OFF_RE.match(text_stripped) or
        _AUTONOMY_LOW_RE.match(text_stripped) or
        _AUTONOMY_NORMAL_RE.match(text_stripped) or
        _AUTONOMY_HIGH_RE.match(text_stripped) or
        _AUTONOMY_STATUS_RE.match(text_stripped)):
        return False
    
    # Phase 12: Check for window watcher commands BEFORE hybrid router
    # These must be handled deterministically, never streamed to LLM
    if (_WHATS_ON_MONITOR_RE.match(text_stripped) or
        _CLOSE_ALL_ON_MONITOR_RE.match(text_stripped) or
        _WHAT_DID_I_OPEN_RE.match(text_stripped) or
        _WHERE_AM_I_RE.match(text_stripped)):
        return False
    
    # Phase 10: Check for reference resolution patterns BEFORE hybrid router
    # Pronoun-based commands must be handled deterministically, not streamed
    from wyzer.core.reference_resolver import is_move_it_to_monitor_request, is_other_one_request, is_pronoun_action_request
    if is_move_it_to_monitor_request(text_stripped) or is_other_one_request(text_stripped) or is_pronoun_action_request(text_stripped):
        return False
    
    # Phase 11: Check for confirmation responses ONLY if there's a pending confirmation
    from wyzer.context.world_state import get_pending_confirmation
    pending = get_pending_confirmation()
    if pending is not None:
        # Block streaming for ANY input when pending confirmation exists
        # The brain worker's resolve_pending() will handle it deterministically
        return False
    
    # Check hybrid router - if it returns a tool_plan, don't stream
    hybrid_decision = hybrid_router.decide(text)
    if hybrid_decision.mode == "tool_plan" and hybrid_decision.intents:
        return False
    
    # Stream for any LLM-routed query (conversational responses)
    # The hybrid router already determined this should go to LLM, so stream it
    if hybrid_decision.mode == "llm":
        return True
    
    # Check if this is a continuation/informational query (reply-only mode)
    is_continuation = is_continuation_phrase(text)
    explicit_topic = is_explicit_continuation(text)
    is_info_query = is_informational_query(text)
    
    # Only stream for reply-only queries (conversational)
    if is_continuation or explicit_topic or is_info_query:
        return True
    
    # For other queries that might involve tools, don't stream
    return False


def handle_user_text_streaming(
    text: str,
    on_segment: "Callable[[str], None]",
    cancel_check: "Optional[Callable[[], bool]]" = None
) -> Dict[str, Any]:
    """
    Handle user text with streaming TTS output.
    
    This is an alternative to handle_user_text() that streams LLM tokens
    to TTS as they arrive. ONLY use this for reply-only/conversational
    queries - check should_use_streaming_tts() first.
    
    If streaming fails, falls back to non-streaming.
    
    WARNING: This function should ONLY be used for conversational queries.
    Tool commands must use handle_user_text() to ensure proper routing.
    
    Args:
        text: User's input text
        on_segment: Callback to receive TTS segments as they're ready
        cancel_check: Optional callable returning True to cancel streaming
        
    Returns:
        Dict with "reply", "latency_ms", and "meta" keys
    """
    from wyzer.brain.tts_stream_buffer import TTSStreamBuffer, now_ms as buffer_now_ms
    from wyzer.brain.ollama_client import OllamaClient
    
    start_time = time.perf_counter()
    logger = get_logger_instance()
    
    # Double-check we should actually stream
    if not should_use_streaming_tts(text):
        logger.debug("[STREAM_TTS] Streaming not appropriate, using non-streaming path")
        result = handle_user_text(text)
        # Emit full reply as single TTS segment
        if result.get("reply") and on_segment:
            try:
                on_segment(result["reply"])
            except Exception as e:
                logger.error(f"[STREAM_TTS] TTS callback error: {e}")
        return result
    
    logger.info("[STREAM_TTS] Using streaming TTS path")
    
    try:
        # Build reply-only prompt (no JSON, plain text output for streaming TTS)
        # This mirrors _call_llm_reply_only but outputs plain text instead of JSON
        ctx = _gather_context_blocks(text, max_session_turns=2)
        
        # Check for smalltalk directive
        smalltalk_directive = ""
        try:
            from wyzer.brain.llm_engine import _is_smalltalk_request, SMALLTALK_SYSTEM_DIRECTIVE
            if _is_smalltalk_request(text):
                smalltalk_directive = f"\n{SMALLTALK_SYSTEM_DIRECTIVE}\n"
        except Exception:
            pass
        
        # Check if user wants detailed/long response (story, "in depth", "in detail", etc.)
        is_detail_request = False
        try:
            from wyzer.brain.llm_engine import _is_story_creative_request
            is_detail_request = _is_story_creative_request(text)
        except Exception:
            pass
        
        # Adjust length instruction based on request type
        if is_detail_request:
            length_instruction = "Provide a thorough, detailed response."
            logger.debug("[STREAM_TTS] Detail/story request detected - allowing longer response")
        else:
            length_instruction = "Reply in 1-2 sentences. Be direct and concise."
        
        prompt = f"""You are Wyzer, a local voice assistant.
{ctx['session']}{ctx['promoted']}{ctx['redaction']}{ctx['memories']}{smalltalk_directive}
{length_instruction}
You ARE allowed to generate stories, poems, jokes, and creative content when asked - keep those spoken-friendly: no markdown, no bullet lists.

User: {text}

Wyzer:"""
        logger.debug(f"[STREAM_TTS] Using reply-only prompt (plain text)")
        
        # Get appropriate LLM client based on mode (supports Ollama and llamacpp)
        client = _get_llm_client()
        if client is None:
            # LLM unavailable
            logger.warning("[STREAM_TTS] No LLM available, falling back")
            result = handle_user_text(text)
            if result.get("reply") and on_segment:
                try:
                    on_segment(result["reply"])
                except Exception as e:
                    logger.error(f"[STREAM_TTS] TTS callback error: {e}")
            return result
        
        # Use higher num_predict for detail requests
        num_predict = Config.VOICE_FAST_STORY_MAX_TOKENS if is_detail_request else Config.OLLAMA_NUM_PREDICT
        
        options = {
            "temperature": Config.OLLAMA_TEMPERATURE,
            "top_p": Config.OLLAMA_TOP_P,
            "num_ctx": Config.OLLAMA_NUM_CTX,
            "num_predict": num_predict
        }
        
        # Apply voice-fast preset overrides (smalltalk detection, story mode, etc.)
        try:
            from wyzer.brain.llm_engine import get_voice_fast_options
            llm_mode = getattr(Config, "LLM_MODE", "ollama")
            voice_fast_opts = get_voice_fast_options(text, llm_mode)
            if voice_fast_opts:
                # Filter out internal keys (prefixed with _)
                for key, value in voice_fast_opts.items():
                    if not key.startswith("_"):
                        options[key] = value
        except Exception as e:
            logger.debug(f"[STREAM_TTS] voice_fast_options error: {e}")
        
        logger.debug("[STREAM_TTS] LLM stream started")
        
        # Get streaming generator
        token_stream = client.generate_stream(
            prompt=prompt,
            model=Config.OLLAMA_MODEL,  # Model param is mainly for Ollama; llamacpp uses loaded model
            options=options
        )
        
        # Process stream using sentence-gated buffer for best UX
        # This accumulates text and only flushes on sentence boundaries
        buffer = TTSStreamBuffer(
            min_chars=getattr(Config, 'TTS_STREAM_MIN_CHARS', 60),
            min_words=getattr(Config, 'TTS_STREAM_MIN_WORDS', 10),
            max_wait_ms=getattr(Config, 'TTS_STREAM_MAX_WAIT_MS', 900),
            boundaries=getattr(Config, 'TTS_STREAM_BOUNDARIES', '.!?:'),
        )
        
        full_reply_parts = []
        segment_count = 0
        cancelled = False
        
        try:
            for token in token_stream:
                # Check for cancellation (barge-in)
                if cancel_check and cancel_check():
                    logger.debug("[STREAM_TTS] Cancelled by cancel_check (barge-in)")
                    cancelled = True
                    break
                
                # Accumulate full reply for logging/display
                full_reply_parts.append(token)
                
                # Feed to sentence-gated buffer
                chunks = buffer.add_text(token, buffer_now_ms())
                
                # Send any ready chunks to TTS
                for chunk in chunks:
                    segment_count += 1
                    if on_segment:
                        try:
                            on_segment(chunk)
                        except Exception as seg_err:
                            logger.error(f"[STREAM_TTS] on_segment callback error: {seg_err}")
            
            # Flush remaining buffer at end of stream (unless cancelled)
            if not cancelled:
                final_chunks = buffer.flush_final()
                for chunk in final_chunks:
                    segment_count += 1
                    if on_segment:
                        try:
                            on_segment(chunk)
                        except Exception as seg_err:
                            logger.error(f"[STREAM_TTS] on_segment callback error (final): {seg_err}")
            
            logger.debug(f"[STREAM_TTS] Emitted {segment_count} segments total")
            
        except Exception as stream_err:
            logger.error(f"[STREAM_TTS] Stream processing error: {stream_err}")
            raise
        
        reply = "".join(full_reply_parts).strip()
        
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        logger.debug(f"[STREAM_TTS] LLM stream ended")
        
        return {
            "reply": reply or "I'm not sure how to respond to that.",
            "latency_ms": latency_ms,
            "meta": {
                "reply_only": True,
                "streamed": True,
            },
        }
        
    except Exception as e:
        # Streaming failed - fall back to non-streaming
        logger.warning(f"[STREAM_TTS] Streaming failed, falling back: {e}")
        
        result = handle_user_text(text)
        result.setdefault("meta", {})["streamed"] = False
        result["meta"]["stream_fallback"] = True
        
        # Emit full reply as single TTS segment
        if result.get("reply") and on_segment:
            try:
                on_segment(result["reply"])
            except Exception as tts_err:
                logger.error(f"[STREAM_TTS] TTS callback error: {tts_err}")
        
        return result


def _hybrid_tool_plan_is_registered(intents: List[Dict[str, Any]], registry) -> bool:
    """Ensure a deterministic plan only references registered tools."""
    for intent in intents or []:
        tool_name = (intent or {}).get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return False
        if not registry.has_tool(tool_name.strip()):
            return False
    return True


def execute_tool_plan(intents: List[Dict[str, Any]], registry) -> Tuple[ExecutionSummary, List[Intent]]:
    """Execute a list of tool-call dicts using the existing intent pipeline.
    
    Phase 10: Applies reference resolution to each intent before execution.
    This resolves pronouns (it/that/this) to concrete targets using world state.
    """
    logger = get_logger_instance()
    
    # =========================================================================
    # PHASE 10: REFERENCE RESOLUTION FOR INTENTS
    # =========================================================================
    # Before converting to Intent objects, resolve any pronoun references
    # in the intent arguments using world state.
    from wyzer.core.reference_resolver import resolve_intent_args, has_unresolved_pronoun
    from wyzer.context.world_state import get_world_state, update_last_intents
    
    ws = get_world_state()
    resolved_intents: List[Dict[str, Any]] = []
    clarification_needed = None
    
    for raw in intents or []:
        tool = (raw or {}).get("tool")
        args = (raw or {}).get("args")
        continue_on_error = bool((raw or {}).get("continue_on_error", False))
        
        if not isinstance(tool, str) or not tool.strip():
            continue
        if not isinstance(args, dict):
            args = {}
        
        intent_dict = {"tool": tool.strip(), "args": args}
        
        # Check if this intent has unresolved pronouns
        if has_unresolved_pronoun(intent_dict):
            resolved_args, clarification = resolve_intent_args(intent_dict, ws)
            if clarification:
                # Resolution failed - need to ask user
                clarification_needed = clarification
                logger.info(f"[REF] intent={tool} needs clarification: {clarification}")
            else:
                args = resolved_args
                logger.info(f"[REF] intent={tool} resolved args={args}")
        
        resolved_intents.append({
            "tool": tool.strip(),
            "args": args,
            "continue_on_error": continue_on_error,
        })
    
    # If any intent needs clarification, return early with a failed result
    if clarification_needed:
        # Create a summary indicating clarification is needed
        from wyzer.core.intent_plan import ExecutionResult, ExecutionSummary
        failed_result = ExecutionResult(
            tool="__reference_resolution__",
            ok=False,
            result=None,
            error={"type": "clarification_needed", "message": clarification_needed}
        )
        return ExecutionSummary(ran=[failed_result], stopped_early=True), []
    
    # Convert to Intent objects
    converted: List[Intent] = []
    for raw in resolved_intents:
        tool = raw.get("tool")
        args = raw.get("args", {})
        continue_on_error = raw.get("continue_on_error", False)
        converted.append(Intent(tool=tool, args=args, continue_on_error=continue_on_error))

    # Store intents for "do that again" before execution
    update_last_intents(resolved_intents)
    
    # Preserve existing validation/gating behavior.
    validate_intents(converted, registry)
    return _execute_intents(converted, registry), converted


def execute_tool_plan_with_autonomy(
    intents: List[Dict[str, Any]],
    registry,
    confidence: float,
    start_time: float,
    original_text: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Execute a tool plan with autonomy policy applied.
    
    Phase 11: Applies autonomy policy before execution.
    - If mode is "off": execute immediately (current behavior)
    - If mode is active: apply policy decision (execute/ask/deny)
    
    Args:
        intents: List of tool call dicts
        registry: Tool registry
        confidence: Router confidence (0.0-1.0)
        start_time: Performance timer start
        original_text: Original user text for logging
        
    Returns:
        Response dict if autonomy blocked/deferred execution, None to proceed normally
    """
    logger = get_logger_instance()
    
    from wyzer.context.world_state import (
        get_autonomy_mode,
        set_pending_confirmation,
        set_last_autonomy_decision,
        get_world_state,
    )
    from wyzer.policy.risk import classify_plan
    from wyzer.policy.autonomy_policy import assess, summarize_plan
    
    mode = get_autonomy_mode()
    
    # Mode OFF: preserve current behavior exactly (no policy check)
    if mode == "off":
        return None  # Signal to proceed with normal execution
    
    # Compute risk classification
    ws = get_world_state()
    risk = classify_plan(intents, ws)
    
    # Get confirmation setting from config
    confirm_sensitive = getattr(Config, "AUTONOMY_CONFIRM_SENSITIVE", True)
    
    # Assess policy decision
    decision = assess(intents, confidence, mode, risk, confirm_sensitive)
    
    # Log the decision
    plan_summary = summarize_plan(intents)
    logger.info(
        f"[AUTONOMY] mode={mode} confidence={confidence:.2f} risk={risk} "
        f"action={decision['action']} confirm={1 if decision['needs_confirmation'] else 0} "
        f"reason=\"{decision['reason'][:60]}...\""
    )
    
    # Store decision for "why did you do that" command
    set_last_autonomy_decision(
        mode=mode,
        confidence=confidence,
        risk=risk,
        action=decision["action"],
        reason=decision["reason"],
        plan_summary=plan_summary,
    )
    
    # Handle decision
    if decision["action"] == "execute":
        # Policy says execute - proceed with normal execution
        return None
    
    elif decision["action"] == "ask":
        if decision["needs_confirmation"]:
            # High-risk confirmation flow
            timeout_sec = getattr(Config, "AUTONOMY_CONFIRM_TIMEOUT_SEC", 20.0)
            prompt = decision["question"] or "Do you want me to proceed?"
            
            set_pending_confirmation(
                plan=intents,
                prompt=prompt,
                timeout_sec=timeout_sec,
                decision=dict(decision),
            )
            logger.info(f"[CONFIRM] set expires_in={timeout_sec}s")
            
            end_time = time.perf_counter()
            return {
                "reply": prompt,
                "latency_ms": int((end_time - start_time) * 1000),
                "meta": {
                    "autonomy_action": "ask_confirm",
                    "risk": risk,
                    "confidence": confidence,
                    "plan_summary": plan_summary,
                    "has_pending_confirmation": True,  # Flag for Core process
                    "confirmation_timeout_sec": timeout_sec,
                },
            }
        else:
            # Clarification question (not high-risk, just uncertain)
            # ALSO set pending confirmation so "Yes" can trigger execution
            question = decision["question"] or "Can you clarify what you'd like me to do?"
            timeout_sec = getattr(Config, "AUTONOMY_CONFIRM_TIMEOUT_SEC", 20.0)
            
            set_pending_confirmation(
                plan=intents,
                prompt=question,
                timeout_sec=timeout_sec,
                decision=dict(decision),
            )
            logger.info(f"[CLARIFY] set pending, expires_in={timeout_sec}s")
            
            end_time = time.perf_counter()
            return {
                "reply": question,
                "latency_ms": int((end_time - start_time) * 1000),
                "meta": {
                    "autonomy_action": "ask_clarify",
                    "risk": risk,
                    "confidence": confidence,
                    "plan_summary": plan_summary,
                    "has_pending_confirmation": True,  # Flag for Core process
                    "confirmation_timeout_sec": timeout_sec,
                },
            }
    
    else:  # deny
        reason = decision["reason"]
        end_time = time.perf_counter()
        return {
            "reply": "I'm not confident enough to do that. Can you be more specific?",
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {
                "autonomy_action": "deny",
                "risk": risk,
                "confidence": confidence,
                "reason": reason,
                "plan_summary": plan_summary,
            },
        }


def _try_extract_explicit_tool_request(user_text: str, registry) -> Optional[Tuple[str, str]]:
    """Detect an explicit 'tool/run/execute <tool> ...' request.

    Returns (tool_name_lower, rest_text).
    """
    raw = (user_text or "").strip()
    if not raw:
        return None

    m = _FASTPATH_EXPLICIT_TOOL_RE.match(raw)
    if not m:
        return None

    tool_name = (m.group("tool") or "").strip().lower()
    if not tool_name:
        return None

    if not registry.has_tool(tool_name):
        return None

    rest = (m.group("rest") or "").strip()
    return tool_name, rest


def _execute_intents(intents, registry) -> ExecutionSummary:
    """
    Execute multiple intents sequentially and collect results.
    
    Args:
        intents: List of Intent objects to execute
        registry: Tool registry
        
    Returns:
        ExecutionSummary with results of all executed intents
    """
    logger = get_logger_instance()
    results = []
    stopped_early = False
    
    for idx, intent in enumerate(intents):
        logger.info(f"[INTENT {idx + 1}/{len(intents)}] Executing: {intent.tool}")
        
        # Execute the tool
        tool_result = _execute_tool(registry, intent.tool, intent.args)
        
        # Check if execution was successful
        has_error = "error" in tool_result

        error_type = None
        if has_error:
            try:
                error_type = (tool_result.get("error") or {}).get("type")
            except Exception:
                error_type = None
        
        # Create execution result
        exec_result = ExecutionResult(
            tool=intent.tool,
            ok=not has_error,
            result=tool_result if not has_error else None,
            error=tool_result.get("error") if has_error else None
        )
        
        results.append(exec_result)
        
        # If error occurred and continue_on_error is False, stop execution
        if has_error and not intent.continue_on_error:
            # Focus is often an optional first step before window operations.
            # If it fails to locate the window, still try subsequent actions.
            if intent.tool == "focus_window" and idx < (len(intents) - 1) and error_type == "window_not_found":
                logger.info(f"[INTENT {idx + 1}/{len(intents)}] Focus failed (window_not_found), continuing")
                continue
            logger.info(f"[INTENT {idx + 1}/{len(intents)}] Failed, stopping execution")
            stopped_early = True
            break
        
        logger.info(f"[INTENT {idx + 1}/{len(intents)}] {'Success' if not has_error else 'Failed (continuing)'}")
    
    return ExecutionSummary(ran=results, stopped_early=stopped_early)


def _normalize_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _strip_trailing_punct(text: str) -> str:
    return (text or "").strip().rstrip(".?!,;:\"")


def _extract_int(text: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,3})\b", text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_volume_percent(text: str) -> Optional[int]:
    """Extract a 0-100 volume percent, supporting '100' and optional '%' sign."""
    m = re.search(r"\b(\d{1,3})\s*%?\b", text or "")
    if not m:
        return None
    try:
        v = int(m.group(1))
    except Exception:
        return None
    if 0 <= v <= 100:
        return v
    return None


def _parse_volume_delta_hint(text_lower: str) -> int:
    """Heuristic delta in percent for phrases like 'a bit', 'a lot'."""
    tl = text_lower or ""
    if any(k in tl for k in ["a little", "a bit", "slightly", "tiny bit", "small"]):
        return 5
    if any(k in tl for k in ["a lot", "much", "way", "significantly"]):
        return 20
    return 10


def _parse_volume_scope_and_process(clause: str) -> Tuple[str, str]:
    """Return (scope, process) where scope is 'master' or 'app'."""
    c = (clause or "").strip()
    cl = c.lower()

    # Strip common query prefixes so we don't treat them as app names.
    # Examples:
    #   "get volume" -> "volume"
    #   "what is the volume" -> "volume"
    #   "what is spotify volume" -> "spotify volume"
    cl = re.sub(
        r"^(?:what\s+is|what's|whats|get|check|show|tell\s+me|current)\s+",
        "",
        cl,
    )
    cl = re.sub(r"^the\s+", "", cl)

    # "set <proc> volume ..." should treat <proc> as app.
    m = re.match(r"^set\s+(?P<proc>.+?)\s+(?:volume|sound|audio)\b", cl)
    if m:
        proc = _strip_trailing_punct(m.group("proc")).strip()
        if proc and proc not in {"the", "my", "this", "that", "volume", "sound", "audio"}:
            return "app", proc

    # Examples:
    #   "spotify volume 30"
    #   "set spotify volume to 30"
    #   "volume 30 for spotify"
    #   "turn down chrome"
    m = re.search(r"\b(?:for|in|on)\s+(?P<proc>[a-z0-9][a-z0-9 _\-\.]{1,60})$", cl)
    if m:
        proc = _strip_trailing_punct(m.group("proc")).strip()
        if proc:
            return "app", proc

    # "spotify volume ..." / "chrome sound ..." (but NOT "set volume ...")
    m = re.match(r"^(?P<first>[a-z0-9][a-z0-9 _\-\.]{1,60})\s+(?:volume|sound|audio)\b", cl)
    if m:
        first = _strip_trailing_punct(m.group("first")).strip()
        if first and first not in {"the", "my", "this", "that", "set", "volume", "sound", "audio"}:
            return "app", first

    # "turn down spotify" / "mute discord" etc.
    m = re.match(r"^(?:turn\s+(?:up|down)|mute|unmute|quieter|louder)\s+(?P<proc>.+)$", cl)
    if m:
        proc = _strip_trailing_punct(m.group("proc")).strip()
        proc_l = proc.lower()
        if proc_l in {"it", "it up", "it down", "the volume", "volume", "sound", "audio"}:
            return "master", ""
        if proc and proc not in {"it", "volume", "sound", "audio"}:
            return "app", proc

    return "master", ""


def _coerce_int_like(value: Any) -> Any:
    """Coerce common numeric string forms (e.g., '35', '35%') to int.

    Returns original value if it can't be safely coerced.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # Accept integral floats.
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("%"): 
            s = s[:-1].strip()
        if re.fullmatch(r"-?\d{1,3}", s):
            try:
                return int(s)
            except Exception:
                return value
    return value


def _normalize_tool_args(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight arg normalization before schema validation."""
    if tool_name != "volume_control" or not isinstance(tool_args, dict):
        return tool_args

    normalized = dict(tool_args)
    if "level" in normalized:
        normalized["level"] = _coerce_int_like(normalized.get("level"))
    if "delta" in normalized:
        normalized["delta"] = _coerce_int_like(normalized.get("delta"))
    # Normalize scope/action strings.
    if isinstance(normalized.get("scope"), str):
        normalized["scope"] = str(normalized.get("scope") or "").strip().lower()
    if isinstance(normalized.get("action"), str):
        normalized["action"] = str(normalized.get("action") or "").strip().lower()
    return normalized


def _parse_scalar_value(raw: str) -> Any:
    v = (raw or "").strip()
    if not v:
        return ""

    vl = v.lower()
    if vl in {"true", "yes", "y", "on"}:
        return True
    if vl in {"false", "no", "n", "off"}:
        return False

    # int
    if re.fullmatch(r"-?\d+", v):
        try:
            return int(v)
        except Exception:
            pass

    # float
    if re.fullmatch(r"-?\d+\.\d+", v):
        try:
            return float(v)
        except Exception:
            pass

    return v


def _pick_single_arg_key_from_schema(schema: Dict[str, Any]) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return None

    required = schema.get("required")
    if isinstance(required, list) and len(required) == 1 and isinstance(required[0], str):
        return required[0]

    if len(props) == 1:
        return next(iter(props.keys()))

    return None


def _try_parse_explicit_tool_clause(clause: str, registry) -> Optional[Intent]:
    """Parse an explicit tool invocation.

    Supported forms (case-insensitive):
      - tool <tool_name> {"json": "args"}
      - tool <tool_name> key=value key2="value with spaces"
      - tool <tool_name> <free text>    (only when schema has a single arg)
      - optional: continue_on_error=true
    """
    text = (clause or "").strip()
    if not text:
        return None

    m = _FASTPATH_EXPLICIT_TOOL_RE.match(text)
    if not m:
        return None

    tool_name = (m.group("tool") or "").strip()
    if not tool_name:
        return None

    # Tool names in registry are lowercase snake_case.
    tool_name = tool_name.strip()
    if not registry.has_tool(tool_name):
        return None

    rest = (m.group("rest") or "").strip()
    args: Dict[str, Any] = {}
    continue_on_error = False

    if rest:
        # JSON args dict (optionally includes continue_on_error)
        if rest.startswith("{") and rest.endswith("}"):
            try:
                payload = json.loads(rest)
            except Exception:
                return None
            if not isinstance(payload, dict):
                return None

            # Allow either direct args or nested args.
            if "args" in payload and isinstance(payload.get("args"), dict):
                args = dict(payload.get("args") or {})
            else:
                args = dict(payload)

            if isinstance(args.get("continue_on_error"), bool):
                continue_on_error = bool(args.pop("continue_on_error"))
            return Intent(tool=tool_name, args=args, continue_on_error=continue_on_error)

        # key=value tokens / quoted strings
        try:
            tokens = shlex.split(rest, posix=False)
        except Exception:
            tokens = rest.split()

        tail: List[str] = []
        for tok in tokens:
            if "=" not in tok:
                tail.append(tok)
                continue
            k, v = tok.split("=", 1)
            k = (k or "").strip()
            v = (v or "").strip().strip('"').strip("'")
            if not k:
                return None

            if k in {"continue_on_error", "continue", "co"}:
                continue_on_error = bool(_parse_scalar_value(v))
                continue

            args[k] = _parse_scalar_value(v)

        if tail:
            # If we already saw key=value args, trailing free-text is ambiguous.
            if args:
                return None
            tool = registry.get(tool_name)
            schema = getattr(tool, "args_schema", {}) if tool is not None else {}
            key = _pick_single_arg_key_from_schema(schema)
            if not key:
                return None
            args[key] = _strip_trailing_punct(" ".join(tail)).strip()
            if not args[key]:
                return None

    return Intent(tool=tool_name, args=args, continue_on_error=continue_on_error)


def _extract_days(text: str) -> Optional[int]:
    """Extract a day count from phrases like 'next 5 days' or '5 day forecast'."""
    if not text:
        return None
    m = re.search(r"\b(?:next\s+)?(\d{1,2})\s+day(?:s)?\b", text.lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_location_after_in(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b(?:in|for)\s+(.+)$", text.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    loc = _strip_trailing_punct(m.group(1)).strip()
    return loc or None


def _parse_audio_device_target(text: str) -> Optional[str]:
    """Extract device name from explicit audio output switching commands."""
    if not text:
        return None

    t = text.strip()
    tl = t.lower()

    # Require explicit audio/output context to avoid accidental triggers.
    explicit_markers = [
        "audio output",
        "output device",
        "audio device",
        "playback device",
        "default audio",
        "default output",
        "speakers",
        "headphones",
        "headset",
        "earbuds",
    ]
    if not any(m in tl for m in explicit_markers):
        return None

    # Common patterns with an explicit target.
    patterns = [
        r"^(?:set|switch|change)\s+(?:the\s+)?(?:default\s+)?(?:audio\s+output|output\s+device|audio\s+device|playback\s+device)(?:\s+device)?\s+(?:to|as)\s+(.+)$",
        r"^(?:set|switch|change)\s+(?:to)\s+(.+?)\s+(?:audio\s+output|output\s+device|audio\s+device|playback\s+device)$",
        r"^(?:use)\s+(.+?)\s+(?:as)\s+(?:audio\s+output|output\s+device|audio\s+device|playback\s+device)$",
        r"^(?:set|switch|change)\s+(?:speakers|headphones|headset|earbuds)\s+(?:to)\s+(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, tl)
        if not m:
            continue
        dev = _strip_trailing_punct(m.group(1)).strip()
        if not dev:
            return None
        # Reject placeholder / ambiguous targets.
        if dev in {"audio", "output", "device", "speakers", "headphones", "headset", "earbuds"}:
            return None
        return dev

    # A very explicit fallback: "set audio output to <dev>" somewhere in the clause.
    m = re.search(r"\b(?:audio\s+output|output\s+device|playback\s+device)\s+(?:to|as)\s+(.+)$", tl)
    if m:
        dev = _strip_trailing_punct(m.group(1)).strip()
        if dev and dev not in {"audio", "output", "device"}:
            return dev

    return None


def _looks_like_url(token: str) -> bool:
    t = (token or "").strip().lower()
    if not t:
        return False
    if t.startswith("http://") or t.startswith("https://") or t.startswith("www."):
        return True
    # Very small heuristic for domain-ish strings.
    return bool(re.search(r"\.[a-z]{2,}$", t))


def _try_fastpath_intents(user_text: str, registry) -> Optional[List[Intent]]:
    """Return a list of intents when the command is unambiguous.

    This is a deterministic, conservative parser intended to bypass the LLM.
    If unsure, return None and let the LLM handle it.
    """
    raw = (user_text or "").strip()
    if not raw:
        return None

    # Split basic multi-step commands: "pause music and mute volume".
    parts = [p.strip() for p in _FASTPATH_SPLIT_RE.split(raw) if p and p.strip()]

    # Expand comma-separated command lists inside each part, so mixed separators work:
    #   "open spotify, then open chrome and open youtube"
    # We keep this conservative to avoid breaking things like "Paris, France".
    expanded: List[str] = []
    for p in parts:
        # Always allow ';' as an explicit command separator.
        if ";" in p:
            expanded.extend([s.strip() for s in _FASTPATH_SEMI_SPLIT_RE.split(p) if s and s.strip()])
            continue

        if "," not in p:
            expanded.append(p)
            continue

        tl = p.lower()
        # Avoid splitting common location strings in weather/forecast commands.
        comma_split_allowed = not (("weather" in tl or "forecast" in tl) and " in " in tl)
        token_hits = len(_FASTPATH_COMMAND_TOKEN_RE.findall(p))

        if comma_split_allowed and token_hits >= 2:
            expanded.extend([s.strip() for s in _FASTPATH_COMMA_SPLIT_RE.split(p) if s and s.strip()])
        else:
            expanded.append(p)

    parts = expanded

    if not parts:
        return None
    if len(parts) > 5:
        return None

    intents: List[Intent] = []
    for part in parts:
        part_intents = _fastpath_parse_clause(part)
        if not part_intents:
            return None
        intents.extend(part_intents)
        if len(intents) > 5:
            return None

    # Final conservative sanity: ensure every referenced tool exists.
    if not intents:
        return None
    if any(not registry.has_tool(i.tool) for i in intents):
        return None
    return intents


def _fastpath_parse_clause(clause: str) -> Optional[List[Intent]]:
    """Parse a single command clause into intents."""
    c_raw = _strip_trailing_punct(clause)
    c = (c_raw or "").strip()
    if not c:
        return None

    c_lower = c.lower()

    # Explicit tool invocation supports ANY tool in the registry.
    explicit = _try_parse_explicit_tool_clause(c, registry=get_registry())
    if explicit:
        return [explicit]

    # --- Info tools ---
    if re.fullmatch(r"(what\s+time\s+is\s+it|what\s+is\s+the\s+time|time)\??", c_lower):
        return [Intent(tool="get_time", args={})]

    if "system info" in c_lower or "system information" in c_lower or re.fullmatch(r"system\s+info", c_lower):
        return [Intent(tool="get_system_info", args={})]

    if "monitor info" in c_lower or "monitors" == c_lower or re.fullmatch(r"monitor\s+info", c_lower):
        return [Intent(tool="monitor_info", args={})]

    if re.fullmatch(r"(where\s+am\s+i|what\s+is\s+my\s+location|my\s+location)\??", c_lower):
        return [Intent(tool="get_location", args={})]

    # Weather / forecast (safe, internet required)
    if "weather" in c_lower or "forecast" in c_lower:
        args: Dict[str, Any] = {}

        loc = _parse_location_after_in(c)
        if loc:
            args["location"] = loc

        days = _extract_days(c)
        if days is not None:
            args["days"] = max(1, min(14, days))
        elif "tomorrow" in c_lower:
            # Include today+tomorrow; tool returns current + daily list.
            args["days"] = 2
        elif "weekly" in c_lower or "week" in c_lower:
            args["days"] = 7

        if any(k in c_lower for k in ["fahrenheit", "imperial", " f ", " f."]):
            args["units"] = "fahrenheit"
        elif any(k in c_lower for k in ["celsius", "metric", " c ", " c."]):
            args["units"] = "celsius"

        return [Intent(tool="get_weather_forecast", args=args)]

    # --- Library refresh ---
    # "scan files", "scan apps" -> tier 3 (full file system + apps scan)
    # "refresh library", "rebuild library" -> normal mode
    if "scan files" in c_lower or "scan apps" in c_lower:
        return [Intent(tool="local_library_refresh", args={"mode": "tier3"})]
    
    if "refresh library" in c_lower or "rebuild library" in c_lower or c_lower == "local library refresh":
        return [Intent(tool="local_library_refresh", args={})]

    # --- Audio output device listing (list/show devices) ---
    # "list audio devices", "show audio devices", "what audio devices", "which devices are available"
    list_devices_patterns = [
        r"\blist\s+(?:audio\s+)?devices?\b",
        r"\bshow\s+(?:audio\s+)?devices?\b",
        r"\bwhat\s+(?:audio\s+)?devices?\b",
        r"\bwhich\s+(?:audio\s+)?devices?\b",
        r"\bavailable\s+(?:audio\s+)?devices?\b",
        r"(?:audio|sound|output)\s+devices?\s+available\b",
        r"\bcurrent\s+(?:audio\s+)?device\b",
        r"\bwhich\s+(?:audio\s+)?device\s+is\s+(?:active|selected|current)\b",
    ]
    if any(re.search(pat, c_lower) for pat in list_devices_patterns):
        return [Intent(tool="set_audio_output_device", args={"action": "list"})]

    # --- Audio output device switching (explicit only) ---
    dev = _parse_audio_device_target(c)
    if dev:
        return [Intent(tool="set_audio_output_device", args={"device": dev})]

    # --- Media / volume ---
    # Prefer true volume control (pycaw) for all volume/mute commands.
    scope, proc = _parse_volume_scope_and_process(c_raw)

    # "turn <target> down to 35%" (target can be master volume keywords or an app name)
    m = re.match(
        r"^turn\s+(?P<target>.+?)\s+(?P<dir>up|down)\s+to\s+(?P<pct>\d{1,3})\s*%?$",
        c_lower,
    )
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        pct = _extract_volume_percent(m.group("pct"))
        if pct is not None:
            target_l = target.lower()
            is_master = target_l in {"volume", "sound", "audio", "the volume", "the sound", "the audio", "master volume"}
            args: Dict[str, Any] = {"scope": "master" if is_master else "app", "action": "set", "level": pct}
            if not is_master:
                args["process"] = target
            return [Intent(tool="volume_control", args=args)]

    # "get volume" / "what is the volume" / per-app "what is spotify volume"
    is_volume_worded = any(k in c_lower for k in ["volume", "sound", "audio"])
    is_query = bool(
        re.search(
            r"\b(?:get|check|show|tell\s+me|what\s+is|what's|whats|current)\b",
            c_lower,
        )
    )
    has_adjust_words = any(
        k in c_lower
        for k in [
            "volume up",
            "volume down",
            "turn up",
            "turn down",
            "louder",
            "quieter",
            "raise volume",
            "lower volume",
            "increase volume",
            "decrease volume",
            "sound up",
            "sound down",
        ]
    )
    if is_volume_worded and is_query and not has_adjust_words and _extract_volume_percent(c_lower) is None:
        args: Dict[str, Any] = {"scope": scope, "action": "get"}
        if scope == "app":
            args["process"] = proc
        return [Intent(tool="volume_control", args=args)]

    # "turn it up" / "turn it down" shorthand
    if re.search(r"\bturn\s+it\s+up\b", c_lower):
        delta = _parse_volume_delta_hint(c_lower)
        return [Intent(tool="volume_control", args={"scope": "master", "action": "change", "delta": int(delta)})]

    if re.search(r"\bturn\s+it\s+down\b", c_lower):
        delta = _parse_volume_delta_hint(c_lower)
        return [Intent(tool="volume_control", args={"scope": "master", "action": "change", "delta": -int(delta)})]

    # Mute/unmute
    if "unmute" in c_lower:
        args: Dict[str, Any] = {"scope": scope, "action": "unmute"}
        if scope == "app":
            args["process"] = proc
        return [Intent(tool="volume_control", args=args)]

    # 'mute' (but not 'unmute')
    if re.search(r"\bmute\b", c_lower) and "unmute" not in c_lower:
        args = {"scope": scope, "action": "mute"}
        if scope == "app":
            args["process"] = proc
        return [Intent(tool="volume_control", args=args)]

    # Explicit set: "volume 35" / "set volume to 35" / "spotify volume 35"
    # Avoid catching "volume down 10" / "turn down" (handled below).
    if "volume" in c_lower or "sound" in c_lower or "audio" in c_lower:
        has_adjust_words = any(
            k in c_lower
            for k in [
                "volume up",
                "volume down",
                "turn up",
                "turn down",
                "louder",
                "quieter",
                "raise",
                "lower",
                "increase",
                "decrease",
                "sound up",
                "sound down",
            ]
        )
        if not has_adjust_words:
            percent = _extract_volume_percent(c_lower)
            if percent is not None:
                # Require a reasonably clear set pattern.
                if re.search(r"\b(set\s+)?(?:the\s+)?(?:master\s+)?(?:volume|sound|audio)\s+(?:to\s+)?\d{1,3}\s*%?\b", c_lower) or re.fullmatch(
                    r"(?:volume|sound|audio)\s+\d{1,3}\s*%?", c_lower
                ):
                    args = {"scope": scope, "action": "set", "level": max(0, min(100, int(percent)))}
                    if scope == "app":
                        args["process"] = proc
                    return [Intent(tool="volume_control", args=args)]

    # Up/down adjustments
    if any(k in c_lower for k in ["volume up", "turn up", "louder", "raise volume", "increase volume", "sound up"]):
        delta = _parse_volume_delta_hint(c_lower)
        n = _extract_int(c_lower)
        if n is not None and 0 <= n <= 100:
            delta = max(1, min(100, int(n)))
        args = {"scope": scope, "action": "change", "delta": int(delta)}
        if scope == "app":
            args["process"] = proc
        return [Intent(tool="volume_control", args=args)]

    if any(k in c_lower for k in ["volume down", "turn down", "quieter", "lower volume", "decrease volume", "sound down"]):
        delta = _parse_volume_delta_hint(c_lower)
        n = _extract_int(c_lower)
        if n is not None and 0 <= n <= 100:
            delta = max(1, min(100, int(n)))
        args = {"scope": scope, "action": "change", "delta": -int(delta)}
        if scope == "app":
            args["process"] = proc
        return [Intent(tool="volume_control", args=args)]

    # Common shorthand: "sound down a bit" / "sound up"
    if re.fullmatch(r"(?:sound|audio)\s+(?:up|down)(?:\s+.*)?", c_lower):
        is_up = " up" in f" {c_lower} "
        delta = _parse_volume_delta_hint(c_lower)
        args = {"scope": scope, "action": "change", "delta": int(delta if is_up else -delta)}
        if scope == "app":
            args["process"] = proc
        return [Intent(tool="volume_control", args=args)]

    if any(k in c_lower for k in ["next track", "next song", "skip", "next"]):
        return [Intent(tool="media_next", args={})]

    if any(k in c_lower for k in ["previous track", "previous song", "prev", "previous", "back track"]):
        return [Intent(tool="media_previous", args={})]

    if any(k in c_lower for k in ["pause", "play", "resume", "play pause", "play/pause"]):
        return [Intent(tool="media_play_pause", args={})]

    # --- Window management ---
    m = re.match(r"^(focus|activate|switch\s+to)\s+(?P<target>.+)$", c_lower)
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        if target:
            return [Intent(tool="focus_window", args={"process": target})]

    m = re.match(r"^minimize\s+(?P<target>.+)$", c_lower)
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        if target:
            return [Intent(tool="minimize_window", args={"process": target})]

    m = re.match(r"^(maximize|fullscreen)\s+(?P<target>.+)$", c_lower)
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        if target:
            return [Intent(tool="maximize_window", args={"process": target})]

    m = re.match(r"^(close|exit|quit)\s+(?P<target>.+)$", c_lower)
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        if target:
            # Force close is intentionally not supported in fast-path.
            return [Intent(tool="close_window", args={"process": target, "force": False})]

    # Ordinal word to number mapping for monitor references
    _ORDINAL_MAP = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2, "secondary": 2, "other": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5,
        "sixth": 6, "6th": 6,
    }
    _ORDINAL_PATTERN = "|".join(_ORDINAL_MAP.keys())

    m = re.match(
        rf"^move\s+(?P<target>.+?)\s+to\s+(?:(?:the\s+)?(?P<primary>primary|main)\s+monitor|(?:the\s+)?(?P<ordinal>{_ORDINAL_PATTERN})\s+monitor|monitor\s+(?P<mon>\d+)|(?P<mon2>\d+))(?P<rest>.*)$",
        c_lower,
    )
    if m:
        target = _strip_trailing_punct(m.group("target")).strip()
        if not target:
            return None

        if m.group("primary"):
            monitor: Any = "primary"
        elif m.group("ordinal"):
            monitor = _ORDINAL_MAP.get(m.group("ordinal").lower(), 1)
        else:
            mon_s = m.group("mon") or m.group("mon2")
            if not mon_s:
                return None
            try:
                monitor = int(mon_s)
            except Exception:
                return None

        rest = (m.group("rest") or "").strip()
        position = "maximize"
        if " left" in f" {rest} " or rest.startswith("left"):
            position = "left"
        elif " right" in f" {rest} " or rest.startswith("right"):
            position = "right"
        elif " center" in f" {rest} " or rest.startswith("center"):
            position = "center"
        elif any(k in rest for k in ["maximize", "full", "fullscreen"]):
            position = "maximize"

        return [
            Intent(
                tool="move_window_to_monitor",
                args={"process": target, "monitor": monitor, "position": position},
            )
        ]

    # --- Open / launch ---
    m = re.match(r"^(open|launch|start)\s+(?P<q>.+)$", c_lower)
    if m:
        query = _strip_trailing_punct(m.group("q")).strip()
        if not query:
            return None

        # If the user explicitly indicates web intent OR query looks like a URL/domain, open as website.
        if _user_explicitly_requested_website(c_raw) or _looks_like_url(query) or query.lower() in _FASTPATH_COMMON_WEBSITES:
            return [Intent(tool="open_website", args={"url": query})]

        # Default: open locally (apps/games/files/folders) via local library resolution.
        return [Intent(tool="open_target", args={"query": query})]

    # "go to X" is usually web.
    m = re.match(r"^(go\s+to)\s+(?P<q>.+)$", c_lower)
    if m:
        query = _strip_trailing_punct(m.group("q")).strip()
        if query:
            return [Intent(tool="open_website", args={"url": query})]

    return None


def _format_fastpath_reply(user_text: str, intents: List[Intent], execution_summary: ExecutionSummary) -> str:
    # =========================================================================
    # PHASE 10: Handle clarification_needed errors from reference resolution
    # =========================================================================
    if execution_summary.ran:
        first_result = execution_summary.ran[0]
        if first_result.tool == "__reference_resolution__" and first_result.error:
            error = first_result.error
            if isinstance(error, dict) and error.get("type") == "clarification_needed":
                return error.get("message", "Which one do you mean?")
    
    def format_info(tool: str, args: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        if tool == "get_time":
            t = result.get("time")
            return f"It is {t}." if t else "Here's the current time."

        if tool == "get_system_info":
            os_name = result.get("os")
            arch = result.get("architecture")
            cores = result.get("cpu_cores")
            if os_name and arch and isinstance(cores, int):
                return f"{os_name} ({arch}), {cores} CPU cores."
            return "Here's your system information."

        if tool == "monitor_info":
            count = result.get("count")
            if isinstance(count, int):
                return f"You have {count} monitor(s)."
            return "Here's your monitor information."

        if tool == "get_window_monitor":
            monitor = result.get("monitor") or {}
            matched = result.get("matched_window") or {}
            process = matched.get("process", "").replace(".exe", "")
            title = matched.get("title", "")
            mon_num = monitor.get("number")
            is_primary = monitor.get("primary", False)
            
            app_name = process.capitalize() if process else (title or "That window")
            
            if mon_num is not None:
                primary_str = " (your primary monitor)" if is_primary else ""
                return f"{app_name} is on monitor {mon_num}{primary_str}."
            elif monitor.get("error"):
                return f"I found {app_name} but couldn't determine which monitor it's on."
            return "Here's the window monitor information."

        # Phase 9: Screen Awareness - get_window_context
        if tool == "get_window_context":
            app = result.get("app", "").replace(".exe", "") if result.get("app") else None
            title = result.get("title")
            
            # Format a natural response
            if app and title:
                app_name = app.capitalize()
                # Truncate title if too long for speech
                if len(title) > 60:
                    title = title[:57] + "..."
                return f"You're looking at {app_name}. The window title is: {title}."
            elif app:
                app_name = app.capitalize()
                return f"You're looking at {app_name}."
            elif title:
                if len(title) > 60:
                    title = title[:57] + "..."
                return f"The active window is: {title}."
            return "I couldn't detect the active window."

        if tool == "get_location":
            city = result.get("city")
            region = result.get("region")
            country = result.get("country")
            parts = [p for p in [city, region, country] if isinstance(p, str) and p.strip()]
            if parts:
                return "You're in " + ", ".join(parts) + "."
            return "Here's your approximate location."

        if tool == "get_weather_forecast":
            loc = (result.get("location") or {}).get("name") if isinstance(result.get("location"), dict) else None
            
            # Check if a specific day was requested via day_offset in args
            day_offset = (args.get("day_offset") or 0) if isinstance(args, dict) else 0
            
            if day_offset > 0:
                # User asked about a future day (e.g., "tomorrow", "Thursday")
                forecast_daily = result.get("forecast_daily") or []
                if isinstance(forecast_daily, list) and len(forecast_daily) > day_offset:
                    day_data = forecast_daily[day_offset]
                    weather = day_data.get("weather")
                    temp_max = day_data.get("temp_max")
                    temp_min = day_data.get("temp_min")
                    
                    # Generate a friendly day label
                    if day_offset == 1:
                        day_label = "Tomorrow"
                    else:
                        # Try to get weekday name from the date
                        date_str = day_data.get("date", "")
                        try:
                            import datetime
                            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                            day_label = dt.strftime("%A")  # e.g., "Thursday"
                        except Exception:
                            day_label = date_str if date_str else "That day"
                    
                    if loc and weather:
                        if temp_max is not None and temp_min is not None:
                            return f"{day_label}: {weather}, high of {temp_max}°, low of {temp_min}° in {loc}."
                        return f"{day_label}: {weather} in {loc}."
                    if weather:
                        if temp_max is not None and temp_min is not None:
                            return f"{day_label}: {weather}, high of {temp_max}°, low of {temp_min}°."
                        return f"{day_label}: {weather}."
                # Fallback if day_offset data not available
            
            # Default: use current weather
            current = result.get("current") if isinstance(result.get("current"), dict) else {}
            temp = current.get("temperature")
            weather = current.get("weather")
            if loc and (temp is not None or weather):
                if temp is not None and weather:
                    return f"{weather}, {temp}° in {loc}."
                if temp is not None:
                    return f"It's {temp}° in {loc}."
                return f"{weather} in {loc}."
            return "Here's the forecast."

        if tool == "system_storage_scan":
            drives = result.get("drives", [])
            if drives:
                summaries = []
                for drive in drives:
                    name = drive.get("name", "").rstrip(":\\").strip()  # Remove backslash/colon
                    free_gb = drive.get("free_gb", 0)
                    total_gb = drive.get("total_gb", 0)
                    if name and free_gb is not None and total_gb:
                        # Round with 0.5 threshold (0.5+ rounds up)
                        free_gb_rounded = int(free_gb + 0.5)
                        total_gb_rounded = int(total_gb + 0.5)
                        summaries.append(f"{name} has {free_gb_rounded} gigs free of {total_gb_rounded} gigs total")
                if summaries:
                    return "Storage scan complete. " + ", ".join(summaries) + "."
            return "Storage scan complete."

        if tool == "system_storage_list":
            drives = result.get("drives", [])
            if drives:
                summaries = []
                for drive in drives:
                    name = drive.get("name", "").rstrip(":\\").strip()  # Remove backslash/colon
                    free_gb = drive.get("free_gb", 0)
                    if name and free_gb is not None:
                        # Round with 0.5 threshold (0.5+ rounds up)
                        free_gb_rounded = int(free_gb + 0.5)
                        summaries.append(f"{name} has {free_gb_rounded} gigs free")
                if summaries:
                    return "You have " + ", ".join(summaries) + "."
            return "Storage information retrieved."

        if tool == "system_storage_open":
            opened = result.get("opened")
            if opened:
                # Remove backslash/colon for cleaner speech output
                opened_clean = opened.rstrip(":\\").strip()
                return f"Opened {opened_clean}."
            return "Drive opened."

        if tool == "get_now_playing":
            status = result.get("status")
            if status == "no_media":
                return "Nothing is currently playing."
            
            title = result.get("title")
            artist = result.get("artist")
            playback_status = result.get("playback_status", "playing")
            
            # Build response - keep it short (title and artist only)
            if title and artist:
                response = f"{title} by {artist}"
                if playback_status == "paused":
                    response = f"Paused: {response}"
                return response + "."
            elif title:
                if playback_status == "paused":
                    return f"Paused: {title}."
                return f"Playing {title}."
            return "Media is playing but I couldn't get the details."

        # ═══════════════════════════════════════════════════════════════════
        # Timer tool: format timer responses for speech
        # ═══════════════════════════════════════════════════════════════════
        if tool == "timer":
            status = result.get("status")
            action = str(args.get("action") or "").lower()
            
            if status == "finished":
                return "Your timer is finished."
            
            if action == "start" and status == "running":
                duration = result.get("duration", 0)
                if duration >= 3600:
                    hours = duration // 3600
                    mins = (duration % 3600) // 60
                    if mins > 0:
                        return f"Timer set for {hours} hour{'s' if hours != 1 else ''} and {mins} minute{'s' if mins != 1 else ''}."
                    return f"Timer set for {hours} hour{'s' if hours != 1 else ''}."
                elif duration >= 60:
                    mins = duration // 60
                    secs = duration % 60
                    if secs > 0:
                        return f"Timer set for {mins} minute{'s' if mins != 1 else ''} and {secs} second{'s' if secs != 1 else ''}."
                    return f"Timer set for {mins} minute{'s' if mins != 1 else ''}."
                else:
                    return f"Timer set for {duration} second{'s' if duration != 1 else ''}."
            
            if action == "cancel" and status == "cancelled":
                return "Timer cancelled."
            
            if action == "status":
                if status == "idle":
                    return "No timer is running."
                if status == "running":
                    remaining = result.get("remaining_seconds", 0)
                    if remaining >= 3600:
                        hours = remaining // 3600
                        mins = (remaining % 3600) // 60
                        if mins > 0:
                            return f"About {hours} hour{'s' if hours != 1 else ''} and {mins} minute{'s' if mins != 1 else ''} remaining."
                        return f"About {hours} hour{'s' if hours != 1 else ''} remaining."
                    elif remaining >= 60:
                        mins = remaining // 60
                        secs = remaining % 60
                        if secs > 0:
                            return f"About {mins} minute{'s' if mins != 1 else ''} and {secs} second{'s' if secs != 1 else ''} remaining."
                        return f"About {mins} minute{'s' if mins != 1 else ''} remaining."
                    else:
                        return f"About {remaining} second{'s' if remaining != 1 else ''} remaining."
            
            return "Timer updated."

        return None

    # Prefer first failure.
    for idx, r in enumerate(execution_summary.ran):
        if not r.ok:
            # Get args for this intent if available
            intent_args = intents[idx].args if idx < len(intents) else {}
            
            # Special case for open_target - keep existing behavior
            if r.tool == "open_target":
                query = (intent_args or {}).get("query", "")
                if query:
                    return f"Could not find {query}."
                return "Could not find that target."
            
            # Use the error mapper for user-friendly messages
            if r.error:
                return _tool_error_to_speech(r.error, r.tool, intent_args)
            return "I couldn't complete that. Please try again."

    if not execution_summary.ran:
        return "Done."

    # Single-intent.
    if len(intents) == 1 and len(execution_summary.ran) == 1:
        tool = intents[0].tool
        args = intents[0].args or {}
        result = execution_summary.ran[0].result or {}

        info = format_info(tool, args, result)
        if info:
            return info

        if tool == "open_target":
            q = args.get("query")
            return f"Opening {q}." if q else "Opening."

        if tool == "open_website":
            url = args.get("url")
            return f"Opening {url}." if url else "Opening."

        if tool == "set_audio_output_device":
            action = str(args.get("action") or "").lower()
            if action == "list":
                devices = result.get("devices") if isinstance(result, dict) else []
                current = result.get("current_device") if isinstance(result, dict) else None
                if not devices:
                    return "No audio devices found."
                current_name = (current or {}).get("name") if isinstance(current, dict) else None
                device_list = ", ".join(str(d.get("name", "Unknown")) for d in (devices or []) if isinstance(d, dict))
                if current_name:
                    return f"Available devices: {device_list}. Current: {current_name}."
                return f"Available devices: {device_list}."
            
            # "set" action
            chosen = result.get("chosen") if isinstance(result, dict) else None
            chosen_name = (chosen or {}).get("name") if isinstance(chosen, dict) else None
            if isinstance(chosen_name, str) and chosen_name.strip():
                return f"Audio output set to {chosen_name}."
            requested = args.get("device")
            return f"Switching audio output to {requested}." if requested else "Switching audio output."

        if tool in {"media_play_pause", "media_next", "media_previous", "volume_up", "volume_down", "volume_mute_toggle"}:
            return "OK."

        if tool == "volume_control":
            action = str(args.get("action") or "").lower()
            scope = str(args.get("scope") or "master").lower()
            requested_proc = args.get("process")

            display = result.get("display") if isinstance(result, dict) else None
            proc_res = result.get("process") if isinstance(result, dict) else None
            target = "volume" if scope != "app" else (display or proc_res or requested_proc or "app volume")

            level = result.get("level") if isinstance(result, dict) else None
            new_level = result.get("new_level") if isinstance(result, dict) else None
            muted = result.get("muted") if isinstance(result, dict) else None

            if action == "get":
                if isinstance(level, int):
                    return f"{target} is {level}%."
                return "OK."

            if action == "set":
                if isinstance(new_level, int):
                    return f"Set {target} to {new_level}%."
                if isinstance(args.get("level"), int):
                    return f"Set {target} to {args.get('level')}%."
                return "OK."

            if action == "change":
                if isinstance(new_level, int):
                    return f"Set {target} to {new_level}%."
                return "OK."

            if action in {"mute", "unmute", "toggle_mute"}:
                if isinstance(muted, bool):
                    if scope == "app":
                        return f"{target} {'muted' if muted else 'unmuted'}."
                    return "Muted." if muted else "Unmuted."
                return "OK."

        return "Done."

    # Multi-intent: summarize key actions + include the last info response (if any).
    opened: List[str] = []
    closed: List[str] = []
    audio_switched: Optional[str] = None
    audio_list: Optional[str] = None
    info_sentence: Optional[str] = None
    volume_fragments: List[str] = []
    media_fragments: List[str] = []

    for idx, intent in enumerate(intents[: len(execution_summary.ran)]):
        res = execution_summary.ran[idx].result or {}
        args = intent.args or {}

        info = format_info(intent.tool, args, res)
        if info:
            info_sentence = info
            continue

        if intent.tool == "open_target":
            q = args.get("query")
            if isinstance(q, str) and q.strip():
                opened.append(q.strip())
            continue

        if intent.tool == "open_website":
            url = args.get("url")
            if isinstance(url, str) and url.strip():
                opened.append(url.strip())
            continue

        if intent.tool == "close_window":
            title = args.get("title")
            if isinstance(title, str) and title.strip():
                closed.append(title.strip())
            continue

        # Media controls
        if intent.tool == "media_play_pause":
            media_fragments.append("Toggled play/pause")
            continue

        if intent.tool == "media_next":
            media_fragments.append("Skipped to next")
            continue

        if intent.tool == "media_previous":
            media_fragments.append("Skipped to previous")
            continue

        if intent.tool == "set_audio_output_device":
            action = str(args.get("action") or "").lower()
            if action == "list":
                devices = res.get("devices") if isinstance(res, dict) else []
                if devices and isinstance(devices, list):
                    device_names = [str(d.get("name", "")) for d in devices if isinstance(d, dict) and d.get("name")]
                    if device_names:
                        audio_list = ", ".join(device_names)
            else:
                # "set" action
                chosen = res.get("chosen") if isinstance(res, dict) else None
                chosen_name = (chosen or {}).get("name") if isinstance(chosen, dict) else None
                if isinstance(chosen_name, str) and chosen_name.strip():
                    audio_switched = chosen_name.strip()
                else:
                    requested = args.get("device")
                    if isinstance(requested, str) and requested.strip():
                        audio_switched = requested.strip()

        if intent.tool == "volume_control":
            action = str(args.get("action") or "").lower()
            scope = str(args.get("scope") or "master").lower()
            requested_proc = args.get("process")

            display = res.get("display") if isinstance(res, dict) else None
            proc_res = res.get("process") if isinstance(res, dict) else None
            target = "volume" if scope != "app" else (display or proc_res or requested_proc or "app volume")

            if action in {"set", "change"}:
                lvl = res.get("new_level") if isinstance(res, dict) else None
                if not isinstance(lvl, int):
                    lvl = args.get("level") if isinstance(args.get("level"), int) else lvl
                if isinstance(lvl, int):
                    volume_fragments.append(f"set {target} to {lvl}%")
                continue

            if action in {"mute", "unmute", "toggle_mute"}:
                muted = res.get("muted") if isinstance(res, dict) else None
                if isinstance(muted, bool):
                    volume_fragments.append(f"{'muted' if muted else 'unmuted'} {target}")
                continue

            if action == "get":
                lvl = res.get("level") if isinstance(res, dict) else None
                if isinstance(lvl, int):
                    volume_fragments.append(f"{target} is {lvl}%")
                continue

    action_fragments: List[str] = []
    if opened:
        unique_opened: List[str] = []
        seen: set[str] = set()
        for o in opened:
            key = o.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_opened.append(o)

        if len(unique_opened) == 1:
            action_fragments.append(f"Opened {unique_opened[0]}")
        elif len(unique_opened) == 2:
            action_fragments.append(f"Opened {unique_opened[0]} and {unique_opened[1]}")
        else:
            action_fragments.append("Opened " + ", ".join(unique_opened[:-1]) + f", and {unique_opened[-1]}")

    if closed:
        unique_closed: List[str] = []
        seen: set[str] = set()
        for c in closed:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_closed.append(c)

        if len(unique_closed) == 1:
            action_fragments.append(f"Closed {unique_closed[0]}")
        elif len(unique_closed) == 2:
            action_fragments.append(f"Closed {unique_closed[0]} and {unique_closed[1]}")
        else:
            action_fragments.append("Closed " + ", ".join(unique_closed[:-1]) + f", and {unique_closed[-1]}")

    if media_fragments:
        action_fragments.extend(media_fragments)

    if audio_switched:
        action_fragments.append(f"Set audio output to {audio_switched}")

    if audio_list:
        action_fragments.append(f"Available devices: {audio_list}")

    if volume_fragments:
        # Keep the last 2 volume-related statements to stay concise.
        for frag in volume_fragments[-2:]:
            # Capitalize first letter
            capitalized = frag[0].upper() + frag[1:] if frag else frag
            action_fragments.append(capitalized)

    action_sentence: Optional[str] = None
    if action_fragments:
        action_sentence = "; ".join(action_fragments) + "."

    if action_sentence and info_sentence:
        return f"{action_sentence} {info_sentence}"
    if info_sentence:
        return info_sentence
    if action_sentence:
        return action_sentence
    return "Done."


def _user_explicitly_requested_website(text: str) -> bool:
    t = (text or "").lower()
    # Keep this conservative to avoid breaking "open youtube" patterns.
    # Only treat as explicit web intent when the user says so.
    explicit_markers = [
        "website",
        "site",
        "web site",
        "web",
        "url",
        "link",
        "http://",
        "https://",
        "www.",
        ".com",
        ".net",
        ".org",
        ".io",
        "dot com",
    ]
    return any(m in t for m in explicit_markers)


def _extract_host_base(url: str) -> Optional[str]:
    if not url:
        return None

    raw = url.strip()
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or "").split(":")[0].strip().lower()
    if not host:
        return None

    # Drop common prefixes
    if host.startswith("www."):
        host = host[4:]

    # Use the left-most label as base (rocketleague.com -> rocketleague)
    base = host.split(".")[0]
    base_norm = _normalize_alnum(base)
    return base_norm or None


def _find_matching_phrase_in_text(text: str, target_norm: str, max_ngram: int = 6) -> Optional[str]:
    """Find a phrase in the user text whose normalized form matches target_norm."""
    if not text or not target_norm:
        return None

    # Tokenize while preserving original tokens for reconstruction
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens:
        return None

    # Try longer n-grams first (more specific)
    for n in range(min(max_ngram, len(tokens)), 0, -1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i + n])
            phrase_norm = _normalize_alnum(phrase)
            if phrase_norm == target_norm:
                return phrase

    return None


def _rewrite_open_website_intents(user_text: str, intents) -> None:
    """Rewrite certain open_website intents to open_target when they likely refer to an installed app/game."""
    if not intents:
        return

    if _user_explicitly_requested_website(user_text):
        return

    for intent in intents:
        if intent.tool != "open_website":
            continue

        url = (intent.args or {}).get("url", "")
        base_norm = _extract_host_base(url)
        if not base_norm:
            continue

        phrase = _find_matching_phrase_in_text(user_text, base_norm)
        if not phrase:
            continue

        try:
            resolved = resolve_target(phrase)
        except Exception:
            continue

        r_type = resolved.get("type")
        confidence = float(resolved.get("confidence", 0) or 0)

        # Only rewrite when we're pretty confident it's a local thing.
        if r_type in {"game", "app", "uwp", "folder", "file"} and confidence >= 0.6:
            intent.tool = "open_target"
            intent.args = {"query": phrase}


def _update_world_state_from_result(tool_name: str, tool_args: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Update WorldState after successful tool execution.
    
    Phase 10: Called after ANY tool successfully runs to enable reference resolution.
    Fails silently if update cannot be performed.
    """
    try:
        # Only update on successful execution (no error in result)
        if "error" in result:
            return
        
        from wyzer.context.world_state import update_from_tool_execution
        update_from_tool_execution(tool_name, tool_args, result)
    except Exception:
        # Fail silently - world state updates are best-effort
        pass


def _handle_replay_last_action(original_text: str, start_time: float) -> Dict[str, Any]:
    """
    Handle deterministic replay of the last successful action.
    
    Phase 10.1: This is the core replay mechanism. When user says "do that again",
    "repeat that", etc., this function replays the exact last tool execution
    without re-routing through the LLM.
    
    Args:
        original_text: The original user text (for logging/meta)
        start_time: Start time for latency calculation
        
    Returns:
        Dict with reply, latency_ms, and execution_summary
    """
    logger = get_logger_instance()
    
    from wyzer.context.world_state import get_world_state
    ws = get_world_state()
    
    # Check if there's a last action to replay
    if not ws.has_replay_action() or ws.last_action is None:
        logger.info("[REPLAY] no last_action; asking user")
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        return {
            "reply": "What should I repeat?",
            "latency_ms": latency_ms,
            "meta": {"replay": True, "no_last_action": True, "original_text": original_text},
        }
    
    last_action = ws.last_action
    tool_name = last_action.tool
    original_args = last_action.args
    resolved_info = last_action.resolved
    
    # Build replay args - prefer resolved info for stable replays
    replay_args = _build_replay_args(tool_name, original_args, resolved_info)
    
    # Log the replay
    resolved_summary = _summarize_resolved_for_log(resolved_info)
    logger.info(f"[REPLAY] tool={tool_name} args={replay_args} resolved={resolved_summary}")
    
    # Execute the replay
    try:
        registry = get_registry()
        result = _execute_tool(registry, tool_name, replay_args)
        
        # Determine success/failure
        has_error = "error" in result
        
        # Format reply
        if has_error:
            error_msg = result.get("error", {}).get("message", "Unknown error")
            reply = f"I couldn't repeat that: {error_msg}"
        else:
            reply = _format_replay_success_reply(tool_name, replay_args, result)
        
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        return {
            "reply": reply,
            "latency_ms": latency_ms,
            "execution_summary": {
                "replayed_tool": tool_name,
                "replayed_args": replay_args,
                "success": not has_error,
                "result": result,
            },
            "meta": {"replay": True, "original_text": original_text},
        }
        
    except Exception as e:
        logger.error(f"[REPLAY] Exception during replay: {e}")
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        return {
            "reply": "I couldn't repeat that action.",
            "latency_ms": latency_ms,
            "meta": {"replay": True, "error": str(e), "original_text": original_text},
        }


def _build_replay_args(tool_name: str, original_args: dict, resolved_info: Optional[dict]) -> dict:
    """
    Build the arguments for replaying a tool.
    
    Phase 10.1: For stable replays, we prefer using resolved info when available.
    This avoids re-searching/re-resolving and ensures deterministic behavior.
    
    Special cases:
    - open_target with UWP: Use the resolved UWP app ID path for direct launch
    - open_target with game: Use the resolved launch target
    - close_window/focus_window: Use resolved process/title
    
    Args:
        tool_name: The tool to replay
        original_args: Original args used in the first execution
        resolved_info: Resolved/canonical info from the result
        
    Returns:
        Args dict to use for replay
    """
    # Start with original args as fallback
    replay_args = dict(original_args) if original_args else {}
    
    if not resolved_info:
        return replay_args
    
    # Special handling for open_target - use stable launch target
    if tool_name == "open_target":
        resolved_type = resolved_info.get("type")
        
        if resolved_type == "uwp":
            # UWP apps: use the app ID path for stable launch
            uwp_path = resolved_info.get("path")
            if uwp_path:
                # Keep the query but add internal resolved hint
                replay_args["_resolved_uwp_path"] = uwp_path
        
        elif resolved_type == "game":
            # Games: store launch info for stable replay
            launch = resolved_info.get("launch")
            if launch and isinstance(launch, dict):
                replay_args["_resolved_launch"] = launch
        
        elif resolved_type in ("app", "file", "folder"):
            # For regular apps/files/folders, use path if available
            resolved_path = resolved_info.get("path")
            if resolved_path:
                replay_args["_resolved_path"] = resolved_path
    
    # Special handling for window tools - use stable process/title
    elif tool_name in ("close_window", "focus_window", "minimize_window", "maximize_window"):
        # Prefer process over title for stability
        process = resolved_info.get("process")
        if process:
            replay_args["process"] = process
            # Remove title if we have process (more stable)
            if "title" in replay_args and replay_args.get("title"):
                pass  # Keep both for redundancy
        else:
            # Fall back to title
            title = resolved_info.get("title")
            if title:
                replay_args["title"] = title
    
    return replay_args


def _summarize_resolved_for_log(resolved: Optional[dict]) -> str:
    """Create a short summary of resolved info for logging."""
    if not resolved:
        return "None"
    
    parts = []
    if resolved.get("type"):
        parts.append(f"type={resolved['type']}")
    if resolved.get("matched_name"):
        parts.append(f"name={resolved['matched_name']}")
    elif resolved.get("app_name"):
        parts.append(f"name={resolved['app_name']}")
    elif resolved.get("game_name"):
        parts.append(f"name={resolved['game_name']}")
    elif resolved.get("process"):
        parts.append(f"process={resolved['process']}")
    elif resolved.get("title"):
        title = resolved['title'][:30] + "..." if len(resolved.get('title', '')) > 30 else resolved.get('title', '')
        parts.append(f"title={title}")
    
    return " ".join(parts) if parts else str(resolved)[:50]


def _format_replay_success_reply(tool_name: str, args: dict, result: dict) -> str:
    """Format a success reply for a replayed action."""
    # Get a friendly target name
    target = None
    
    # Check result for target info
    resolved = result.get("resolved", {})
    matched = result.get("matched", {})
    
    if resolved.get("matched_name"):
        target = resolved["matched_name"]
    elif result.get("app_name"):
        target = result["app_name"]
    elif result.get("game_name"):
        target = result["game_name"]
    elif matched.get("process"):
        target = matched["process"]
        if target and target.lower().endswith(".exe"):
            target = target[:-4]
    elif args.get("query"):
        target = args["query"]
    elif args.get("process"):
        target = args["process"]
    elif args.get("title"):
        target = args["title"]
    
    # Format based on tool
    if tool_name == "open_target":
        if target:
            return f"Opened {target} again."
        return "Done."
    elif tool_name == "close_window":
        if target:
            return f"Closed {target} again."
        return "Closed the window again."
    elif tool_name == "focus_window":
        if target:
            return f"Focused {target} again."
        return "Focused the window again."
    elif tool_name == "minimize_window":
        if target:
            return f"Minimized {target} again."
        return "Minimized the window again."
    elif tool_name == "maximize_window":
        if target:
            return f"Maximized {target} again."
        return "Maximized the window again."
    else:
        return "Done."


def _execute_tool(registry, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool with validation and logging (pool-aware)"""
    logger = get_logger_instance()
    
    # Check tool exists
    tool = registry.get(tool_name)
    if tool is None:
        return {
            "error": {
                "type": "tool_not_found",
                "message": f"Tool '{tool_name}' does not exist"
            }
        }
    
    # Normalize arguments
    tool_args = _normalize_tool_args(tool_name, tool_args or {})
    
    # Phase 10.1: Separate internal replay keys (prefixed with _) from regular args
    # Internal keys are used for deterministic replay but should not be validated
    # against the tool's public schema.
    internal_args = {k: v for k, v in tool_args.items() if k.startswith("_")}
    public_args = {k: v for k, v in tool_args.items() if not k.startswith("_")}
    
    # Validate only public arguments against schema
    is_valid, error = validate_args(tool.args_schema, public_args)
    if not is_valid:
        return {"error": error}
    
    # Merge internal args back for the tool to use
    # (tools like open_target check for _resolved_* keys)
    full_args = {**public_args, **internal_args}
    
    # Log BEFORE execution
    logger.info(f"[TOOLS] Executing {tool_name} args={full_args}")
    
    # Try to use worker pool if enabled
    pool = _tool_pool
    if pool is not None:
        try:
            job_id = str(uuid.uuid4())
            request_id = str(uuid.uuid4())
            
            # Submit job to pool
            if pool.submit_job(job_id, request_id, tool_name, full_args):
                # Wait for result with timeout
                result_obj = pool.wait_for_result(job_id, timeout=Config.TOOL_POOL_TIMEOUT_SEC)
                if result_obj is not None:
                    result = result_obj.result
                    logger.info(f"[TOOLS] Pool result {result}")
                    # Phase 10: Update world state for reference resolution
                    _update_world_state_from_result(tool_name, full_args, result)
                    return result
                else:
                    # Timeout, fall back to in-process
                    logger.warning(f"[POOL] Timeout waiting for {tool_name}, falling back to in-process")
            else:
                # Pool submission failed, fall back to in-process
                logger.warning(f"[POOL] Failed to submit {tool_name} to pool, falling back to in-process")
        
        except Exception as e:
            logger.warning(f"[POOL] Error using pool for {tool_name}: {e}, falling back to in-process")
    
    # Fall back to in-process execution
    try:
        result = tool.run(**full_args)
        
        # Log AFTER execution
        logger.info(f"[TOOLS] Result {result}")
        
        # Phase 10: Update world state for reference resolution
        _update_world_state_from_result(tool_name, full_args, result)
        
        return result
    except Exception as e:
        error_result = {
            "error": {
                "type": "execution_error",
                "message": str(e)
            }
        }
        logger.info(f"[TOOLS] Result {error_result}")
        return error_result


def _call_llm(user_text: str, registry) -> Dict[str, Any]:
    """
    Call LLM for initial intent interpretation.
    
    Uses PromptBuilder for token-budgeted prompt construction.
    
    Returns:
        Dict with either {"reply": "..."} or {"intents": [...]} or legacy formats
    """
    from wyzer.brain.prompt_builder import build_llm_prompt
    
    # Gather context blocks
    session_context = ""
    try:
        from wyzer.brain.prompt import get_session_context_block
        session_context = get_session_context_block()
    except Exception:
        pass
    
    promoted_context = ""
    try:
        from wyzer.brain.prompt import get_promoted_memory_block
        promoted_context = get_promoted_memory_block()
    except Exception:
        pass
    
    redaction_context = ""
    try:
        from wyzer.brain.prompt import get_redaction_block
        redaction_context = get_redaction_block()
    except Exception:
        pass
    
    # Get smart memories (relevance-gated inside PromptBuilder)
    memories_context = ""
    try:
        from wyzer.brain.prompt import get_smart_memories_block
        memories_context = get_smart_memories_block(user_text)
    except Exception:
        pass
    
    # Phase 9: Get visual context (screen awareness, read-only)
    visual_context = ""
    try:
        from wyzer.vision.window_context import get_visual_context_block
        visual_context = get_visual_context_block()
    except Exception:
        pass
    
    # Build token-budgeted prompt
    prompt, mode = build_llm_prompt(
        user_text=user_text,
        session_context=session_context,
        promoted_context=promoted_context,
        redaction_context=redaction_context,
        memories_context=memories_context,
        visual_context=visual_context,
    )
    
    return _ollama_request(prompt)


def _call_llm_for_explicit_tool(user_text: str, tool_name: str, registry) -> Dict[str, Any]:
    """Ask the LLM to produce arguments for a specific explicitly-requested tool.

    This is used when the user says: "tool <name> ..." but the deterministic parser
    cannot safely parse the args.
    """
    # Respect config: if LLM is off or NO_OLLAMA is set, we can't do the hybrid step.
    if getattr(Config, "NO_OLLAMA", False) or str(getattr(Config, "LLM_MODE", "ollama") or "ollama").strip().lower() == "off":
        return {
            "reply": _get_no_ollama_reply()
        }

    tool = registry.get(tool_name)
    if tool is None:
        return {"reply": f"Unknown tool '{tool_name}'."}

    tool_desc = getattr(tool, "description", "")
    schema = getattr(tool, "args_schema", {})

    prompt = f"""You are Wyzer, a local voice assistant.

The user is making an EXPLICIT tool call and has already chosen the tool.
Your only job is to produce VALID arguments for that exact tool.

Tool name: {tool_name}
Tool description: {tool_desc}
Tool args schema (JSON): {json.dumps(schema)}

Rules:
- You MUST NOT choose a different tool.
- Output JSON only (no markdown).
- If you can infer valid args confidently, return:
  {{"intents": [{{"tool": "{tool_name}", "args": {{...}}}}], "reply": ""}}
- If required information is missing/ambiguous, do NOT guess; instead ask ONE clarifying question and return:
  {{"reply": "your question"}}
- Do not include unknown fields (schema has additionalProperties=false in many tools).

User: {user_text}

Your response (JSON only):"""

    return _ollama_request(prompt)


def _gather_context_blocks(user_text: str, max_session_turns: int = 2) -> Dict[str, str]:
    """
    Gather context blocks for LLM prompts with size limits.
    
    Args:
        user_text: User input (for memory relevance check)
        max_session_turns: Max session turns to include
        
    Returns:
        Dict with session_context, promoted_context, redaction_context, memories_context
    """
    from wyzer.brain.prompt_builder import should_inject_memories
    
    session_context = ""
    try:
        from wyzer.memory.memory_manager import get_memory_manager
        mem_mgr = get_memory_manager()
        context = mem_mgr.get_session_context(max_turns=max_session_turns)
        if context:
            session_context = f"\n--- Recent ---\n{context}\n---\n"
    except Exception:
        pass
    
    promoted_context = ""
    try:
        from wyzer.brain.prompt import get_promoted_memory_block
        promoted_context = get_promoted_memory_block()
        if promoted_context:
            promoted_context = promoted_context[:300]  # Cap size
    except Exception:
        pass
    
    redaction_context = ""
    try:
        from wyzer.brain.prompt import get_redaction_block
        redaction_context = get_redaction_block()
        if redaction_context:
            redaction_context = redaction_context[:200]  # Cap size
    except Exception:
        pass
    
    memories_context = ""
    # Only inject memories for relevant queries
    if should_inject_memories(user_text):
        try:
            from wyzer.brain.prompt import get_smart_memories_block
            memories_context = get_smart_memories_block(user_text)
            if memories_context:
                memories_context = memories_context[:400]  # Cap size
        except Exception:
            pass
    
    return {
        "session": session_context,
        "promoted": promoted_context,
        "redaction": redaction_context,
        "memories": memories_context,
    }


def _call_llm_reply_only(user_text: str) -> Dict[str, Any]:
    """Force a direct reply with no tool calls.

    Used as a fallback when the LLM returns spurious tool intents,
    and for creative content requests (stories, poems, etc.).
    
    When voice_fast is active and identity/smalltalk query detected,
    uses fast-lane prompt for minimal token count.
    """
    logger = get_logger_instance()
    llm_mode = getattr(Config, "LLM_MODE", "ollama")
    
    # Check if we should use fast-lane prompt (llamacpp + identity/smalltalk)
    use_fastlane = False
    fastlane_reason = ""
    if llm_mode == "llamacpp":
        try:
            from wyzer.brain.llm_engine import get_voice_fast_options
            voice_opts = get_voice_fast_options(user_text, llm_mode)
            use_fastlane = voice_opts.get("_use_fastlane_prompt", False)
            fastlane_reason = voice_opts.get("_fastlane_reason", "unknown")
        except Exception:
            pass
    
    if use_fastlane:
        # Use fast-lane prompt for minimal tokens
        try:
            from wyzer.brain.prompt_builder import build_fastlane_prompt
            from wyzer.memory.memory_manager import get_memory_manager
            
            # Get ONLY the single relevant memory for fastlane (not all memories)
            memories_context = ""
            try:
                mem_mgr = get_memory_manager()
                memories_context = mem_mgr.select_for_fastlane_injection(user_text)
            except Exception:
                pass
            
            prompt, mode, stats = build_fastlane_prompt(
                user_text=user_text,
                memories_context=memories_context,
            )
            
            # Log prompt path at INFO level
            logger.info(
                f'[PROMPT_PATH] used=FAST_LANE reason={fastlane_reason} '
                f'user_text="{user_text[:50]}"'
            )
            
            # Add JSON format instruction (minimal)
            prompt += ' {"reply": "'
            
            return _ollama_request(prompt, user_text=user_text)
        except Exception as e:
            logger.debug(f"[FASTLANE] fallback to normal prompt: {e}")
    
    # Log normal prompt path
    logger.info(
        f'[PROMPT_PATH] used=NORMAL reason=reply_only '
        f'user_text="{user_text[:50]}"'
    )
    
    # Normal prompt path
    ctx = _gather_context_blocks(user_text, max_session_turns=2)
    
    # Check for smalltalk directive
    smalltalk_directive = ""
    try:
        from wyzer.brain.llm_engine import _is_smalltalk_request, SMALLTALK_SYSTEM_DIRECTIVE
        if _is_smalltalk_request(user_text):
            smalltalk_directive = f"\n{SMALLTALK_SYSTEM_DIRECTIVE}\n"
    except Exception:
        pass
    
    prompt = f"""You are Wyzer, a local voice assistant.
{ctx['session']}{ctx['promoted']}{ctx['redaction']}{ctx['memories']}{smalltalk_directive}
Reply naturally. Be direct.
You ARE allowed to generate stories, poems, jokes, and creative content when asked.
Keep creative content spoken-friendly: no markdown, no bullet lists.

NEVER invent or request tools. Respond directly in plain text.

JSON format: {{"reply": "your response"}}

User: {user_text}

JSON:"""
    return _ollama_request(prompt, user_text=user_text)


def _call_llm_with_execution_summary(
    user_text: str,
    execution_summary: ExecutionSummary,
    registry
) -> Dict[str, Any]:
    """
    Call LLM after tool execution(s) to generate final reply.
    
    Args:
        user_text: Original user request
        execution_summary: Summary of executed intents
        registry: Tool registry
        
    Returns:
        Dict with {"reply": "..."}
    """
    ctx = _gather_context_blocks(user_text, max_session_turns=2)
    
    # Build a concise summary of what was executed
    summary_parts = []
    for result in execution_summary.ran:
        if result.ok:
            # Truncate long results
            result_str = json.dumps(result.result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            summary_parts.append(f"- {result.tool}: OK - {result_str}")
        else:
            summary_parts.append(f"- {result.tool}: FAILED - {result.error}")
    
    if execution_summary.stopped_early:
        summary_parts.append("- Stopped early")
    
    summary_text = "\n".join(summary_parts)
    
    prompt = f"""You are Wyzer, a local voice assistant.
{ctx['session']}{ctx['promoted']}{ctx['redaction']}{ctx['memories']}
User asked: {user_text}

Results:
{summary_text}

Reply naturally in 1-2 sentences based on results.

JSON: {{"reply": "your response"}}

JSON:"""

    return _ollama_request(prompt)


def _call_llm_with_tool_result(
    user_text: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_result: Dict[str, Any],
    registry
) -> Dict[str, Any]:
    """
    Call LLM after tool execution to generate final reply.
    
    Returns:
        Dict with {"reply": "..."}
    """
    ctx = _gather_context_blocks(user_text, max_session_turns=2)
    
    # Truncate tool result if too long
    result_str = json.dumps(tool_result)
    if len(result_str) > 300:
        result_str = result_str[:300] + "..."
    
    args_str = json.dumps(tool_args)
    if len(args_str) > 100:
        args_str = args_str[:100] + "..."
    
    prompt = f"""You are Wyzer, a local voice assistant.
{ctx['session']}{ctx['promoted']}{ctx['redaction']}{ctx['memories']}
User asked: {user_text}

Tool: {tool_name}({args_str})
Result: {result_str}

Reply naturally in 1-2 sentences.

JSON: {{"reply": "your response"}}

JSON:"""

    return _ollama_request(prompt)


def _ollama_request(prompt: str, user_text: str = "") -> Dict[str, Any]:
    """
    Make request to LLM (Ollama or llama.cpp).
    
    Args:
        prompt: The full prompt to send to the LLM
        user_text: Original user text (for voice_fast options detection)
    
    Returns:
        Parsed JSON response or fallback dict
    """
    logger = get_logger_instance()
    
    # Check if LLM is disabled
    if getattr(Config, "NO_OLLAMA", False):
        logger.debug("[LLM] LLM disabled, returning not supported")
        return {
            "reply": _get_no_ollama_reply()
        }
    
    llm_mode = getattr(Config, "LLM_MODE", "ollama")
    if llm_mode == "off":
        logger.debug("[LLM] LLM mode is off, returning not supported")
        return {
            "reply": _get_no_ollama_reply()
        }
    
    # Estimate tokens before sending (for comparison/debugging)
    try:
        from wyzer.brain.prompt_builder import estimate_tokens
        est_tokens = estimate_tokens(prompt)
    except Exception:
        est_tokens = len(prompt) // 4
    
    try:
        timeout = Config.LLM_TIMEOUT
        
        options = {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_ctx": 4096,
            "num_predict": 150
        }
        
        # Apply voice-fast preset overrides if user_text provided
        if user_text:
            try:
                from wyzer.brain.llm_engine import get_voice_fast_options
                voice_fast_opts = get_voice_fast_options(user_text, llm_mode)
                if voice_fast_opts:
                    for key, value in voice_fast_opts.items():
                        if not key.startswith("_"):
                            options[key] = value
            except Exception as e:
                logger.debug(f"[LLM] voice_fast_options error: {e}")
        
        if llm_mode == "llamacpp":
            # Use llama.cpp client
            client = _get_llm_client()
            if client is None:
                return {"reply": _get_no_ollama_reply()}
            
            # For llama.cpp, add instruction to respond in JSON format
            json_prompt = prompt + "\n\nIMPORTANT: Respond with valid JSON only."
            reply_text = client.generate(prompt=json_prompt, options=options, stream=False)
            
            logger.debug(f"[LLAMACPP] est_tokens={est_tokens}")
        else:
            # Use Ollama direct HTTP (default)
            base_url = Config.OLLAMA_BASE_URL.rstrip("/")
            model = Config.OLLAMA_MODEL
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # Request JSON output
                "options": options
            }
            
            req = urllib.request.Request(
                f"{base_url}/api/generate",
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_data = json.loads(response.read().decode('utf-8'))
            
            # Log token counts from Ollama
            prompt_tokens = response_data.get("prompt_eval_count", 0)
            eval_tokens = response_data.get("eval_count", 0)
            logger.debug(f"[OLLAMA] est_tokens={est_tokens}, prompt_tokens={prompt_tokens}, eval_tokens={eval_tokens}")
            
            reply_text = response_data.get("response", "").strip()

        def _extract_reply_from_args(args: Any) -> str:
            if not isinstance(args, dict):
                return ""
            for key in ("reply", "message", "text", "content"):
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        def _postprocess_llm_json(parsed: Any) -> Any:
            """Compat layer for models that return a pseudo-tool `reply` inside intents."""
            if not isinstance(parsed, dict):
                return parsed

            raw_intents = parsed.get("intents")
            if isinstance(raw_intents, list) and raw_intents:
                extracted_reply = ""
                filtered_intents = []
                for intent in raw_intents:
                    if not isinstance(intent, dict):
                        continue
                    tool = intent.get("tool")
                    if isinstance(tool, str) and tool.strip().lower() == "reply":
                        if not extracted_reply:
                            extracted_reply = _extract_reply_from_args(intent.get("args", {}))
                        continue
                    filtered_intents.append(intent)

                # If the only thing returned was a reply-intent, convert to a direct reply.
                if not filtered_intents and extracted_reply:
                    return {"reply": extracted_reply}

                # Otherwise, keep intents but remove the pseudo reply intent.
                parsed["intents"] = filtered_intents
                if extracted_reply and (not isinstance(parsed.get("reply"), str) or not parsed.get("reply", "").strip()):
                    parsed["reply"] = extracted_reply

            # Legacy single-intent format
            raw_intent = parsed.get("intent")
            if isinstance(raw_intent, dict):
                tool = raw_intent.get("tool")
                if isinstance(tool, str) and tool.strip().lower() == "reply":
                    extracted_reply = _extract_reply_from_args(raw_intent.get("args", {}))
                    if extracted_reply:
                        return {"reply": extracted_reply}

            # Legacy single-tool format
            tool = parsed.get("tool")
            if isinstance(tool, str) and tool.strip().lower() == "reply":
                extracted_reply = _extract_reply_from_args(parsed.get("args", {}))
                if extracted_reply:
                    return {"reply": extracted_reply}

            return parsed

        # Clean up reply_text - some models return doubled braces from template escaping
        cleaned_text = reply_text.strip()
        # Handle doubled braces: {{"reply":...}} -> {"reply":...}
        if cleaned_text.startswith("{{") and cleaned_text.endswith("}}"):
            cleaned_text = cleaned_text[1:-1]
        # Remove markdown code fences if present
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Try to parse as JSON
        try:
            parsed = json.loads(cleaned_text)
            parsed = _postprocess_llm_json(parsed)
            if isinstance(parsed, dict):
                return parsed
            return {"reply": str(parsed)}
        except json.JSONDecodeError:
            # LLM didn't return valid JSON, extract reply if possible
            return {"reply": cleaned_text if cleaned_text else "I couldn't process that."}

    except urllib.error.URLError as e:
        # Distinguish slow-model timeouts from true connection failures.
        reason = getattr(e, "reason", None)
        is_timeout = isinstance(reason, socket.timeout) or "timed out" in str(e).lower()
        llm_name = "llama.cpp" if llm_mode == "llamacpp" else "Ollama"
        if is_timeout:
            logger.warning(f"{llm_name} request timed out after {Config.LLM_TIMEOUT}s: {e}")
            return {
                "reply": f"{llm_name} is taking too long to respond (timeout: {Config.LLM_TIMEOUT}s). Increase --llm-timeout or WYZER_LLM_TIMEOUT."
            }

        logger.warning(f"{llm_name} request failed (URL error): {e}")
        return {"reply": f"I couldn't reach {llm_name}. Is it running?"}

    except socket.timeout as e:
        llm_name = "llama.cpp" if llm_mode == "llamacpp" else "Ollama"
        logger.warning(f"{llm_name} request timed out after {Config.LLM_TIMEOUT}s: {e}")
        return {
            "reply": f"{llm_name} is taking too long to respond (timeout: {Config.LLM_TIMEOUT}s). Increase --llm-timeout or WYZER_LLM_TIMEOUT."
        }

    except Exception as e:
        # Keep generic fallback, but log the underlying error for debugging.
        llm_name = "llama.cpp" if llm_mode == "llamacpp" else "Ollama"
        logger.error(f"Unexpected {llm_name} error: {e}")
        return {"reply": f"I had trouble talking to {llm_name}."}


# ============================================================================
# PHASE 11: AUTONOMY COMMAND HANDLERS (deterministic, no LLM)
# ============================================================================
# NOTE: Regex patterns for autonomy commands are defined at the top of the file
# (near line 45) so they can be used by should_use_streaming_tts().


def _check_autonomy_commands(text: str, start_time: float) -> Optional[Dict[str, Any]]:
    """
    Check for autonomy-related voice commands.
    
    These are handled purely deterministically (no LLM).
    
    Args:
        text: User input text
        start_time: Performance timer start
        
    Returns:
        Response dict if command matched, None otherwise
    """
    logger = get_logger_instance()
    text = text.strip()
    
    from wyzer.context.world_state import (
        get_autonomy_mode,
        set_autonomy_mode,
        get_last_autonomy_decision,
    )
    
    # "autonomy off"
    if _AUTONOMY_OFF_RE.match(text):
        old_mode = get_autonomy_mode()
        set_autonomy_mode("off")
        logger.info("[AUTONOMY] mode changed to off (source=voice)")
        end_time = time.perf_counter()
        return {
            "reply": "Autonomy set to off.",
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {"autonomy_command": "off", "previous_mode": old_mode},
        }
    
    # "autonomy low"
    if _AUTONOMY_LOW_RE.match(text):
        old_mode = get_autonomy_mode()
        set_autonomy_mode("low")
        logger.info("[AUTONOMY] mode changed to low (source=voice)")
        end_time = time.perf_counter()
        return {
            "reply": "Autonomy set to low.",
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {"autonomy_command": "low", "previous_mode": old_mode},
        }
    
    # "autonomy normal"
    if _AUTONOMY_NORMAL_RE.match(text):
        old_mode = get_autonomy_mode()
        set_autonomy_mode("normal")
        logger.info("[AUTONOMY] mode changed to normal (source=voice)")
        end_time = time.perf_counter()
        return {
            "reply": "Autonomy set to normal.",
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {"autonomy_command": "normal", "previous_mode": old_mode},
        }
    
    # "autonomy high"
    if _AUTONOMY_HIGH_RE.match(text):
        old_mode = get_autonomy_mode()
        set_autonomy_mode("high")
        logger.info("[AUTONOMY] mode changed to high (source=voice)")
        end_time = time.perf_counter()
        return {
            "reply": "Autonomy set to high.",
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {"autonomy_command": "high", "previous_mode": old_mode},
        }
    
    # "what's my autonomy" / "what is your autonomy"
    if _AUTONOMY_STATUS_RE.match(text):
        mode = get_autonomy_mode()
        # Respond with "My" if user said "your", otherwise "Your" if user said "my"
        if " your " in text.lower():
            reply = f"My autonomy mode is {mode}."
        else:
            reply = f"Your autonomy mode is {mode}."
        end_time = time.perf_counter()
        return {
            "reply": reply,
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {"autonomy_command": "status", "current_mode": mode},
        }
    
    # "why did you do that"
    if _WHY_DID_YOU_RE.match(text):
        logger.info("[CMD] why_did_you_do_that matched")
        
        from wyzer.context.world_state import get_world_state
        
        mode = get_autonomy_mode()
        decision = get_last_autonomy_decision()
        ws = get_world_state()
        last_action = ws.last_action
        
        if decision is not None:
            # We have an autonomy decision - format it
            from wyzer.policy.autonomy_policy import format_decision_for_speech
            decision_dict = {
                "action": decision.action,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "risk": decision.risk,
                "needs_confirmation": False,
                "question": None,
            }
            reply = format_decision_for_speech(decision_dict)
            if decision.plan_summary:
                reply = f"For '{decision.plan_summary}': {reply}"
            logger.info(f"[EXPLAIN] mode={mode} confidence={decision.confidence:.2f} risk={decision.risk} action={decision.action} reason=\"{decision.reason}\"")
        elif last_action is not None:
            # No autonomy decision, but we have a last action (autonomy off or not triggered)
            tool_name = last_action.tool
            args_summary = ", ".join(f"{k}={v}" for k, v in last_action.args.items()) if last_action.args else ""
            target_info = ""
            if last_action.resolved:
                if "matched_name" in last_action.resolved:
                    target_info = f" target={last_action.resolved['matched_name']}"
                elif "title" in last_action.resolved:
                    target_info = f" target={last_action.resolved['title']}"
            
            logger.info(f"[EXPLAIN] mode={mode} last_action={tool_name}{target_info}")
            
            if mode == "off":
                reply = f"I executed {tool_name}({args_summary}). Autonomy is off, so no policy check was applied."
            else:
                reply = f"I executed {tool_name}({args_summary}). This action didn't require an autonomy decision."
        else:
            # Nothing to explain
            logger.info(f"[EXPLAIN] mode={mode} no_action=true")
            reply = "I haven't performed any actions yet."
        
        end_time = time.perf_counter()
        return {
            "reply": reply,
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {
                "autonomy_command": "why",
                "mode": mode,
                "had_decision": decision is not None,
                "had_last_action": last_action is not None,
            },
        }
    
    return None


def _check_confirmation_response(text: str, start_time: float) -> Optional[Dict[str, Any]]:
    """
    Check if user is responding to a pending confirmation.
    
    Only active when there's a pending confirmation from autonomy policy.
    
    Args:
        text: User input text
        start_time: Performance timer start
        
    Returns:
        Response dict if confirmation handled, None otherwise
    """
    logger = get_logger_instance()
    text = text.strip()
    
    from wyzer.context.world_state import (
        get_pending_confirmation,
        consume_pending_confirmation,
        clear_pending_confirmation,
        get_autonomy_mode,
    )
    
    # Only check for confirmation responses if there's a pending confirmation
    pending = get_pending_confirmation()
    if pending is None:
        return None
    
    # Check for "yes" confirmation
    if _CONFIRM_YES_RE.match(text):
        plan = consume_pending_confirmation()
        if plan is None:
            # Expired or already consumed
            logger.info("[CONFIRM] Confirmation expired or already used")
            end_time = time.perf_counter()
            return {
                "reply": "That confirmation has expired. Please try again.",
                "latency_ms": int((end_time - start_time) * 1000),
                "meta": {"confirmation": "expired"},
            }
        
        # Execute the confirmed plan
        logger.info(f"[CONFIRM] confirmed -> executing {len(plan)} tools")
        try:
            registry = get_registry()
            execution_summary, executed_intents = execute_tool_plan(plan, registry)
            reply = _format_fastpath_reply("", executed_intents, execution_summary)
            
            end_time = time.perf_counter()
            return {
                "reply": reply,
                "latency_ms": int((end_time - start_time) * 1000),
                "execution_summary": {
                    "ran": [
                        {
                            "tool": r.tool,
                            "ok": r.ok,
                            "result": r.result,
                            "error": r.error,
                        }
                        for r in execution_summary.ran
                    ],
                    "stopped_early": execution_summary.stopped_early,
                },
                "meta": {"confirmation": "confirmed", "confirmed_plan": [t.get("tool") for t in plan]},
            }
        except Exception as e:
            logger.error(f"[CONFIRM] Error executing confirmed plan: {e}")
            end_time = time.perf_counter()
            return {
                "reply": "Something went wrong executing that action.",
                "latency_ms": int((end_time - start_time) * 1000),
                "meta": {"confirmation": "error", "error": str(e)},
            }
    
    # Check for "no" cancellation
    if _CONFIRM_NO_RE.match(text):
        clear_pending_confirmation()
        logger.info("[CONFIRM] cancelled")
        end_time = time.perf_counter()
        return {
            "reply": "Okay, cancelled.",
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {"confirmation": "cancelled"},
        }
    
    # Text doesn't match yes/no patterns - not a confirmation response
    # Let it fall through to normal processing, but note the pending confirmation
    return None


# ============================================================================
# PHASE 12: WINDOW WATCHER COMMAND HANDLERS (deterministic, no LLM)
# ============================================================================

def _check_window_watcher_commands(text: str, start_time: float) -> Optional[Dict[str, Any]]:
    """
    Check for Phase 12 window watcher voice commands.
    
    These are handled purely deterministically (no LLM).
    Commands:
    - "what's on monitor 2" - list windows on that monitor
    - "close all on screen 1" - close all windows on that monitor (with confirmation)
    - "what did I just open" - list recently opened windows
    - "where am I" - report focused window and monitor
    
    Args:
        text: User input text
        start_time: Performance timer start
        
    Returns:
        Response dict if command matched, None otherwise
    """
    logger = get_logger_instance()
    text = text.strip()
    
    from wyzer.context.world_state import (
        get_windows_on_monitor,
        get_focused_window_info,
        get_recent_window_events,
        get_monitor_count,
        set_pending_confirmation,
        get_autonomy_mode,
    )
    from wyzer.policy.risk import classify_plan
    
    # Known junk window titles to exclude
    JUNK_TITLES = frozenset({
        "program manager",
        "windows input experience",
        "windows shell experience host",
        "textinputhost",
        "search",
        "start",
        "task view",
        "nvidia geforce overlay",
        "nvidia notification helper window",
        "settings",
    })
    
    # Known junk process names to exclude
    JUNK_PROCESSES = frozenset({
        "explorer.exe",
        "applicationframehost.exe",
        "textinputhost.exe",
        "searchhost.exe",
        "shellexperiencehost.exe",
        "startmenuexperiencehost.exe",
        "lockapp.exe",
        "dwm.exe",
        "taskhostw.exe",
    })
    
    def _is_junk_window(w: Dict[str, Any], is_focused: bool = False) -> bool:
        """Check if a window is junk that should be excluded from listings."""
        title = (w.get("title") or "").strip()
        process = (w.get("process") or "").lower()
        
        # Empty titles are junk UNLESS this is the focused window
        if not title and not is_focused:
            return True
        
        # Check known junk titles
        if title.lower() in JUNK_TITLES:
            return True
        
        # Check known junk processes
        if process in JUNK_PROCESSES:
            return True
        
        return False
    
    # "what's on monitor 2" or "what's on the second monitor"
    match = _WHATS_ON_MONITOR_RE.match(text)
    if match:
        # Extract from group 1 (number after monitor) or group 2 (ordinal before monitor)
        monitor_ref = match.group(1) or match.group(2)
        monitor = _parse_monitor_number(monitor_ref)
        detected_count = get_monitor_count()
        
        logger.info(f"[CMD_P12] whats_on_monitor monitor={monitor} detected_count={detected_count}")
        
        # Check if requested monitor exists
        if monitor > detected_count:
            reply = f"I only detect {detected_count} monitor{'s' if detected_count != 1 else ''} right now."
            end_time = time.perf_counter()
            return {
                "reply": reply,
                "latency_ms": int((end_time - start_time) * 1000),
                "meta": {
                    "window_watcher_command": "whats_on_monitor",
                    "monitor": monitor,
                    "detected_count": detected_count,
                    "error": "monitor_not_found",
                },
            }
        
        windows = get_windows_on_monitor(monitor)
        focused = get_focused_window_info()
        focused_hwnd = focused.get("hwnd") if focused else None
        
        # Filter out junk windows
        filtered_windows = []
        for w in windows:
            is_focused = w.get("hwnd") == focused_hwnd
            if not _is_junk_window(w, is_focused):
                filtered_windows.append(w)
        
        logger.info(f"[CMD_P12] whats_on_monitor monitor={monitor} raw={len(windows)} filtered={len(filtered_windows)}")
        
        if not filtered_windows:
            reply = f"Monitor {monitor} appears empty."
        else:
            # Format window list (show titles, max 6), prioritize focused window
            # Sort: focused window first, then by title
            sorted_windows = sorted(
                filtered_windows,
                key=lambda w: (0 if w.get("hwnd") == focused_hwnd else 1, w.get("title", "").lower())
            )
            
            lines = []
            for w in sorted_windows[:6]:
                title = w.get("title", "Untitled")[:40]
                if len(w.get("title", "")) > 40:
                    title += "..."
                is_focused = w.get("hwnd") == focused_hwnd
                prefix = "→ " if is_focused else "• "
                lines.append(f"{prefix}{title}")
            
            extra = len(filtered_windows) - 6
            if extra > 0:
                lines.append(f"...and {extra} more")
            
            reply = f"Monitor {monitor} has {len(filtered_windows)} windows:\n" + "\n".join(lines)
        
        end_time = time.perf_counter()
        return {
            "reply": reply,
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {"window_watcher_command": "whats_on_monitor", "monitor": monitor},
        }
    
    # "close all on screen 1"
    match = _CLOSE_ALL_ON_MONITOR_RE.match(text)
    if match:
        monitor_ref = match.group(1)
        monitor = _parse_monitor_number(monitor_ref)
        detected_count = get_monitor_count()
        
        logger.info(f"[CMD_P12] close_all_on_monitor monitor={monitor} detected_count={detected_count}")
        
        # Check if requested monitor exists
        if monitor > detected_count:
            reply = f"I only detect {detected_count} monitor{'s' if detected_count != 1 else ''} right now."
            end_time = time.perf_counter()
            return {
                "reply": reply,
                "latency_ms": int((end_time - start_time) * 1000),
                "meta": {
                    "window_watcher_command": "close_all_on_monitor",
                    "monitor": monitor,
                    "detected_count": detected_count,
                    "error": "monitor_not_found",
                },
            }
        
        windows = get_windows_on_monitor(monitor)
        
        # Filter windows: exclude junk and Wyzer's own process
        closeable = []
        for w in windows:
            title = (w.get("title") or "").strip()
            process = (w.get("process") or "").lower()
            
            # Skip junk windows
            if _is_junk_window(w):
                continue
            # Skip Wyzer's own process
            if "wyzer" in process or "python" in process:
                continue
            
            closeable.append(w)
        
        logger.info(f"[CMD_P12] close_all_on_monitor monitor={monitor} raw={len(windows)} closeable={len(closeable)}")
        
        if not closeable:
            reply = f"No closeable windows found on monitor {monitor}."
            end_time = time.perf_counter()
            return {
                "reply": reply,
                "latency_ms": int((end_time - start_time) * 1000),
                "meta": {"window_watcher_command": "close_all_on_monitor", "monitor": monitor, "count": 0},
            }
        
        # Build tool plan to close each window
        max_close = getattr(Config, "WINDOW_WATCHER_MAX_BULK_CLOSE", 10)
        if len(closeable) > max_close:
            # Too many - ask for explicit confirmation with warning
            titles_preview = ", ".join(w.get("title", "?")[:20] for w in closeable[:3])
            reply = f"There are {len(closeable)} windows on monitor {monitor}. That's a lot! Say 'close all' again to confirm, or close specific windows."
            end_time = time.perf_counter()
            return {
                "reply": reply,
                "latency_ms": int((end_time - start_time) * 1000),
                "meta": {"window_watcher_command": "close_all_on_monitor", "monitor": monitor, "count": len(closeable), "needs_explicit": True},
            }
        
        # Build the tool plan
        tool_plan = []
        for w in closeable:
            tool_plan.append({
                "tool": "close_window",
                "args": {"title": w.get("title", "")},
                "continue_on_error": True,  # Keep going even if one fails
            })
        
        # Check risk - bulk close is HIGH risk
        risk = classify_plan(tool_plan)
        mode = get_autonomy_mode()
        
        # Format confirmation prompt
        titles_preview = ", ".join(w.get("title", "?")[:20] for w in closeable[:3])
        if len(closeable) > 3:
            titles_preview += f"... and {len(closeable) - 3} more"
        
        # Always require confirmation for bulk close (HIGH risk)
        if mode == "off" or risk == "high":
            # Set pending confirmation
            prompt = f"I found {len(closeable)} windows on monitor {monitor}: {titles_preview}. Close them all?"
            set_pending_confirmation(
                plan=tool_plan,
                prompt=prompt,
                timeout_sec=getattr(Config, "AUTONOMY_CONFIRM_TIMEOUT_SEC", 45.0),
            )
            
            logger.info(f"[CMD_P12] close_all_on_monitor awaiting confirmation count={len(closeable)} risk={risk}")
            
            end_time = time.perf_counter()
            return {
                "reply": prompt,
                "latency_ms": int((end_time - start_time) * 1000),
                "meta": {
                    "window_watcher_command": "close_all_on_monitor",
                    "monitor": monitor,
                    "count": len(closeable),
                    "has_pending_confirmation": True,
                    "confirmation_timeout_sec": getattr(Config, "AUTONOMY_CONFIRM_TIMEOUT_SEC", 45.0),
                },
            }
        
        # In high autonomy mode with non-high risk (shouldn't happen for bulk close but just in case)
        # Execute directly
        registry = get_registry()
        execution_summary, executed_intents = execute_tool_plan(tool_plan, registry)
        closed_count = sum(1 for r in execution_summary.ran if r.ok)
        
        reply = f"Closed {closed_count} of {len(closeable)} windows on monitor {monitor}."
        
        end_time = time.perf_counter()
        return {
            "reply": reply,
            "latency_ms": int((end_time - start_time) * 1000),
            "execution_summary": {
                "ran": [{"tool": r.tool, "ok": r.ok} for r in execution_summary.ran],
                "stopped_early": execution_summary.stopped_early,
            },
            "meta": {"window_watcher_command": "close_all_on_monitor", "monitor": monitor, "closed": closed_count},
        }
    
    # "what did I just open"
    if _WHAT_DID_I_OPEN_RE.match(text):
        # Get recent opened and focus_changed events
        opened_events = get_recent_window_events(event_type="opened", limit=5)
        focus_events = get_recent_window_events(event_type="focus_changed", limit=3)
        
        # Combine and sort by timestamp
        all_events = opened_events + focus_events
        all_events.sort(key=lambda e: e.get("ts", 0), reverse=True)
        
        logger.info(f"[CMD_P12] what_did_i_open events={len(all_events)}")
        
        if not all_events:
            reply = "I haven't tracked any recently opened windows yet."
        else:
            lines = []
            seen_titles = set()
            for ev in all_events[:5]:
                title = ev.get("title", "Untitled")[:40]
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                ev_type = ev.get("type", "")
                if ev_type == "opened":
                    lines.append(f"• Opened: {title}")
                else:
                    lines.append(f"• Focused: {title}")
            
            reply = "Recently opened:\n" + "\n".join(lines)
        
        end_time = time.perf_counter()
        return {
            "reply": reply,
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {"window_watcher_command": "what_did_i_open", "count": len(all_events)},
        }
    
    # "where am I"
    if _WHERE_AM_I_RE.match(text):
        focused = get_focused_window_info()
        
        logger.info(f"[CMD_P12] where_am_i focused={focused is not None}")
        
        if not focused:
            reply = "I can't detect the focused window right now."
        else:
            title = focused.get("title", "Unknown")[:50]
            process = focused.get("process", "")
            monitor = focused.get("monitor", 1)
            
            if process:
                reply = f"You're in {title} ({process}) on monitor {monitor}."
            else:
                reply = f"You're in {title} on monitor {monitor}."
        
        end_time = time.perf_counter()
        return {
            "reply": reply,
            "latency_ms": int((end_time - start_time) * 1000),
            "meta": {"window_watcher_command": "where_am_i", "focused": focused},
        }
    
    return None
