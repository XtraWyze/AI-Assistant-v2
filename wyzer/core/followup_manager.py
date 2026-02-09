"""
FOLLOWUP listening window manager for Wyzer AI Assistant.

After every completed TTS response, Wyzer enters a FOLLOWUP state where hotword
detection is temporarily disabled. The user can say a follow-up WITHOUT hotword
detection for approximately FOLLOWUP_TIMEOUT_SEC (default 3.0 seconds).

The timer is extended (reset) on each detected speech. FOLLOWUP ends when:
1. Silence/no valid speech occurs for FOLLOWUP_TIMEOUT_SEC, OR
2. User says an explicit end phrase like "no", "nope", "that's all", etc.

If user says any other text, treat it as a normal query and allow chaining
(re-enter FOLLOWUP after responding).
"""
import re
import time
from typing import Optional, List, Dict, Any, Union
from wyzer.core.logger import get_logger
from wyzer.core.config import Config


# ============================================================================
# PENDING TOOL COMPLETION
# ============================================================================
# When a tool fails and requests clarification (e.g., close_window returns
# window_not_found), a PendingToolCompletion is stored.  On the next
# follow-up utterance the orchestrator will:
#   1. Extract the missing slot value from user speech
#   2. Re-run the original tool with the resolved slot
#   3. Speak ONLY the tool-confirmed result (NO LLM)
# ============================================================================

class PendingToolCompletion:
    """Holds state for a tool that failed and is waiting for user clarification."""

    __slots__ = ("tool", "slots", "original_args", "expires_ts")

    def __init__(
        self,
        tool: str,
        slots: Dict[str, Any],
        original_args: Dict[str, Any],
        expires_s: float = 20.0,
    ):
        self.tool = tool
        self.slots = dict(slots)          # e.g. {"title": None}
        self.original_args = dict(original_args)
        self.expires_ts = time.time() + expires_s

    def is_expired(self) -> bool:
        return time.time() > self.expires_ts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "complete_tool",
            "tool": self.tool,
            "slots": self.slots,
            "original_args": self.original_args,
            "expires_ts": self.expires_ts,
        }

    def __repr__(self) -> str:
        return f"PendingToolCompletion(tool={self.tool!r}, slots={self.slots!r})"


# ============================================================================
# EXIT PHRASE SENTINEL
# ============================================================================
# When an exit phrase is detected, this sentinel dict is returned instead of
# the raw transcript. This creates a single source of truth for exit-phrase
# handling and prevents double-detection in downstream pipeline stages.
#
# Sentinel structure (JSON-safe for IPC):
#   {"type": "exit_followup", "phrase": "<normalized>", "original": "<raw>"}
#
# Usage:
#   result = followup_manager.check_exit_phrase(text)
#   if is_exit_sentinel(result):
#       # Handle exit: skip LLM, skip tools, return to IDLE
#   else:
#       # result is None, text is a normal query
# ============================================================================

EXIT_SENTINEL_TYPE = "exit_followup"


def make_exit_sentinel(normalized_phrase: str, original_text: str) -> Dict[str, str]:
    """
    Create an exit phrase sentinel dict.
    
    Args:
        normalized_phrase: The normalized exit phrase that was matched
        original_text: The original user transcript
        
    Returns:
        A JSON-safe dict sentinel: {"type": "exit_followup", "phrase": ..., "original": ...}
    """
    return {
        "type": EXIT_SENTINEL_TYPE,
        "phrase": normalized_phrase,
        "original": original_text,
    }


def is_exit_sentinel(value: Any) -> bool:
    """
    Check if a value is an exit phrase sentinel.
    
    Args:
        value: Any value to check
        
    Returns:
        True if value is an exit sentinel dict, False otherwise
    """
    return (
        isinstance(value, dict) 
        and value.get("type") == EXIT_SENTINEL_TYPE
    )


class FollowupManager:
    """Manager for FOLLOWUP listening window behavior"""
    
    # Exit phrases: if transcript matches any of these (case-insensitive),
    # exit FOLLOWUP and return to IDLE
    EXIT_PHRASES: List[str] = [
        "no",
        "nope",
        "nothing",
        "that's all",
        "thats all",
        "stop",
        "cancel",
        "never mind",
        "nevermind",
        "nothing else",
        "all good",
    ]
    
    def __init__(self):
        """Initialize FollowupManager"""
        self.logger = get_logger()
        self._followup_active: bool = False
        self._window_start_time: float = 0.0
        self._last_speech_time: float = 0.0
        self._chain_count: int = 0
        # Grace period to ignore residual TTS echo picked up by microphone.
        # TTS playback is already complete when FOLLOWUP starts, so this only
        # needs to cover potential echo/reverb - 0.3s should be sufficient.
        self._grace_period_duration: float = 0.3
        # Pending tool completion: when a tool fails and asks for clarification,
        # this stores the tool + unresolved slots so the next follow-up bypasses
        # the LLM and re-runs the tool directly.
        self._pending_tool: Optional[PendingToolCompletion] = None
    
    def start_followup_window(self) -> None:
        """
        Start a new FOLLOWUP listening window.
        Called after TTS completes and we want to listen for follow-ups.
        """
        if not Config.FOLLOWUP_ENABLED:
            return
        
        self._followup_active = True
        self._window_start_time = time.time()
        self._last_speech_time = self._window_start_time
        self._chain_count = 0
        
        self.logger.info("[STATE] FOLLOWUP: listening (hotword disabled)")
    
    def is_followup_active(self) -> bool:
        """Check if FOLLOWUP window is currently active"""
        return self._followup_active and Config.FOLLOWUP_ENABLED
    
    def reset_speech_timer(self) -> None:
        """Reset the silence timer (called when speech is detected)"""
        if not self.is_followup_active():
            return
        self._last_speech_time = time.time()
    
    def check_timeout(self) -> bool:
        """
        Check if FOLLOWUP window has timed out (silence for too long).
        
        Returns:
            True if timed out (should exit FOLLOWUP), False otherwise
        """
        if not self.is_followup_active():
            return False
        
        current_time = time.time()
        time_since_speech = current_time - self._last_speech_time
        
        if time_since_speech >= Config.FOLLOWUP_TIMEOUT_SEC:
            self.logger.info(
                f"[STATE] FOLLOWUP: timeout after {time_since_speech:.2f}s silence"
            )
            self._followup_active = False
            return True
        
        return False
    
    def is_in_grace_period(self) -> bool:
        """
        Check if FOLLOWUP is in grace period (TTS prompt still playing).
        During grace period, ignore VAD speech detection to avoid picking up TTS audio.
        
        Returns:
            True if still in grace period, False if listening is active
        """
        if not self.is_followup_active():
            return False
        
        current_time = time.time()
        time_since_start = current_time - self._window_start_time
        return time_since_start < self._grace_period_duration
    
    def check_no_speech_timeout(self) -> bool:
        """
        Check if FOLLOWUP window has timed out waiting for speech to start.
        
        This handles the case where the user speaks during grace period (audio is
        ignored to avoid TTS echo) but doesn't speak again after grace period ends.
        Without this check, FOLLOWUP would listen indefinitely.
        
        The timeout is: grace_period + FOLLOWUP_TIMEOUT_SEC from window start.
        
        Returns:
            True if timed out waiting for speech, False otherwise
        """
        if not self.is_followup_active():
            return False
        
        # Only check after grace period has ended
        if self.is_in_grace_period():
            return False
        
        current_time = time.time()
        # Total allowed time = grace period + followup timeout
        max_wait_time = self._grace_period_duration + Config.FOLLOWUP_TIMEOUT_SEC
        time_since_start = current_time - self._window_start_time
        
        if time_since_start >= max_wait_time:
            self.logger.info(
                f"[STATE] FOLLOWUP: no speech detected after {time_since_start:.2f}s"
            )
            self._followup_active = False
            return True
        
        return False
    
    def is_exit_phrase(self, text: str) -> bool:
        """
        Check if transcript matches an exit phrase.
        
        Normalizes text and checks if:
        1. Text is exactly an exit phrase, OR
        2. Text STARTS with an exit phrase (e.g., "no thanks"), OR
        3. Text ENDS with an exit phrase (e.g., "forget it, cancel"), OR
        4. A single-word exit phrase appears as first word (e.g., "stop right there")
        
        This allows catching exit phrases at the beginning or end of longer sentences.
        
        Args:
            text: User's transcript
            
        Returns:
            True if text matches exit phrase, False otherwise
        """
        if not text:
            return False
        
        # Normalize: lowercase, remove punctuation, strip whitespace
        normalized = self._normalize_text(text)
        words = normalized.split()
        
        if not words:
            return False
        
        # Check each exit phrase
        for phrase in self.EXIT_PHRASES:
            phrase_normalized = self._normalize_text(phrase)
            phrase_words = phrase_normalized.split()
            
            if not phrase_words:
                continue
            
            # Exact match
            if normalized == phrase_normalized:
                self.logger.info(f"[FOLLOWUP] Exit phrase detected: '{text}' -> '{normalized}'")
                return True
            
            # Text starts with the phrase (e.g., "no thanks" starts with "no")
            if len(words) >= len(phrase_words):
                if words[:len(phrase_words)] == phrase_words:
                    self.logger.info(f"[FOLLOWUP] Exit phrase detected: '{text}' -> '{normalized}'")
                    return True
            
            # Text ends with the phrase (e.g., "forget it, cancel" ends with "cancel")
            if len(words) >= len(phrase_words):
                if words[-len(phrase_words):] == phrase_words:
                    self.logger.info(f"[FOLLOWUP] Exit phrase detected: '{text}' -> '{normalized}'")
                    return True
            
            # Check if first word alone matches a single-word exit phrase
            # This handles cases like "Nothing" not matching "nothing else"
            # but allows checking individual words
            if len(phrase_words) == 1 and len(words) >= 1:
                # Single word exit phrase - check if it's the first word
                if words[0] == phrase_words[0]:
                    self.logger.info(f"[FOLLOWUP] Exit phrase detected: '{text}' -> '{normalized}'")
                    return True
        
        return False
    
    def check_exit_phrase(self, text: str, log_detection: bool = True) -> Optional[Dict[str, str]]:
        """
        Check if transcript matches an exit phrase and return a sentinel if so.
        
        This is the PREFERRED method for exit phrase detection as it returns
        a sentinel that can be propagated through the pipeline to prevent
        double-detection. The sentinel is JSON-safe for IPC.
        
        Args:
            text: User's transcript
            log_detection: Whether to log the detection (default True).
                           Set False to suppress duplicate log entries.
            
        Returns:
            Exit sentinel dict if text matches exit phrase, None otherwise.
            Sentinel format: {"type": "exit_followup", "phrase": "<normalized>", "original": "<raw>"}
        """
        if not text:
            return None
        
        # Normalize: lowercase, remove punctuation, strip whitespace
        normalized = self._normalize_text(text)
        words = normalized.split()
        
        if not words:
            return None
        
        # Check each exit phrase
        for phrase in self.EXIT_PHRASES:
            phrase_normalized = self._normalize_text(phrase)
            phrase_words = phrase_normalized.split()
            
            if not phrase_words:
                continue
            
            matched = False
            
            # Exact match
            if normalized == phrase_normalized:
                matched = True
            
            # Text starts with the phrase (e.g., "no thanks" starts with "no")
            elif len(words) >= len(phrase_words) and words[:len(phrase_words)] == phrase_words:
                matched = True
            
            # Text ends with the phrase (e.g., "forget it, cancel" ends with "cancel")
            elif len(words) >= len(phrase_words) and words[-len(phrase_words):] == phrase_words:
                matched = True
            
            # Single word exit phrase - check if it's the first word
            elif len(phrase_words) == 1 and len(words) >= 1 and words[0] == phrase_words[0]:
                matched = True
            
            if matched:
                if log_detection:
                    self.logger.info(f"[EXIT] Exit phrase detected: '{text}' -> '{normalized}'")
                return make_exit_sentinel(normalized, text)
        
        return None
    
    def end_followup_window(self) -> None:
        """Explicitly end the FOLLOWUP window (e.g., when user says exit phrase)"""
        if self._followup_active:
            self.logger.info("[STATE] FOLLOWUP: ended")
            self._followup_active = False
            self._chain_count = 0
    
    def increment_chain(self) -> bool:
        """
        Increment the follow-up chain counter.
        
        Returns:
            True if chain limit not exceeded, False if max chain depth reached
        """
        self._chain_count += 1
        if self._chain_count > Config.FOLLOWUP_MAX_CHAIN:
            self.logger.info(
                f"[FOLLOWUP] Max chain depth ({Config.FOLLOWUP_MAX_CHAIN}) reached"
            )
            self._followup_active = False
            return False
        return True
    
    def get_chain_count(self) -> int:
        """Get current chain count"""
        return self._chain_count
    
    def get_remaining_time(self) -> float:
        """
        Get remaining time in FOLLOWUP window (seconds).
        
        Returns:
            Remaining time in seconds, or 0 if window is expired/inactive
        """
        if not self.is_followup_active():
            return 0.0
        
        current_time = time.time()
        time_since_speech = current_time - self._last_speech_time
        remaining = Config.FOLLOWUP_TIMEOUT_SEC - time_since_speech
        
        return max(0.0, remaining)

    # ========================================================================
    # PENDING TOOL COMPLETION API
    # ========================================================================

    def set_pending_tool(self, pending: PendingToolCompletion) -> None:
        """Store a pending tool completion request.

        This is called when a tool fails and needs clarification (e.g.
        close_window returns window_not_found).  On the next follow-up
        utterance the orchestrator will intercept the input, resolve the
        missing slot, and re-run the tool WITHOUT the LLM.
        """
        self._pending_tool = pending
        self.logger.info(f"[FOLLOWUP] Pending tool set: {pending}")

    def get_pending_tool(self) -> Optional[PendingToolCompletion]:
        """Return the current pending tool completion, or None if expired/absent."""
        if self._pending_tool is None:
            return None
        if self._pending_tool.is_expired():
            self.logger.info("[FOLLOWUP] Pending tool expired, clearing")
            self._pending_tool = None
            return None
        return self._pending_tool

    def consume_pending_tool(self) -> Optional[PendingToolCompletion]:
        """Return and clear the pending tool completion (one-shot)."""
        pending = self.get_pending_tool()
        self._pending_tool = None
        return pending

    def clear_pending_tool(self) -> None:
        """Explicitly discard any pending tool completion."""
        self._pending_tool = None
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for exit phrase matching.
        
        - Lowercase
        - Remove punctuation (keep spaces)
        - Strip leading/trailing whitespace
        - Collapse multiple spaces to single space
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Lowercase
        normalized = text.lower()
        
        # Remove punctuation (keep alphanumeric, spaces)
        # This regex keeps a-z, 0-9, and spaces only
        normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
        
        # Collapse multiple spaces to single space
        normalized = re.sub(r"\s+", " ", normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized

    # ========================================================================
    # PHASE 17: AGENT-GRADE FOLLOW-UP TARGETS
    # ========================================================================
    # These attributes are set by the agent loop (_update_agent_followup_targets)
    # and consumed by the reference resolver for "yes it's X" / "that one" / "it"
    # resolution.
    _agent_last_window_candidates: List[Dict[str, Any]] = []
    _agent_last_control_candidates: List[Dict[str, Any]] = []
    _agent_last_tool: Optional[Dict[str, Any]] = None          # {name, args, result_summary}
    _agent_last_confirmable_target: Optional[Dict[str, Any]] = None  # {type, value}

    def set_last_tool(self, name: str, args: Dict[str, Any], result_summary: str = "") -> None:
        """Record the last tool call for follow-up resolution."""
        self._agent_last_tool = {
            "name": name,
            "args": dict(args),
            "result_summary": result_summary,
        }

    def get_last_confirmable_target(self) -> Optional[Dict[str, Any]]:
        """Return the last confirmable target (window or control)."""
        return self._agent_last_confirmable_target

    def get_window_candidates(self) -> List[Dict[str, Any]]:
        """Return cached window candidates from last perception."""
        return self._agent_last_window_candidates

    def resolve_yes_its_x(self, text: str) -> Optional[Dict[str, Any]]:
        """Check if user says 'yes, it's X' and X matches a window.

        Returns:
            Dict with {action: "close_window", title: str} if matched,
            None otherwise.
        """
        s = (text or "").strip().lower()
        # Pattern: "yes it's X", "yeah it's X", "yes X", "it's X"
        import re as _re
        m = _re.match(
            r"^(?:yes|yeah|yep|yup|sure|right|ok|okay)"
            r"(?:\s*,?\s*(?:it(?:'?s|\s+is)?\s+)?|\s+)"
            r"(.+?)\.?$",
            s,
            _re.IGNORECASE,
        )
        if not m:
            return None

        candidate = m.group(1).strip()
        if not candidate or len(candidate) < 2:
            return None

        # Try to match against window candidates
        for win in self._agent_last_window_candidates:
            title = (win.get("title") or "").lower()
            app = (win.get("app") or "").lower().replace(".exe", "")
            if candidate in title or candidate == app or title.startswith(candidate):
                return {
                    "action": "close_window",
                    "title": win.get("title") or app,
                }

        # Try foreground match
        if self._agent_last_confirmable_target:
            val = self._agent_last_confirmable_target.get("value") or {}
            tgt_title = (val.get("title") or "").lower()
            tgt_app = (val.get("app") or "").lower().replace(".exe", "")
            if candidate in tgt_title or candidate == tgt_app:
                return {
                    "action": "close_window",
                    "title": val.get("title") or tgt_app,
                }

        return None
