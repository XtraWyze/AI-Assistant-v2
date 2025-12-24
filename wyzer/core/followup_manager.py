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
