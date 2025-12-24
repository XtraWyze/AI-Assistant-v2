"""
Pending Confirmation Resolver for Phase 11 Confirmation Flow.

This module provides deterministic yes/no/cancel pattern matching and resolution
for pending confirmations. It is the SINGLE source of truth for handling
user responses to pending confirmation prompts.

HARD RULES:
- Regex/string only - NO LLM parsing for yes/no/cancel
- Must execute/cancel/expire deterministically
- Must clear pending BEFORE execution to prevent double-run

Usage:
    from wyzer.policy.pending_confirmation import resolve_pending
    
    result = resolve_pending(transcript, now_ts)
    if result == "executed":
        # Plan was executed
    elif result == "cancelled":
        # User cancelled
    elif result == "expired":
        # Confirmation timed out
    elif result == "ignored":
        # Pending exists but user said something other than yes/no
        # Re-ask the confirmation prompt
    elif result == "none":
        # No pending confirmation - continue normal flow
"""

import re
import time
from typing import Literal, Optional, Callable, List, Dict, Any

from wyzer.core.logger import get_logger

# ============================================================================
# YES/NO PATTERNS (compiled regexes)
# ============================================================================
# These patterns match if the response STARTS with a confirmation word
# to handle natural replies like "Yes, do it" or "No, don't do that"

YES_PATTERN = re.compile(
    r"^(?:yes|yeah|yep|yup|do\s+it|proceed|confirm|go\s+ahead|sure|ok|okay|absolutely|affirmative)(?:\b|$|[.,!?\s])",
    re.IGNORECASE
)

NO_PATTERN = re.compile(
    r"^(?:no|nope|nah|cancel|stop|don'?t|do\s+not|nevermind|never\s+mind|abort|negative)(?:\b|$|[.,!?\s])",
    re.IGNORECASE
)

# Confirmation timeout (seconds) - used when setting new pending confirmations
# Increased to 45s to give user more time to respond (decoupled from FOLLOWUP timing)
CONFIRMATION_TIMEOUT_SEC = 45.0

# Grace period (ms) to accept confirmation response while TTS is still playing
CONFIRMATION_GRACE_MS = 1500


def normalize(text: str) -> str:
    """
    Normalize text for pattern matching.
    
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple whitespace to single space
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Lowercase and strip
    normalized = text.lower().strip()
    
    # Collapse multiple whitespace to single space
    normalized = re.sub(r"\s+", " ", normalized)
    
    return normalized


def is_yes(text: str) -> bool:
    """
    Check if text is an affirmative response.
    
    Args:
        text: User's response text
        
    Returns:
        True if text matches yes pattern
    """
    normalized = normalize(text)
    return bool(YES_PATTERN.match(normalized))


def is_no(text: str) -> bool:
    """
    Check if text is a negative/cancel response.
    
    Args:
        text: User's response text
        
    Returns:
        True if text matches no pattern
    """
    normalized = normalize(text)
    return bool(NO_PATTERN.match(normalized))


def is_confirmation_response(text: str) -> bool:
    """
    Check if text is any kind of confirmation response (yes OR no).
    
    Args:
        text: User's response text
        
    Returns:
        True if text matches yes or no pattern
    """
    return is_yes(text) or is_no(text)


# Type alias for resolve_pending return values
ResolveResult = Literal["executed", "cancelled", "expired", "ignored", "none"]


def resolve_pending(
    transcript: str,
    now_ts: Optional[float] = None,
    executor: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    speak_fn: Optional[Callable[[str], None]] = None,
) -> ResolveResult:
    """
    Resolve a pending confirmation based on user transcript.
    
    This is the MAIN entry point for confirmation handling. Call this BEFORE
    any router/LLM processing when a transcript arrives.
    
    Behavior:
    - If no pending_confirmation exists -> "none" (continue normal flow)
    - If pending exists and expired -> clear, log, return "expired"
    - If is_yes(transcript) -> pop pending, execute via existing orchestrator path, return "executed"
    - If is_no(transcript) -> clear pending, speak "Okay, cancelled.", return "cancelled"
    - Else -> return "ignored" (pending exists but transcript is not yes/no)
    
    IMPORTANT:
    - Clears pending BEFORE execution to prevent double-run on retries
    - Uses the provided executor (which should be the same tool_plan executor
      used for normal tool execution)
    
    Args:
        transcript: User's input text
        now_ts: Current timestamp (defaults to time.time())
        executor: Optional callable to execute the pending plan.
                  Signature: executor(plan: List[Dict]) -> Any
                  If None, will use default orchestrator execute_tool_plan.
        speak_fn: Optional callable to speak text (for "Okay, cancelled")
                  If None, speaking is skipped.
        
    Returns:
        One of: "executed", "cancelled", "expired", "ignored", "none"
    """
    logger = get_logger()
    
    if now_ts is None:
        now_ts = time.time()
    
    # Import here to avoid circular imports
    from wyzer.context.world_state import (
        get_pending_confirmation,
        consume_pending_confirmation,
        clear_pending_confirmation,
    )
    
    # Check if there's a pending confirmation
    pending = get_pending_confirmation()
    if pending is None:
        return "none"
    
    # Calculate age for logging
    age_ms = int((now_ts - (pending.expires_ts - CONFIRMATION_TIMEOUT_SEC)) * 1000)
    
    # Check for expiry
    is_expired = now_ts > pending.expires_ts
    if is_expired:
        clear_pending_confirmation()
        logger.info("[CONFIRM] expired -> cleared")
        return "expired"
    
    # Check for YES confirmation
    if is_yes(transcript):
        logger.info(f"[CONFIRM] received reply=\"yes\" age_ms={age_ms} expired=0")
        # Pop the pending plan BEFORE execution to prevent double-run
        plan = consume_pending_confirmation()
        if plan is None:
            # Race condition: expired or consumed between get and consume
            logger.info("[CONFIRM] expired (race)")
            return "expired"
        
        logger.info(f"[CONFIRM] confirmed -> executing {len(plan)} tools")
        
        # Execute using provided executor or default
        if executor is not None:
            try:
                executor(plan)
            except Exception as e:
                logger.error(f"[CONFIRM] execution error: {e}")
        else:
            # Use default orchestrator executor
            _execute_plan_default(plan, logger)
        
        return "executed"
    
    # Check for NO cancellation
    if is_no(transcript):
        logger.info(f"[CONFIRM] received reply=\"no\" age_ms={age_ms} expired=0")
        clear_pending_confirmation()
        logger.info("[CONFIRM] cancelled")
        
        # Speak cancellation acknowledgment
        if speak_fn is not None:
            try:
                speak_fn("Okay, cancelled.")
            except Exception as e:
                logger.debug(f"[CONFIRM] speak error: {e}")
        
        return "cancelled"
    
    # Transcript is something else - not a yes/no response
    # The caller should re-ask the pending prompt
    logger.debug(f"[CONFIRM] ignored - transcript not yes/no: '{transcript[:50]}...'")
    return "ignored"


def _execute_plan_default(plan: List[Dict[str, Any]], logger) -> None:
    """
    Execute a plan using the default orchestrator path.
    
    This is a fallback when no executor is provided to resolve_pending.
    
    Args:
        plan: The tool plan to execute
        logger: Logger instance
    """
    try:
        from wyzer.core.orchestrator import execute_tool_plan, get_registry
        
        registry = get_registry()
        
        # Execute the plan
        execution_summary, executed_intents = execute_tool_plan(plan, registry)
        logger.info(f"[CONFIRM] executed {len(execution_summary.ran)} tools")
    except Exception as e:
        logger.error(f"[CONFIRM] default execution failed: {e}")


def check_passive_expiry() -> bool:
    """
    Check and clear expired pending confirmations (passive expiry).
    
    Call this from an existing periodic tick (heartbeat) to ensure
    confirmations expire even if the user never speaks again.
    
    Returns:
        True if a pending confirmation was expired and cleared, False otherwise
    """
    logger = get_logger()
    
    from wyzer.context.world_state import get_world_state, _world_state_lock
    
    ws = get_world_state()
    with _world_state_lock:
        pending = ws.pending_confirmation
        if pending is None:
            return False
        
        if time.time() > pending.expires_ts:
            ws.pending_confirmation = None
            logger.info("[CONFIRM] expired (passive)")
            return True
    
    return False


def get_pending_prompt() -> Optional[str]:
    """
    Get the prompt text for the current pending confirmation.
    
    Used when re-asking the user after they said something other than yes/no.
    
    Returns:
        The prompt text, or None if no pending confirmation
    """
    from wyzer.context.world_state import get_pending_confirmation
    
    pending = get_pending_confirmation()
    if pending is None:
        return None
    
    return pending.prompt


def has_active_pending() -> bool:
    """
    Check if there's an active (non-expired) pending confirmation.
    
    This is a convenience function for guards that need to know
    if confirmation handling should take precedence.
    
    Returns:
        True if there's a valid pending confirmation
    """
    from wyzer.context.world_state import get_pending_confirmation
    return get_pending_confirmation() is not None
