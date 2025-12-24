"""wyzer.policy.llm_observability

Phase 11.5 - LLM Observability Module.

Structured logging for every LLM invocation with:
- Invocation reason (why LLM was called)
- Whether speech was allowed or suppressed
- Whether autonomy logic was involved
- Outcome: acted / asked / deferred / silent

HARD RULES:
- No user-visible debug output - logs only
- Does not modify tool schemas or execution
- All logging is local-first and offline
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from wyzer.core.logger import get_logger


class LLMOutcome(Enum):
    """Possible outcomes of an LLM invocation."""
    ACTED = "acted"           # LLM produced action (tool call)
    ASKED = "asked"           # LLM asked clarification question
    DEFERRED = "deferred"     # LLM deferred to autonomy policy
    SPOKE = "spoke"           # LLM produced speech reply
    SILENT = "silent"         # LLM speech was suppressed


@dataclass
class LLMInvocationLog:
    """
    Structured log entry for an LLM invocation.
    
    Fields:
        timestamp: Unix timestamp of invocation
        invocation_reason: Why the LLM was called
        speech_allowed: Whether speech gating allowed output
        speech_reason: Detailed reason for speech decision
        autonomy_involved: Whether autonomy logic was involved
        outcome: Final outcome of the invocation
        user_text: Truncated user input (for debugging)
        reply_length: Length of generated reply
        latency_ms: LLM inference latency (if available)
        tool_called: Tool name if a tool was called
        confidence: Router/LLM confidence score
        suppression_reason: Why speech was suppressed (if applicable)
    """
    invocation_reason: str
    speech_allowed: bool
    autonomy_involved: bool
    outcome: str
    user_text: str = ""
    reply_length: int = 0
    latency_ms: int = 0
    tool_called: Optional[str] = None
    confidence: float = 1.0
    suppression_reason: str = ""
    speech_reason: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return asdict(self)
    
    def to_log_line(self) -> str:
        """Format as single-line log entry."""
        parts = [
            f"reason={self.invocation_reason}",
            f"speech={'allowed' if self.speech_allowed else 'SUPPRESSED'}",
            f"autonomy={self.autonomy_involved}",
            f"outcome={self.outcome}",
        ]
        
        if self.tool_called:
            parts.append(f"tool={self.tool_called}")
        
        if self.latency_ms > 0:
            parts.append(f"latency_ms={self.latency_ms}")
        
        if self.reply_length > 0:
            parts.append(f"reply_len={self.reply_length}")
        
        if self.confidence < 1.0:
            parts.append(f"confidence={self.confidence:.2f}")
        
        if self.suppression_reason:
            parts.append(f"suppression={self.suppression_reason}")
        
        return " ".join(parts)


# Module-level invocation history (limited to last N entries for memory)
_invocation_history: List[LLMInvocationLog] = []
_MAX_HISTORY_SIZE = 100


def log_llm_invocation(entry: LLMInvocationLog) -> None:
    """
    Log an LLM invocation with structured data.
    
    Logs to file/console via logger and maintains in-memory history.
    No user-visible output - logs only.
    
    Args:
        entry: The invocation log entry
    """
    global _invocation_history
    
    logger = get_logger()
    
    # Add to history (bounded)
    _invocation_history.append(entry)
    if len(_invocation_history) > _MAX_HISTORY_SIZE:
        _invocation_history = _invocation_history[-_MAX_HISTORY_SIZE:]
    
    # Log structured entry (INFO level, filtered in quiet mode)
    log_line = entry.to_log_line()
    logger.info(f"[LLM_OBS] {log_line}")


def log_tool_execution(
    tool_name: str,
    success: bool,
    latency_ms: int = 0,
    error: Optional[str] = None,
) -> None:
    """
    Log a tool execution event.
    
    Args:
        tool_name: Name of the tool executed
        success: Whether execution succeeded
        latency_ms: Execution latency in milliseconds
        error: Error message if failed
    """
    logger = get_logger()
    
    outcome = "success" if success else "failed"
    parts = [
        f"tool={tool_name}",
        f"outcome={outcome}",
        f"latency_ms={latency_ms}",
    ]
    
    if error:
        parts.append(f"error={error[:50]}")
    
    logger.info(f"[TOOL_OBS] {' '.join(parts)}")


def log_speech_gate(
    allowed: bool,
    reason: str,
    user_text: str = "",
    reply_preview: str = "",
) -> None:
    """
    Log a speech gating decision.
    
    Args:
        allowed: Whether speech was allowed
        reason: Reason for the decision
        user_text: Truncated user input
        reply_preview: Preview of the reply (if any)
    """
    logger = get_logger()
    
    status = "ALLOWED" if allowed else "SUPPRESSED"
    parts = [
        f"speech={status}",
        f"reason={reason}",
    ]
    
    if user_text:
        parts.append(f'input="{user_text[:30]}..."')
    
    if reply_preview and allowed:
        parts.append(f'reply_preview="{reply_preview[:30]}..."')
    
    logger.info(f"[SPEECH_GATE] {' '.join(parts)}")


def log_autonomy_decision(
    action: str,
    confidence: float,
    risk: str,
    mode: str,
    reason: str = "",
) -> None:
    """
    Log an autonomy policy decision.
    
    Args:
        action: The action taken (execute/ask/deny)
        confidence: Router confidence score
        risk: Risk classification (low/medium/high)
        mode: Autonomy mode (off/low/normal/high)
        reason: Detailed reason for decision
    """
    logger = get_logger()
    
    parts = [
        f"action={action}",
        f"confidence={confidence:.2f}",
        f"risk={risk}",
        f"mode={mode}",
    ]
    
    if reason:
        parts.append(f"reason={reason[:50]}")
    
    logger.info(f"[AUTONOMY_OBS] {' '.join(parts)}")


def get_recent_invocations(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent LLM invocations for debugging/analysis.
    
    Args:
        limit: Maximum number of entries to return
        
    Returns:
        List of invocation log entries as dicts
    """
    global _invocation_history
    return [entry.to_dict() for entry in _invocation_history[-limit:]]


def get_invocation_stats() -> Dict[str, Any]:
    """
    Get statistics about LLM invocations.
    
    Returns:
        Dict with counts by outcome, speech status, etc.
    """
    global _invocation_history
    
    if not _invocation_history:
        return {
            "total": 0,
            "by_outcome": {},
            "speech_allowed": 0,
            "speech_suppressed": 0,
            "autonomy_involved": 0,
        }
    
    by_outcome: Dict[str, int] = {}
    speech_allowed = 0
    speech_suppressed = 0
    autonomy_involved = 0
    
    for entry in _invocation_history:
        outcome = entry.outcome
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        
        if entry.speech_allowed:
            speech_allowed += 1
        else:
            speech_suppressed += 1
        
        if entry.autonomy_involved:
            autonomy_involved += 1
    
    return {
        "total": len(_invocation_history),
        "by_outcome": by_outcome,
        "speech_allowed": speech_allowed,
        "speech_suppressed": speech_suppressed,
        "autonomy_involved": autonomy_involved,
    }


def clear_history() -> None:
    """Clear invocation history (for testing)."""
    global _invocation_history
    _invocation_history = []
