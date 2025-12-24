"""wyzer.policy.llm_speech_gate

Phase 11.5 - LLM Speech Gating Module.

Strict gating function that determines whether the LLM may produce
user-facing text. The LLM may speak ONLY if at least one condition is true:

1. A tool result exists and needs explanation
2. The user explicitly asked for reasoning or explanation  
3. Autonomy policy requested evaluation or justification
4. Clarification is required to safely act

If NONE are true → suppress LLM speech entirely.

HARD RULES:
- This module NEVER modifies tool schemas or execution logic
- Gating decisions are deterministic (no LLM)
- All decisions are logged via LLM observability
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SpeechReason(Enum):
    """Enumeration of valid reasons for LLM speech."""
    TOOL_RESULT_EXPLANATION = "tool_result_explanation"
    USER_REQUESTED_REASONING = "user_requested_reasoning"
    AUTONOMY_JUSTIFICATION = "autonomy_justification"
    CLARIFICATION_REQUIRED = "clarification_required"
    REPLY_ONLY_QUERY = "reply_only_query"
    ERROR_REPORTING = "error_reporting"
    SUPPRESSED = "suppressed"


@dataclass
class SpeechGateResult:
    """
    Result of speech gate evaluation.
    
    Fields:
        allowed: Whether LLM speech is allowed
        reason: Why speech is allowed/suppressed
        confidence: Gate confidence (1.0 = certain)
        details: Additional context for logging
    """
    allowed: bool
    reason: SpeechReason
    confidence: float = 1.0
    details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "allowed": self.allowed,
            "reason": self.reason.value,
            "confidence": self.confidence,
            "details": self.details,
        }


# ============================================================================
# REASONING/EXPLANATION REQUEST PATTERNS
# ============================================================================
# These patterns detect when user explicitly asks for reasoning/explanation.
# Match must be explicit - no fuzzy matching.

_REASONING_REQUEST_RE = re.compile(
    r"(?:"
    r"^why\b|"                                     # "why did you..."
    r"^explain\b|"                                 # "explain..."
    r"^how (?:does|do|did|can|could|would)\b|"    # "how does X work"
    r"^what (?:is|are|was|were)\b|"               # "what is X"
    r"^tell me (?:about|why|how)\b|"              # "tell me about X"
    r"^describe\b|"                               # "describe X"
    r"^can you (?:explain|tell me|describe)\b|"   # "can you explain..."
    r"\bwhy\s+(?:is|are|did|do|does|would|should)\b|"  # mid-sentence "why is..."
    r"\bexplain\b|"                               # any "explain" in text
    r"\bhow (?:does|do) (?:it|this|that) work\b"  # "how does this work"
    r")",
    re.IGNORECASE
)

_JUSTIFICATION_REQUEST_RE = re.compile(
    r"(?:"
    r"^why did you\b|"                            # "why did you do that"
    r"^why'd you\b|"                              # contraction
    r"^explain (?:your|that|the)\s*(?:decision|action|choice)\b|"
    r"^what was your (?:reason|reasoning)\b|"
    r"^justify\b"
    r")",
    re.IGNORECASE
)


def _is_reasoning_request(user_text: str) -> bool:
    """Check if user explicitly asked for reasoning/explanation."""
    if not user_text:
        return False
    return bool(_REASONING_REQUEST_RE.search(user_text.strip()))


def _is_justification_request(user_text: str) -> bool:
    """Check if user asked for autonomy justification."""
    if not user_text:
        return False
    return bool(_JUSTIFICATION_REQUEST_RE.search(user_text.strip()))


def _has_tool_result(context: Dict[str, Any]) -> bool:
    """Check if context contains a tool result that needs explanation."""
    if not context:
        return False
    
    # Check for execution summary with results
    exec_summary = context.get("execution_summary")
    if exec_summary:
        ran = exec_summary.get("ran", [])
        if ran and len(ran) > 0:
            # Tool was executed - may need explanation
            return True
    
    # Check for tool_calls in context
    tool_calls = context.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        return True
    
    # Check for tool_result directly
    tool_result = context.get("tool_result")
    if tool_result is not None:
        return True
    
    return False


def _needs_clarification(context: Dict[str, Any]) -> bool:
    """Check if clarification is required to safely act."""
    if not context:
        return False
    
    # Check for clarification flag
    if context.get("needs_clarification"):
        return True
    
    # Check for ambiguous reference
    if context.get("ambiguous_reference"):
        return True
    
    # Check for low confidence routing
    confidence = context.get("confidence", 1.0)
    if confidence < 0.5:
        return True
    
    # Check for multiple possible intents
    if context.get("multiple_intents") and len(context.get("possible_intents", [])) > 1:
        return True
    
    # Check for autonomy policy asking
    autonomy = context.get("autonomy_decision", {})
    if autonomy.get("action") == "ask":
        return True
    
    return False


def _is_autonomy_justification_context(context: Dict[str, Any]) -> bool:
    """Check if autonomy policy requested evaluation or justification."""
    if not context:
        return False
    
    # Check for autonomy evaluation flag
    if context.get("autonomy_evaluation"):
        return True
    
    # Check for justification request from autonomy system
    autonomy = context.get("autonomy_decision", {})
    if autonomy.get("needs_justification"):
        return True
    
    return False


def _is_reply_only_query(context: Dict[str, Any]) -> bool:
    """Check if this is a reply-only query (no tools, just conversation)."""
    if not context:
        return False
    
    # Reply-only mode explicitly set
    if context.get("reply_only"):
        return True
    
    # Continuation phrase
    if context.get("is_continuation"):
        return True
    
    # Informational query
    if context.get("is_informational"):
        return True
    
    return False


def _is_error_context(context: Dict[str, Any]) -> bool:
    """Check if context contains an error that needs reporting."""
    if not context:
        return False
    
    # Check for error flag
    if context.get("error"):
        return True
    
    # Check for tool execution errors
    exec_summary = context.get("execution_summary", {})
    ran = exec_summary.get("ran", [])
    for result in ran:
        if result.get("error") or not result.get("ok", True):
            return True
    
    return False


def llm_may_speak(
    user_text: str,
    context: Optional[Dict[str, Any]] = None,
) -> SpeechGateResult:
    """
    Strict gating function to determine if LLM may produce user-facing text.
    
    The LLM may speak ONLY if at least ONE of these conditions is true:
    1. A tool result exists and needs explanation
    2. The user explicitly asked for reasoning or explanation
    3. Autonomy policy requested evaluation or justification
    4. Clarification is required to safely act
    5. This is a reply-only conversational query
    6. An error occurred that needs reporting
    
    Args:
        user_text: The user's input text
        context: Additional context dict with tool results, autonomy state, etc.
        
    Returns:
        SpeechGateResult with allowed flag and reason
    """
    context = context or {}
    
    # 1. Tool result exists and needs explanation
    if _has_tool_result(context):
        return SpeechGateResult(
            allowed=True,
            reason=SpeechReason.TOOL_RESULT_EXPLANATION,
            details="Tool execution result requires explanation",
        )
    
    # 2. User explicitly asked for reasoning or explanation
    if _is_reasoning_request(user_text):
        return SpeechGateResult(
            allowed=True,
            reason=SpeechReason.USER_REQUESTED_REASONING,
            details="User explicitly requested reasoning/explanation",
        )
    
    # 3. Autonomy policy requested justification
    if _is_autonomy_justification_context(context) or _is_justification_request(user_text):
        return SpeechGateResult(
            allowed=True,
            reason=SpeechReason.AUTONOMY_JUSTIFICATION,
            details="Autonomy policy requires justification",
        )
    
    # 4. Clarification is required to safely act
    if _needs_clarification(context):
        return SpeechGateResult(
            allowed=True,
            reason=SpeechReason.CLARIFICATION_REQUIRED,
            details="Clarification needed for safe action",
        )
    
    # 5. Reply-only conversational query (no tools)
    if _is_reply_only_query(context):
        return SpeechGateResult(
            allowed=True,
            reason=SpeechReason.REPLY_ONLY_QUERY,
            details="Reply-only conversational query",
        )
    
    # 6. Error occurred that needs reporting
    if _is_error_context(context):
        return SpeechGateResult(
            allowed=True,
            reason=SpeechReason.ERROR_REPORTING,
            details="Error requires user notification",
        )
    
    # No valid reason to speak - suppress LLM speech
    return SpeechGateResult(
        allowed=False,
        reason=SpeechReason.SUPPRESSED,
        details="No valid reason for LLM speech - suppressing",
    )


def gate_reply(
    reply: str,
    user_text: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Apply speech gating to an LLM reply.
    
    If speech is not allowed, returns empty string.
    If speech is allowed, returns the original reply.
    
    This function also logs the gating decision via LLM observability.
    
    Args:
        reply: The LLM-generated reply
        user_text: The user's input text
        context: Additional context dict
        
    Returns:
        The reply if allowed, empty string if suppressed
    """
    from wyzer.policy.llm_observability import log_llm_invocation, LLMInvocationLog
    
    gate_result = llm_may_speak(user_text, context)
    
    # Log the gating decision
    log_llm_invocation(LLMInvocationLog(
        invocation_reason=gate_result.reason.value,
        speech_allowed=gate_result.allowed,
        speech_reason=gate_result.details,
        autonomy_involved=_is_autonomy_justification_context(context or {}),
        outcome="spoke" if gate_result.allowed and reply else "silent",
        user_text=user_text[:100] if user_text else "",
        reply_length=len(reply) if reply else 0,
    ))
    
    if gate_result.allowed:
        return reply
    else:
        return ""
