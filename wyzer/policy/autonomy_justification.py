"""wyzer.policy.autonomy_justification

Phase 11.5 - Autonomy Justification Contract.

When autonomy mode is enabled and an action is evaluated, the LLM may
ONLY justify decisions using explicit inputs:
- Confidence score
- Risk level (low/medium/high)
- Autonomy mode (off/low/normal/high)
- Policy thresholds

FORBIDDEN:
- Emotional language ("I felt", "I thought it would be nice")
- Speculation ("you probably wanted", "I assumed")
- Assumptions not backed by explicit data
- Making up reasons not derived from policy

HARD RULES:
- All justifications are deterministic (no LLM)
- Justifications cite exact confidence and threshold values
- No personality or opinion in justifications
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wyzer.policy.autonomy_policy import AutonomyDecision


# ============================================================================
# JUSTIFICATION TEMPLATES
# ============================================================================
# These templates generate deterministic, factual justifications.
# No emotional language, no speculation, just policy facts.

EXECUTE_JUSTIFICATION_TEMPLATE = """Decision: EXECUTE
Confidence: {confidence:.0%}
Risk Level: {risk}
Autonomy Mode: {mode}
Threshold: {threshold:.0%}
Reason: Confidence exceeded threshold for {risk}-risk action in {mode} mode."""

ASK_CLARIFICATION_TEMPLATE = """Decision: ASK (clarification)
Confidence: {confidence:.0%}
Risk Level: {risk}
Autonomy Mode: {mode}
Threshold Range: {low_threshold:.0%} - {high_threshold:.0%}
Reason: Confidence fell within clarification range; asked for confirmation."""

ASK_CONFIRMATION_TEMPLATE = """Decision: ASK (confirmation)
Confidence: {confidence:.0%}
Risk Level: {risk}
Autonomy Mode: {mode}
Reason: High-risk action requires explicit user confirmation per policy."""

DENY_JUSTIFICATION_TEMPLATE = """Decision: DENY
Confidence: {confidence:.0%}
Risk Level: {risk}
Autonomy Mode: {mode}
Threshold: {threshold:.0%}
Reason: Confidence below minimum threshold for action."""

OFF_MODE_TEMPLATE = """Decision: EXECUTE (autonomy off)
Autonomy Mode: off
Reason: Autonomy is disabled; current behavior preserved."""


def get_justification(
    decision: "AutonomyDecision",
    mode: str,
    tool_plan: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Generate a deterministic justification for an autonomy decision.
    
    Uses ONLY explicit policy inputs - no speculation or assumptions.
    
    Args:
        decision: The autonomy decision that was made
        mode: The autonomy mode (off/low/normal/high)
        tool_plan: Optional tool plan for context
        
    Returns:
        Factual justification string
    """
    action = decision.get("action", "execute")
    confidence = decision.get("confidence", 0.0)
    risk = decision.get("risk", "low")
    needs_confirmation = decision.get("needs_confirmation", False)
    
    # Import thresholds from policy
    from wyzer.policy.autonomy_policy import (
        LOW_THRESHOLD_EXECUTE,
        NORMAL_THRESHOLD_EXECUTE,
        NORMAL_THRESHOLD_ASK,
        HIGH_THRESHOLD_EXECUTE,
        HIGH_THRESHOLD_ASK,
        HIGH_MODE_SENSITIVE_THRESHOLD,
    )
    
    if mode == "off":
        return OFF_MODE_TEMPLATE
    
    if action == "execute":
        # Determine which threshold was used
        if mode == "low":
            threshold = LOW_THRESHOLD_EXECUTE
        elif mode == "normal":
            threshold = NORMAL_THRESHOLD_EXECUTE
        elif mode == "high":
            if risk == "high":
                threshold = HIGH_MODE_SENSITIVE_THRESHOLD
            else:
                threshold = HIGH_THRESHOLD_EXECUTE
        else:
            threshold = 0.9
        
        return EXECUTE_JUSTIFICATION_TEMPLATE.format(
            confidence=confidence,
            risk=risk,
            mode=mode,
            threshold=threshold,
        )
    
    elif action == "ask":
        if needs_confirmation:
            return ASK_CONFIRMATION_TEMPLATE.format(
                confidence=confidence,
                risk=risk,
                mode=mode,
            )
        else:
            # Clarification - show threshold range
            if mode == "low":
                low_threshold = 0.0
                high_threshold = LOW_THRESHOLD_EXECUTE
            elif mode == "normal":
                low_threshold = NORMAL_THRESHOLD_ASK
                high_threshold = NORMAL_THRESHOLD_EXECUTE
            elif mode == "high":
                low_threshold = HIGH_THRESHOLD_ASK
                high_threshold = HIGH_THRESHOLD_EXECUTE
            else:
                low_threshold = 0.75
                high_threshold = 0.90
            
            return ASK_CLARIFICATION_TEMPLATE.format(
                confidence=confidence,
                risk=risk,
                mode=mode,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
    
    elif action == "deny":
        # Determine which threshold was failed
        if mode == "low":
            threshold = 0.0  # Low mode doesn't really deny
        elif mode == "normal":
            threshold = NORMAL_THRESHOLD_ASK
        elif mode == "high":
            threshold = HIGH_THRESHOLD_ASK
        else:
            threshold = 0.75
        
        return DENY_JUSTIFICATION_TEMPLATE.format(
            confidence=confidence,
            risk=risk,
            mode=mode,
            threshold=threshold,
        )
    
    # Fallback
    return f"Decision: {action.upper()}\nConfidence: {confidence:.0%}\nRisk: {risk}\nMode: {mode}"


def get_brief_justification(
    decision: "AutonomyDecision",
    mode: str,
) -> str:
    """
    Generate a brief, spoken justification for an autonomy decision.
    
    This is for voice output - short and factual.
    
    Args:
        decision: The autonomy decision
        mode: The autonomy mode
        
    Returns:
        Brief spoken justification
    """
    action = decision.get("action", "execute")
    confidence = decision.get("confidence", 0.0)
    risk = decision.get("risk", "low")
    needs_confirmation = decision.get("needs_confirmation", False)
    
    if mode == "off":
        return "Autonomy is off. I followed standard behavior."
    
    confidence_pct = int(confidence * 100)
    
    if action == "execute":
        return f"I acted because my confidence was {confidence_pct}% with {risk} risk."
    
    elif action == "ask":
        if needs_confirmation:
            return f"I asked because {risk}-risk actions need confirmation."
        else:
            return f"I asked because my confidence was {confidence_pct}%, below the threshold."
    
    elif action == "deny":
        return f"I declined because my confidence was only {confidence_pct}%."
    
    return f"My confidence was {confidence_pct}% with {risk} risk."


def validate_justification(justification: str) -> bool:
    """
    Validate that a justification follows the contract.
    
    Checks for forbidden patterns:
    - Emotional language
    - Speculation
    - Assumptions
    
    Args:
        justification: The justification text to validate
        
    Returns:
        True if justification is valid, False if it violates contract
    """
    import re
    
    # Forbidden patterns - emotional/speculative language
    forbidden_patterns = [
        r"\bI felt\b",
        r"\bI thought\b",
        r"\bI assumed\b",
        r"\bI believed\b",
        r"\bI guessed\b",
        r"\bI imagined\b",
        r"\bprobably\b",
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bmight have\b",
        r"\bcould have\b",
        r"\bseemed like\b",
        r"\bappeared to\b",
        r"\bI wanted\b",
        r"\bI hoped\b",
        r"\bI wished\b",
        r"\bnice\b",
        r"\bgreat\b",
        r"\bhelpful\b",
        r"\bfriendly\b",
    ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, justification, re.IGNORECASE):
            return False
    
    return True


def sanitize_justification(justification: str) -> str:
    """
    Sanitize a justification to remove forbidden patterns.
    
    If the justification contains forbidden language, this function
    returns a generic factual replacement.
    
    Args:
        justification: The justification to sanitize
        
    Returns:
        Sanitized justification
    """
    if validate_justification(justification):
        return justification
    
    # Replace with generic factual statement
    return "Decision made according to configured autonomy policy and confidence thresholds."


# ============================================================================
# LAST DECISION TRACKING
# ============================================================================
# Track the last autonomy decision for "why did you do that" queries.

_last_decision: Optional["AutonomyDecision"] = None
_last_mode: Optional[str] = None
_last_tool_plan: Optional[List[Dict[str, Any]]] = None


def record_decision(
    decision: "AutonomyDecision",
    mode: str,
    tool_plan: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Record the last autonomy decision for later retrieval.
    
    Args:
        decision: The autonomy decision
        mode: The autonomy mode
        tool_plan: The tool plan that was evaluated
    """
    global _last_decision, _last_mode, _last_tool_plan
    _last_decision = decision
    _last_mode = mode
    _last_tool_plan = tool_plan
    
    # Log via observability
    from wyzer.policy.llm_observability import log_autonomy_decision
    log_autonomy_decision(
        action=decision.get("action", "execute"),
        confidence=decision.get("confidence", 0.0),
        risk=decision.get("risk", "low"),
        mode=mode,
        reason=decision.get("reason", ""),
    )


def get_last_decision_justification() -> Optional[str]:
    """
    Get the justification for the last autonomy decision.
    
    Returns:
        Justification string, or None if no decision recorded
    """
    global _last_decision, _last_mode
    
    if _last_decision is None or _last_mode is None:
        return None
    
    return get_justification(_last_decision, _last_mode, _last_tool_plan)


def get_last_decision_brief() -> Optional[str]:
    """
    Get a brief spoken justification for the last autonomy decision.
    
    Returns:
        Brief justification, or None if no decision recorded
    """
    global _last_decision, _last_mode
    
    if _last_decision is None or _last_mode is None:
        return None
    
    return get_brief_justification(_last_decision, _last_mode)


def clear_last_decision() -> None:
    """Clear the last decision record."""
    global _last_decision, _last_mode, _last_tool_plan
    _last_decision = None
    _last_mode = None
    _last_tool_plan = None
