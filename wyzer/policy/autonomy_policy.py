"""wyzer.policy.autonomy_policy

Phase 11 - Autonomy Policy Assessment Module.

Deterministic policy layer that decides whether to:
- Execute immediately (action="execute")
- Ask a clarification question (action="ask")
- Deny/refuse the action (action="deny")

Based on:
- Router confidence (float 0.0-1.0)
- Risk classification (low/medium/high)
- Autonomy mode (off/low/normal/high)
- Configuration flags (AUTONOMY_CONFIRM_SENSITIVE)

HARD RULES:
- Mode OFF preserves current behavior exactly (no new confirmations)
- High-risk actions require confirmation unless explicitly overridden
- All decisions are deterministic (no LLM)
- Policy never modifies tool routing or schemas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict


# ============================================================================
# TYPES
# ============================================================================

class AutonomyDecision(TypedDict):
    """
    Result of autonomy policy assessment.
    
    Fields:
        action: What to do - "execute", "ask", or "deny"
        reason: Human-readable explanation of the decision
        question: Clarification question to ask (if action="ask")
        confidence: Router confidence that was evaluated
        risk: Risk classification that was evaluated
        needs_confirmation: Whether this requires yes/no confirmation (high-risk)
    """
    action: Literal["execute", "ask", "deny"]
    reason: str
    question: Optional[str]
    confidence: float
    risk: str  # low|medium|high
    needs_confirmation: bool


# Valid autonomy modes
AUTONOMY_MODES = frozenset({"off", "low", "normal", "high"})


# ============================================================================
# MODE THRESHOLDS
# ============================================================================
# These thresholds define when to execute vs ask vs deny based on confidence.
# Each mode has different risk tolerance.

# Mode LOW: Very conservative
# - Execute low/medium risk only if confidence >= 0.95
# - High risk: always ask with confirmation
LOW_THRESHOLD_EXECUTE = 0.95
LOW_THRESHOLD_ASK = 0.0  # Always ask if below execute threshold

# Mode NORMAL: Balanced
# - Execute low/medium risk if confidence >= 0.90
# - Ask if 0.75 <= confidence < 0.90
# - Deny if confidence < 0.75 (but prefer ask if possible)
# - High risk: always ask with confirmation
NORMAL_THRESHOLD_EXECUTE = 0.90
NORMAL_THRESHOLD_ASK = 0.75
NORMAL_THRESHOLD_DENY = 0.75  # Below this, deny or ask

# Mode HIGH: More permissive
# - Execute low/medium risk if confidence >= 0.85
# - Ask if 0.70 <= confidence < 0.85
# - Deny if confidence < 0.70 (but prefer ask if possible)
# - High risk: depends on AUTONOMY_CONFIRM_SENSITIVE
HIGH_THRESHOLD_EXECUTE = 0.85
HIGH_THRESHOLD_ASK = 0.70
HIGH_THRESHOLD_DENY = 0.70

# For high mode with AUTONOMY_CONFIRM_SENSITIVE=False:
# Execute high-risk only if confidence >= 0.97
HIGH_MODE_SENSITIVE_THRESHOLD = 0.97


# ============================================================================
# POLICY FUNCTIONS
# ============================================================================

def assess(
    tool_plan: List[Dict[str, Any]],
    confidence: float,
    mode: str,
    risk: str,
    confirm_sensitive: bool = True,
) -> AutonomyDecision:
    """
    Assess a tool plan and return an autonomy decision.
    
    Args:
        tool_plan: List of tool call dicts
        confidence: Router confidence (0.0-1.0)
        mode: Autonomy mode ("off", "low", "normal", "high")
        risk: Risk classification ("low", "medium", "high")
        confirm_sensitive: Whether to require confirmation for high-risk in high mode
        
    Returns:
        AutonomyDecision with action, reason, and metadata
    """
    # Validate inputs
    mode = mode.lower().strip() if mode else "off"
    if mode not in AUTONOMY_MODES:
        mode = "off"
    
    risk = risk.lower().strip() if risk else "low"
    if risk not in ("low", "medium", "high"):
        risk = "low"
    
    confidence = max(0.0, min(1.0, float(confidence) if confidence else 0.0))
    
    # ==========================================================================
    # MODE OFF: Preserve current behavior exactly
    # ==========================================================================
    if mode == "off":
        return _assess_off_mode(tool_plan, confidence, risk)
    
    # ==========================================================================
    # AUTONOMY ACTIVE: Apply policy based on mode
    # ==========================================================================
    if mode == "low":
        return _assess_low_mode(tool_plan, confidence, risk)
    elif mode == "normal":
        return _assess_normal_mode(tool_plan, confidence, risk)
    elif mode == "high":
        return _assess_high_mode(tool_plan, confidence, risk, confirm_sensitive)
    
    # Fallback (should not reach here)
    return _assess_off_mode(tool_plan, confidence, risk)


def _assess_off_mode(
    tool_plan: List[Dict[str, Any]],
    confidence: float,
    risk: str,
) -> AutonomyDecision:
    """
    OFF mode: Preserve current behavior exactly.
    
    Always returns execute with no confirmation requirements.
    This ensures autonomy=off doesn't change any existing behavior.
    """
    return AutonomyDecision(
        action="execute",
        reason="Autonomy off - current behavior preserved",
        question=None,
        confidence=confidence,
        risk=risk,
        needs_confirmation=False,
    )


def _assess_low_mode(
    tool_plan: List[Dict[str, Any]],
    confidence: float,
    risk: str,
) -> AutonomyDecision:
    """
    LOW mode: Very conservative.
    
    - Low/medium risk: execute only if confidence >= 0.95, else ask
    - High risk: always ask with confirmation
    """
    if risk == "high":
        # High risk: always ask with confirmation in low mode
        return AutonomyDecision(
            action="ask",
            reason=f"High-risk action requires confirmation (mode=low)",
            question=_generate_confirmation_question(tool_plan, risk),
            confidence=confidence,
            risk=risk,
            needs_confirmation=True,
        )
    
    # Low/medium risk
    if confidence >= LOW_THRESHOLD_EXECUTE:
        return AutonomyDecision(
            action="execute",
            reason=f"Confidence {confidence:.2f} >= {LOW_THRESHOLD_EXECUTE} threshold (mode=low, risk={risk})",
            question=None,
            confidence=confidence,
            risk=risk,
            needs_confirmation=False,
        )
    else:
        return AutonomyDecision(
            action="ask",
            reason=f"Confidence {confidence:.2f} < {LOW_THRESHOLD_EXECUTE} threshold (mode=low, risk={risk})",
            question=_generate_clarification_question(tool_plan, confidence),
            confidence=confidence,
            risk=risk,
            needs_confirmation=False,
        )


def _assess_normal_mode(
    tool_plan: List[Dict[str, Any]],
    confidence: float,
    risk: str,
) -> AutonomyDecision:
    """
    NORMAL mode: Balanced.
    
    - Low/medium risk: execute >= 0.90, ask 0.75-0.89, deny < 0.75
    - High risk: always ask with confirmation
    """
    if risk == "high":
        # High risk: always ask with confirmation in normal mode
        return AutonomyDecision(
            action="ask",
            reason=f"High-risk action requires confirmation (mode=normal)",
            question=_generate_confirmation_question(tool_plan, risk),
            confidence=confidence,
            risk=risk,
            needs_confirmation=True,
        )
    
    # Low/medium risk
    if confidence >= NORMAL_THRESHOLD_EXECUTE:
        return AutonomyDecision(
            action="execute",
            reason=f"Confidence {confidence:.2f} >= {NORMAL_THRESHOLD_EXECUTE} threshold (mode=normal, risk={risk})",
            question=None,
            confidence=confidence,
            risk=risk,
            needs_confirmation=False,
        )
    elif confidence >= NORMAL_THRESHOLD_ASK:
        return AutonomyDecision(
            action="ask",
            reason=f"Confidence {confidence:.2f} in ask range [{NORMAL_THRESHOLD_ASK}, {NORMAL_THRESHOLD_EXECUTE}) (mode=normal, risk={risk})",
            question=_generate_clarification_question(tool_plan, confidence),
            confidence=confidence,
            risk=risk,
            needs_confirmation=False,
        )
    else:
        # Very low confidence - prefer ask over deny for better UX
        return AutonomyDecision(
            action="ask",
            reason=f"Confidence {confidence:.2f} < {NORMAL_THRESHOLD_ASK} - asking for clarification (mode=normal, risk={risk})",
            question=_generate_clarification_question(tool_plan, confidence),
            confidence=confidence,
            risk=risk,
            needs_confirmation=False,
        )


def _assess_high_mode(
    tool_plan: List[Dict[str, Any]],
    confidence: float,
    risk: str,
    confirm_sensitive: bool,
) -> AutonomyDecision:
    """
    HIGH mode: More permissive.
    
    - Low/medium risk: execute >= 0.85, ask 0.70-0.84, deny < 0.70
    - High risk: 
        - If confirm_sensitive=True: ask with confirmation
        - If confirm_sensitive=False: execute only if >= 0.97, else ask
    """
    if risk == "high":
        if confirm_sensitive:
            # High risk with confirm_sensitive: ask with confirmation
            return AutonomyDecision(
                action="ask",
                reason=f"High-risk action requires confirmation (mode=high, confirm_sensitive=True)",
                question=_generate_confirmation_question(tool_plan, risk),
                confidence=confidence,
                risk=risk,
                needs_confirmation=True,
            )
        else:
            # High risk without confirm_sensitive: execute only if very high confidence
            if confidence >= HIGH_MODE_SENSITIVE_THRESHOLD:
                return AutonomyDecision(
                    action="execute",
                    reason=f"High-risk allowed: confidence {confidence:.2f} >= {HIGH_MODE_SENSITIVE_THRESHOLD} (mode=high, confirm_sensitive=False)",
                    question=None,
                    confidence=confidence,
                    risk=risk,
                    needs_confirmation=False,
                )
            else:
                return AutonomyDecision(
                    action="ask",
                    reason=f"High-risk: confidence {confidence:.2f} < {HIGH_MODE_SENSITIVE_THRESHOLD} (mode=high, confirm_sensitive=False)",
                    question=_generate_confirmation_question(tool_plan, risk),
                    confidence=confidence,
                    risk=risk,
                    needs_confirmation=True,
                )
    
    # Low/medium risk
    if confidence >= HIGH_THRESHOLD_EXECUTE:
        return AutonomyDecision(
            action="execute",
            reason=f"Confidence {confidence:.2f} >= {HIGH_THRESHOLD_EXECUTE} threshold (mode=high, risk={risk})",
            question=None,
            confidence=confidence,
            risk=risk,
            needs_confirmation=False,
        )
    elif confidence >= HIGH_THRESHOLD_ASK:
        return AutonomyDecision(
            action="ask",
            reason=f"Confidence {confidence:.2f} in ask range [{HIGH_THRESHOLD_ASK}, {HIGH_THRESHOLD_EXECUTE}) (mode=high, risk={risk})",
            question=_generate_clarification_question(tool_plan, confidence),
            confidence=confidence,
            risk=risk,
            needs_confirmation=False,
        )
    else:
        # Very low confidence - prefer ask over deny for better UX
        return AutonomyDecision(
            action="ask",
            reason=f"Confidence {confidence:.2f} < {HIGH_THRESHOLD_ASK} - asking for clarification (mode=high, risk={risk})",
            question=_generate_clarification_question(tool_plan, confidence),
            confidence=confidence,
            risk=risk,
            needs_confirmation=False,
        )


def _generate_clarification_question(
    tool_plan: List[Dict[str, Any]],
    confidence: float,
) -> str:
    """
    Generate a short clarification question when uncertain.
    
    Args:
        tool_plan: The planned tools
        confidence: Router confidence
        
    Returns:
        Short question string
    """
    if not tool_plan:
        return "What would you like me to do?"
    
    # Get first tool for context
    first_tool = tool_plan[0] if tool_plan else {}
    tool_name = first_tool.get("tool", "")
    args = first_tool.get("args", {})
    
    # Generate context-aware question based on tool type
    if tool_name in ("open_target", "open_app", "launch_app"):
        target = args.get("query", args.get("target", args.get("app", "")))
        if target:
            return f"Did you want me to open {target}?"
        return "What did you want me to open?"
    
    if tool_name == "close_window":
        target = args.get("process", args.get("title", args.get("query", "")))
        if target:
            return f"Did you want me to close {target}?"
        return "Which window should I close?"
    
    if tool_name in ("focus_window", "minimize_window", "maximize_window"):
        target = args.get("process", args.get("title", args.get("query", "")))
        if target:
            action = tool_name.replace("_window", "").replace("_", " ")
            return f"Did you want me to {action} {target}?"
        return "Which window?"
    
    if tool_name == "volume_control":
        return "What volume level?"
    
    if tool_name == "timer":
        return "How long for the timer?"
    
    # Generic fallback
    if len(tool_plan) > 1:
        return "I understood multiple commands. Should I proceed?"
    
    return "Can you clarify what you'd like me to do?"


def _generate_confirmation_question(
    tool_plan: List[Dict[str, Any]],
    risk: str,
) -> str:
    """
    Generate a confirmation prompt for high-risk actions.
    
    Args:
        tool_plan: The planned tools
        risk: Risk level
        
    Returns:
        Confirmation prompt string
    """
    if not tool_plan:
        return "Are you sure you want me to proceed?"
    
    # Get first tool for context
    first_tool = tool_plan[0] if tool_plan else {}
    tool_name = first_tool.get("tool", "")
    args = first_tool.get("args", {})
    
    # Generate context-aware confirmation based on tool
    if "delete" in tool_name or "remove" in tool_name:
        target = args.get("path", args.get("file", args.get("target", "")))
        if target:
            return f"This will delete {target}. Are you sure?"
        return "This will delete data. Are you sure?"
    
    if "shutdown" in tool_name or "restart" in tool_name or "reboot" in tool_name:
        return "This will restart or shut down your computer. Are you sure?"
    
    if "kill" in tool_name or "terminate" in tool_name:
        target = args.get("process", args.get("name", ""))
        if target:
            return f"This will force-close {target}. Are you sure?"
        return "This will force-close a process. Are you sure?"
    
    # Generic high-risk confirmation
    return "This action may be destructive. Do you want me to proceed?"


def format_decision_for_speech(decision: AutonomyDecision) -> str:
    """
    Format an autonomy decision for spoken output.
    
    Used by "why did you do that" command.
    
    Args:
        decision: The autonomy decision
        
    Returns:
        Human-readable explanation
    """
    action = decision["action"]
    reason = decision["reason"]
    confidence = decision["confidence"]
    risk = decision["risk"]
    
    if action == "execute":
        return f"I executed because confidence was {confidence:.0%} with {risk} risk. {reason}"
    elif action == "ask":
        if decision["needs_confirmation"]:
            return f"I asked for confirmation because of {risk} risk. {reason}"
        else:
            return f"I asked for clarification because confidence was {confidence:.0%}. {reason}"
    else:  # deny
        return f"I declined because confidence was too low at {confidence:.0%}. {reason}"


def summarize_plan(tool_plan: List[Dict[str, Any]]) -> str:
    """
    Create a short summary of a tool plan for logging.
    
    Args:
        tool_plan: List of tool call dicts
        
    Returns:
        Short summary string
    """
    if not tool_plan:
        return "(empty plan)"
    
    tools = [intent.get("tool", "?") for intent in tool_plan if isinstance(intent, dict)]
    if len(tools) == 0:
        return "(no tools)"
    elif len(tools) == 1:
        return tools[0]
    elif len(tools) <= 3:
        return " + ".join(tools)
    else:
        return f"{tools[0]} + {len(tools) - 1} more"
