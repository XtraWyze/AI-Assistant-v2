"""
Intent plan normalization and validation for multi-intent commands.
Part of Phase 6 - Multi-Intent Support.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


# Maximum number of intents allowed per request
MAX_INTENTS = 7


@dataclass
class Intent:
    """Represents a single intent/tool call"""
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    continue_on_error: bool = False


@dataclass
class IntentPlan:
    """Normalized intent plan from LLM output"""
    reply: str
    intents: List[Intent]
    confidence: float = 0.8


@dataclass
class ExecutionResult:
    """Result of executing a single intent"""
    tool: str
    ok: bool
    result: Optional[Dict[str, Any]] = None
    error: Any = None  # Can be dict with 'type'/'message', string, or None


@dataclass
class ExecutionSummary:
    """Summary of multi-intent execution"""
    ran: List[ExecutionResult]
    stopped_early: bool = False


def normalize_plan(model_output: Dict[str, Any]) -> IntentPlan:
    """
    Normalize LLM output to standard IntentPlan format.
    
    Handles backward compatibility with legacy formats:
    - {"tool": "...", "args": {...}}  -> Single intent
    - {"intent": {...}}               -> Single intent
    - {"intents": [...]}              -> Multi-intent (new format)
    
    Args:
        model_output: Raw LLM response dict
        
    Returns:
        IntentPlan with normalized intents list
    """
    reply = model_output.get("reply", "")
    confidence = model_output.get("confidence", 0.8)
    intents_list: List[Intent] = []
    
    # Check for new multi-intent format
    if "intents" in model_output and model_output["intents"]:
        raw_intents = model_output["intents"]
        if isinstance(raw_intents, list):
            for raw_intent in raw_intents:
                if isinstance(raw_intent, dict):
                    tool_name = raw_intent.get("tool", "")
                    if tool_name:
                        intents_list.append(Intent(
                            tool=tool_name,
                            args=raw_intent.get("args", {}),
                            continue_on_error=raw_intent.get("continue_on_error", False)
                        ))
    
    # Check for legacy single intent format ({"intent": {...}})
    elif "intent" in model_output and model_output["intent"]:
        intent_obj = model_output["intent"]
        if isinstance(intent_obj, dict):
            tool_name = intent_obj.get("tool", "")
            if tool_name:
                intents_list.append(Intent(
                    tool=tool_name,
                    args=intent_obj.get("args", {}),
                    continue_on_error=False
                ))
    
    # Check for legacy single tool format ({"tool": "..."})
    elif "tool" in model_output and model_output["tool"]:
        tool_name = model_output["tool"]
        if isinstance(tool_name, str) and tool_name:
            intents_list.append(Intent(
                tool=tool_name,
                args=model_output.get("args", {}),
                continue_on_error=False
            ))
    
    return IntentPlan(
        reply=reply,
        intents=intents_list,
        confidence=confidence
    )


def filter_unknown_tools(intents: List[Intent], tool_registry) -> tuple[List[Intent], List[str]]:
    """
    Filter out intents with unknown tools (non-fatal handling).
    
    Instead of throwing an error for unknown tools, we filter them out
    and return info about what was removed. This allows partial execution
    of valid intents even when some are invalid.
    
    Args:
        intents: List of Intent objects
        tool_registry: ToolRegistry instance for checking tool existence
        
    Returns:
        Tuple of (valid_intents, unknown_tool_names)
    """
    valid = []
    unknown = []
    
    for intent in intents:
        if not isinstance(intent.tool, str) or not intent.tool:
            unknown.append("<empty>")
            continue
        if not tool_registry.has_tool(intent.tool):
            unknown.append(intent.tool)
            continue
        valid.append(intent)
    
    return valid, unknown


def validate_intents(intents: List[Intent], tool_registry) -> None:
    """
    Validate intent list against tool registry and constraints.
    
    NOTE: As of unknown-tool graceful handling, this function no longer
    throws for unknown tools. Use filter_unknown_tools() first to remove
    unknown tools, then call this to validate the remaining intents.
    
    Args:
        intents: List of Intent objects
        tool_registry: ToolRegistry instance for checking tool existence
        
    Raises:
        ValueError: If validation fails with descriptive message
    """
    # Check max intents limit
    if len(intents) > MAX_INTENTS:
        raise ValueError(
            f"Too many intents requested ({len(intents)}). "
            f"Maximum allowed: {MAX_INTENTS}"
        )
    
    # Validate each intent
    for idx, intent in enumerate(intents):
        # Check tool name is non-empty string
        if not isinstance(intent.tool, str) or not intent.tool:
            raise ValueError(
                f"Intent #{idx + 1}: Tool name must be a non-empty string"
            )
        
        # NOTE: Unknown tool check removed - handled by filter_unknown_tools()
        # This allows graceful degradation instead of hard failure.
        
        # Check args is a dict
        if not isinstance(intent.args, dict):
            raise ValueError(
                f"Intent #{idx + 1}: Tool '{intent.tool}' args must be a dict, "
                f"got {type(intent.args).__name__}"
            )
        
        # Check continue_on_error is bool
        if not isinstance(intent.continue_on_error, bool):
            raise ValueError(
                f"Intent #{idx + 1}: Tool '{intent.tool}' continue_on_error "
                f"must be a bool"
            )
