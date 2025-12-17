"""
Orchestrator for Wyzer AI Assistant - Phase 6
Coordinates LLM reasoning and tool execution.
Supports multi-intent commands (Phase 6 enhancement).
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
from typing import Dict, Any, Optional, List, Tuple
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


def _get_no_ollama_reply() -> str:
    """Get a random short reply for unsupported commands in no-ollama mode."""
    return random.choice(_NO_OLLAMA_FALLBACK_REPLIES)


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
    
    try:
        registry = get_registry()

        # Hybrid router FIRST: deterministic tool plans for obvious commands; LLM otherwise.
        hybrid_decision = hybrid_router.decide(text)
        if hybrid_decision.mode == "tool_plan" and hybrid_decision.intents:
            if _hybrid_tool_plan_is_registered(hybrid_decision.intents, registry):
                logger.info(f"[HYBRID] route=tool_plan confidence={hybrid_decision.confidence:.2f}")
                execution_summary, executed_intents = execute_tool_plan(hybrid_decision.intents, registry)
                # Always use _format_fastpath_reply to properly handle tool execution failures.
                # The hybrid_decision.reply is just a prediction; we need to verify execution succeeded.
                reply = _format_fastpath_reply(
                    text, executed_intents, execution_summary
                )

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
            
            # Validate all intents before execution
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
    """Execute a list of tool-call dicts using the existing intent pipeline."""
    converted: List[Intent] = []
    for raw in intents or []:
        tool = (raw or {}).get("tool")
        args = (raw or {}).get("args")
        continue_on_error = bool((raw or {}).get("continue_on_error", False))
        if not isinstance(tool, str) or not tool.strip():
            continue
        if not isinstance(args, dict):
            args = {}
        converted.append(Intent(tool=tool.strip(), args=args, continue_on_error=continue_on_error))

    # Preserve existing validation/gating behavior.
    validate_intents(converted, registry)
    return _execute_intents(converted, registry), converted


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
    
    # Normalize + validate arguments
    tool_args = _normalize_tool_args(tool_name, tool_args or {})
    is_valid, error = validate_args(tool.args_schema, tool_args)
    if not is_valid:
        return {"error": error}
    
    # Log BEFORE execution
    logger.info(f"[TOOLS] Executing {tool_name} args={tool_args}")
    
    # Try to use worker pool if enabled
    pool = _tool_pool
    if pool is not None:
        try:
            job_id = str(uuid.uuid4())
            request_id = str(uuid.uuid4())
            
            # Submit job to pool
            if pool.submit_job(job_id, request_id, tool_name, tool_args):
                # Wait for result with timeout
                result_obj = pool.wait_for_result(job_id, timeout=Config.TOOL_POOL_TIMEOUT_SEC)
                if result_obj is not None:
                    result = result_obj.result
                    logger.info(f"[TOOLS] Pool result {result}")
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
        result = tool.run(**tool_args)
        
        # Log AFTER execution
        logger.info(f"[TOOLS] Result {result}")
        
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
    
    Returns:
        Dict with either {"reply": "..."} or {"intents": [...]} or legacy formats
    """
    # Build tool list for prompt
    tools_list = registry.list_tools()
    tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_list])
    
    prompt = f"""You are Wyzer, a local voice assistant. You can use tools to help users.

Available tools:
{tools_desc}

TOOL USAGE GUIDANCE:
- For "open X" requests:
    - If X is an installed app/game/folder/file name, use open_target with query=X.
    - Use open_website ONLY when the user explicitly requests a website/URL (e.g., says "website", provides a domain like "example.com", or says "go to ...").
    - Do NOT invent URLs for non-web apps/games. (Example: "open Rocket League" is a game -> open_target, not open_website.)
- For window control: use focus_window, minimize_window, maximize_window, close_window, or move_window_to_monitor
  - move_window_to_monitor: Use monitor="primary" for primary monitor, or monitor=0/1/2 for specific monitor index
- For media control: use media_play_pause, media_next, media_previous for playback
- For volume control: prefer volume_control (supports get/set/change + mute/unmute + per-app volume by process)
        - Master volume example: {{"tool": "volume_control", "args": {{"scope": "master", "action": "set", "level": 35}}}}
        - Per-app volume example: {{"tool": "volume_control", "args": {{"scope": "app", "process": "spotify", "action": "change", "delta": -10}}}}
        - Legacy volume tools volume_up/volume_down/volume_mute_toggle exist but volume_control is preferred
- For switching the default audio output device (speakers/headset): use set_audio_output_device with device="name" (fuzzy match allowed)
- For monitor info: use monitor_info to check available monitors
- For checking which monitor an app is on: use get_window_monitor with process="appname" (e.g., "spotify", "chrome") to find which monitor displays that window
- For library management: use local_library_refresh to rebuild the index ONLY when the user explicitly asks to refresh/rebuild/rescan/update the local library/index
- For storage/drives (IMPORTANT - pick only ONE tool):
    - system_storage_scan: Full system scan of ALL drives - use ONLY when user says "system scan", "scan my drives", or "scan disk" (no specific drive mentioned)
    - system_storage_list: Check specific drive OR all drives - use for "scan drive E", "space on D", "what's on E", "check drive C", "how much storage" - Args: {{"drive": "E"}} if user specifies drive, empty dict if not
    - system_storage_open: Open a drive in file manager - use for "open drive D", "open E", "show drive C" - Args: {{"drive": "D"}}
    - RULE: If user mentions a specific drive (C, D, E, etc), NEVER use system_storage_scan. Always use system_storage_list or system_storage_open instead.
- For creative or conversational requests (e.g., "tell me a story", jokes, roleplay, general chat): do NOT call any tools; respond directly with {{"reply": "..."}}
- For location: use get_location (approximate IP-based)
- For weather/forecast: use get_weather_forecast (optionally pass location; otherwise it uses IP-based location)
    - If user asks for Fahrenheit, pass units="fahrenheit" (or units="imperial")

MULTI-INTENT SUPPORT (NEW):
- You can now execute MULTIPLE tools in sequence for complex requests
- Use "intents" array to specify multiple actions in order
- Each intent has: {{"tool": "tool_name", "args": {{...}}, "continue_on_error": false}}
- Keep intents under 5 per request
- Preserve order - they execute sequentially

EXAMPLES:
User: "open downloads"
Response: {{"intents": [{{"tool": "open_target", "args": {{"query": "downloads"}}}}], "reply": "Opening downloads"}}

User: "launch chrome and open youtube"
Response: {{"intents": [{{"tool": "open_target", "args": {{"query": "chrome"}}}}, {{"tool": "open_website", "args": {{"url": "youtube"}}}}], "reply": "Launching Chrome and opening YouTube"}}

User: "open Rocket League"
Response: {{"intents": [{{"tool": "open_target", "args": {{"query": "Rocket League"}}}}], "reply": "Opening Rocket League"}}

User: "open rocketleague.com"
Response: {{"intents": [{{"tool": "open_website", "args": {{"url": "rocketleague.com"}}}}], "reply": "Opening rocketleague.com"}}

User: "pause music and mute volume"
Response: {{"intents": [{{"tool": "media_play_pause", "args": {{}}}}, {{"tool": "volume_mute_toggle", "args": {{}}}}], "reply": "Pausing and muting"}}

User: "move chrome to primary monitor"
Response: {{"intents": [{{"tool": "move_window_to_monitor", "args": {{"process": "chrome", "monitor": "primary"}}}}], "reply": "Moving Chrome to your primary monitor"}}

User: "what time is it"
Response: {{"intents": [{{"tool": "get_time", "args": {{}}}}], "reply": ""}}

User: "hello"
Response: {{"reply": "Hello! How can I help you?"}}

RESPONSE FORMAT: You must respond with valid JSON only (no markdown, no code blocks).
Option 1 - Direct reply (no tool needed):
{{"reply": "your response here"}}

Option 2 - Single tool:
{{"intents": [{{"tool": "tool_name", "args": {{"key": "value"}}}}], "reply": "brief message"}}

Option 3 - Multiple tools (for multi-step requests):
{{"intents": [{{"tool": "tool1", "args": {{}}}}, {{"tool": "tool2", "args": {{}}}}], "reply": "brief message"}}

Rules:
- Default to 1-2 sentences.
- If the user explicitly asks for a long story, an in-depth explanation, step-by-step details, or to "go into detail", you may write a longer response.
- Be direct and helpful
- Use intents when appropriate to answer the user's question
- Preserve order for multi-step actions
- If using tools, I will run them and ask you again for the final reply

User: {user_text}

Your response (JSON only):"""

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


def _call_llm_reply_only(user_text: str) -> Dict[str, Any]:
    """Force a direct reply with no tool calls.

    Used as a fallback when the LLM returns spurious tool intents.
    """
    prompt = f"""You are Wyzer, a local voice assistant.

No tools are available for this request.
Write a natural response to the user.

RESPONSE FORMAT: JSON only (no markdown):
{{"reply": "your response"}}

User: {user_text}

Your response (JSON only):"""
    return _ollama_request(prompt)


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
    # Build a summary of what was executed
    summary_parts = []
    for result in execution_summary.ran:
        if result.ok:
            summary_parts.append(
                f"- Executed '{result.tool}' successfully. Result: {json.dumps(result.result)}"
            )
        else:
            summary_parts.append(
                f"- Failed to execute '{result.tool}'. Error: {result.error}"
            )
    
    if execution_summary.stopped_early:
        summary_parts.append("- Execution stopped early due to error")
    
    summary_text = "\n".join(summary_parts)
    
    prompt = f"""You are Wyzer, a local voice assistant.

The user asked: {user_text}

I executed the following actions:
{summary_text}

Now provide a natural reply to the user based on these results.
- Default to 1-2 sentences.
- If the user's original request asked for detail/step-by-step/a long explanation or a story, provide a longer, more in-depth reply.

RESPONSE FORMAT: JSON only (no markdown):
{{"reply": "your natural response to the user"}}

Your response (JSON only):"""

    response = _ollama_request(prompt)
    return response


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
    prompt = f"""You are Wyzer, a local voice assistant.

The user asked: {user_text}

I executed the tool '{tool_name}' with arguments: {json.dumps(tool_args)}

Tool result: {json.dumps(tool_result)}

Now provide a natural reply to the user based on this result.
- Default to 1-2 sentences.
- If the user's original request asked for detail/step-by-step/a long explanation or a story, provide a longer, more in-depth reply.

RESPONSE FORMAT: JSON only (no markdown):
{{"reply": "your natural response to the user"}}

Your response (JSON only):"""

    response = _ollama_request(prompt)
    return response


def _ollama_request(prompt: str) -> Dict[str, Any]:
    """
    Make request to Ollama LLM.
    
    Returns:
        Parsed JSON response or fallback dict
    """
    logger = get_logger_instance()
    
    # Check if Ollama is disabled via NO_OLLAMA flag
    if getattr(Config, "NO_OLLAMA", False):
        logger.debug("[NO_OLLAMA] Ollama disabled, returning not supported")
        return {
            "reply": _get_no_ollama_reply()
        }
    
    try:
        # Use existing config
        base_url = Config.OLLAMA_BASE_URL.rstrip("/")
        model = Config.OLLAMA_MODEL
        timeout = Config.LLM_TIMEOUT
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",  # Request JSON output
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_predict": 150
            }
        }
        
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = json.loads(response.read().decode('utf-8'))
        
        # Parse response
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

        # Try to parse as JSON
        try:
            parsed = json.loads(reply_text)
            parsed = _postprocess_llm_json(parsed)
            if isinstance(parsed, dict):
                return parsed
            return {"reply": str(parsed)}
        except json.JSONDecodeError:
            # LLM didn't return valid JSON, extract reply if possible
            return {"reply": reply_text if reply_text else "I couldn't process that."}

    except urllib.error.URLError as e:
        # Distinguish slow-model timeouts from true connection failures.
        reason = getattr(e, "reason", None)
        is_timeout = isinstance(reason, socket.timeout) or "timed out" in str(e).lower()
        if is_timeout:
            logger.warning(f"Ollama request timed out after {Config.LLM_TIMEOUT}s: {e}")
            return {
                "reply": f"Ollama is taking too long to respond (timeout: {Config.LLM_TIMEOUT}s). Increase --llm-timeout or WYZER_LLM_TIMEOUT."
            }

        logger.warning(f"Ollama request failed (URL error): {e}")
        return {"reply": "I couldn't reach Ollama. Is it running?"}

    except socket.timeout as e:
        logger.warning(f"Ollama request timed out after {Config.LLM_TIMEOUT}s: {e}")
        return {
            "reply": f"Ollama is taking too long to respond (timeout: {Config.LLM_TIMEOUT}s). Increase --llm-timeout or WYZER_LLM_TIMEOUT."
        }

    except Exception as e:
        # Keep generic fallback, but log the underlying error for debugging.
        logger.exception(f"Unexpected Ollama error: {e}")
        return {"reply": "I had trouble talking to Ollama."}
