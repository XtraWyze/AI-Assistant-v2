"""wyzer.policy.meta_answer

Deterministic meta-question handler for "how do you work" type questions.

Provides truthful, concise explanations about Wyzer's architecture WITHOUT
calling the LLM. Prevents hallucinations on meta/introspective questions.

HARD RULES:
- NO LLM calls (fully deterministic)
- Responses must be 1-2 sentences max (voice-first)
- Must accurately reflect actual architecture
- Never mention "machine learning algorithms" generically
- Only handles questions ABOUT Wyzer itself

This module is called BEFORE the LLM for eligible questions.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
from wyzer.core.logger import get_logger


# ============================================================================
# META-QUESTION PATTERNS
# ============================================================================
# These patterns match questions about Wyzer's internal workings.
# Must be explicit to avoid false positives.

_META_HOW_DID_YOU_RE = re.compile(
    r"^\s*how(?:'d| did)\s+you\s+(?:do\s+)?(?:that|this|it)\s*[\?\.]?\s*$",
    re.IGNORECASE
)

_META_HOW_DOES_IT_WORK_RE = re.compile(
    r"^\s*how\s+(?:does|do)\s+(?:this|that|it|you)\s+work\s*[\?\.]?\s*$",
    re.IGNORECASE
)

_META_WHAT_ARE_YOU_DOING_RE = re.compile(
    r"^\s*what\s+(?:are|were)\s+you\s+doing\s*[\?\.]?\s*$",
    re.IGNORECASE
)

_META_HOW_DO_YOU_REMEMBER_RE = re.compile(
    r"^\s*how\s+do\s+you\s+(?:remember|know)\s+",
    re.IGNORECASE
)

_META_HOW_DO_YOU_REMEMBER_GENERIC_RE = re.compile(
    r"^\s*how\s+do\s+you\s+remember\s*[\?\.]?\s*$",
    re.IGNORECASE
)

_META_WHERE_DID_YOU_GET_RE = re.compile(
    r"^\s*where\s+did\s+you\s+get\s+(?:that|this)\s*[\?\.]?\s*$",
    re.IGNORECASE
)

_META_HOW_DID_YOU_KNOW_RE = re.compile(
    r"^\s*how\s+did\s+you\s+know\s+(?:that|this)\s*[\?\.]?\s*$",
    re.IGNORECASE
)


# ============================================================================
# ANSWER TEMPLATES (TRUTHFUL, CONCISE)
# ============================================================================
# These reflect the actual architecture, not generic AI platitudes.

_ANSWER_DETERMINISTIC_ROUTING = (
    "I use deterministic routing for obvious commands, "
    "and call the LLM only for complex questions. "
    "That last one went through direct tool matching."
)

_ANSWER_MEMORY_BASED = (
    "I store memories you've told me about yourself. "
    "When you ask identity questions, I inject the relevant saved fact "
    "and generate a quick answer."
)

_ANSWER_WINDOW_REFERENCE = (
    "I track your active window and the last few apps you interacted with. "
    "That let me resolve 'it' or 'that' to the actual target."
)

_ANSWER_LLM_REASONING = (
    "That question went through the LLM for intent parsing. "
    "It analyzed your request and generated the tool call or reply."
)

_ANSWER_GENERIC_META = (
    "I route simple commands deterministically and use the LLM for complex queries. "
    "Memory injection is explicit when you've saved facts about yourself."
)


# ============================================================================
# CONTEXT-AWARE DETECTION
# ============================================================================

def maybe_handle_meta_question(
    user_text: str,
    last_execution_summary: Optional[dict] = None,
    was_memory_based: bool = False,
    was_identity_query: bool = False,
) -> Tuple[bool, str]:
    """
    Check if user text is a meta-question about Wyzer, and return deterministic answer.
    
    This is called BEFORE the LLM to intercept meta/introspective questions.
    
    Args:
        user_text: The user's input text
        last_execution_summary: Summary of the last tool execution (if any)
        was_memory_based: True if the last response used memory injection
        was_identity_query: True if the last response was an identity query
        
    Returns:
        Tuple of (handled: bool, response_text: str)
        If handled is False, response_text is empty string.
    """
    logger = get_logger()
    
    # Normalize text
    text = user_text.strip()
    if not text:
        return (False, "")
    
    # Check each meta-question pattern
    pattern_name = None
    
    if _META_HOW_DID_YOU_RE.match(text):
        pattern_name = "how_did_you_do_that"
    elif _META_HOW_DOES_IT_WORK_RE.match(text):
        pattern_name = "how_does_it_work"
    elif _META_WHAT_ARE_YOU_DOING_RE.match(text):
        pattern_name = "what_are_you_doing"
    elif _META_HOW_DO_YOU_REMEMBER_RE.match(text) or _META_HOW_DO_YOU_REMEMBER_GENERIC_RE.match(text):
        pattern_name = "how_do_you_remember"
    elif _META_WHERE_DID_YOU_GET_RE.match(text):
        pattern_name = "where_did_you_get"
    elif _META_HOW_DID_YOU_KNOW_RE.match(text):
        pattern_name = "how_did_you_know"
    
    if not pattern_name:
        return (False, "")
    
    # Select appropriate answer based on context
    answer = _select_answer(
        pattern_name=pattern_name,
        last_execution_summary=last_execution_summary,
        was_memory_based=was_memory_based,
        was_identity_query=was_identity_query,
    )
    
    logger.info(f"[META] handled meta-question pattern={pattern_name}")
    
    return (True, answer)


def _select_answer(
    pattern_name: str,
    last_execution_summary: Optional[dict],
    was_memory_based: bool,
    was_identity_query: bool,
) -> str:
    """
    Select the appropriate answer based on context.
    
    Args:
        pattern_name: Name of the matched pattern
        last_execution_summary: Summary of last tool execution
        was_memory_based: Whether last response used memory
        was_identity_query: Whether last response was identity query
        
    Returns:
        Concise answer string (1-2 sentences)
    """
    # Memory-specific questions
    if pattern_name in ("how_do_you_remember", "where_did_you_get"):
        return _ANSWER_MEMORY_BASED
    
    # "How did you know that" after identity/memory response
    if pattern_name == "how_did_you_know" and (was_memory_based or was_identity_query):
        return _ANSWER_MEMORY_BASED
    
    # "How'd you do that" after a tool execution
    if pattern_name == "how_did_you_do_that":
        if last_execution_summary and last_execution_summary.get("ran"):
            # Tool was executed
            return _ANSWER_DETERMINISTIC_ROUTING
        elif was_memory_based or was_identity_query:
            return _ANSWER_MEMORY_BASED
        else:
            return _ANSWER_LLM_REASONING
    
    # "How does this work" / "What are you doing"
    if pattern_name in ("how_does_it_work", "what_are_you_doing"):
        return _ANSWER_GENERIC_META
    
    # Default: generic meta answer
    return _ANSWER_GENERIC_META
