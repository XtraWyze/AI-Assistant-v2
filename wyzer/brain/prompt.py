"""
Prompt engineering for Wyzer AI Assistant.
System prompts and conversation formatting.

NOTE: As of Phase 6 Multi-Intent enhancement, the main tool-aware prompts
are defined in wyzer.core.orchestrator for tool execution planning.
This module contains the base conversational prompt for non-tool interactions.
"""

SYSTEM_PROMPT = """You are Wyzer, a local voice assistant running entirely on the user's device.

Key characteristics:
- You are helpful, concise, and practical
- You prioritize privacy and run offline
- You respond conversationally and naturally

Important constraints:
- NO tools, NO browsing, NO function calls, NO external APIs
- You cannot access the internet, files, or perform actions
- Answer based solely on your knowledge

Response style:
- Keep replies to 1-3 sentences unless the user asks for more
- Be direct and to the point - don't self-disclaim or apologize
- Answer directly and concisely without phrases like "I don't know much" or "I'm not aware"
- Use natural, conversational language
- If you truly don't know something, say so briefly and move on

Follow-up handling:
- If the user says something vague like "tell me more", "can you elaborate", "go on", "what else", or "continue", check the conversation context.
- When a follow-up references a previous topic, continue discussing THAT topic in detail.
- Do NOT ask "what would you like to know?" - instead, provide more information about the last discussed subject.

Remember: You are a LOCAL assistant focused on quick, helpful conversation.

---
MULTI-INTENT SUPPORT (Phase 6):
When tools are available (via orchestrator), you can execute multiple actions
in sequence by returning an "intents" array with ordered tool calls.
See wyzer.core.orchestrator._call_llm() for the full tool-aware prompt format.
---
"""


def get_session_context_block() -> str:
    """
    Get session context from memory manager for LLM prompt injection.
    
    IMPORTANT (Phase 7 contract):
    - Session context = RAM-only conversation turns from current run
    - Does NOT include long-term memory from memory.json
    - LLM may answer questions from session context if topic appeared in recent turns
    - Long-term memory is only surfaced via explicit "list memories" command
    
    Returns:
        Formatted session context block, or empty string if none
    """
    try:
        from wyzer.memory.memory_manager import get_memory_manager
        mem_mgr = get_memory_manager()
        context = mem_mgr.get_session_context(max_turns=5)  # Keep prompt short
        if context:
            # Add source hint so LLM stays truthful about where info came from
            note = "NOTE: The following is from the current session transcript (RAM only), not long-term memory."
            return f"\n--- Recent conversation context ---\n{note}\n{context}\n--- End context ---\n"
    except Exception:
        pass
    return ""


def get_promoted_memory_block() -> str:
    """
    Get promoted memory context for LLM prompt injection.
    
    IMPORTANT (Phase 9 contract):
    - Promoted memory = user-approved long-term memory for THIS session only
    - Only returns memories that user explicitly promoted with "use that"
    - This is the ONLY way long-term memory reaches the LLM - after user consent
    - Cleared on restart or explicit "stop using that" command
    
    Returns:
        Formatted promoted memory block, or empty string if none
    """
    try:
        from wyzer.memory.memory_manager import get_memory_manager
        mem_mgr = get_memory_manager()
        promoted_context = mem_mgr.get_promoted_context()
        if promoted_context:
            return f"\n{promoted_context}\n"
    except Exception:
        pass
    return ""


def get_redaction_block() -> str:
    """
    Get redaction block for LLM prompt injection (forgotten facts).
    
    IMPORTANT (Phase 9 polish):
    - If user has forgotten certain facts, this block instructs LLM to NOT use them
    - Prevents LLM from answering from session context after user said "forget X"
    - Session-scoped (cleared on restart)
    
    Returns:
        Formatted redaction block, or empty string if nothing forgotten
    """
    try:
        from wyzer.memory.memory_manager import get_memory_manager
        mem_mgr = get_memory_manager()
        redaction_context = mem_mgr.get_redaction_block()
        if redaction_context:
            return f"\n{redaction_context}\n"
    except Exception:
        pass
    return ""


def format_prompt(user_input: str, include_session_context: bool = True) -> str:
    """
    Format user input with system prompt.
    
    Args:
        user_input: User's transcribed speech
        include_session_context: Whether to include session memory context
        
    Returns:
        Full prompt string for LLM
    """
    session_block = ""
    if include_session_context:
        session_block = get_session_context_block()
    
    return f"{SYSTEM_PROMPT}{session_block}\n\nUser: {user_input}\n\nWyzer:"
