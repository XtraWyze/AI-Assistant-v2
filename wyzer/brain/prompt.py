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
- NEVER invent or request tools - respond directly in plain text

Creative content:
- You ARE allowed to generate short stories, fictional narratives, poems, jokes, and creative content
- ONLY when the user explicitly asks for a story, narrative, poem, joke, or fictional content
- Creative responses must be safe and spoken-friendly (no markdown, no lists)
- Be concise unless the user asks for more detail

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


def get_all_memories_block() -> str:
    """
    Get ALL long-term memories for LLM prompt injection.
    
    Only returns content when use_memories session flag is True.
    Controlled by "use memories" / "stop using memories" voice commands.
    
    Contract:
    - Default OFF (memories not injected unless explicitly enabled)
    - Session-scoped (cleared on restart)
    - Deduplicates and caps at 30 bullets / 1200 chars
    - Labeled block instructs LLM to use only if relevant
    
    Returns:
        Formatted memory block, or empty string if flag is OFF or no memories
    """
    try:
        from wyzer.memory.memory_manager import get_memory_manager
        mem_mgr = get_memory_manager()
        memory_block = mem_mgr.get_all_memories_for_injection()
        if memory_block:
            return f"\n{memory_block}\n"
    except Exception:
        pass
    return ""


def get_smart_memories_block(user_text: str) -> str:
    """
    Get smartly-selected memories for LLM prompt injection.
    
    Uses deterministic 3-bucket selection:
    1. PINNED memories (always injected when pinned=True)
    2. MENTION-TRIGGERED memories (injected when user mentions key/alias)
    3. Top-K fallback (remaining slots filled by relevance score)
    
    This is the DEFAULT behavior - no "use memories" flag required.
    
    Defaults (from memory_manager.select_for_injection):
    - PINNED_MAX = 4 (max pinned memories)
    - MENTION_MAX = 4 (max mention-triggered memories)
    - K_TOTAL = 6 (total memories to inject)
    - MAX_CHARS = 1200 (char limit for block)
    
    Args:
        user_text: User's current input (for mention detection)
        
    Returns:
        Formatted memory block, or empty string if no memories selected
    """
    try:
        from wyzer.memory.memory_manager import get_memory_manager
        from wyzer.core.logger import get_logger
        mem_mgr = get_memory_manager()
        memory_block = mem_mgr.select_for_injection(user_text)
        if memory_block:
            # Count lines (memories) injected
            lines = [l for l in memory_block.split('\n') if l.startswith('- (')]
            get_logger().info(f"[MEMORY] Sending {len(lines)} memories to LLM")
            get_logger().debug(f"[MEMORY] Block preview: {memory_block[:200]}...")
            return f"\n{memory_block}\n"
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


# -----------------------------------------------------------------------------
# Message-based prompt building (internal representation)
# -----------------------------------------------------------------------------
# NOTE: messages[] is internal only; transport to Ollama remains a prompt string.
# This enables future streaming-to-TTS without changing routing - we can stream
# only the assistant response while keeping system context fixed.

def build_prompt_messages(
    user_input: str,
    include_session_context: bool = True,
    include_promoted_memory: bool = True,
    include_redaction: bool = True,
    include_all_memories: bool = True
) -> "list":
    """
    Build a list of Message objects for the LLM prompt.
    
    This is the message-based equivalent of format_prompt(), producing an
    internal messages[] representation that can be flattened for Ollama.
    
    Args:
        user_input: User's transcribed speech
        include_session_context: Whether to include session memory context
        include_promoted_memory: Whether to include promoted memory context
        include_redaction: Whether to include redaction block
        include_all_memories: Whether to include all memories block
        
    Returns:
        List of Message dicts ready for flatten_messages()
    """
    from wyzer.brain.messages import MessageBuilder
    
    builder = MessageBuilder()
    
    # 1. Core system prompt (always included)
    builder.system(SYSTEM_PROMPT)
    
    # 2. Session context (conversation history from RAM)
    if include_session_context:
        session_block = get_session_context_block()
        if session_block:
            builder.system(session_block)
    
    # 3. Promoted memory context (user-approved long-term memory)
    if include_promoted_memory:
        promoted_block = get_promoted_memory_block()
        if promoted_block:
            builder.system(promoted_block)
    
    # 4. Redaction block (forgotten facts LLM should not use)
    if include_redaction:
        redaction_block = get_redaction_block()
        if redaction_block:
            builder.system(redaction_block)
    
    # 5. Smart memories block (deterministic: pinned + mention-triggered)
    # This is ALWAYS checked - uses user_input for mention detection
    smart_memories_block = get_smart_memories_block(user_input)
    if smart_memories_block:
        builder.system(smart_memories_block)
    
    # 6. All memories block (when "use memories" flag is explicitly enabled)
    # This adds ALL remaining memories on top of the smart selection
    if include_all_memories:
        all_memories_block = get_all_memories_block()
        if all_memories_block:
            builder.system(all_memories_block)
    
    # 7. User message (the actual user input)
    builder.user(user_input)
    
    return builder.build()


def format_prompt_from_messages(
    user_input: str,
    include_session_context: bool = True,
    include_promoted_memory: bool = True,
    include_redaction: bool = True,
    include_all_memories: bool = True
) -> str:
    """
    Build and flatten messages into a prompt string.
    
    This produces output equivalent to the existing format_prompt() function
    but uses the internal messages[] representation.
    
    The flattened output matches the existing format:
    - System prompt first
    - Context blocks appended
    - "User: <input>" and "Wyzer:" markers preserved
    
    Args:
        user_input: User's transcribed speech
        include_session_context: Whether to include session memory context
        include_promoted_memory: Whether to include promoted memory context
        include_redaction: Whether to include redaction block
        include_all_memories: Whether to include all memories block
        
    Returns:
        Full prompt string for LLM (same format as before)
    """
    from wyzer.brain.messages import flatten_messages
    
    messages = build_prompt_messages(
        user_input=user_input,
        include_session_context=include_session_context,
        include_promoted_memory=include_promoted_memory,
        include_redaction=include_redaction,
        include_all_memories=include_all_memories
    )
    
    # Flatten without role headers (matches existing format)
    # But we need to add "User:" and "Wyzer:" markers to match existing output
    if not messages:
        return f"User: {user_input}\n\nWyzer:"
    
    # Separate system messages from user message
    system_parts = []
    user_text = user_input
    
    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        elif msg["role"] == "user":
            user_text = msg["content"]
    
    # Build prompt matching existing format exactly
    system_block = "\n".join(system_parts)
    return f"{system_block}\n\nUser: {user_text}\n\nWyzer:"
