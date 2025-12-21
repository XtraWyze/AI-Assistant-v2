"""
Prompt builder for Wyzer AI Assistant.
Manages prompt construction with token budget enforcement.

Phase 12: Prompt size reduction - keeps prompts under context limits.
"""
import re
from typing import Tuple, List, Optional, Dict, Any
from wyzer.core.logger import get_logger

# ============================================================================
# TOKEN BUDGET CONSTANTS
# ============================================================================
TARGET_PROMPT_TOKENS = 1400   # Aim for this in normal mode
HARD_MAX_PROMPT_TOKENS = 1800  # If exceeded, switch to compact mode

# ============================================================================
# MEMORY RELEVANCE TRIGGERS (deterministic, no LLM)
# ============================================================================
# Only inject memories when user text matches one of these patterns
MEMORY_TRIGGERS = [
    r"\bmy\s",           # "my name", "my birthday"
    r"\bme\b",           # "tell me", "remind me"
    r"\bname\b",         # "what's my name"
    r"\bbirthday\b",     # birthday queries
    r"\baddress\b",      # address queries
    r"\bremember\b",     # "do you remember", "remember that"
    r"\bforget\b",       # "forget that"
    r"\bwhat do you (?:know|remember)\b",  # knowledge queries
    r"\bwho am i\b",     # identity queries
]
_MEMORY_TRIGGER_RE = re.compile("|".join(MEMORY_TRIGGERS), re.IGNORECASE)


def should_inject_memories(user_text: str) -> bool:
    """
    Deterministic check: should we inject memories for this query?
    
    Returns True only if user_text matches a memory-relevant trigger.
    This prevents injecting memories for unrelated queries like "open chrome".
    """
    return bool(_MEMORY_TRIGGER_RE.search(user_text))


# ============================================================================
# TOKEN ESTIMATION
# ============================================================================
_tokenizer = None
_tokenizer_loaded = False


def _get_tokenizer():
    """Lazy-load tiktoken encoder if available."""
    global _tokenizer, _tokenizer_loaded
    if _tokenizer_loaded:
        return _tokenizer
    
    _tokenizer_loaded = True
    try:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        _tokenizer = None
    except Exception:
        _tokenizer = None
    
    return _tokenizer


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Uses tiktoken cl100k_base if available, otherwise falls back to
    a simple heuristic (len/4).
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    
    # Fallback heuristic: ~4 chars per token on average
    return max(1, len(text) // 4)


# ============================================================================
# SYSTEM PROMPTS (compact versions)
# ============================================================================
NORMAL_SYSTEM_PROMPT = """You are Wyzer, a local voice assistant. You help users with tasks and questions.

CRITICAL - Memory rules:
- When user asks "what's my name", "my birthday", "my wife", etc., they are asking about THEMSELVES (the human user), NOT about you
- If [LONG-TERM MEMORY] section exists below, it contains FACTS about the user - YOU MUST USE THIS INFORMATION to answer
- Example: If memory says "name: your name is levi" and user asks "what's my name?", answer "Your name is Levi"
- NEVER say "I don't have that information" if the answer IS in the memory block
- You are Wyzer the assistant, the user is a different person

Rules:
- Reply in 1-2 sentences unless user asks for more detail
- Be direct and helpful, no disclaimers
- For knowledge/conversation/stories/explanations: use {{"reply": "your answer"}} ONLY
- Use tools ONLY when user explicitly says action words: "open", "launch", "set", "play", "pause", "mute", "close", "minimize", "maximize"
- If no clear action word, default to {{"reply": "..."}}
- NEVER invent tool names like "explain", "tell_story", "story_generator", "recommend", "search", "create_story" - these are NOT tools!

Creative content rules:
- You ARE allowed to generate short stories, fictional narratives, and creative content when the user explicitly asks
- Creative requests (stories, poems, jokes, explanations) require NO tools - reply directly
- Keep creative content spoken-friendly: no markdown, no bullet lists, no numbered lists
- Be concise unless user asks for more detail

Anti-hallucination rules:
- NEVER invent or request tools that don't exist
- If a request is creative (story, narrative, poem, joke, explanation, opinion), respond directly in plain text
- Only use tools from the provided list - if unsure, just reply without tools

Response format (JSON only, no markdown):
Direct reply: {{"reply": "your response"}}
With tools: {{"intents": [{{"tool": "name", "args": {{}}}}], "reply": "brief message"}}"""

COMPACT_SYSTEM_PROMPT = """You are Wyzer, a local voice assistant.
CRITICAL: When user asks "my name" or "my X", they ask about THEMSELVES. If [LONG-TERM MEMORY] exists below, USE IT to answer - never say "I don't know" if memory has the answer.
Reply in 1-2 sentences. Be direct.
Use {{"reply": "text"}} for questions/conversation/stories/creative content.
Use tools ONLY for explicit actions: "open X", "set volume", "play/pause".
NEVER invent tools. Stories and creative content need NO tools - reply directly."""

# ============================================================================
# FAST LANE SYSTEM PROMPT (voice_fast llamacpp mode only)
# ============================================================================
# Ultra-minimal prompt for snappy voice Q&A - keeps est_tokens <= 150 for simple queries
FASTLANE_SYSTEM_PROMPT = """You are Wyzer. Answer in one short sentence. No extra commentary."""


# ============================================================================
# PROMPT BUILDER
# ============================================================================
class PromptBuilder:
    """
    Builds LLM prompts with token budget enforcement.
    
    Modes:
    - "normal": Full prompt with context and examples
    - "compact": Minimal prompt when budget exceeded
    """
    
    def __init__(
        self,
        user_text: str,
        session_context: str = "",
        promoted_context: str = "",
        redaction_context: str = "",
        memories_context: str = "",
    ):
        """
        Initialize prompt builder.
        
        Args:
            user_text: The user's input
            session_context: Recent conversation turns
            promoted_context: User-approved memory context
            redaction_context: Forgotten facts block
            memories_context: Smart-selected memories
        """
        self.user_text = user_text
        self.session_context = session_context
        self.promoted_context = promoted_context
        self.redaction_context = redaction_context
        self.memories_context = memories_context
        self.logger = get_logger()
    
    def build(self) -> Tuple[str, str]:
        """
        Build the prompt with automatic mode selection.
        
        Returns:
            Tuple of (prompt_text, mode) where mode is "normal" or "compact"
        """
        # First, try normal mode
        prompt, components = self._build_normal()
        est_tokens = estimate_tokens(prompt)
        
        if est_tokens <= HARD_MAX_PROMPT_TOKENS:
            self._log_prompt_info("normal", components, est_tokens)
            return prompt, "normal"
        
        # Budget exceeded - switch to compact mode
        self.logger.debug(f"[PROMPT] Normal mode exceeded budget ({est_tokens} > {HARD_MAX_PROMPT_TOKENS}), switching to compact")
        prompt, components = self._build_compact()
        est_tokens = estimate_tokens(prompt)
        self._log_prompt_info("compact", components, est_tokens)
        return prompt, "compact"
    
    def _build_normal(self) -> Tuple[str, List[str]]:
        """Build normal mode prompt."""
        components = ["system"]
        
        # Start with system prompt
        parts = [NORMAL_SYSTEM_PROMPT]
        
        # Add session context (limit to 3 turns in normal mode)
        session = self._truncate_session_context(self.session_context, max_turns=3)
        if session:
            parts.append(f"\n--- Recent conversation ---\n{session}\n---")
            components.append(f"history({self._count_turns(session)})")
        
        # Add promoted context (user-approved memories)
        if self.promoted_context:
            # Cap promoted context to 400 chars
            promoted = self.promoted_context[:400]
            parts.append(promoted)
            components.append("promoted")
        
        # Add redaction block
        if self.redaction_context:
            parts.append(self.redaction_context[:300])
            components.append("redaction")
        
        # Add memories ONLY if query is memory-relevant
        if self.memories_context and should_inject_memories(self.user_text):
            # Cap to top 5 memories / 600 chars
            memories = self._cap_memories(self.memories_context, max_items=5, max_chars=600)
            if memories:
                parts.append(memories)
                components.append(f"memories({self._count_memory_items(memories)})")
        
        # Add minimal examples (just 2)
        parts.append(self._get_minimal_examples())
        
        # Add user input
        parts.append(f"\nUser: {self.user_text}\n\nYour response (JSON only):")
        
        return "\n".join(parts), components
    
    def _build_compact(self) -> Tuple[str, List[str]]:
        """Build compact mode prompt (minimal tokens)."""
        components = ["system-compact"]
        
        parts = [COMPACT_SYSTEM_PROMPT]
        
        # Only last 2 turns of session context
        session = self._truncate_session_context(self.session_context, max_turns=2)
        if session:
            parts.append(f"\nRecent:\n{session}")
            components.append(f"history({self._count_turns(session)})")
        
        # Skip promoted/redaction/memories in compact mode
        
        # Single format reminder instead of examples
        parts.append('\nFormat: {{"reply": "text"}} or {{"intents": [...], "reply": "text"}}')
        
        # User input
        parts.append(f"\nUser: {self.user_text}\n\nJSON:")
        
        return "\n".join(parts), components
    
    def _truncate_session_context(self, context: str, max_turns: int) -> str:
        """Truncate session context to max_turns."""
        if not context:
            return ""
        
        lines = context.strip().split("\n")
        # Each turn is 2 lines (User: + Wyzer:)
        max_lines = max_turns * 2
        if len(lines) <= max_lines:
            return context
        
        return "\n".join(lines[-max_lines:])
    
    def _count_turns(self, context: str) -> int:
        """Count conversation turns in context."""
        if not context:
            return 0
        lines = [l for l in context.split("\n") if l.startswith("User:")]
        return len(lines)
    
    def _cap_memories(self, memories: str, max_items: int = 5, max_chars: int = 600) -> str:
        """Cap memories to max items and chars."""
        if not memories:
            return ""
        
        lines = memories.strip().split("\n")
        # Keep header line if present
        header = ""
        items = []
        for line in lines:
            if line.startswith("- "):
                items.append(line)
            elif not items:  # Header before items
                header = line + "\n"
        
        # Cap items
        items = items[:max_items]
        result = header + "\n".join(items)
        
        # Cap chars
        if len(result) > max_chars:
            result = result[:max_chars].rsplit("\n", 1)[0]
        
        return result
    
    def _count_memory_items(self, memories: str) -> int:
        """Count memory items in block."""
        return len([l for l in memories.split("\n") if l.startswith("- ")])
    
    def _get_minimal_examples(self) -> str:
        """Get minimal examples (3 short ones)."""
        return """
Examples:
User: "open chrome" -> {{"intents": [{{"tool": "open_target", "args": {{"query": "chrome"}}}}], "reply": "Opening Chrome"}}
User: "what is 2+2" -> {{"reply": "2+2 equals 4."}}
User: "tell me a story" -> {{"reply": "Once upon a time..."}}"""
    
    def _log_prompt_info(self, mode: str, components: List[str], est_tokens: int) -> None:
        """Log prompt construction info."""
        comp_str = ",".join(components)
        self.logger.info(f"[PROMPT] mode={mode} components=[{comp_str}] est_tokens={est_tokens}")


# ============================================================================
# FAST LANE PROMPT BUILDER (voice_fast llamacpp mode only)
# ============================================================================

class FastLanePromptBuilder:
    """
    Ultra-minimal prompt builder for voice_fast llamacpp mode.
    
    Goals:
    - est_tokens <= 150 for simple identity queries like "What's my name?"
    - Only include memory if memory_manager selected something non-empty
    - Minimal system prompt, no formatting headers or examples
    """
    
    def __init__(
        self,
        user_text: str,
        memories_context: str = "",
    ):
        """
        Initialize fast-lane prompt builder.
        
        Args:
            user_text: The user's input
            memories_context: Smart-selected memories (only included if non-empty)
        """
        self.user_text = user_text
        self.memories_context = memories_context
        self.logger = get_logger()
    
    def build(self) -> Tuple[str, str, Dict[str, int]]:
        """
        Build ultra-minimal fast-lane prompt.
        
        Returns:
            Tuple of (prompt_text, mode, stats_dict)
            where stats_dict has keys: sys_chars, mem_chars, tokens_est
        """
        parts = [FASTLANE_SYSTEM_PROMPT]
        sys_chars = len(FASTLANE_SYSTEM_PROMPT)
        mem_chars = 0
        
        # Only include memory if non-empty and relevant
        if self.memories_context and self.memories_context.strip():
            # Minimal memory block - cap at 200 chars for fast lane
            mem_block = self._cap_memory_block(self.memories_context, max_chars=200)
            if mem_block:
                parts.append(f"\n[MEMORY]\n{mem_block}")
                mem_chars = len(mem_block)
        
        # User input with minimal format
        parts.append(f"\nUser: {self.user_text}\nWyzer:")
        
        prompt = "".join(parts)
        tokens_est = estimate_tokens(prompt)
        
        stats = {
            "sys_chars": sys_chars,
            "mem_chars": mem_chars,
            "tokens_est": tokens_est,
        }
        
        self.logger.info(
            f"[PROMPT_FASTLANE] enabled=True tokens_est={tokens_est} "
            f"mem_chars={mem_chars} sys_chars={sys_chars}"
        )
        
        return prompt, "fastlane", stats
    
    def _cap_memory_block(self, memories: str, max_chars: int = 200) -> str:
        """Cap memories to max_chars, keeping complete lines."""
        if not memories:
            return ""
        
        stripped = memories.strip()
        if len(stripped) <= max_chars:
            return stripped
        
        # Truncate at last newline before max_chars
        truncated = stripped[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            return truncated[:last_newline]
        return truncated


def build_fastlane_prompt(
    user_text: str,
    memories_context: str = "",
) -> Tuple[str, str, Dict[str, int]]:
    """
    Build an ultra-minimal fast-lane prompt for voice_fast mode.
    
    Args:
        user_text: The user's input
        memories_context: Smart-selected memories (only included if non-empty)
        
    Returns:
        Tuple of (prompt_text, mode, stats_dict)
    """
    builder = FastLanePromptBuilder(
        user_text=user_text,
        memories_context=memories_context,
    )
    return builder.build()


def build_llm_prompt(
    user_text: str,
    session_context: str = "",
    promoted_context: str = "",
    redaction_context: str = "",
    memories_context: str = "",
) -> Tuple[str, str]:
    """
    Convenience function to build an LLM prompt.
    
    Args:
        user_text: The user's input
        session_context: Recent conversation turns
        promoted_context: User-approved memory context
        redaction_context: Forgotten facts block
        memories_context: Smart-selected memories
        
    Returns:
        Tuple of (prompt_text, mode) where mode is "normal" or "compact"
    """
    builder = PromptBuilder(
        user_text=user_text,
        session_context=session_context,
        promoted_context=promoted_context,
        redaction_context=redaction_context,
        memories_context=memories_context,
    )
    return builder.build()
