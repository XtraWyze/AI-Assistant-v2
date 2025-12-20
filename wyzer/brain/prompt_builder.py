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

Rules:
- Reply in 1-2 sentences unless user asks for more detail
- Be direct and helpful, no disclaimers
- For knowledge questions: reply directly with {{"reply": "your answer"}}
- For actions (open apps, control volume, etc.): use intents array
- Do NOT invent tool names - only use tools I provide

Response format (JSON only, no markdown):
Direct reply: {{"reply": "your response"}}
With tools: {{"intents": [{{"tool": "name", "args": {{}}}}], "reply": "brief message"}}"""

COMPACT_SYSTEM_PROMPT = """You are Wyzer, a local voice assistant.
Reply in 1-2 sentences. Be direct.
Response format (JSON only): {{"reply": "text"}} or {{"intents": [{{"tool": "name", "args": {{}}}}], "reply": "text"}}"""


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
        """Get minimal examples (2 short ones)."""
        return """
Examples:
User: "open chrome" -> {{"intents": [{{"tool": "open_target", "args": {{"query": "chrome"}}}}], "reply": "Opening Chrome"}}
User: "what is 2+2" -> {{"reply": "2+2 equals 4."}}"""
    
    def _log_prompt_info(self, mode: str, components: List[str], est_tokens: int) -> None:
        """Log prompt construction info."""
        comp_str = ",".join(components)
        self.logger.info(f"[PROMPT] mode={mode} components=[{comp_str}] est_tokens={est_tokens}")


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
