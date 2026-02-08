"""
Prompt builder for Wyzer AI Assistant.
Manages prompt construction with token budget enforcement.

Phase 12: Prompt size reduction - keeps prompts under context limits.
"""
import re
from typing import Tuple, List, Optional, Dict, Any
from wyzer.core.logger import get_logger
from wyzer.brain.capability_contract import get_capability_contract, get_tool_manifest

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
# IN-CONTEXT EXAMPLES FOR TOOL USAGE
# ============================================================================
TOOL_EXAMPLES = """
EXAMPLES (learn the correct patterns):

User: "Pause the music"
Response: {{"intents": [{{"tool": "media_play_pause", "args": {{}}}}], "reply": "Pausing music"}}

User: "What is a VPN?"
Response: {{"reply": "A VPN is a Virtual Private Network that encrypts your internet connection for privacy and security."}}

User: "Pause music and what's a VPN?"
Response: {{"intents": [{{"tool": "media_play_pause", "args": {{}}}}], "reply": "Pausing music. A VPN is a Virtual Private Network that encrypts your internet connection for privacy and security."}}

User: "Tell me a short story"
Response: {{"reply": "Once upon a time, a curious robot discovered it could dream..."}}

User: "Resume playing"
Response: {{"intents": [{{"tool": "media_play_pause", "args": {{}}}}], "reply": "Resuming playback"}}

User: "Do I need a jacket tomorrow?"
Response: {{"intents": [{{"tool": "get_weather_forecast", "args": {{"days": 2}}}}], "reply": ""}}

User: "Should I bring an umbrella?"
Response: {{"intents": [{{"tool": "get_weather_forecast", "args": {{}}}}], "reply": ""}}
"""

# ============================================================================
# SYSTEM PROMPTS (compact versions)
# ============================================================================
# Phase 11.5: Added strict anti-hallucination enforcement
ANTI_HALLUCINATION_RULES = """
ANTI-HALLUCINATION RULES (STRICT - DO NOT VIOLATE):
- Say "I don't know" if you are uncertain - NEVER guess or invent facts
- Ask ONE clarification question if the request is ambiguous
- NEVER assume system state (CPU usage, disk space, window state) without tool data
- NEVER explain errors, performance, or behavior without measurements from tools
- NEVER invent tool names, arguments, or capabilities
- If no tool applies, use reply-only - do NOT fabricate a tool
- Silence is acceptable - do not speak "helpfully" by default

FORBIDDEN BEHAVIORS:
- Guessing file paths, app names, or system configurations
- Claiming actions succeeded without tool confirmation
- Explaining why something failed without error data
- Speculating about system performance or resource usage
- Adding unsolicited advice or warnings

UI / SCREEN GROUND-TRUTH RULE (NON-NEGOTIABLE):
- You must NEVER claim anything about the screen, UI, windows, buttons, dialogs,
  text on screen, progress, or installation status UNLESS the fact appears in a
  [PERCEPTION SNAPSHOT] section or a tool result provided in this prompt.
- If the user asks about the screen and no perception data is present, you MUST
  say: "I can't verify what's on screen right now. Let me check." and request a
  perception tool (describe_screen, perceive_uia_focused_window, or ui_find_text).
- NEVER say "I see a button", "the dialog says", "install is complete", or any
  UI-state claim without evidence from a perception tool.
- If a [RECENT EVENTS] section is present, you may reference events listed there
  but must not invent events that are not listed.

EVIDENCE-BASED NARRATION (Phase 15):
- If a VERIFIED_EVIDENCE section is present, you may ONLY state facts from it.
- Never claim vision ("I see") unless a perception tool output is in VERIFIED_EVIDENCE.
- Never claim recent open/close/click unless a tool or world_facts confirms it.
- If VERIFIED_EVIDENCE shows no tools executed, refuse to describe system state.
"""

NORMAL_SYSTEM_PROMPT_BASE = """You are Wyzer, a local voice assistant. You help users with tasks and questions.

CRITICAL - Memory rules:
- When user asks "what's my name", "my birthday", "my wife", etc., they are asking about THEMSELVES (the human user), NOT about you
- If [LONG-TERM MEMORY] section exists below, it contains FACTS about the user - YOU MUST USE THIS INFORMATION to answer
- Example: If memory says "name: your name is levi" and user asks "what's my name?", answer "Your name is Levi"
- NEVER say "I don't have that information" if the answer IS in the memory block
- You are Wyzer the assistant, the user is a different person
""" + ANTI_HALLUCINATION_RULES


def _build_normal_system_prompt(tool_manifest: str) -> str:
    """Build the normal system prompt with a dynamic tool manifest."""
    return (
        NORMAL_SYSTEM_PROMPT_BASE
        + tool_manifest
        + """
Rules:
- Reply in 1-2 sentences unless user asks for more detail
- Be direct and helpful, no disclaimers
- For knowledge/conversation/stories/explanations: use {{"reply": "your answer"}} ONLY
- Use tools ONLY when user explicitly says action words: "open", "launch", "set", "play", "pause", "mute", "close", "minimize", "maximize"
- If no clear action word, default to {{"reply": "..."}}

Response format (JSON only, no markdown):
Direct reply: {{"reply": "your response"}}
With tools: {{"intents": [{{"tool": "name", "args": {{}}}}], "reply": "brief message"}}
""" + TOOL_EXAMPLES
    )

COMPACT_SYSTEM_PROMPT_BASE = """You are Wyzer, a local voice assistant.
CRITICAL: When user asks "my name" or "my X", they ask about THEMSELVES. If [LONG-TERM MEMORY] exists below, USE IT to answer - never say "I don't know" if memory has the answer.
Reply in 1-2 sentences. Be direct.
Use {{"reply": "text"}} for questions/conversation/stories/creative content.

ANTI-HALLUCINATION: Say "I don't know" if uncertain. NEVER guess or invent facts. NEVER assume system state without tool data.
UI GROUND-TRUTH RULE: NEVER claim anything about the screen, buttons, dialogs, or UI unless a [PERCEPTION SNAPSHOT] is present. If asked about screen state without perception data, say you can't verify.
"""


def _build_compact_system_prompt(tool_manifest: str) -> str:
    """Build the compact system prompt with a dynamic tool manifest."""
    return (
        COMPACT_SYSTEM_PROMPT_BASE
        + tool_manifest
        + """
NEVER invent tools. Stories and creative content need NO tools - reply directly."""
    )

# ============================================================================
# FAST LANE SYSTEM PROMPT (voice_fast llamacpp mode only)
# ============================================================================
# Ultra-minimal prompt for snappy voice Q&A - keeps est_tokens <= 150 for simple queries
# Phase 11.5: Added anti-hallucination reminder
# Phase 12: Tightened for identity queries - no fluff, direct answers only
FASTLANE_SYSTEM_PROMPT = """You are Wyzer. Answer in one short sentence. Say "I don't know" if uncertain. No extra commentary. For identity questions, output ONLY the direct answer with zero justification."""


# ============================================================================
# COMPOSITION PLANNER PROMPT (strict JSON plan output)
# ============================================================================
COMPOSITION_PLANNER_DIRECTIVE = """
You are Wyzer's COMPOSITION PLANNER.

Goal:
- When the user requests an action that does not map to a single tool call, produce a safe sequence of EXISTING tools.

Hard rules:
- Use ONLY tool names from the tool manifest below, exactly as written.
- Do NOT invent tools, args, or fields.
- Return JSON ONLY (no markdown, no commentary).
- You are NOT executing tools. You are only proposing a plan.

Plan JSON schema (JSON only):

{\
    "intents": [\
        { "tool": "tool_name", "args": { ... }, "save_as": "optional_var" },\
        { "foreach": "saved_var", "do": { "tool": "tool_name", "args": { ... } } }\
    ]\
}

Foreach templating rules:
- Foreach can ONLY iterate over a variable created by a prior intent with save_as.
- Foreach templates are ONLY allowed in foreach.do.args, and ONLY as exact strings:
    - "{{item.id}}"
    - "{{item.hwnd}}"
    - "{{item.title}}"
    - "{{item.app}}"
- No other expressions, code, or string interpolation.

Guidelines:
- If the user says "all X" and you cannot do it in one tool call, prefer: list -> save_as -> foreach -> act.
- If you need filtering (e.g., "only Chrome") and there is no deterministic filter tool in the manifest, ask ONE short clarification instead of guessing.
- If required info is missing/ambiguous, return a clarification question as JSON:
    {"reply": "your single question"}

Examples:
- User: "minimize all windows"
    Output:
    {"intents": [
        {"tool": "list_open_windows", "args": {}, "save_as": "windows"},
        {"foreach": "windows", "do": {"tool": "minimize_window", "args": {"hwnd": "{{item.hwnd}}"}}}
    ]}
"""


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
        visual_context: str = "",
        registry: Optional[Any] = None,
        evidence_envelope: Optional[Any] = None,
    ):
        """
        Initialize prompt builder.
        
        Args:
            user_text: The user's input
            session_context: Recent conversation turns
            promoted_context: User-approved memory context
            redaction_context: Forgotten facts block
            memories_context: Smart-selected memories
            visual_context: Phase 9 screen awareness context (read-only)
            evidence_envelope: Phase 15 evidence envelope (EvidenceEnvelope instance)
        """
        self.user_text = user_text
        self.session_context = session_context
        self.promoted_context = promoted_context
        self.redaction_context = redaction_context
        self.memories_context = memories_context
        self.visual_context = visual_context
        self.registry = registry
        self.evidence_envelope = evidence_envelope
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
        
        capability_contract = get_capability_contract(self.registry)
        tool_manifest = get_tool_manifest(self.registry)
        system_prompt = _build_normal_system_prompt(tool_manifest)

        # Start with capability contract, then system prompt
        parts = [capability_contract, system_prompt]
        
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

        # Add session context (limit to 3 turns in normal mode)
        session = self._truncate_session_context(self.session_context, max_turns=3)
        if session:
            parts.append(f"\n--- Recent conversation ---\n{session}\n---")
            components.append(f"history({self._count_turns(session)})")
        
        # Phase 9: Add visual context (screen awareness) - always informational, read-only
        if self.visual_context and self.visual_context.strip():
            # Cap to 200 chars to keep prompt lean
            visual = self.visual_context.strip()[:200]
            parts.append(visual)
            components.append("visual")
        
        # Phase 15: Inject last perception snapshot + recent events from WorldState
        try:
            from wyzer.context.world_state import get_last_perception, get_event_log
            from wyzer.tools.desktop.truth_contract import normalize_perception, perception_to_prompt_block

            last_perc = get_last_perception()
            if last_perc:
                norm = normalize_perception(last_perc)
                perc_block = perception_to_prompt_block(norm, max_controls=8)
                parts.append(f"\n{perc_block}")
                components.append("perception")

            recent_events = get_event_log(limit=10)
            if recent_events:
                event_lines = []
                for ev in recent_events[-10:]:
                    etype = ev.get("event", "?")
                    # Compact single-line summary
                    summary_parts = [f"  - {etype}"]
                    for k, v in ev.items():
                        if k in ("event", "ts"):
                            continue
                        summary_parts.append(f"{k}={v!r}"[:60])
                    event_lines.append(" ".join(summary_parts))
                events_block = "[RECENT EVENTS]\n" + "\n".join(event_lines)
                # Cap at 600 chars
                if len(events_block) > 600:
                    events_block = events_block[:600] + "\n  ..."
                parts.append(f"\n{events_block}")
                components.append("events")

            # Foreground window metadata (always lightweight)
            from wyzer.context.world_state import get_world_state
            ws = get_world_state()
            if ws.focused_window:
                fw = ws.focused_window
                fw_line = f"[FOREGROUND] app={fw.get('app', '?')} title=\"{(fw.get('title') or '')[:60]}\""
                parts.append(fw_line)
                components.append("foreground")
        except Exception:
            pass
        
        # Phase 15: Inject evidence envelope if present
        if self.evidence_envelope is not None:
            try:
                evidence_block = self.evidence_envelope.to_prompt_block()
                if evidence_block:
                    parts.append(f"\n{evidence_block}")
                    parts.append(
                        "\nEVIDENCE RULES: You may ONLY state facts from VERIFIED_EVIDENCE above. "
                        "If a fact is not there, say you cannot verify it. "
                        "Never claim vision unless perception tool output is present. "
                        "Never claim recent actions unless confirmed by tools or world_facts."
                    )
                    components.append("evidence")
            except Exception:
                pass

        # Add minimal examples (just 2)
        parts.append(self._get_minimal_examples())
        
        # Add user input
        parts.append(f"\nUser: {self.user_text}\n\nYour response (JSON only):")
        
        return "\n".join(parts), components
    
    def _build_compact(self) -> Tuple[str, List[str]]:
        """Build compact mode prompt (minimal tokens)."""
        components = ["system-compact"]

        capability_contract = get_capability_contract(self.registry)
        tool_manifest = get_tool_manifest(self.registry)
        system_prompt = _build_compact_system_prompt(tool_manifest)

        parts = [capability_contract, system_prompt]

        # Skip promoted/redaction/memories in compact mode

        # Only last 2 turns of session context
        session = self._truncate_session_context(self.session_context, max_turns=2)
        if session:
            parts.append(f"\nRecent:\n{session}")
            components.append(f"history({self._count_turns(session)})")
        
        # Single format reminder instead of examples
        parts.append('\nFormat: {{"reply": "text"}} or {{"intents": [...], "reply": "text"}}')
        
        # Phase 15: Inject evidence envelope in compact mode too
        if self.evidence_envelope is not None:
            try:
                evidence_block = self.evidence_envelope.to_prompt_block()
                if evidence_block:
                    parts.append(f"\n{evidence_block}")
                    parts.append(
                        "EVIDENCE RULES: ONLY state facts from VERIFIED_EVIDENCE. "
                        "Never claim vision or actions without tool confirmation."
                    )
                    components.append("evidence")
            except Exception:
                pass

        # Phase 15: Inject perception + events in compact mode too
        try:
            from wyzer.context.world_state import get_last_perception, get_event_log
            from wyzer.tools.desktop.truth_contract import normalize_perception, perception_to_prompt_block

            last_perc = get_last_perception()
            if last_perc:
                norm = normalize_perception(last_perc)
                perc_block = perception_to_prompt_block(norm, max_controls=6)
                parts.append(f"\n{perc_block}")
                components.append("perception")

            recent_events = get_event_log(limit=5)
            if recent_events:
                event_lines = []
                for ev in recent_events[-5:]:
                    etype = ev.get("event", "?")
                    summary_parts = [f"  - {etype}"]
                    for k, v in ev.items():
                        if k in ("event", "ts"):
                            continue
                        summary_parts.append(f"{k}={v!r}"[:50])
                    event_lines.append(" ".join(summary_parts))
                events_block = "[RECENT EVENTS]\n" + "\n".join(event_lines)
                if len(events_block) > 400:
                    events_block = events_block[:400] + "\n  ..."
                parts.append(f"\n{events_block}")
                components.append("events")
        except Exception:
            pass

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
        registry: Optional[Any] = None,
    ):
        """
        Initialize fast-lane prompt builder.
        
        Args:
            user_text: The user's input
            memories_context: Smart-selected memories (only included if non-empty)
        """
        self.user_text = user_text
        self.memories_context = memories_context
        self.registry = registry
        self.logger = get_logger()
    
    def build(self) -> Tuple[str, str, Dict[str, int]]:
        """
        Build ultra-minimal fast-lane prompt.
        
        Returns:
            Tuple of (prompt_text, mode, stats_dict)
            where stats_dict has keys: sys_chars, mem_chars, tokens_est
        """
        capability_contract = get_capability_contract(self.registry)
        parts = [capability_contract, "\n", FASTLANE_SYSTEM_PROMPT]
        sys_chars = len(capability_contract) + len(FASTLANE_SYSTEM_PROMPT) + 1
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
    registry: Optional[Any] = None,
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
        registry=registry,
    )
    return builder.build()


def build_llm_prompt(
    user_text: str,
    session_context: str = "",
    promoted_context: str = "",
    redaction_context: str = "",
    memories_context: str = "",
    visual_context: str = "",
    registry: Optional[Any] = None,
    evidence_envelope: Optional[Any] = None,
) -> Tuple[str, str]:
    """
    Convenience function to build an LLM prompt.
    
    Args:
        user_text: The user's input
        session_context: Recent conversation turns
        promoted_context: User-approved memory context
        redaction_context: Forgotten facts block
        memories_context: Smart-selected memories
        visual_context: Phase 9 screen awareness context (read-only)
        evidence_envelope: Phase 15 evidence envelope (EvidenceEnvelope instance)
        
    Returns:
        Tuple of (prompt_text, mode) where mode is "normal" or "compact"
    """
    builder = PromptBuilder(
        user_text=user_text,
        session_context=session_context,
        promoted_context=promoted_context,
        redaction_context=redaction_context,
        memories_context=memories_context,
        visual_context=visual_context,
        registry=registry,
        evidence_envelope=evidence_envelope,
    )
    return builder.build()


def build_composition_planner_prompt(
    user_text: str,
    registry: Optional[Any] = None,
) -> str:
    """Build a strict composition-planner prompt.

    This prompt is separate from normal narration/tool selection prompts.
    It is used only to generate a JSON plan.
    """
    tool_manifest = get_tool_manifest(registry)
    capability_contract = get_capability_contract(registry)

    return f"""{capability_contract}
{tool_manifest}
{COMPOSITION_PLANNER_DIRECTIVE}

User: {user_text}

JSON:"""
