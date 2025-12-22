"""
LLM Engine for Wyzer AI Assistant - Phase 4
Handles local LLM inference via Ollama HTTP API or llama.cpp server.
Phase 7 enhancement: Uses OllamaClient for connection reuse and streaming support.
Phase 8 enhancement: Added llamacpp mode for embedded llama.cpp server.

Architecture note (Phase 10 refactor):
- Internally uses messages[] representation for structured prompt building.
- Transport to Ollama/llama.cpp remains a single flattened prompt string.
- This enables future streaming-to-TTS without changing routing.
"""
import re
import time
from typing import Dict, Optional, Any, List, Callable, Union
from wyzer.core.logger import get_logger
from wyzer.core.config import Config
from wyzer.brain.prompt import format_prompt
from wyzer.brain.ollama_client import OllamaClient
from wyzer.brain.llamacpp_client import LlamaCppClient
from wyzer.brain.prompt_compact import compact_prompt
from wyzer.brain.messages import (
    Message,
    msg_system,
    msg_user,
    msg_assistant,
    flatten_messages,
    MessageBuilder
)


# ============================================================================
# Voice-Fast Preset: Concise, snappy responses for voice assistant Q&A
# ============================================================================

# Pattern to detect "story" / "creative" requests that need longer output
_STORY_CREATIVE_RE = re.compile(
    r"\b(tell(?:\s+me)?\s+(?:a\s+)?story|"
    r"write(?:\s+me)?\s+(?:a\s+)?(?:story|poem|song|essay|letter|script)|"
    r"(?:compose|create)(?:\s+me)?\s+(?:a\s+)?(?:poem|song|story|narrative)|"
    r"continue\s+(?:the\s+)?story|make\s+up\s+a\s+story|"
    r"explain\s+in\s+detail|(?:explain|describe|tell\s+me\s+about).*\s+in\s+(?:detail|depth)|"
    r"in\s+depth\s+(?:explanation|description)|"
    r"give\s+(?:me\s+)?(?:a\s+)?(?:full|detailed|long|in-depth)\s+(?:explanation|description)|"
    r"write(?:\s+me)?\s+(?:a\s+)?(?:long|detailed))\b",
    re.IGNORECASE
)

# Pattern to detect identity queries that need ultra-short responses (1 sentence)
# These are simple factual questions about user identity
# Patterns match common STT outputs: "what's my name", "whats my name", "what is my name"
_IDENTITY_QUERY_RE = re.compile(
    r"^\s*(?:"
    r"what(?:'?s|\s+is)\s+my\s+name|"
    r"who\s+am\s+i|"
    r"what(?:'?s|\s+is)\s+my\s+(?:wife|husband|spouse|partner)(?:'?s)?\s*name|"
    r"what(?:'?s|\s+is)\s+my\s+(?:dog|cat|pet)(?:'?s)?\s*name|"
    r"when(?:'?s|\s+is)\s+my\s+birthday|"
    r"what(?:'?s|\s+is)\s+my\s+birthday|"
    r"where\s+do\s+i\s+live|"
    r"what(?:'?s|\s+is)\s+my\s+(?:address|location|city)"
    r")\s*[\?\.]?\s*$",
    re.IGNORECASE
)


def _is_identity_query(text: str) -> bool:
    """
    Detect if user is asking a simple identity question.
    
    These queries require ultra-short responses (1 sentence, max_tokens=24).
    Examples: "What's my name?", "Who am I?", "What's my dog's name?"
    
    Args:
        text: User's input text
        
    Returns:
        True if identity query detected
    """
    if not text:
        return False
    return bool(_IDENTITY_QUERY_RE.match(text))


def _is_story_creative_request(text: str) -> bool:
    """
    Detect if user is asking for story/creative content that needs longer output.
    
    Args:
        text: User's input text
        
    Returns:
        True if creative/story request detected
    """
    if not text:
        return False
    return bool(_STORY_CREATIVE_RE.search(text))


# Pattern to detect generic "smalltalk" prompts that need short, focused responses
_SMALLTALK_RE = re.compile(
    r"^\s*(?:"
    r"tell\s+me\s+something(?:\s*\.)?$|"  # "tell me something" (not "tell me something about X")
    r"say\s+something(?:\s*\.)?$|"        # "say something"
    r"talk\s+to\s+me(?:\s*\.)?$|"         # "talk to me"
    r"what'?s?\s+up(?:\s*[\?\.])?$|"      # "what's up"
    r"how\s+are\s+you(?:\s*[\?\.])?$|"    # "how are you"
    r"hello(?:\s*[\!\.])?$|"              # "hello"
    r"hi(?:\s*[\!\.])?$|"                  # "hi"
    r"hey(?:\s*[\!\.])?$"                  # "hey"
    r")\s*$",
    re.IGNORECASE
)

# Smalltalk system directive for strict concise responses
SMALLTALK_SYSTEM_DIRECTIVE = "Reply in 1–2 short sentences. Do not ask follow-up questions."


def _is_smalltalk_request(text: str) -> bool:
    """
    Detect if user is making a generic smalltalk request.
    
    These are open-ended prompts like "tell me something" that tend to
    produce rambling responses if not constrained.
    
    Args:
        text: User's input text
        
    Returns:
        True if smalltalk request detected
    """
    if not text:
        return False
    return bool(_SMALLTALK_RE.match(text))


def get_voice_fast_options(user_text: str, llm_mode: str) -> Dict[str, Any]:
    """
    Get generation options for voice-fast preset.
    
    This preset is designed for snappy voice assistant responses:
    - Ultra-low max_tokens (24) for identity queries ("what's my name")
    - Low max_tokens (48) for smalltalk ("tell me something")
    - Normal max_tokens (64) for quick Q&A
    - Higher max_tokens (320) for story/creative requests
    - Low temperature (0.2) for deterministic answers
    
    Args:
        user_text: User's input text
        llm_mode: Current LLM mode ("llamacpp", "ollama", "off")
        
    Returns:
        Dict with optimized generation options, or empty dict if not applicable
    """
    logger = get_logger()
    
    # Only apply to llamacpp by default, unless explicitly enabled
    voice_fast_env = Config.VOICE_FAST_ENABLED
    if llm_mode != "llamacpp" and voice_fast_env:
        # Check if explicitly enabled via env (not just "auto")
        import os
        if os.environ.get("WYZER_VOICE_FAST", "auto").lower() not in ("true", "1", "yes"):
            logger.debug(f"[VOICE_FAST] skipped: mode={llm_mode} (not llamacpp, env=auto)")
            return {}
    
    if not voice_fast_env:
        logger.debug(f"[VOICE_FAST] skipped: VOICE_FAST_ENABLED=False")
        return {}
    
    # Detect request type priority: story > identity > smalltalk > normal
    is_story = _is_story_creative_request(user_text)
    is_identity = _is_identity_query(user_text)
    is_smalltalk = _is_smalltalk_request(user_text)
    
    # Determine preset and max_tokens based on request type
    system_directive = None
    if is_story:
        max_tokens = Config.VOICE_FAST_STORY_MAX_TOKENS
        preset_reason = "story"
        temperature = Config.VOICE_FAST_TEMPERATURE
        top_p = Config.VOICE_FAST_TOP_P
        stop_sequences = ["\n\n", "\nUser:", "\nWyzer:"]
    elif is_identity:
        # FAST_LANE: Ultra-tight params for identity queries
        max_tokens = 12  # Ultra-short for identity facts (e.g., "Your name is Levi.")
        preset_reason = "identity"
        temperature = 0.0  # Deterministic for factual recall
        top_p = 1.0  # No nucleus sampling needed at temp=0
        # NOTE: Do NOT use "." as stop - it can cut off mid-sentence on abbreviations
        # Rely on max_tokens=12 and system prompt "one short sentence" instead
        stop_sequences = ["\n", "\n\n", "\nUser:", "\nWyzer:"]
    elif is_smalltalk:
        # FAST_LANE: Tight params for smalltalk
        max_tokens = 16  # Very short for generic smalltalk
        preset_reason = "smalltalk"
        temperature = 0.2  # Low temp for concise responses
        top_p = 0.9
        stop_sequences = ["\n", "\nUser:", "\nWyzer:"]
        system_directive = SMALLTALK_SYSTEM_DIRECTIVE
    else:
        max_tokens = Config.VOICE_FAST_MAX_TOKENS
        preset_reason = "normal"
        temperature = Config.VOICE_FAST_TEMPERATURE
        top_p = Config.VOICE_FAST_TOP_P
        stop_sequences = ["\n\n", "\nUser:", "\nWyzer:"]
    
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
        # Add stop sequences for cleaner endings (llama.cpp supports this)
        "stop": stop_sequences,
    }
    
    # Add system directive for smalltalk (used by prompt builder)
    if system_directive:
        options["_smalltalk_directive"] = system_directive
    
    # Signal to use fast-lane prompt for identity queries and smalltalk
    if is_identity or is_smalltalk:
        options["_use_fastlane_prompt"] = True
        options["_fastlane_reason"] = preset_reason
        logger.debug(
            f"[GEN_FAST_LANE] max_tokens={max_tokens} temp={temperature} "
            f"top_p={top_p} stop={stop_sequences}"
        )
    
    # Log generation parameters at INFO level
    logger.info(
        f"[GEN] max_tokens={max_tokens} temp={temperature} "
        f"reason={preset_reason}"
    )
    
    return options


class LLMEngine:
    """Local LLM engine using Ollama or llama.cpp"""
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.1:latest",
        timeout: int = 30,
        enabled: bool = True,
        llm_mode: str = "ollama"
    ):
        """
        Initialize LLM engine
        
        Args:
            base_url: LLM API base URL (Ollama or llama.cpp server)
            model: Model name to use (mainly for Ollama)
            timeout: Request timeout in seconds
            enabled: Whether LLM is enabled
            llm_mode: LLM backend mode ("ollama", "llamacpp", or "off")
        """
        self.logger = get_logger()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.enabled = enabled
        self.llm_mode = llm_mode
        
        # Client can be OllamaClient or LlamaCppClient
        self.client: Optional[Union[OllamaClient, LlamaCppClient]] = None
        
        # Check for NO_OLLAMA mode from config (legacy, sets mode to off)
        if getattr(Config, "NO_OLLAMA", False):
            self.enabled = False
            self.llm_mode = "off"
            self.logger.info("LLM brain disabled (no-ollama mode)")
            return
        
        # Check for disabled mode
        if llm_mode == "off" or not enabled:
            self.enabled = False
            self.llm_mode = "off"
            self.logger.info("LLM brain disabled (running in STT-only mode)")
            return
        
        # Initialize client based on mode
        if llm_mode == "llamacpp":
            self._init_llamacpp_client()
        else:
            # Default: Ollama
            self._init_ollama_client()
    
    def _init_ollama_client(self) -> None:
        """Initialize Ollama client."""
        self.client = OllamaClient(base_url=self.base_url, timeout=self.timeout)
        self.logger.info(f"Initializing LLM brain (Ollama): {self.model} @ {self.base_url}")
        
        # Verify connection
        try:
            ping_start = time.time()
            if self.client.ping():
                ping_ms = int((time.time() - ping_start) * 1000)
                self.logger.info(f"LLM brain initialized successfully ({ping_ms}ms)")
            else:
                self.logger.warning("LLM connection check inconclusive")
                self.logger.warning("Will attempt to connect on first request")
        except Exception as e:
            self.logger.warning(f"LLM connection check failed: {e}")
            self.logger.warning("Will attempt to connect on first request")
    
    def _init_llamacpp_client(self) -> None:
        """Initialize llama.cpp client."""
        self.client = LlamaCppClient(base_url=self.base_url, timeout=self.timeout)
        self.logger.info(f"Initializing LLM brain (llama.cpp) @ {self.base_url}")
        
        # Verify connection
        try:
            ping_start = time.time()
            if self.client.ping():
                ping_ms = int((time.time() - ping_start) * 1000)
                self.logger.info(f"LLM brain (llama.cpp) initialized successfully ({ping_ms}ms)")
            else:
                self.logger.warning("llama.cpp connection check inconclusive")
                self.logger.warning("Will attempt to connect on first request")
        except Exception as e:
            self.logger.warning(f"llama.cpp connection check failed: {e}")
            self.logger.warning("Will attempt to connect on first request")
    
    def update_base_url(self, new_url: str) -> None:
        """
        Update the base URL (useful when llamacpp server starts on dynamic port).
        
        Args:
            new_url: New base URL for the LLM server
        """
        self.base_url = new_url.rstrip("/")
        if self.llm_mode == "llamacpp":
            self.client = LlamaCppClient(base_url=self.base_url, timeout=self.timeout)
        elif self.llm_mode == "ollama":
            self.client = OllamaClient(base_url=self.base_url, timeout=self.timeout)
        self.logger.info(f"LLM base URL updated to {self.base_url}")
    
    def think(self, text: str) -> Dict[str, Any]:
        """
        Process user input and generate response.
        
        Internally builds a messages[] representation for structured prompt
        construction, then flattens to a single string for Ollama transport.
        
        Args:
            text: User's transcribed speech
            
        Returns:
            Dictionary with:
                - reply (str): Assistant's response
                - confidence (float): Confidence score
                - model (str): Model used
                - latency_ms (int): Processing time in milliseconds
        """
        if not self.enabled:
            return {
                "reply": "(LLM disabled)",
                "confidence": 0.0,
                "model": "none",
                "latency_ms": 0
            }
        
        start_time = time.time()
        
        try:
            # Detect identity query FIRST (before any prompt building)
            is_identity = _is_identity_query(text)
            
            # Get voice-fast options to check for fast-lane mode
            voice_fast_opts = get_voice_fast_options(text, self.llm_mode)
            use_fastlane = voice_fast_opts.get("_use_fastlane_prompt", False) if voice_fast_opts else False
            fastlane_reason = voice_fast_opts.get("_fastlane_reason", "") if voice_fast_opts else ""
            
            # Track which prompt path was used for logging sanity
            prompt_path_used = None
            
            # Force FAST_LANE for identity queries in llamacpp mode
            # Identity queries MUST use fastlane - never normal path
            if use_fastlane and self.llm_mode == "llamacpp":
                # FAST_LANE: Use minimal prompt for identity/smalltalk queries
                prompt_path_used = "FAST_LANE"
                compacted_prompt = self._build_fastlane_prompt(text, fastlane_reason)
                was_compacted = False
                
                # Log prompt path at INFO level for visibility
                self.logger.info(
                    f'[PROMPT_PATH] used=FAST_LANE reason={fastlane_reason} '
                    f'user_text="{text[:50]}"'
                )
            else:
                # Normal path: Build messages[] internally, then flatten for Ollama transport
                prompt_path_used = "NORMAL"
                
                # SANITY CHECK: Identity queries should NEVER reach NORMAL path in llamacpp mode
                if is_identity and self.llm_mode == "llamacpp":
                    self.logger.warning(
                        f"[PROMPT_SANITY] Identity query fell back to NORMAL (this should not happen). "
                        f"user_text=\"{text[:50]}\" voice_fast_enabled={bool(voice_fast_opts)}"
                    )
                
                # Log prompt path at INFO level
                reason = "non-llamacpp" if self.llm_mode != "llamacpp" else (
                    "voice_fast_disabled" if not voice_fast_opts else "not_identity_or_smalltalk"
                )
                self.logger.info(
                    f'[PROMPT_PATH] used=NORMAL reason={reason} '
                    f'user_text="{text[:50]}"'
                )
                
                # NOTE: messages[] is internal only; Ollama receives a single prompt string
                messages = self._build_messages(text)
                full_prompt = self._flatten_to_prompt(messages, text)
                
                # Inject smalltalk directive if applicable (for concise responses)
                if _is_smalltalk_request(text):
                    # Insert directive right before "User:" marker
                    directive = f"\n[STYLE DIRECTIVE: {SMALLTALK_SYSTEM_DIRECTIVE}]\n"
                    user_marker = "\n\nUser:"
                    if user_marker in full_prompt:
                        full_prompt = full_prompt.replace(user_marker, f"{directive}{user_marker}", 1)
                
                # Apply prompt compaction if needed
                compacted_prompt, was_compacted = compact_prompt(
                    full_prompt, 
                    max_chars=Config.LLM_MAX_PROMPT_CHARS
                )
            
            # Prepare request options from config (base defaults)
            options = {
                "temperature": self._safe_float(Config.OLLAMA_TEMPERATURE, 0.4),
                "top_p": self._safe_float(Config.OLLAMA_TOP_P, 0.9),
                "num_ctx": self._safe_int(Config.OLLAMA_NUM_CTX, 4096),
                "num_predict": self._safe_int(Config.OLLAMA_NUM_PREDICT, 120)
            }
            
            # Apply voice-fast preset overrides for llamacpp mode
            if voice_fast_opts:
                options.update(voice_fast_opts)
            
            # Use streaming or non-streaming based on config
            use_stream = Config.OLLAMA_STREAM
            
            try:
                # Call the client
                reply = self.client.generate(
                    prompt=compacted_prompt,
                    model=self.model,
                    options=options,
                    stream=use_stream
                ).strip()
            except ValueError as e:
                # Model not found or invalid response
                error_msg = str(e)
                self.logger.error(f"LLM error: {error_msg}")
                latency_ms = int((time.time() - start_time) * 1000)
                
                return {
                    "reply": error_msg,
                    "confidence": 0.3,
                    "model": self.model,
                    "latency_ms": latency_ms
                }
            except ConnectionError as e:
                # Cannot reach Ollama
                error_msg = str(e)
                self.logger.error(f"LLM connection error: {error_msg}")
                latency_ms = int((time.time() - start_time) * 1000)
                
                return {
                    "reply": error_msg,
                    "confidence": 0.3,
                    "model": self.model,
                    "latency_ms": latency_ms
                }
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Determine confidence based on response length
            if not reply:
                confidence = 0.3
            elif len(reply) < 100:
                confidence = 0.8
            else:
                confidence = 0.6
            
            self.logger.debug(f"LLM response received in {latency_ms}ms")
            
            return {
                "reply": reply or "(No response generated)",
                "confidence": confidence,
                "model": self.model,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            # Generic error
            latency_ms = int((time.time() - start_time) * 1000)
            error_msg = "I encountered an error processing your request."
            self.logger.error(f"LLM unexpected error: {e}")
            
            return {
                "reply": error_msg,
                "confidence": 0.3,
                "model": self.model,
                "latency_ms": latency_ms
            }
    
    def _build_messages(self, user_text: str) -> List[Message]:
        """
        Build internal messages[] representation for the prompt.
        
        This assembles the conversation context in a structured way:
        1. System prompt (persona, instructions)
        2. Session context (if available)
        3. Promoted memory (if enabled)
        4. Redaction block (if any facts forgotten)
        5. All memories (if use_memories flag is on)
        6. User message
        
        NOTE: This is internal only; Ollama receives a flattened string.
        
        Args:
            user_text: The user's transcribed speech
            
        Returns:
            List of Message dicts
        """
        from wyzer.brain.prompt import (
            SYSTEM_PROMPT,
            get_session_context_block,
            get_promoted_memory_block,
            get_redaction_block,
            get_all_memories_block
        )
        
        builder = MessageBuilder()
        
        # 1. Core system prompt (always included)
        builder.system(SYSTEM_PROMPT)
        
        # 2. Session context (conversation history from RAM)
        session_block = get_session_context_block()
        if session_block:
            builder.system(session_block)
        
        # 3. Promoted memory context (user-approved long-term memory)
        promoted_block = get_promoted_memory_block()
        if promoted_block:
            builder.system(promoted_block)
        
        # 4. Redaction block (forgotten facts LLM should not use)
        redaction_block = get_redaction_block()
        if redaction_block:
            builder.system(redaction_block)
        
        # 5. All memories block (when use_memories flag is enabled)
        all_memories_block = get_all_memories_block()
        if all_memories_block:
            builder.system(all_memories_block)
        
        # 6. User message (the actual user input)
        builder.user(user_text)
        
        return builder.build()
    
    def _flatten_to_prompt(self, messages: List[Message], user_text: str) -> str:
        """
        Flatten messages[] to a single prompt string for Ollama.
        
        Produces output matching the existing format:
        - System blocks concatenated
        - "User: <input>" and "Wyzer:" markers preserved
        
        NOTE: Ollama transport remains a single prompt string.
        
        Args:
            messages: List of Message dicts from _build_messages()
            user_text: Original user text (for fallback)
            
        Returns:
            Flattened prompt string ready for Ollama
        """
        if not messages:
            return f"User: {user_text}\n\nWyzer:"
        
        # Separate system messages from user message
        system_parts = []
        final_user_text = user_text
        
        for msg in messages:
            if msg["role"] == "system":
                content = msg["content"].strip()
                if content:
                    system_parts.append(content)
            elif msg["role"] == "user":
                final_user_text = msg["content"]
        
        # Build prompt matching existing format exactly
        # System blocks are joined, then "User: X" and "Wyzer:" appended
        if system_parts:
            # Join system parts - preserve existing formatting (blocks may include their own newlines)
            system_block = "".join(system_parts)
            return f"{system_block}\n\nUser: {final_user_text}\n\nWyzer:"
        else:
            return f"User: {final_user_text}\n\nWyzer:"
    
    def _build_fastlane_prompt(self, user_text: str, reason: str) -> str:
        """
        Build ultra-minimal fast-lane prompt for identity/smalltalk queries.
        
        FAST_LANE mode goals:
        - est_tokens <= 150 for simple identity queries
        - Minimal system prompt
        - ONLY the single relevant memory line (if any)
        - No formatting headers, no examples
        
        Args:
            user_text: The user's input
            reason: "identity" or "smalltalk"
            
        Returns:
            Minimal prompt string ready for LLM
        """
        from wyzer.brain.prompt_builder import (
            FASTLANE_SYSTEM_PROMPT,
            estimate_tokens
        )
        from wyzer.memory.memory_manager import get_memory_manager
        
        # Get ONLY the relevant memory for this identity query
        try:
            mem_mgr = get_memory_manager()
            memory_context = mem_mgr.select_for_fastlane_injection(user_text)
        except Exception:
            memory_context = ""
        
        # Build minimal prompt
        parts = [FASTLANE_SYSTEM_PROMPT]
        
        mem_chars = 0
        if memory_context and memory_context.strip():
            parts.append(f"\n[MEMORY]\n{memory_context.strip()}")
            mem_chars = len(memory_context.strip())
        
        parts.append(f"\nUser: {user_text}\nWyzer:")
        
        prompt = "".join(parts)
        prompt_chars = len(prompt)
        est_tokens = estimate_tokens(prompt)
        
        self.logger.debug(
            f"[FAST_LANE] enabled=True reason={reason} "
            f"prompt_chars={prompt_chars} est_tokens={est_tokens} mem_chars={mem_chars}"
        )
        
        return prompt
    
    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        """Safely convert to float with fallback."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        """Safely convert to int with fallback."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def think_streaming(
        self,
        text: str,
        on_segment: Callable[[str], None],
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Process user input with streaming TTS output.
        
        As tokens arrive from Ollama, they are buffered and emitted to TTS
        via on_segment callback when sentence/paragraph boundaries are detected.
        The full reply is accumulated and returned.
        
        Falls back to non-streaming think() if streaming fails.
        
        Args:
            text: User's transcribed speech
            on_segment: Callback to receive TTS segments as they're ready
            cancel_check: Optional callable returning True to cancel streaming
            
        Returns:
            Dictionary with:
                - reply (str): Complete assistant response
                - confidence (float): Confidence score
                - model (str): Model used
                - latency_ms (int): Processing time in milliseconds
                - streamed (bool): Whether streaming was used
        """
        from wyzer.brain.stream_tts import accumulate_full_reply
        
        if not self.enabled:
            return {
                "reply": "(LLM disabled)",
                "confidence": 0.0,
                "model": "none",
                "latency_ms": 0,
                "streamed": False
            }
        
        start_time = time.time()
        
        try:
            # Build and flatten prompt (same as non-streaming)
            messages = self._build_messages(text)
            full_prompt = self._flatten_to_prompt(messages, text)
            
            compacted_prompt, was_compacted = compact_prompt(
                full_prompt,
                max_chars=Config.LLM_MAX_PROMPT_CHARS
            )
            
            options = {
                "temperature": self._safe_float(Config.OLLAMA_TEMPERATURE, 0.4),
                "top_p": self._safe_float(Config.OLLAMA_TOP_P, 0.9),
                "num_ctx": self._safe_int(Config.OLLAMA_NUM_CTX, 4096),
                "num_predict": self._safe_int(Config.OLLAMA_NUM_PREDICT, 120)
            }
            
            # Apply voice-fast preset overrides for llamacpp mode
            voice_fast_opts = get_voice_fast_options(text, self.llm_mode)
            if voice_fast_opts:
                options.update(voice_fast_opts)
            
            self.logger.debug("[STREAM_TTS] LLM stream started")
            
            # Get streaming generator from client
            token_stream = self.client.generate_stream(
                prompt=compacted_prompt,
                model=self.model,
                options=options
            )
            
            # Process stream: emit TTS segments, accumulate full reply
            min_chars = getattr(Config, 'STREAM_TTS_BUFFER_CHARS', 150)
            first_emit_chars = getattr(Config, 'STREAM_TTS_FIRST_EMIT_CHARS', 24)
            reply = accumulate_full_reply(
                token_stream=token_stream,
                on_segment=on_segment,
                min_chars=min_chars,
                first_emit_chars=first_emit_chars,
                cancel_check=cancel_check
            )
            
            reply = reply.strip()
            latency_ms = int((time.time() - start_time) * 1000)
            
            self.logger.debug(f"[STREAM_TTS] LLM stream ended")
            
            # Determine confidence
            if not reply:
                confidence = 0.3
            elif len(reply) < 100:
                confidence = 0.8
            else:
                confidence = 0.6
            
            return {
                "reply": reply or "(No response generated)",
                "confidence": confidence,
                "model": self.model,
                "latency_ms": latency_ms,
                "streamed": True
            }
            
        except Exception as e:
            # Streaming failed - fall back to non-streaming
            self.logger.warning(f"[STREAM_TTS] Streaming failed, falling back: {e}")
            
            result = self.think(text)
            result["streamed"] = False
            
            # Emit full reply as single TTS segment (fallback behavior)
            if result.get("reply"):
                try:
                    on_segment(result["reply"])
                except Exception as tts_err:
                    self.logger.error(f"TTS callback error: {tts_err}")
            
            return result
