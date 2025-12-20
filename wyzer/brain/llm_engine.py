"""
LLM Engine for Wyzer AI Assistant - Phase 4
Handles local LLM inference via Ollama HTTP API.
Phase 7 enhancement: Uses OllamaClient for connection reuse and streaming support.

Architecture note (Phase 10 refactor):
- Internally uses messages[] representation for structured prompt building.
- Transport to Ollama remains a single flattened prompt string.
- This enables future streaming-to-TTS without changing routing.
"""
import time
from typing import Dict, Optional, Any, List, Callable
from wyzer.core.logger import get_logger
from wyzer.core.config import Config
from wyzer.brain.prompt import format_prompt
from wyzer.brain.ollama_client import OllamaClient
from wyzer.brain.prompt_compact import compact_prompt
from wyzer.brain.messages import (
    Message,
    msg_system,
    msg_user,
    msg_assistant,
    flatten_messages,
    MessageBuilder
)


class LLMEngine:
    """Local LLM engine using Ollama"""
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.1:latest",
        timeout: int = 30,
        enabled: bool = True
    ):
        """
        Initialize LLM engine
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use
            timeout: Request timeout in seconds
            enabled: Whether LLM is enabled
        """
        self.logger = get_logger()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.enabled = enabled
        
        # Check for NO_OLLAMA mode from config
        if getattr(Config, "NO_OLLAMA", False):
            self.enabled = False
            self.logger.info("LLM brain disabled (no-ollama mode)")
            self.client = None
            return
        
        # Initialize HTTP client for Ollama
        self.client = OllamaClient(base_url=self.base_url, timeout=timeout)
        
        if not self.enabled:
            self.logger.info("LLM brain disabled (running in STT-only mode)")
            return
        
        self.logger.info(f"Initializing LLM brain: {model} @ {base_url}")
        
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
            # Build messages[] internally, then flatten for Ollama transport
            # NOTE: messages[] is internal only; Ollama receives a single prompt string
            messages = self._build_messages(text)
            full_prompt = self._flatten_to_prompt(messages, text)
            
            # Apply prompt compaction if needed
            compacted_prompt, was_compacted = compact_prompt(
                full_prompt, 
                max_chars=Config.LLM_MAX_PROMPT_CHARS
            )
            
            # Prepare request options from config
            options = {
                "temperature": self._safe_float(Config.OLLAMA_TEMPERATURE, 0.4),
                "top_p": self._safe_float(Config.OLLAMA_TOP_P, 0.9),
                "num_ctx": self._safe_int(Config.OLLAMA_NUM_CTX, 4096),
                "num_predict": self._safe_int(Config.OLLAMA_NUM_PREDICT, 120)
            }
            
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
            
            self.logger.debug("[STREAM_TTS] LLM stream started")
            
            # Get streaming generator from client
            token_stream = self.client.generate_stream(
                prompt=compacted_prompt,
                model=self.model,
                options=options
            )
            
            # Process stream: emit TTS segments, accumulate full reply
            min_chars = getattr(Config, 'STREAM_TTS_BUFFER_CHARS', 150)
            reply = accumulate_full_reply(
                token_stream=token_stream,
                on_segment=on_segment,
                min_chars=min_chars,
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
