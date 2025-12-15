"""
LLM Engine for Wyzer AI Assistant - Phase 4
Handles local LLM inference via Ollama HTTP API.
Phase 7 enhancement: Uses OllamaClient for connection reuse and streaming support.
"""
import time
from typing import Dict, Optional, Any
from wyzer.core.logger import get_logger
from wyzer.core.config import Config
from wyzer.brain.prompt import format_prompt
from wyzer.brain.ollama_client import OllamaClient
from wyzer.brain.prompt_compact import compact_prompt


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
        Process user input and generate response
        
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
            # Format prompt
            full_prompt = format_prompt(text)
            
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
