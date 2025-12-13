"""
LLM Engine for Wyzer AI Assistant - Phase 4
Handles local LLM inference via Ollama HTTP API.
"""
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Optional
from wyzer.core.logger import get_logger
from wyzer.brain.prompt import format_prompt


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
        
        if not self.enabled:
            self.logger.info("LLM brain disabled (running in STT-only mode)")
            return
        
        self.logger.info(f"Initializing LLM brain: {model} @ {base_url}")
        
        # Verify connection
        try:
            self._check_connection()
            self.logger.info("LLM brain initialized successfully")
        except Exception as e:
            self.logger.warning(f"LLM connection check failed: {e}")
            self.logger.warning("Will attempt to connect on first request")
    
    def _check_connection(self) -> None:
        """Check if Ollama is running"""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = [m.get("name", "") for m in data.get("models", [])]
                self.logger.debug(f"Available models: {models}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot reach Ollama at {self.base_url}") from e
    
    def think(self, text: str) -> Dict[str, any]:
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
            
            # Prepare request
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "num_ctx": 4096
                }
            }
            
            # Make request
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method="POST"
            )
            
            self.logger.debug(f"Sending request to Ollama: {self.model}")
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode('utf-8'))
            
            # Extract response
            reply = response_data.get("response", "").strip()
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Determine confidence
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
            
        except urllib.error.URLError as e:
            # Connection error
            latency_ms = int((time.time() - start_time) * 1000)
            
            if "Connection refused" in str(e) or "timed out" in str(e).lower():
                error_msg = "I couldn't reach the local model. Is Ollama running?"
                self.logger.error(f"Ollama connection failed: {e}")
                self.logger.info("Try: ollama serve")
            else:
                error_msg = "I encountered a network error trying to reach the local model."
                self.logger.error(f"Network error: {e}")
            
            return {
                "reply": error_msg,
                "confidence": 0.3,
                "model": self.model,
                "latency_ms": latency_ms
            }
            
        except json.JSONDecodeError as e:
            # JSON parsing error
            latency_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"Failed to parse Ollama response: {e}")
            
            return {
                "reply": "I received an invalid response from the local model.",
                "confidence": 0.3,
                "model": self.model,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            # Generic error
            latency_ms = int((time.time() - start_time) * 1000)
            error_str = str(e).lower()
            
            # Check for model not found
            if "model" in error_str and ("not found" in error_str or "pull" in error_str):
                error_msg = f"Model '{self.model}' not found. Try: ollama pull {self.model}"
                self.logger.error(f"Model not found: {self.model}")
            else:
                error_msg = "I encountered an error processing your request."
                self.logger.error(f"LLM error: {e}")
            
            return {
                "reply": error_msg,
                "confidence": 0.3,
                "model": self.model,
                "latency_ms": latency_ms
            }
