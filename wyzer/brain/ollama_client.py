"""
HTTP client for Ollama API with connection reuse and streaming support.
Handles all communication with local Ollama instance.
"""
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Iterator, Any, Optional
from wyzer.core.logger import get_logger


class OllamaClient:
    """Client for Ollama API with connection reuse and streaming."""
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 30
    ):
        """
        Initialize Ollama HTTP client.
        
        Args:
            base_url: Ollama API base URL (e.g., http://127.0.0.1:11434)
            timeout: Default timeout for requests in seconds
        """
        self.logger = get_logger()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        # Build a reusable opener with connection pooling support
        # This helps with keep-alive and connection reuse across requests
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPHandler(debuglevel=0),
            urllib.request.HTTPSHandler(debuglevel=0)
        )
    
    def ping(self) -> bool:
        """
        Check if Ollama server is running and accessible.
        
        Returns:
            True if Ollama is reachable, False otherwise
        """
        try:
            start_time = time.time()
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET"
            )
            with self.opener.open(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = [m.get("name", "") for m in data.get("models", [])]
                elapsed_ms = int((time.time() - start_time) * 1000)
                self.logger.debug(f"Ollama ping successful ({elapsed_ms}ms). Available models: {models}")
                return True
        except Exception as e:
            self.logger.debug(f"Ollama ping failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        model: str,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from Ollama (non-streaming).
        
        Args:
            prompt: Input prompt text
            model: Model name to use
            options: Generation options (temperature, top_p, num_ctx, num_predict, etc.)
            stream: If True, use streaming endpoint (but still returns final string)
        
        Returns:
            Generated text response
            
        Raises:
            ConnectionError: If cannot reach Ollama
            ValueError: If response is invalid or model not found
        """
        if options is None:
            options = {}
        
        start_time = time.time()
        
        try:
            if stream:
                # Use streaming but accumulate into final string
                result = ""
                for chunk in self.generate_stream(prompt, model, options):
                    result += chunk
                return result
            else:
                # Non-streaming mode
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": options
                }
                
                req = urllib.request.Request(
                    f"{self.base_url}/api/generate",
                    data=json.dumps(payload).encode('utf-8'),
                    headers={
                        'Content-Type': 'application/json',
                        'Connection': 'keep-alive'
                    },
                    method="POST"
                )
                
                self.logger.debug(f"Generating (non-stream) with {model}")
                
                with self.opener.open(req, timeout=self.timeout) as response:
                    response_data = json.loads(response.read().decode('utf-8'))
                
                elapsed_ms = int((time.time() - start_time) * 1000)
                prompt_tokens = response_data.get("prompt_eval_count", 0)
                eval_tokens = response_data.get("eval_count", 0)
                self.logger.debug(f"Generation completed in {elapsed_ms}ms (prompt_tokens={prompt_tokens}, eval_tokens={eval_tokens})")
                
                return response_data.get("response", "").strip()
        
        except urllib.error.HTTPError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_body = ""
            try:
                error_body = e.read().decode('utf-8')
            except:
                pass
            
            self.logger.error(f"HTTP {e.code} from Ollama after {elapsed_ms}ms: {error_body}")
            
            # Check for model not found
            if e.code == 404 or "model" in error_body.lower():
                raise ValueError(f"Model '{model}' not found. Try: ollama pull {model}") from e
            
            raise ConnectionError(f"Ollama HTTP error: {e.code}") from e
        
        except urllib.error.URLError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"Connection error after {elapsed_ms}ms: {e}")
            
            if "Connection refused" in str(e):
                raise ConnectionError(f"Cannot reach Ollama at {self.base_url}. Try: ollama serve") from e
            else:
                raise ConnectionError(f"Network error: {e}") from e
    
    def generate_stream(
        self,
        prompt: str,
        model: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """
        Generate text from Ollama with streaming.
        Yields text deltas as they arrive.
        
        Args:
            prompt: Input prompt text
            model: Model name to use
            options: Generation options
            
        Yields:
            Text chunks/tokens as they arrive from the model
            
        Raises:
            ConnectionError: If cannot reach Ollama
            ValueError: If streaming parsing fails
        """
        if options is None:
            options = {}
        
        start_time = time.time()
        first_token_time = None
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": options
            }
            
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Connection': 'keep-alive'
                },
                method="POST"
            )
            
            self.logger.debug(f"Generating (stream) with {model}")
            
            with self.opener.open(req, timeout=self.timeout) as response:
                # Read streaming response line by line (NDJSON format)
                for line in response:
                    line = line.decode('utf-8').strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse stream line: {line[:100]}")
                        raise ValueError(f"Invalid JSON in stream: {e}") from e
                    
                    # Record first token time
                    if first_token_time is None:
                        first_token_time = time.time()
                        time_to_first = int((first_token_time - start_time) * 1000)
                        self.logger.debug(f"First token in {time_to_first}ms")
                    
                    # Extract and yield the text delta
                    response_text = data.get("response", "")
                    if response_text:
                        yield response_text
                    
                    # Check if done
                    if data.get("done", False):
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        prompt_tokens = data.get("prompt_eval_count", 0)
                        eval_tokens = data.get("eval_count", 0)
                        self.logger.debug(f"Stream completed in {elapsed_ms}ms (prompt_tokens={prompt_tokens}, eval_tokens={eval_tokens})")
                        break
        
        except urllib.error.HTTPError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_body = ""
            try:
                error_body = e.read().decode('utf-8')
            except:
                pass
            
            self.logger.error(f"HTTP {e.code} from Ollama stream after {elapsed_ms}ms")
            
            if e.code == 404 or "model" in error_body.lower():
                raise ValueError(f"Model '{model}' not found. Try: ollama pull {model}") from e
            
            raise ConnectionError(f"Ollama HTTP error: {e.code}") from e
        
        except urllib.error.URLError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"Stream connection error after {elapsed_ms}ms: {e}")
            
            if "Connection refused" in str(e):
                raise ConnectionError(f"Cannot reach Ollama at {self.base_url}. Try: ollama serve") from e
            else:
                raise ConnectionError(f"Network error: {e}") from e
