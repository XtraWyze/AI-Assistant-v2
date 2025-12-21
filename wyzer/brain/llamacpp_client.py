"""
HTTP client for llama.cpp server API.

Provides a similar interface to OllamaClient for llama.cpp server,
supporting both the native /completion endpoint and OpenAI-compatible
/v1/chat/completions endpoint.
"""
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Iterator, Any, Optional, List

from wyzer.core.logger import get_logger


class LlamaCppClient:
    """
    Client for llama.cpp server HTTP API.
    
    Supports:
    - OpenAI-compatible /v1/chat/completions (preferred)
    - Native /completion endpoint (fallback)
    - Streaming and non-streaming modes
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        timeout: int = 30
    ):
        """
        Initialize llama.cpp HTTP client.
        
        Args:
            base_url: Server base URL (e.g., http://127.0.0.1:8081)
            timeout: Default timeout for requests in seconds
        """
        self.logger = get_logger()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        # Build a reusable opener with connection pooling support
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPHandler(debuglevel=0),
            urllib.request.HTTPSHandler(debuglevel=0)
        )
        
        # Track which endpoint style is supported (auto-detected on first call)
        self._use_openai_compat: Optional[bool] = None
    
    def ping(self) -> bool:
        """
        Check if llama.cpp server is running and accessible.
        
        Returns:
            True if server is reachable, False otherwise
        """
        try:
            start_time = time.time()
            req = urllib.request.Request(
                f"{self.base_url}/health",
                method="GET"
            )
            with self.opener.open(req, timeout=5) as response:
                elapsed_ms = int((time.time() - start_time) * 1000)
                self.logger.debug(f"llama.cpp ping successful ({elapsed_ms}ms)")
                return True
        except Exception as e:
            # Try /v1/models as fallback
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/v1/models",
                    method="GET"
                )
                with self.opener.open(req, timeout=5) as response:
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    self.logger.debug(f"llama.cpp ping via /v1/models ({elapsed_ms}ms)")
                    return True
            except Exception:
                pass
            
            self.logger.debug(f"llama.cpp ping failed: {e}")
            return False
    
    def _detect_endpoint_style(self) -> bool:
        """
        Auto-detect whether the server supports OpenAI-compatible endpoints.
        
        Returns:
            True if OpenAI-compatible endpoints available, False otherwise
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/models",
                method="GET"
            )
            with self.opener.open(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                if isinstance(data, dict) and "data" in data:
                    self.logger.debug("[LLAMACPP] OpenAI-compatible endpoints available")
                    return True
        except Exception:
            pass
        
        self.logger.debug("[LLAMACPP] Using native /completion endpoint")
        return False
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from llama.cpp server (non-streaming).
        
        Args:
            prompt: Input prompt text
            model: Model name (ignored for llama.cpp, uses loaded model)
            options: Generation options (temperature, top_p, n_ctx, n_predict)
            stream: If True, use streaming internally but return final string
            
        Returns:
            Generated text response
            
        Raises:
            ConnectionError: If cannot reach server
            ValueError: If response is invalid
        """
        if options is None:
            options = {}
        
        if stream:
            # Use streaming but accumulate into final string
            result = ""
            for chunk in self.generate_stream(prompt, model, options):
                result += chunk
            return result
        
        # Auto-detect endpoint style on first call
        if self._use_openai_compat is None:
            self._use_openai_compat = self._detect_endpoint_style()
        
        start_time = time.time()
        
        if self._use_openai_compat:
            return self._generate_openai_compat(prompt, options, start_time)
        else:
            return self._generate_native(prompt, options, start_time)
    
    def _generate_native(
        self,
        prompt: str,
        options: Dict[str, Any],
        start_time: float
    ) -> str:
        """Generate using native /completion endpoint."""
        try:
            payload = {
                "prompt": prompt,
                "stream": False,
                "temperature": options.get("temperature", 0.7),
                "top_p": options.get("top_p", 0.9),
                "n_predict": options.get("num_predict", options.get("n_predict", 128)),
            }
            
            # Add optional parameters
            if "num_ctx" in options or "n_ctx" in options:
                payload["n_ctx"] = options.get("num_ctx", options.get("n_ctx", 2048))
            
            req = urllib.request.Request(
                f"{self.base_url}/completion",
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Connection': 'keep-alive'
                },
                method="POST"
            )
            
            self.logger.debug("[LLAMACPP] Generating (native, non-stream)")
            
            with self.opener.open(req, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode('utf-8'))
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.debug(f"[LLAMACPP] Generation completed in {elapsed_ms}ms")
            
            return response_data.get("content", "").strip()
            
        except urllib.error.HTTPError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_body = ""
            try:
                error_body = e.read().decode('utf-8')
            except:
                pass
            
            self.logger.error(f"[LLAMACPP] HTTP {e.code} after {elapsed_ms}ms: {error_body}")
            raise ConnectionError(f"llama.cpp HTTP error: {e.code}") from e
            
        except urllib.error.URLError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"[LLAMACPP] Connection error after {elapsed_ms}ms: {e}")
            
            if "Connection refused" in str(e):
                raise ConnectionError(f"Cannot reach llama.cpp at {self.base_url}") from e
            else:
                raise ConnectionError(f"Network error: {e}") from e
    
    def _generate_openai_compat(
        self,
        prompt: str,
        options: Dict[str, Any],
        start_time: float
    ) -> str:
        """Generate using OpenAI-compatible /v1/chat/completions endpoint."""
        try:
            # Build messages array for chat completions
            messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "messages": messages,
                "stream": False,
                "temperature": options.get("temperature", 0.7),
                "top_p": options.get("top_p", 0.9),
                "max_tokens": options.get("num_predict", options.get("n_predict", 128)),
            }
            
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Connection': 'keep-alive'
                },
                method="POST"
            )
            
            self.logger.debug("[LLAMACPP] Generating (OpenAI-compat, non-stream)")
            
            with self.opener.open(req, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode('utf-8'))
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.debug(f"[LLAMACPP] Generation completed in {elapsed_ms}ms")
            
            # Extract content from OpenAI-format response
            choices = response_data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "").strip()
            
            return ""
            
        except urllib.error.HTTPError as e:
            # If OpenAI endpoint fails, fall back to native
            if e.code == 404:
                self.logger.warning("[LLAMACPP] OpenAI endpoint not found, falling back to native")
                self._use_openai_compat = False
                return self._generate_native(prompt, options, start_time)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_body = ""
            try:
                error_body = e.read().decode('utf-8')
            except:
                pass
            
            self.logger.error(f"[LLAMACPP] HTTP {e.code} after {elapsed_ms}ms: {error_body}")
            raise ConnectionError(f"llama.cpp HTTP error: {e.code}") from e
            
        except urllib.error.URLError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"[LLAMACPP] Connection error after {elapsed_ms}ms: {e}")
            
            if "Connection refused" in str(e):
                raise ConnectionError(f"Cannot reach llama.cpp at {self.base_url}") from e
            else:
                raise ConnectionError(f"Network error: {e}") from e
    
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """
        Generate text with streaming.
        Yields text deltas as they arrive.
        
        Args:
            prompt: Input prompt text
            model: Model name (ignored, uses loaded model)
            options: Generation options
            
        Yields:
            Text chunks/tokens as they arrive from the model
            
        Raises:
            ConnectionError: If cannot reach server
            ValueError: If streaming parsing fails
        """
        if options is None:
            options = {}
        
        # Auto-detect endpoint style on first call
        if self._use_openai_compat is None:
            self._use_openai_compat = self._detect_endpoint_style()
        
        start_time = time.time()
        first_token_time = None
        
        if self._use_openai_compat:
            yield from self._generate_stream_openai_compat(prompt, options, start_time)
        else:
            yield from self._generate_stream_native(prompt, options, start_time)
    
    def _generate_stream_native(
        self,
        prompt: str,
        options: Dict[str, Any],
        start_time: float
    ) -> Iterator[str]:
        """Stream using native /completion endpoint."""
        first_token_time = None
        
        try:
            payload = {
                "prompt": prompt,
                "stream": True,
                "temperature": options.get("temperature", 0.7),
                "top_p": options.get("top_p", 0.9),
                "n_predict": options.get("num_predict", options.get("n_predict", 128)),
            }
            
            req = urllib.request.Request(
                f"{self.base_url}/completion",
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Connection': 'keep-alive'
                },
                method="POST"
            )
            
            self.logger.debug("[LLAMACPP] Generating (native, stream)")
            
            with self.opener.open(req, timeout=self.timeout) as response:
                for line in response:
                    line = line.decode('utf-8').strip()
                    
                    if not line:
                        continue
                    
                    # Handle SSE format (data: prefix)
                    if line.startswith("data: "):
                        line = line[6:]
                    
                    if line == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    if first_token_time is None:
                        first_token_time = time.time()
                        time_to_first = int((first_token_time - start_time) * 1000)
                        self.logger.debug(f"[LLAMACPP] First token in {time_to_first}ms")
                    
                    content = data.get("content", "")
                    if content:
                        yield content
                    
                    if data.get("stop", False):
                        break
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.debug(f"[LLAMACPP] Stream completed in {elapsed_ms}ms")
            
        except urllib.error.HTTPError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"[LLAMACPP] HTTP {e.code} from stream after {elapsed_ms}ms")
            raise ConnectionError(f"llama.cpp HTTP error: {e.code}") from e
            
        except urllib.error.URLError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"[LLAMACPP] Stream connection error after {elapsed_ms}ms: {e}")
            
            if "Connection refused" in str(e):
                raise ConnectionError(f"Cannot reach llama.cpp at {self.base_url}") from e
            else:
                raise ConnectionError(f"Network error: {e}") from e
    
    def _generate_stream_openai_compat(
        self,
        prompt: str,
        options: Dict[str, Any],
        start_time: float
    ) -> Iterator[str]:
        """Stream using OpenAI-compatible /v1/chat/completions endpoint."""
        first_token_time = None
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "messages": messages,
                "stream": True,
                "temperature": options.get("temperature", 0.7),
                "top_p": options.get("top_p", 0.9),
                "max_tokens": options.get("num_predict", options.get("n_predict", 128)),
            }
            
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Connection': 'keep-alive'
                },
                method="POST"
            )
            
            self.logger.debug("[LLAMACPP] Generating (OpenAI-compat, stream)")
            
            with self.opener.open(req, timeout=self.timeout) as response:
                for line in response:
                    line = line.decode('utf-8').strip()
                    
                    if not line:
                        continue
                    
                    # Handle SSE format (data: prefix)
                    if line.startswith("data: "):
                        line = line[6:]
                    
                    if line == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    if first_token_time is None:
                        first_token_time = time.time()
                        time_to_first = int((first_token_time - start_time) * 1000)
                        self.logger.debug(f"[LLAMACPP] First token in {time_to_first}ms")
                    
                    # Extract delta content from OpenAI format
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                        
                        # Check for stop
                        if choices[0].get("finish_reason"):
                            break
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.debug(f"[LLAMACPP] Stream completed in {elapsed_ms}ms")
            
        except urllib.error.HTTPError as e:
            # If OpenAI endpoint fails with 404, fall back to native
            if e.code == 404:
                self.logger.warning("[LLAMACPP] OpenAI stream endpoint not found, falling back to native")
                self._use_openai_compat = False
                yield from self._generate_stream_native(prompt, options, start_time)
                return
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"[LLAMACPP] HTTP {e.code} from stream after {elapsed_ms}ms")
            raise ConnectionError(f"llama.cpp HTTP error: {e.code}") from e
            
        except urllib.error.URLError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"[LLAMACPP] Stream connection error after {elapsed_ms}ms: {e}")
            
            if "Connection refused" in str(e):
                raise ConnectionError(f"Cannot reach llama.cpp at {self.base_url}") from e
            else:
                raise ConnectionError(f"Network error: {e}") from e
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from a chat messages array.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            options: Generation options
            stream: Whether to use streaming internally
            
        Returns:
            Generated text response
        """
        if options is None:
            options = {}
        
        # Flatten messages to a prompt for non-OpenAI endpoints
        # Or use directly for OpenAI-compat
        if self._use_openai_compat is None:
            self._use_openai_compat = self._detect_endpoint_style()
        
        if self._use_openai_compat:
            return self._generate_chat_openai(messages, options, stream)
        else:
            # Flatten to prompt for native endpoint
            prompt = self._flatten_messages_to_prompt(messages)
            return self.generate(prompt, options=options, stream=stream)
    
    def _flatten_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a flat prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    def _generate_chat_openai(
        self,
        messages: List[Dict[str, str]],
        options: Dict[str, Any],
        stream: bool
    ) -> str:
        """Generate using OpenAI chat format directly."""
        start_time = time.time()
        
        try:
            payload = {
                "messages": messages,
                "stream": stream,
                "temperature": options.get("temperature", 0.7),
                "top_p": options.get("top_p", 0.9),
                "max_tokens": options.get("num_predict", options.get("n_predict", 128)),
            }
            
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Connection': 'keep-alive'
                },
                method="POST"
            )
            
            if stream:
                result = ""
                with self.opener.open(req, timeout=self.timeout) as response:
                    for line in response:
                        line = line.decode('utf-8').strip()
                        if not line or line.startswith("data: [DONE]"):
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        try:
                            data = json.loads(line)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                result += delta.get("content", "")
                        except json.JSONDecodeError:
                            continue
                return result.strip()
            else:
                with self.opener.open(req, timeout=self.timeout) as response:
                    response_data = json.loads(response.read().decode('utf-8'))
                
                choices = response_data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    return message.get("content", "").strip()
                return ""
                
        except Exception as e:
            self.logger.error(f"[LLAMACPP] Chat generation error: {e}")
            raise ConnectionError(f"llama.cpp chat error: {e}") from e
