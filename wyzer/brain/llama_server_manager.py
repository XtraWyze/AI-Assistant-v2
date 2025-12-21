"""
Llama.cpp Server Manager for Wyzer AI Assistant.

Manages the llama.cpp HTTP server as a subprocess:
- Start/stop server lifecycle
- Health checks with retry logic
- Logging to rotating file
- Windows console hiding
- PID tracking to prevent double-starts
- Auto-optimization with GPU detection, flash attention, optimal threading
"""
import os
import sys
import time
import subprocess
import threading
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from wyzer.core.logger import get_logger


def kill_existing_llama_servers() -> int:
    """
    Kill any existing llama-server processes to ensure a fresh start.
    
    Returns:
        Number of processes killed
    """
    killed = 0
    
    if sys.platform == "win32":
        try:
            # Use taskkill to terminate all llama-server.exe processes
            result = subprocess.run(
                ["taskkill", "/F", "/IM", "llama-server.exe"],
                capture_output=True, text=True,
                creationflags=0x08000000  # CREATE_NO_WINDOW
            )
            if result.returncode == 0:
                # Count how many were killed from output
                output = result.stdout + result.stderr
                killed = output.lower().count("success")
                if killed == 0 and "terminated" in output.lower():
                    killed = 1
        except Exception:
            pass
    else:
        # Unix-like systems
        try:
            # Use pkill to terminate llama-server processes
            result = subprocess.run(
                ["pkill", "-9", "-f", "llama-server"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                killed = 1  # pkill doesn't report count easily
        except Exception:
            try:
                # Fallback: use killall
                subprocess.run(
                    ["killall", "-9", "llama-server"],
                    capture_output=True, text=True
                )
                killed = 1
            except Exception:
                pass
    
    return killed


def detect_gpu() -> Tuple[bool, str, int]:
    """
    Detect available GPU for llama.cpp acceleration.
    
    Returns:
        Tuple of (has_gpu, gpu_type, vram_mb)
        gpu_type: "cuda", "vulkan", "metal", or "none"
    """
    # Try NVIDIA CUDA first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=0x08000000 if sys.platform == "win32" else 0
        )
        if result.returncode == 0:
            # Get total VRAM in MB
            vram = int(result.stdout.strip().split('\n')[0])
            return (True, "cuda", vram)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    
    # Check for AMD GPU via rocm-smi (Linux) or via environment
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION") or os.environ.get("ROCM_PATH"):
        return (True, "vulkan", 0)  # AMD ROCm detected
    
    # Check for Vulkan support (works with AMD/Intel on Windows)
    try:
        # Check if vulkan DLLs exist
        if sys.platform == "win32":
            vulkan_paths = [
                os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "System32", "vulkan-1.dll"),
                os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "SysWOW64", "vulkan-1.dll"),
            ]
            if any(os.path.exists(p) for p in vulkan_paths):
                # Vulkan available, but we don't know VRAM - assume it works
                return (True, "vulkan", 0)
    except Exception:
        pass
    
    # macOS Metal
    if sys.platform == "darwin":
        return (True, "metal", 0)
    
    return (False, "none", 0)


def get_optimal_threads() -> int:
    """
    Get optimal thread count for llama.cpp based on CPU.
    
    Returns:
        Optimal number of threads (typically physical cores)
    """
    try:
        # Get physical CPU count (not hyperthreaded)
        physical_cores = multiprocessing.cpu_count()
        
        # Try to get physical cores only (not HT) on Windows
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # GetLogicalProcessorInformation is complex, use simpler heuristic
                # Assume hyperthreading: physical = logical / 2
                physical_cores = max(1, physical_cores // 2)
            except Exception:
                physical_cores = max(1, physical_cores // 2)
        
        # Leave 1-2 cores for system/TTS
        optimal = max(1, physical_cores - 1)
        return optimal
    except Exception:
        return 4  # Safe default


def build_auto_optimize_args(
    gpu_layers: int = -1,
    enable_flash_attn: bool = True,
    enable_mlock: bool = True,
    batch_size: int = 512,
) -> List[str]:
    """
    Build auto-optimized command line arguments for llama.cpp server.
    
    Args:
        gpu_layers: Number of layers to offload to GPU (-1 = all)
        enable_flash_attn: Enable flash attention if supported
        enable_mlock: Lock model in RAM to prevent swapping
        batch_size: Batch size for processing
        
    Returns:
        List of command line arguments
    """
    logger = get_logger()
    args = []
    
    # GPU detection and layer offloading
    has_gpu, gpu_type, vram_mb = detect_gpu()
    
    if has_gpu:
        logger.info(f"[LLAMACPP-OPT] GPU detected: {gpu_type}" + 
                   (f" ({vram_mb}MB VRAM)" if vram_mb > 0 else ""))
        
        # Offload layers to GPU
        if gpu_layers == -1:
            # Auto: offload all layers
            args.extend(["--n-gpu-layers", "99"])  # 99 = effectively all
            logger.info("[LLAMACPP-OPT] GPU offload: all layers")
        elif gpu_layers > 0:
            args.extend(["--n-gpu-layers", str(gpu_layers)])
            logger.info(f"[LLAMACPP-OPT] GPU offload: {gpu_layers} layers")
        
        # Flash attention (CUDA/Metal only, significant speedup)
        if enable_flash_attn and gpu_type in ("cuda", "metal"):
            args.extend(["--flash-attn", "on"])  # Newer llama.cpp requires explicit value
            logger.info("[LLAMACPP-OPT] Flash attention: enabled")
    else:
        logger.info("[LLAMACPP-OPT] No GPU detected, using CPU only")
    
    # Memory optimization
    if enable_mlock:
        args.append("--mlock")
        logger.info("[LLAMACPP-OPT] Memory lock: enabled")
    
    # Batch size optimization
    if batch_size > 0:
        args.extend(["--batch-size", str(batch_size)])
        logger.info(f"[LLAMACPP-OPT] Batch size: {batch_size}")
    
    # Continuous batching for better throughput
    args.append("--cont-batching")
    
    return args


class LlamaServerManager:
    """
    Singleton manager for the llama.cpp HTTP server subprocess.
    
    Provides functions to start, stop, and healthcheck the server.
    Server process is started with configurable port on 127.0.0.1.
    """
    
    _instance: Optional["LlamaServerManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "LlamaServerManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self.logger = get_logger()
        self.process: Optional[subprocess.Popen] = None
        self.base_url: Optional[str] = None
        self._started_by_wyzer: bool = False
        self._log_file_handle = None
        self._warmup_done: bool = False  # Track if warmup has been performed
        self._initialized = True
    
    def ensure_server_running(
        self,
        binary_path: str,
        model_path: str,
        port: int = 8081,
        ctx_size: int = 2048,
        n_threads: int = 4,
        extra_args: Optional[list] = None,
        auto_optimize: bool = True,
        gpu_layers: int = -1,
    ) -> Optional[str]:
        """
        Ensure the llama.cpp server is running. Start if needed.
        
        Args:
            binary_path: Path to llama-server executable
            model_path: Path to GGUF model file
            port: HTTP port to listen on (default 8081)
            ctx_size: Context window size (default 2048)
            n_threads: Number of threads (default 4, use 0 for auto)
            extra_args: Additional command line arguments
            auto_optimize: Auto-detect GPU and optimize settings (default True)
            gpu_layers: Number of GPU layers (-1 = auto/all)
            
        Returns:
            Base URL (e.g., "http://127.0.0.1:8081") if successful, None if failed
        """
        target_url = f"http://127.0.0.1:{port}"
        
        # Kill any existing llama-server processes for a fresh start
        killed = kill_existing_llama_servers()
        if killed > 0:
            self.logger.info(f"[LLAMACPP] Killed {killed} existing llama-server process(es) for fresh start")
            time.sleep(0.5)  # Brief pause to ensure port is released
        
        # Reset our tracked process since we killed everything
        self.process = None
        self._started_by_wyzer = False
        
        # Validate binary exists
        binary = Path(binary_path)
        if not binary.exists():
            self.logger.error(f"[LLAMACPP] Binary not found at {binary_path}")
            self.logger.error("[LLAMACPP] Please place llama-server.exe in wyzer/llm_bin/")
            return None
        
        # Validate model exists
        model = Path(model_path)
        if not model.exists():
            self.logger.error(f"[LLAMACPP] Model not found at {model_path}")
            self.logger.error("[LLAMACPP] Please place a GGUF model in wyzer/llm_models/")
            return None
        
        # Build auto-optimization arguments if enabled
        all_extra_args = list(extra_args) if extra_args else []
        
        if auto_optimize:
            self.logger.info("[LLAMACPP] Auto-optimization enabled, detecting hardware...")
            opt_args = build_auto_optimize_args(gpu_layers=gpu_layers)
            all_extra_args.extend(opt_args)
            
            # Auto-detect optimal threads if set to 0 or using auto-optimize
            if n_threads == 0:
                n_threads = get_optimal_threads()
                self.logger.info(f"[LLAMACPP-OPT] Auto threads: {n_threads}")
        else:
            # Safe performance defaults when auto-optimize is OFF
            # Apply reasonable thread count and batch size from config
            from wyzer.core.config import Config
            import os
            
            if n_threads == 0:
                # Use min(8, cpu_count) as safe default
                n_threads = min(8, os.cpu_count() or 4)
                self.logger.info(f"[LLAMACPP] Safe default threads: {n_threads}")
            
            # Apply batch size from config
            batch_size = getattr(Config, 'LLAMACPP_BATCH_SIZE', 512)
            if batch_size > 0:
                all_extra_args.extend(["--batch-size", str(batch_size)])
                self.logger.info(f"[LLAMACPP] Safe default batch size: {batch_size}")
        
        # Start the server
        started = self.start_server(
            binary_path=str(binary),
            model_path=str(model),
            port=port,
            ctx_size=ctx_size,
            n_threads=n_threads,
            extra_args=all_extra_args if all_extra_args else None
        )
        
        if not started:
            return None
        
        # Wait for server to be ready with retries
        # Model loading can take 30+ seconds for larger models
        max_wait_sec = 60.0
        poll_interval = 0.5
        elapsed = 0.0
        
        self.logger.info(f"[LLAMACPP] Waiting for server to become ready (up to {max_wait_sec:.0f}s)...")
        
        while elapsed < max_wait_sec:
            # Check if process died
            if self.process is not None and self.process.poll() is not None:
                exit_code = self.process.returncode
                self.logger.error(f"[LLAMACPP] Server process exited with code {exit_code}")
                self.process = None
                return None
            
            if self.healthcheck(target_url):
                self.logger.info(f"[LLAMACPP] Server ready at {target_url} (took {elapsed:.1f}s)")
                self.base_url = target_url
                
                # Perform one-time warmup to pre-load model weights into GPU/cache
                self._perform_warmup(target_url)
                
                return target_url
            
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        self.logger.error(f"[LLAMACPP] Server did not become ready within {max_wait_sec}s")
        self.stop_server()
        return None
    
    def start_server(
        self,
        binary_path: str,
        model_path: str,
        port: int = 8081,
        ctx_size: int = 2048,
        n_threads: int = 4,
        extra_args: Optional[list] = None
    ) -> bool:
        """
        Start the llama.cpp server subprocess.
        
        Args:
            binary_path: Path to llama-server executable
            model_path: Path to GGUF model file  
            port: HTTP port to listen on
            ctx_size: Context window size
            n_threads: Number of threads (0 = auto)
            extra_args: Additional command line arguments
            
        Returns:
            True if process started successfully, False otherwise
        """
        if self.process is not None and self.process.poll() is None:
            self.logger.warning("[LLAMACPP] Server already running, not starting again")
            return True
        
        # Resolve to absolute paths to avoid working directory issues
        binary_abs = str(Path(binary_path).resolve())
        model_abs = str(Path(model_path).resolve())
        
        # Build command line with absolute paths
        cmd = [
            binary_abs,
            "--model", model_abs,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--ctx-size", str(ctx_size),
        ]
        
        if n_threads > 0:
            cmd.extend(["--threads", str(n_threads)])
        
        if extra_args:
            cmd.extend(extra_args)
        
        self.logger.info(f"[LLAMACPP] Starting server: {' '.join(cmd[:4])}...")
        
        # Setup log file
        log_dir = Path("wyzer/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "llamacpp_server.log"
        
        try:
            self._log_file_handle = open(log_path, "a", encoding="utf-8", buffering=1)
            self._log_file_handle.write(f"\n{'='*60}\n")
            self._log_file_handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting llama.cpp server\n")
            self._log_file_handle.write(f"Command: {' '.join(cmd)}\n")
            self._log_file_handle.write(f"{'='*60}\n")
            self._log_file_handle.flush()
        except Exception as e:
            self.logger.warning(f"[LLAMACPP] Could not open log file: {e}")
            self._log_file_handle = None
        
        # Platform-specific subprocess flags
        creationflags = 0
        if sys.platform == "win32":
            # CREATE_NO_WINDOW = 0x08000000 - prevents console window popup
            creationflags = 0x08000000
        
        try:
            # Use absolute path for cwd as well
            binary_dir = Path(binary_abs).parent
            self.process = subprocess.Popen(
                cmd,
                stdout=self._log_file_handle if self._log_file_handle else subprocess.DEVNULL,
                stderr=self._log_file_handle if self._log_file_handle else subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                cwd=str(binary_dir) if binary_dir.exists() else None
            )
            self._started_by_wyzer = True
            self.logger.info(f"[LLAMACPP] Server process started (PID: {self.process.pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"[LLAMACPP] Failed to start server: {e}")
            if self._log_file_handle:
                try:
                    self._log_file_handle.write(f"ERROR: {e}\n")
                    self._log_file_handle.close()
                except:
                    pass
                self._log_file_handle = None
            return False
    
    def stop_server(self, force: bool = False) -> None:
        """
        Stop the llama.cpp server.
        
        Args:
            force: If True, stop any llama-server process even if not started by Wyzer
        """
        # If we have a tracked process, stop it
        if self.process is not None:
            if not self._started_by_wyzer and not force:
                self.logger.debug("[LLAMACPP] Not stopping server (not started by Wyzer)")
                return
            
            self.logger.info("[LLAMACPP] Stopping server...")
            
            try:
                # Try graceful termination first
                self.process.terminate()
                
                # Wait up to 5 seconds for graceful shutdown
                try:
                    self.process.wait(timeout=5.0)
                    self.logger.info("[LLAMACPP] Server stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if not responding
                    self.logger.warning("[LLAMACPP] Server not responding, forcing kill")
                    self.process.kill()
                    self.process.wait(timeout=2.0)
                    self.logger.info("[LLAMACPP] Server killed")
                    
            except Exception as e:
                self.logger.error(f"[LLAMACPP] Error stopping server: {e}")
            finally:
                self.process = None
                self._started_by_wyzer = False
                self.base_url = None
                self._warmup_done = False  # Reset warmup flag for next server start
                
                # Close log file
                if self._log_file_handle:
                    try:
                        self._log_file_handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Server stopped\n")
                        self._log_file_handle.close()
                    except:
                        pass
                    self._log_file_handle = None
            return
        
        # No tracked process but force requested - try to kill any running llama-server
        if force:
            self._kill_external_server()
    
    def _kill_external_server(self) -> None:
        """Kill any externally running llama-server process."""
        try:
            if sys.platform == "win32":
                # Use taskkill on Windows
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", "llama-server.exe"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=0x08000000  # No console window
                )
                if result.returncode == 0:
                    self.logger.info("[LLAMACPP] Killed external llama-server process")
                elif "not found" not in result.stderr.lower():
                    self.logger.debug(f"[LLAMACPP] No external llama-server found to kill")
            else:
                # Use pkill on Unix
                result = subprocess.run(
                    ["pkill", "-f", "llama-server"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    self.logger.info("[LLAMACPP] Killed external llama-server process")
        except Exception as e:
            self.logger.debug(f"[LLAMACPP] Could not kill external server: {e}")
        
        self.base_url = None

    def _perform_warmup(self, base_url: str) -> None:
        """
        Perform a one-time warmup request to pre-load model weights.
        
        This reduces cold-start latency for the first real query by:
        - Pre-loading model weights into GPU memory
        - Warming up CPU caches
        - Initializing CUDA/Metal kernels
        
        Uses exponential backoff retry (up to 3 seconds total) to handle
        transient 503 errors when the OpenAI-compatible endpoint isn't
        fully ready immediately after healthcheck passes.
        
        Args:
            base_url: Server base URL
        """
        if self._warmup_done:
            self.logger.debug("[LLAMACPP] Warmup already done, skipping")
            return
        
        import urllib.request
        import urllib.error
        import json
        
        warmup_start = time.time()
        max_warmup_time = 15.0  # Cap total warmup time at 15 seconds (model loading can take 10-20s)
        max_attempts = 50  # Allow many retries during model loading
        base_delay = 0.2  # Start with 200ms delay
        attempt = 0
        last_error = None
        
        while attempt < max_attempts:
            attempt += 1
            elapsed = time.time() - warmup_start
            
            # Check time budget
            if elapsed >= max_warmup_time:
                self.logger.warning(
                    f"[LLAMACPP] Warmup timed out after {int(elapsed * 1000)}ms "
                    f"({attempt - 1} attempts)"
                )
                break
            
            try:
                # Minimal warmup request - just trigger model inference once
                # Uses OpenAI-compatible endpoint for consistency
                warmup_payload = {
                    "model": "local-model",  # Required by OpenAI API format
                    "prompt": "Hi",
                    "max_tokens": 1,  # Generate just 1 token
                    "temperature": 0,
                    "stream": False,
                }
                
                data = json.dumps(warmup_payload).encode("utf-8")
                
                # Try OpenAI-compatible completions endpoint first (more reliable)
                req = urllib.request.Request(
                    f"{base_url}/v1/completions",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                
                with urllib.request.urlopen(req, timeout=10.0) as response:
                    # Discard the output - we just want to warm up the model
                    _ = response.read()
                
                warmup_ms = int((time.time() - warmup_start) * 1000)
                self._warmup_done = True
                self.logger.info(
                    f"[LLAMACPP] Warmup succeeded in {warmup_ms}ms after {attempt} attempt(s)"
                )
                return
                
            except urllib.error.HTTPError as e:
                last_error = e
                # HTTP 503 Service Unavailable - server not ready yet, retry
                if e.code == 503:
                    delay = min(base_delay * (2 ** (attempt - 1)), 0.5)  # Exp backoff, cap at 500ms
                    remaining = max_warmup_time - (time.time() - warmup_start)
                    if remaining > delay:
                        self.logger.debug(
                            f"[LLAMACPP] Warmup attempt {attempt} failed: HTTP 503 (retrying in {int(delay * 1000)}ms)"
                        )
                        time.sleep(delay)
                        continue
                # Other HTTP errors - log and retry once more
                self.logger.debug(
                    f"[LLAMACPP] Warmup attempt {attempt} failed: HTTP {e.code}"
                )
                
            except urllib.error.URLError as e:
                last_error = e
                # Connection refused, timeout, etc. - retry with backoff
                delay = min(base_delay * (2 ** (attempt - 1)), 0.5)
                remaining = max_warmup_time - (time.time() - warmup_start)
                if remaining > delay:
                    self.logger.debug(
                        f"[LLAMACPP] Warmup attempt {attempt} failed: {e.reason} (retrying in {int(delay * 1000)}ms)"
                    )
                    time.sleep(delay)
                    continue
                    
            except Exception as e:
                last_error = e
                # Unexpected error - log and continue
                self.logger.debug(
                    f"[LLAMACPP] Warmup attempt {attempt} failed: {type(e).__name__}: {e}"
                )
                delay = min(base_delay * (2 ** (attempt - 1)), 0.5)
                remaining = max_warmup_time - (time.time() - warmup_start)
                if remaining > delay:
                    time.sleep(delay)
                    continue
        
        # All attempts exhausted or timed out
        warmup_ms = int((time.time() - warmup_start) * 1000)
        self.logger.warning(
            f"[LLAMACPP] Warmup failed after {warmup_ms}ms ({attempt} attempts): {last_error}"
        )
        # Mark as done anyway to avoid blocking startup indefinitely
        self._warmup_done = True

    def healthcheck(self, base_url: Optional[str] = None, timeout: float = 2.0) -> bool:
        """
        Check if the llama.cpp server is healthy and responding.
        
        Args:
            base_url: URL to check (default: self.base_url)
            timeout: Request timeout in seconds
            
        Returns:
            True if server is healthy, False otherwise
        """
        import urllib.request
        import urllib.error
        import json
        
        url = base_url or self.base_url
        if not url:
            return False
        
        # Try health endpoint first (llama.cpp standard)
        health_endpoints = [
            f"{url}/health",
            f"{url}/v1/models",  # OpenAI-compatible endpoint
        ]
        
        for endpoint in health_endpoints:
            try:
                req = urllib.request.Request(endpoint, method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, Exception):
                continue
        
        return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the server state.
        
        Returns:
            Dictionary with server status information
        """
        return {
            "running": self.process is not None and self.process.poll() is None,
            "started_by_wyzer": self._started_by_wyzer,
            "base_url": self.base_url,
            "pid": self.process.pid if self.process and self.process.poll() is None else None,
        }


# Module-level convenience functions
_manager: Optional[LlamaServerManager] = None


def get_llama_server_manager() -> LlamaServerManager:
    """Get the singleton LlamaServerManager instance."""
    global _manager
    if _manager is None:
        _manager = LlamaServerManager()
    return _manager


def ensure_server_running(
    binary_path: str,
    model_path: str,
    port: int = 8081,
    ctx_size: int = 2048,
    n_threads: int = 4,
    extra_args: Optional[list] = None,
    auto_optimize: bool = True,
    gpu_layers: int = -1,
) -> Optional[str]:
    """
    Convenience function to ensure llama.cpp server is running.
    
    Returns base URL if successful, None if failed.
    """
    return get_llama_server_manager().ensure_server_running(
        binary_path=binary_path,
        model_path=model_path,
        port=port,
        ctx_size=ctx_size,
        n_threads=n_threads,
        extra_args=extra_args,
        auto_optimize=auto_optimize,
        gpu_layers=gpu_layers,
    )


def stop_server(force: bool = False) -> None:
    """Convenience function to stop the llama.cpp server.
    
    Args:
        force: If True, kill any llama-server process even if not started by Wyzer
    """
    get_llama_server_manager().stop_server(force=force)


def healthcheck(base_url: Optional[str] = None) -> bool:
    """Convenience function to check server health."""
    return get_llama_server_manager().healthcheck(base_url)
