"""wyzer.core.brain_worker

Brain Worker process:
- STT (Whisper)
- Orchestrator (LLM + tools)
- TTS

Receives requests from core via core_to_brain_q and sends results/logs via brain_to_core_q.
"""

from __future__ import annotations

import os
import sys
import queue
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from wyzer.core.config import Config
from wyzer.core.ipc import now_ms, safe_put
from wyzer.core.logger import get_logger, init_logger
from wyzer.core.followup_manager import FollowupManager, is_exit_sentinel
from wyzer.stt.stt_router import STTRouter
from wyzer.tts.tts_router import TTSRouter
from wyzer.tools.timer_tool import check_timer_finished


def _apply_config(config_dict: Dict[str, Any]) -> None:
    # Logger first - include quiet mode
    log_level = str(config_dict.get("log_level", "INFO")).upper()
    quiet_mode = config_dict.get("quiet_mode", False) or os.environ.get("WYZER_QUIET_MODE", "false").lower() in ("true", "1", "yes")
    init_logger(log_level, quiet_mode=quiet_mode)

    # Ensure orchestrator uses the worker's config (it reads Config.*)
    if "ollama_url" in config_dict:
        Config.OLLAMA_BASE_URL = str(config_dict["ollama_url"])
    if "ollama_model" in config_dict:
        Config.OLLAMA_MODEL = str(config_dict["ollama_model"])
    if "llm_timeout" in config_dict:
        Config.LLM_TIMEOUT = int(config_dict["llm_timeout"])
    
    # Llama.cpp settings (Phase 8)
    if "llamacpp_bin" in config_dict:
        Config.LLAMACPP_BIN_PATH = str(config_dict["llamacpp_bin"])
    if "llamacpp_model" in config_dict:
        Config.LLAMACPP_MODEL_PATH = str(config_dict["llamacpp_model"])
    if "llamacpp_port" in config_dict:
        Config.LLAMACPP_PORT = int(config_dict["llamacpp_port"])
    if "llamacpp_ctx" in config_dict:
        Config.LLAMACPP_CTX_SIZE = int(config_dict["llamacpp_ctx"])
    if "llamacpp_threads" in config_dict:
        Config.LLAMACPP_THREADS = int(config_dict["llamacpp_threads"])
    if "llamacpp_auto_optimize" in config_dict:
        Config.LLAMACPP_AUTO_OPTIMIZE = bool(config_dict["llamacpp_auto_optimize"])
    if "llamacpp_gpu_layers" in config_dict:
        Config.LLAMACPP_GPU_LAYERS = int(config_dict["llamacpp_gpu_layers"])
    if "llm_mode" in config_dict:
        Config.LLM_MODE = str(config_dict["llm_mode"])

    # Whisper defaults
    if "whisper_model" in config_dict:
        Config.WHISPER_MODEL = str(config_dict["whisper_model"])
    if "whisper_device" in config_dict:
        Config.WHISPER_DEVICE = str(config_dict["whisper_device"])
    if "whisper_compute_type" in config_dict:
        Config.WHISPER_COMPUTE_TYPE = str(config_dict["whisper_compute_type"])


def _is_capture_valid(text: str) -> bool:
    """
    Check if a transcript represents a valid speech capture.
    
    Returns False for empty, whitespace-only, or single-token transcripts
    that likely result from silence timeout with no real speech.
    This prevents follow-up mode from triggering after empty captures.
    
    Args:
        text: The transcribed text from STT
        
    Returns:
        True if transcript is valid for processing, False otherwise
    """
    if not text:
        return False
    
    stripped = text.strip()
    if not stripped:
        return False
    
    # Count tokens (words) - single token is likely noise/artifact
    tokens = stripped.split()
    if len(tokens) <= 1:
        # Single token might be noise, filler word, or partial hotword
        # Allow it only if it's a recognized command word
        single_token = tokens[0].lower() if tokens else ""
        # Short filler words that aren't real commands
        noise_words = {"um", "uh", "hmm", "hm", "ah", "oh", "er", "like", "so", "and"}
        if single_token in noise_words:
            return False
    
    return True


def _read_wav_to_float32(wav_path: str) -> np.ndarray:
    import wave

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if channels != 1:
        raise ValueError(f"Expected mono WAV, got channels={channels}")
    if sample_rate != Config.SAMPLE_RATE:
        # Keep it strict for now; core should record at Config.SAMPLE_RATE.
        raise ValueError(f"Unexpected sample_rate={sample_rate}, expected {Config.SAMPLE_RATE}")

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    return audio


class _TTSController:
    """
    TTS Controller with prefetch support.
    
    Implements "synth ahead" pattern for smoother streaming:
    - While segment N is playing, segment N+1 is synthesized in background
    - When playback completes, next segment is ready to play immediately
    - Single playback at a time, but synthesis can overlap with playback
    """
    
    def __init__(self, tts: Optional[TTSRouter], brain_to_core_q, simulate: bool = False):
        self._tts = tts
        self._simulate = simulate
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._running = True
        self._brain_to_core_q = brain_to_core_q
        
        # Prefetch state
        self._prefetch_lock = threading.Lock()
        self._prefetch_wav: Optional[str] = None  # Path to prefetched WAV
        self._prefetch_meta: Optional[Dict[str, Any]] = None  # Meta for prefetched item
        self._prefetch_text: Optional[str] = None  # Text that was prefetched
        self._prefetch_thread: Optional[threading.Thread] = None
        self._is_playing = False  # Track if we're currently playing audio
        
        # Track pending show_followup_prompt for streaming TTS
        self._pending_followup_prompt: bool = False
        
        self._thread = threading.Thread(target=self._loop, name="BrainTTS", daemon=True)
        self._thread.start()

    def enqueue(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        meta = meta or {}
        # Handle stream-end marker: just update pending followup flag, don't queue
        if meta.get("_stream_end"):
            self._pending_followup_prompt = meta.get("show_followup_prompt", False)
            return
        if not text:
            return
        self._queue.put({"text": text, "meta": meta})
        
        # Trigger prefetch if we're currently playing and no prefetch is running
        self._maybe_start_prefetch()

    def _maybe_start_prefetch(self) -> None:
        """Start prefetching if conditions are right (playing, no current prefetch)."""
        # Only prefetch if we're currently playing something
        if not self._is_playing:
            return
        
        # Check if prefetch is already done or in progress
        with self._prefetch_lock:
            if self._prefetch_wav is not None:
                return  # Already have a prefetched item
            if self._prefetch_thread and self._prefetch_thread.is_alive():
                return  # Prefetch already in progress
        
        # Try to start prefetch
        self._start_prefetch()

    def interrupt(self) -> None:
        with self._lock:
            self._stop_event.set()
            self._pending_followup_prompt = False  # Reset on interrupt
            self._is_playing = False
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
        
        # Clear any prefetched audio
        with self._prefetch_lock:
            if self._prefetch_wav:
                try:
                    os.unlink(self._prefetch_wav)
                except:
                    pass
            self._prefetch_wav = None
            self._prefetch_meta = None
            self._prefetch_text = None

    def clear_stop(self) -> None:
        with self._lock:
            self._stop_event.clear()
    
    def is_cancelled(self) -> bool:
        """Check if stop was requested (for streaming cancellation)."""
        return self._stop_event.is_set()

    def shutdown(self) -> None:
        self._running = False
        self.interrupt()
        self._queue.put({"text": "", "meta": {"_shutdown": True}})
        self._thread.join(timeout=1.0)
        # Clean up prefetch thread if running
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=0.5)

    def _simulate_speak(self, duration_sec: float) -> bool:
        end = time.time() + max(0.0, duration_sec)
        while time.time() < end:
            if self._stop_event.is_set():
                return False
            time.sleep(0.05)
        return True
    
    def _do_prefetch(self, text: str, meta: Dict[str, Any]) -> None:
        """Background thread to prefetch (synthesize) next segment."""
        logger = get_logger()
        
        if self._stop_event.is_set():
            return
        
        if not self._tts:
            return
        
        try:
            logger.debug(f"[TTS_PREFETCH] Synthesizing ahead: {text[:50]}...")
            wav_path = self._tts.synthesize(text)
            if wav_path and not self._stop_event.is_set():
                with self._prefetch_lock:
                    self._prefetch_wav = wav_path
                    self._prefetch_meta = meta
                    self._prefetch_text = text
                logger.debug(f"[TTS_PREFETCH] Ready: {text[:30]}...")
        except Exception as e:
            logger.error(f"[TTS_PREFETCH] Prefetch error: {e}")
    
    def _start_prefetch(self) -> None:
        """Start prefetching the next item in queue if available."""
        logger = get_logger()
        
        # Don't prefetch if stopped
        if self._stop_event.is_set():
            return
        
        # Check if there's something to prefetch
        try:
            next_item = self._queue.get_nowait()
        except queue.Empty:
            return
        
        text = (next_item.get("text") or "").strip()
        meta = next_item.get("meta") or {}
        
        if not text or meta.get("_shutdown"):
            # Put it back
            self._queue.put(next_item)
            return
        
        # Clear any old prefetch
        with self._prefetch_lock:
            if self._prefetch_wav:
                try:
                    os.unlink(self._prefetch_wav)
                except:
                    pass
                self._prefetch_wav = None
        
        # Start prefetch in background thread
        self._prefetch_thread = threading.Thread(
            target=self._do_prefetch,
            args=(text, meta),
            name="BrainTTS-Prefetch",
            daemon=True
        )
        self._prefetch_thread.start()
    
    def _get_prefetched(self) -> Optional[Tuple[str, Dict[str, Any], str]]:
        """Get prefetched WAV if available. Returns (wav_path, meta, text) or None."""
        with self._prefetch_lock:
            if self._prefetch_wav and os.path.exists(self._prefetch_wav):
                wav = self._prefetch_wav
                meta = self._prefetch_meta or {}
                text = self._prefetch_text or ""
                self._prefetch_wav = None
                self._prefetch_meta = None
                self._prefetch_text = None
                return (wav, meta, text)
        return None

    def _loop(self) -> None:
        logger = get_logger()
        while self._running:
            # First check if we have a prefetched segment ready
            prefetched = self._get_prefetched()
            
            if prefetched:
                wav_path, meta, text = prefetched
                
                if meta.get("_shutdown"):
                    try:
                        os.unlink(wav_path)
                    except:
                        pass
                    return
                
                # Mark as playing BEFORE we start - this allows enqueue() to trigger prefetch
                self._is_playing = True
                
                # Try to start prefetching next segment
                self._maybe_start_prefetch()
                
                # Play the prefetched audio
                safe_put(self._brain_to_core_q, {"type": "LOG", "level": "DEBUG", "msg": "tts_started"})
                
                ok = False
                try:
                    if self._simulate:
                        ok = self._simulate_speak(2.0)
                    elif self._tts:
                        self.clear_stop()
                        ok = self._tts.play_wav(wav_path, self._stop_event)
                except Exception as e:
                    logger.error(f"TTS playback error: {e}")
                    ok = False
                finally:
                    self._is_playing = False
                    
                    # Clean up WAV file
                    try:
                        os.unlink(wav_path)
                    except:
                        pass
                    
                    # Determine show_followup_prompt
                    is_streaming = meta.get("_streaming", False)
                    if is_streaming and self._queue.empty() and not self._prefetch_wav:
                        show_followup = self._pending_followup_prompt
                        self._pending_followup_prompt = False
                    else:
                        show_followup = meta.get("show_followup_prompt", False)
                    
                    safe_put(
                        self._brain_to_core_q,
                        {
                            "type": "LOG",
                            "level": "DEBUG",
                            "msg": "tts_finished" if ok else "tts_interrupted",
                            "meta": {"show_followup_prompt": show_followup},
                        },
                    )
                continue
            
            # No prefetch available, get from queue normally
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            meta = item.get("meta") or {}
            if meta.get("_shutdown"):
                return

            text = (item.get("text") or "").strip()
            if not text:
                continue

            # Mark as playing BEFORE synthesis starts - allows enqueue() to trigger prefetch
            # for items that arrive during synthesis+playback
            self._is_playing = True

            safe_put(self._brain_to_core_q, {"type": "LOG", "level": "DEBUG", "msg": "tts_started"})

            ok = False
            try:
                if self._simulate or bool(meta.get("simulate_tts_sec")):
                    ok = self._simulate_speak(float(meta.get("simulate_tts_sec", 2.0)))
                elif self._tts:
                    self.clear_stop()
                    ok = self._tts.speak(text, self._stop_event)
                else:
                    ok = False
            except Exception as e:
                logger.error(f"TTS error: {e}")
                ok = False
            finally:
                self._is_playing = False
                
                # Determine show_followup_prompt:
                # - For non-streaming: use meta directly
                # - For streaming: use pending flag if this is the last segment (queue empty)
                is_streaming = meta.get("_streaming", False)
                if is_streaming and self._queue.empty() and not self._prefetch_wav:
                    # Last streaming segment - use pending followup flag
                    show_followup = self._pending_followup_prompt
                    self._pending_followup_prompt = False  # Reset for next response
                else:
                    show_followup = meta.get("show_followup_prompt", False)
                
                safe_put(
                    self._brain_to_core_q,
                    {
                        "type": "LOG",
                        "level": "DEBUG",
                        "msg": "tts_finished" if ok else "tts_interrupted",
                        "meta": {"show_followup_prompt": show_followup},
                    },
                )


def run_brain_worker(core_to_brain_q, brain_to_core_q, config_dict: Dict[str, Any]) -> None:
    """Entrypoint for the Brain Worker process."""

    _apply_config(config_dict)
    logger = get_logger()

    # Log process role for architecture verification
    import time
    _role_log = {
        "role": "Brain (compute-heavy worker)",
        "responsibilities": "STT, LLM inference, tool execution, TTS synthesis",
        "pid": os.getpid(),
        "ppid": os.getppid() if hasattr(os, 'getppid') else "N/A",
        "python_executable": sys.executable,
        "start_time": time.time(),
    }
    logger.info(f"[ROLE] Brain worker startup: pid={_role_log['pid']} ppid={_role_log['ppid']} exec={sys.executable}")
    logger.info(f"[ROLE] Brain responsibilities: {_role_log['responsibilities']}")

    # Init heavy components once
    stt = STTRouter(
        whisper_model=str(config_dict.get("whisper_model", Config.WHISPER_MODEL)),
        whisper_device=str(config_dict.get("whisper_device", Config.WHISPER_DEVICE)),
        whisper_compute_type=str(config_dict.get("whisper_compute_type", Config.WHISPER_COMPUTE_TYPE)),
    )

    llm_mode = str(config_dict.get("llm_mode", Config.LLM_MODE))
    llamacpp_base_url: Optional[str] = None  # Tracks llamacpp server URL if started
    
    # Handle LLM mode initialization
    if llm_mode == "llamacpp":
        # Start embedded llama.cpp server (Phase 8)
        logger.info("[LLAMACPP] Initializing embedded llama.cpp server...")
        try:
            from wyzer.brain.llama_server_manager import ensure_server_running, stop_server
            
            llamacpp_base_url = ensure_server_running(
                binary_path=str(config_dict.get("llamacpp_bin", Config.LLAMACPP_BIN_PATH)),
                model_path=str(config_dict.get("llamacpp_model", Config.LLAMACPP_MODEL_PATH)),
                port=int(config_dict.get("llamacpp_port", Config.LLAMACPP_PORT)),
                ctx_size=int(config_dict.get("llamacpp_ctx", Config.LLAMACPP_CTX_SIZE)),
                n_threads=int(config_dict.get("llamacpp_threads", Config.LLAMACPP_THREADS)),
                auto_optimize=bool(config_dict.get("llamacpp_auto_optimize", Config.LLAMACPP_AUTO_OPTIMIZE)),
                gpu_layers=int(config_dict.get("llamacpp_gpu_layers", Config.LLAMACPP_GPU_LAYERS)),
            )
            
            if llamacpp_base_url:
                logger.info(f"[LLAMACPP] Server ready at {llamacpp_base_url}")
                # Update config so orchestrator uses the right URL
                Config.LLAMACPP_BASE_URL = llamacpp_base_url
            else:
                logger.warning("[LLAMACPP] Failed to start server - continuing in tools-only mode")
                llm_mode = "off"  # Fallback to no-LLM mode
                Config.LLM_MODE = "off"
        except Exception as e:
            logger.error(f"[LLAMACPP] Error starting server: {e}")
            logger.warning("[LLAMACPP] Continuing in tools-only mode")
            llm_mode = "off"
            Config.LLM_MODE = "off"
    elif llm_mode == "ollama":
        logger.info(f"[LLM] Using Ollama at {Config.OLLAMA_BASE_URL}")
    else:
        logger.info("[LLM] LLM disabled in brain worker")

    tts_enabled = bool(config_dict.get("tts_enabled", True))
    tts_router: Optional[TTSRouter] = None
    if tts_enabled:
        try:
            tts_router = TTSRouter(
                engine=str(config_dict.get("tts_engine", "piper")),
                piper_exe_path=str(config_dict.get("piper_exe_path", "./wyzer/assets/piper/piper.exe")),
                piper_model_path=str(config_dict.get("piper_model_path", "./wyzer/assets/piper/en_US-lessac-medium.onnx")),
                piper_speaker_id=config_dict.get("piper_speaker_id"),
                output_device=config_dict.get("tts_output_device"),
                enabled=True,
            )
        except Exception as e:
            logger.error(f"Failed to init TTS: {e}")
            tts_router = None

    simulate_tts = bool(config_dict.get("simulate_tts", False))
    tts_controller = _TTSController(tts_router, brain_to_core_q, simulate=simulate_tts)

    # Initialize tool worker pool if enabled
    from wyzer.core import orchestrator
    orchestrator.init_tool_pool()

    interrupt_generation = 0
    last_job_id = "none"
    last_heartbeat = time.time()

    safe_put(brain_to_core_q, {"type": "LOG", "level": "INFO", "msg": "brain_worker_started"})

    while True:
        # Emit heartbeat every ~10s (configurable)
        current_time = time.time()
        if current_time - last_heartbeat >= Config.HEARTBEAT_INTERVAL_SEC:
            try:
                q_in_size = core_to_brain_q.qsize() if hasattr(core_to_brain_q, 'qsize') else -1
                q_out_size = brain_to_core_q.qsize() if hasattr(brain_to_core_q, 'qsize') else -1
            except Exception:
                q_in_size = q_out_size = -1
            
            # Get tool worker heartbeats
            worker_hbs = orchestrator.get_tool_pool_heartbeats()
            workers_str = ""
            if worker_hbs:
                workers_str = " workers=[" + ",".join(
                    f"W{w['id']}:jobs={w['jobs']}" for w in worker_hbs
                ) + "]"
            
            logger.info(
                f"[HEARTBEAT] role=Brain pid={os.getpid()} "
                f"q_in={q_in_size} q_out={q_out_size} "
                f"last_job={last_job_id} interrupt_gen={interrupt_generation}{workers_str}"
            )
            last_heartbeat = current_time
        
        # Check if a timer has finished and announce it
        if check_timer_finished():
            logger.info("[TIMER] Timer finished, announcing alarm")
            tts_controller.enqueue("Your timer is finished.", meta={"_timer_alarm": True})
        
        # Use timeout so we can poll for timer completion
        try:
            msg = core_to_brain_q.get(timeout=0.1)
        except queue.Empty:
            continue
        mtype = (msg or {}).get("type")

        if mtype == "SHUTDOWN":
            safe_put(brain_to_core_q, {"type": "LOG", "level": "INFO", "msg": "brain_worker_shutdown"})
            try:
                tts_controller.shutdown()
            except Exception:
                pass
            orchestrator.shutdown_tool_pool()
            
            # Stop llamacpp server if we started it (Phase 8)
            if llamacpp_base_url:
                try:
                    from wyzer.brain.llama_server_manager import stop_server
                    logger.info("[LLAMACPP] Stopping embedded server...")
                    stop_server(force=True)  # Force stop any llama-server on Wyzer shutdown
                except Exception as e:
                    logger.warning(f"[LLAMACPP] Error stopping server: {e}")
            
            return

        if mtype == "INTERRUPT":
            interrupt_generation += 1
            tts_controller.interrupt()
            safe_put(brain_to_core_q, {"type": "LOG", "level": "INFO", "msg": "interrupt_ack"})
            continue

        if mtype not in {"AUDIO", "TEXT"}:
            safe_put(
                brain_to_core_q,
                {"type": "LOG", "level": "WARNING", "msg": f"unknown_msg_type:{mtype}"},
            )
            continue

        req_id = msg.get("id") or ""
        last_job_id = req_id  # Track for heartbeat
        request_gen = interrupt_generation
        start_ms = now_ms()
        meta = msg.get("meta") or {}
        
        # Check if this is just a TTS prompt (not a user query to process)
        if meta.get("is_followup_prompt"):
            # Just TTS the prompt, don't process it through orchestrator
            prompt_text = str(msg.get("text") or "")
            if prompt_text:
                tts_controller.enqueue(prompt_text, meta={"_prompt_only": True})
            continue

        try:
            user_text: str = ""
            stt_ms = 0

            if mtype == "AUDIO":
                stt_start = now_ms()

                wav_path = msg.get("wav_path")
                pcm_bytes = msg.get("pcm_bytes")

                if wav_path:
                    audio = _read_wav_to_float32(wav_path)
                    try:
                        os.unlink(wav_path)
                    except Exception:
                        pass
                elif pcm_bytes:
                    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
                else:
                    audio = np.array([], dtype=np.float32)

                user_text = stt.transcribe(audio)
                stt_ms = now_ms() - stt_start

                # Check if transcript is valid (not empty/minimal)
                # This handles cases where VAD picked up noise or hotword bleed-through
                # but no real speech was captured. We set capture_valid=False so the
                # core process knows NOT to enter follow-up mode.
                capture_valid = _is_capture_valid(user_text)

                if not user_text or not capture_valid:
                    safe_put(
                        brain_to_core_q,
                        {
                            "type": "RESULT",
                            "id": req_id,
                            "reply": "(I didn't catch that.)",
                            "tool_calls": None,
                            "tts_text": None,
                            "meta": {
                                "timings": {
                                    "stt_ms": stt_ms,
                                    "llm_ms": 0,
                                    "tool_ms": 0,
                                    "tts_start_ms": None,
                                    "total_ms": now_ms() - start_ms,
                                },
                                "is_followup": meta.get("is_followup", False),  # Preserve followup flag
                                "followup_chain": meta.get("followup_chain"),  # Preserve chain count
                                "capture_valid": False,  # Invalid capture - don't enter follow-up
                                "user_text": user_text,  # Include for debugging even if invalid
                            },
                        },
                    )
                    continue

            else:
                user_text = str(msg.get("text") or "")

            # =========================================================================
            # PHASE 11: PENDING CONFIRMATION CHECK (BEFORE exit phrase detection!)
            # =========================================================================
            # If there's a pending confirmation, handle yes/no/cancel BEFORE exit phrase
            # detection. This prevents "no, cancel that" from being caught by exit phrases.
            is_followup = meta.get("is_followup", False)
            if user_text:
                from wyzer.policy.pending_confirmation import (
                    resolve_pending,
                    get_pending_prompt,
                    has_active_pending,
                )
                
                # Define speak function for "Okay, cancelled" TTS
                def _speak_confirmation(text: str) -> None:
                    if tts_enabled and tts_router and text:
                        tts_controller.enqueue(text, meta={"_confirmation": True})
                
                # Define executor that uses the orchestrator's execute_tool_plan
                def _execute_confirmed_plan(plan):
                    from wyzer.tools.registry import build_default_registry
                    from wyzer.core.intent_plan import Intent
                    from wyzer.core.orchestrator import _execute_intents
                    
                    registry = build_default_registry()
                    # plan is a list of dicts: [{"tool": "...", "args": {...}}]
                    intents = []
                    for p in plan:
                        if isinstance(p, dict) and p.get("tool"):
                            intents.append(Intent(
                                tool=p["tool"],
                                args=p.get("args", {}),
                                continue_on_error=p.get("continue_on_error", False)
                            ))
                    if intents:
                        summary = _execute_intents(intents, registry)
                        # Build a reply from the execution
                        results = []
                        for r in summary.ran:
                            if r.ok:
                                result_str = str(r.result.get("message", "Done")) if isinstance(r.result, dict) else "Done"
                                results.append(result_str)
                            else:
                                results.append(f"Failed: {r.error}")
                        return "; ".join(results) if results else "Done"
                    return "Done"
                
                confirmation_result = resolve_pending(
                    user_text,
                    executor=_execute_confirmed_plan,
                    speak_fn=_speak_confirmation,
                )
                
                if confirmation_result == "executed":
                    # Confirmation was handled - return early with success
                    total_ms = now_ms() - start_ms
                    safe_put(
                        brain_to_core_q,
                        {
                            "type": "RESULT",
                            "id": req_id,
                            "reply": "Done.",
                            "tool_calls": None,
                            "tts_text": "Done.",
                            "meta": {
                                "timings": {
                                    "stt_ms": stt_ms,
                                    "llm_ms": 0,
                                    "tool_ms": 0,
                                    "tts_start_ms": now_ms(),
                                    "total_ms": total_ms,
                                },
                                "user_text": user_text,
                                "is_followup": is_followup,
                                "confirmation_result": "executed",
                                "capture_valid": True,
                                "suppress_exit_once": True,  # Prevent Core exit detection
                            },
                        },
                    )
                    # TTS the result
                    if tts_enabled and tts_router:
                        tts_controller.enqueue("Done.", meta={"_confirmation": True})
                    continue
                
                elif confirmation_result == "cancelled":
                    # User cancelled - "Okay, cancelled" already spoken by resolve_pending
                    total_ms = now_ms() - start_ms
                    safe_put(
                        brain_to_core_q,
                        {
                            "type": "RESULT",
                            "id": req_id,
                            "reply": "Okay, cancelled.",
                            "tool_calls": None,
                            "tts_text": "Okay, cancelled.",
                            "meta": {
                                "timings": {
                                    "stt_ms": stt_ms,
                                    "llm_ms": 0,
                                    "tool_ms": 0,
                                    "tts_start_ms": now_ms(),
                                    "total_ms": total_ms,
                                },
                                "user_text": user_text,
                                "is_followup": is_followup,
                                "confirmation_result": "cancelled",
                                "capture_valid": True,
                                "suppress_exit_once": True,  # Prevent Core exit detection
                            },
                        },
                    )
                    continue
                
                elif confirmation_result == "expired":
                    # Confirmation expired - tell user
                    expired_reply = "That confirmation has expired."
                    total_ms = now_ms() - start_ms
                    safe_put(
                        brain_to_core_q,
                        {
                            "type": "RESULT",
                            "id": req_id,
                            "reply": expired_reply,
                            "tool_calls": None,
                            "tts_text": expired_reply,
                            "meta": {
                                "timings": {
                                    "stt_ms": stt_ms,
                                    "llm_ms": 0,
                                    "tool_ms": 0,
                                    "tts_start_ms": now_ms(),
                                    "total_ms": total_ms,
                                },
                                "user_text": user_text,
                                "is_followup": is_followup,
                                "confirmation_result": "expired",
                                "capture_valid": True,
                                "suppress_exit_once": True,  # Prevent Core exit detection
                            },
                        },
                    )
                    if tts_enabled and tts_router:
                        tts_controller.enqueue(expired_reply, meta={"_confirmation": True})
                    continue
                
                elif confirmation_result == "ignored":
                    # User said something other than yes/no while pending exists
                    # Re-ask the confirmation prompt
                    pending_prompt = get_pending_prompt()
                    if pending_prompt:
                        logger.info(f"[CONFIRM] re-asking: {pending_prompt}")
                        total_ms = now_ms() - start_ms
                        safe_put(
                            brain_to_core_q,
                            {
                                "type": "RESULT",
                                "id": req_id,
                                "reply": pending_prompt,
                                "tool_calls": None,
                                "tts_text": pending_prompt,
                                "meta": {
                                    "timings": {
                                        "stt_ms": stt_ms,
                                        "llm_ms": 0,
                                        "tool_ms": 0,
                                        "tts_start_ms": now_ms(),
                                        "total_ms": total_ms,
                                    },
                                    "user_text": user_text,
                                    "is_followup": is_followup,
                                    "confirmation_result": "ignored",
                                    "capture_valid": True,
                                },
                            },
                        )
                        if tts_enabled and tts_router:
                            tts_controller.enqueue(pending_prompt, meta={"_confirmation": True})
                        continue
                
                # confirmation_result == "none" - no pending confirmation, continue normal flow

            # Check if this is an exit phrase (in any mode) - skip orchestrator processing
            # This is the SINGLE source of truth for exit detection in multiprocess mode.
            # We use check_exit_phrase() which returns a sentinel, preventing double-detection.
            # NOTE: If there's a pending confirmation, we already handled it above.
            exit_sentinel = None
            if user_text:
                # GUARD: Don't treat as exit phrase if there's still a pending confirmation
                # (This shouldn't happen since we handle all pending cases above, but just in case)
                if not has_active_pending():
                    followup_mgr = FollowupManager()
                    exit_sentinel = followup_mgr.check_exit_phrase(user_text, log_detection=True)
                if exit_sentinel:
                    # Exit phrase detected - don't process through orchestrator, return empty
                    # Include the sentinel in meta so core process knows not to re-check
                    reply = ""
                    tts_text = None
                    tool_calls = None
                    tool_ms = 0
                    llm_ms = 0
                    
                    total_ms = now_ms() - start_ms
                    safe_put(
                        brain_to_core_q,
                        {
                            "type": "RESULT",
                            "id": req_id,
                            "reply": reply,
                            "tool_calls": tool_calls,
                            "tts_text": tts_text,
                            "meta": {
                                "timings": {
                                    "stt_ms": stt_ms,
                                    "llm_ms": llm_ms,
                                    "tool_ms": tool_ms,
                                    "tts_start_ms": None,
                                    "total_ms": total_ms,
                                },
                                "tts_interrupted": False,
                                "user_text": user_text,
                                "is_followup": is_followup,
                                "followup_chain": meta.get("followup_chain"),
                                "show_followup_prompt": False,  # CRITICAL: Don't show follow-up prompt on exit
                                "exit_sentinel": exit_sentinel,  # Sentinel for core to skip re-detection
                            },
                        },
                    )
                    continue

            # Check if this is an explicit memory command (Phase 7)
            # Memory commands bypass tools/LLM entirely
            from wyzer.memory.command_detector import handle_memory_command
            memory_result = handle_memory_command(user_text) if user_text else None
            if memory_result:
                reply, memory_meta = memory_result
                tts_text = reply
                tts_start_ms = now_ms()
                
                # Enqueue TTS for memory response
                if tts_enabled and tts_router and reply:
                    tts_controller.enqueue(reply, meta={"_memory_response": True})
                
                # Record this turn in session memory
                # For RECALL commands, preserve the recall result so "use that" can promote it
                from wyzer.memory.memory_manager import get_memory_manager
                mem_mgr = get_memory_manager()
                is_recall = memory_meta.get("memory_action") == "recall"
                mem_mgr.add_session_turn(user_text, reply, preserve_recall=is_recall)
                
                total_ms = now_ms() - start_ms
                safe_put(
                    brain_to_core_q,
                    {
                        "type": "RESULT",
                        "id": req_id,
                        "reply": reply,
                        "tool_calls": None,
                        "tts_text": tts_text,
                        "meta": {
                            "timings": {
                                "stt_ms": stt_ms,
                                "llm_ms": 0,  # No LLM call
                                "tool_ms": 0,  # No tool call
                                "tts_start_ms": tts_start_ms,
                                "total_ms": total_ms,
                            },
                            "tts_interrupted": False,
                            "user_text": user_text,
                            "is_followup": is_followup,
                            "followup_chain": meta.get("followup_chain"),
                            "show_followup_prompt": False,
                            "memory_action": memory_meta.get("memory_action"),
                            "capture_valid": True,
                        },
                    },
                )
                continue

            # Check if this is a "how do you know" source question (Phase 7 polish)
            # These bypass LLM to give truthful deterministic answers
            from wyzer.memory.command_detector import handle_source_question
            source_result = handle_source_question(user_text) if user_text else None
            if source_result:
                reply, source_meta = source_result
                tts_text = reply
                tts_start_ms = now_ms()
                
                # Enqueue TTS for source response
                if tts_enabled and tts_router and reply:
                    tts_controller.enqueue(reply, meta={"_source_response": True})
                
                # Record this turn in session memory
                from wyzer.memory.memory_manager import get_memory_manager
                mem_mgr = get_memory_manager()
                mem_mgr.add_session_turn(user_text, reply)
                
                total_ms = now_ms() - start_ms
                safe_put(
                    brain_to_core_q,
                    {
                        "type": "RESULT",
                        "id": req_id,
                        "reply": reply,
                        "tool_calls": None,
                        "tts_text": tts_text,
                        "meta": {
                            "timings": {
                                "stt_ms": stt_ms,
                                "llm_ms": 0,  # No LLM call
                                "tool_ms": 0,  # No tool call
                                "tts_start_ms": tts_start_ms,
                                "total_ms": total_ms,
                            },
                            "tts_interrupted": False,
                            "user_text": user_text,
                            "is_followup": is_followup,
                            "followup_chain": meta.get("followup_chain"),
                            "show_followup_prompt": False,
                            "source_question": True,
                            "capture_valid": True,
                        },
                    },
                )
                continue

            # LLM + tools via orchestrator
            # Always call orchestrator - it handles NO_OLLAMA mode internally
            # and the hybrid router can handle deterministic commands without LLM
            llm_start = now_ms()
            from wyzer.core.orchestrator import handle_user_text, should_use_streaming_tts, handle_user_text_streaming

            # =========================================================================
            # PHASE 10: REFERENCE RESOLUTION (before routing decision)
            # =========================================================================
            # Resolve vague follow-up phrases BEFORE checking streaming/routing.
            # "close it" → "close Chrome", "do that again" → repeat last action, etc.
            # This MUST happen before should_use_streaming_tts() to ensure resolved
            # commands go through the tool path, not the streaming reply-only path.
            # NOTE: We preserve original_user_text for display - user should see what
            # they actually said, not the resolved version.
            from wyzer.core.reference_resolver import resolve_references, is_replay_sentinel
            from wyzer.context.world_state import get_world_state
            
            original_user_text = user_text  # Preserve for display
            resolved_text = resolve_references(user_text, get_world_state())
            if resolved_text != user_text:
                logger.info(f'[REF_RESOLVE] "{user_text}" → "{resolved_text}"')
                user_text = resolved_text  # Use resolved for routing

            # =========================================================================
            # PHASE 10.1: REPLAY SENTINEL CHECK (before streaming decision)
            # =========================================================================
            # If reference resolver returned the replay sentinel, ALWAYS go through
            # non-streaming path so handle_user_text can execute the deterministic replay.
            # This prevents the sentinel from being treated as a conversational query.
            force_non_streaming = is_replay_sentinel(user_text)

            # Check if we should use streaming TTS for this request
            # Streaming is ONLY for conversational/reply-only queries, NOT for tool commands
            # Phase 10.1: Also skip streaming if this is a replay request
            use_streaming_tts = should_use_streaming_tts(user_text) and not force_non_streaming
            result_streamed = False
            
            if use_streaming_tts:
                # Use streaming path: tokens flow to TTS as they arrive
                # The on_segment callback enqueues each segment for TTS
                tts_controller.clear_stop()  # Ensure stop flag is cleared
                
                def on_tts_segment(segment: str) -> None:
                    """Callback for streaming TTS segments."""
                    if segment and not tts_controller.is_cancelled():
                        tts_controller.enqueue(segment, meta={"_streaming": True})
                
                result_dict = handle_user_text_streaming(
                    user_text,
                    on_segment=on_tts_segment,
                    cancel_check=tts_controller.is_cancelled
                )
                result_streamed = (result_dict or {}).get("meta", {}).get("streamed", False)
            else:
                # Non-streaming path (tools, hybrid router, etc.)
                result_dict = handle_user_text(user_text)
            
            llm_ms = now_ms() - llm_start

            reply = (result_dict or {}).get("reply", "")
            exec_summary = (result_dict or {}).get("execution_summary")
            tool_calls = None
            tool_ms = 0
            if exec_summary and isinstance(exec_summary, dict):
                tool_calls = exec_summary.get("ran")
                try:
                    for r in tool_calls or []:
                        res = (r or {}).get("result")
                        if isinstance(res, dict) and isinstance(res.get("latency_ms"), int):
                            tool_ms += int(res.get("latency_ms") or 0)
                except Exception:
                    tool_ms = 0

            # Record this turn in session memory (after successful orchestrator response)
            # Use original_user_text so conversation history reflects what user actually said
            from wyzer.memory.memory_manager import get_memory_manager
            mem_mgr = get_memory_manager()
            if original_user_text and reply:
                mem_mgr.add_session_turn(original_user_text, reply)

            tts_text: Optional[str] = reply
            tts_start_ms: Optional[int] = None

            tts_interrupted = False
            
            # Determine if we should show follow-up prompt
            # Only show prompt if response ends with a question mark
            # (i.e., the assistant is asking the user something)
            show_followup_prompt = False
            if reply and reply.rstrip().endswith("?"):
                # Response is a question - show follow-up prompt to listen for answer
                show_followup_prompt = True

            # If user interrupted while we were processing, do not speak stale reply.
            if request_gen != interrupt_generation:
                tts_text = None
                tts_interrupted = True
            elif not tts_enabled or not tts_router:
                tts_text = None
            elif result_streamed:
                # Already enqueued TTS segments during streaming - don't enqueue again
                # BUT we need to send the show_followup_prompt flag for the last segment
                tts_text = None
                # Send a stream-end marker so TTS controller knows to trigger followup
                tts_controller.enqueue("", meta={
                    "_stream_end": True,
                    "show_followup_prompt": show_followup_prompt,
                })

            # Enqueue speech (non-blocking)
            if tts_text:
                tts_start_ms = now_ms()
                tts_meta = msg.get("meta") or {}
                tts_meta["show_followup_prompt"] = show_followup_prompt
                tts_controller.enqueue(tts_text, meta=tts_meta)

            total_ms = now_ms() - start_ms

            # Phase 11: Extract orchestrator meta for confirmation flags
            orch_meta = (result_dict or {}).get("meta", {})
            
            safe_put(
                brain_to_core_q,
                {
                    "type": "RESULT",
                    "id": req_id,
                    "reply": reply,
                    "tool_calls": tool_calls,
                    "tts_text": tts_text,
                    "meta": {
                        "timings": {
                            "stt_ms": stt_ms,
                            "llm_ms": llm_ms,
                            "tool_ms": tool_ms,
                            "tts_start_ms": tts_start_ms,
                            "total_ms": total_ms,
                        },
                        "tts_interrupted": tts_interrupted,
                        "user_text": original_user_text,  # Display what user actually said
                        "resolved_text": user_text if user_text != original_user_text else None,  # For debugging
                        "is_followup": meta.get("is_followup", False),  # Preserve followup flag
                        "followup_chain": meta.get("followup_chain"),  # Preserve chain count
                        "show_followup_prompt": show_followup_prompt,  # Whether to show prompt
                        "has_tool_calls": bool(tool_calls),  # Whether response involved tools
                        "capture_valid": True,  # Valid capture - allow follow-up if appropriate
                        # Phase 11: Forward confirmation flags from orchestrator to Core
                        "has_pending_confirmation": orch_meta.get("has_pending_confirmation", False),
                        "confirmation_timeout_sec": orch_meta.get("confirmation_timeout_sec"),
                        "autonomy_action": orch_meta.get("autonomy_action"),
                    },
                },
            )

        except Exception as e:
            err = str(e)
            safe_put(brain_to_core_q, {"type": "LOG", "level": "ERROR", "msg": f"brain_worker_error:{err}", "meta": {"trace": traceback.format_exc()}})
            safe_put(
                brain_to_core_q,
                {
                    "type": "RESULT",
                    "id": req_id,
                    "reply": f"(error: {err})",
                    "tool_calls": None,
                    "tts_text": None,
                    "meta": {
                        "timings": {"total_ms": now_ms() - start_ms},
                        "error": True,
                        "is_followup": meta.get("is_followup", False),  # Preserve followup flag
                        "followup_chain": meta.get("followup_chain"),  # Preserve chain count
                    },
                },
            )
