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
from typing import Any, Dict, Optional

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
    def __init__(self, tts: Optional[TTSRouter], brain_to_core_q, simulate: bool = False):
        self._tts = tts
        self._simulate = simulate
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="BrainTTS", daemon=True)
        self._thread.start()
        self._brain_to_core_q = brain_to_core_q

    def enqueue(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not text:
            return
        self._queue.put({"text": text, "meta": meta or {}})

    def interrupt(self) -> None:
        with self._lock:
            self._stop_event.set()
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass

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

    def _simulate_speak(self, duration_sec: float) -> bool:
        end = time.time() + max(0.0, duration_sec)
        while time.time() < end:
            if self._stop_event.is_set():
                return False
            time.sleep(0.05)
        return True

    def _loop(self) -> None:
        logger = get_logger()
        while self._running:
            item = self._queue.get()
            meta = item.get("meta") or {}
            if meta.get("_shutdown"):
                return

            text = (item.get("text") or "").strip()
            if not text:
                continue

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
                safe_put(
                    self._brain_to_core_q,
                    {
                        "type": "LOG",
                        "level": "DEBUG",
                        "msg": "tts_finished" if ok else "tts_interrupted",
                        "meta": {"show_followup_prompt": meta.get("show_followup_prompt", False)},
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
    if llm_mode != "ollama":
        logger.info("LLM disabled in brain worker")

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

            # Check if this is an exit phrase (in any mode) - skip orchestrator processing
            # This is the SINGLE source of truth for exit detection in multiprocess mode.
            # We use check_exit_phrase() which returns a sentinel, preventing double-detection.
            is_followup = meta.get("is_followup", False)
            exit_sentinel = None
            if user_text:
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

            # Check if we should use streaming TTS for this request
            # Streaming is ONLY for conversational/reply-only queries, NOT for tool commands
            use_streaming_tts = should_use_streaming_tts(user_text)
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
            from wyzer.memory.memory_manager import get_memory_manager
            mem_mgr = get_memory_manager()
            if user_text and reply:
                mem_mgr.add_session_turn(user_text, reply)

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
                tts_text = None

            # Enqueue speech (non-blocking)
            if tts_text:
                tts_start_ms = now_ms()
                tts_meta = msg.get("meta") or {}
                tts_meta["show_followup_prompt"] = show_followup_prompt
                tts_controller.enqueue(tts_text, meta=tts_meta)

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
                            "tts_start_ms": tts_start_ms,
                            "total_ms": total_ms,
                        },
                        "tts_interrupted": tts_interrupted,
                        "user_text": user_text,
                        "is_followup": meta.get("is_followup", False),  # Preserve followup flag
                        "followup_chain": meta.get("followup_chain"),  # Preserve chain count
                        "show_followup_prompt": show_followup_prompt,  # Whether to show prompt
                        "has_tool_calls": bool(tool_calls),  # Whether response involved tools
                        "capture_valid": True,  # Valid capture - allow follow-up if appropriate
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
