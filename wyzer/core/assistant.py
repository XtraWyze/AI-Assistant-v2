"""wyzer.core.assistant

Contains the original single-process `WyzerAssistant` and the new multiprocess
split implementation `WyzerAssistantMultiprocess`.

Multiprocess architecture:
- Process A (realtime core): mic stream + VAD + hotword + state machine
- Process B (brain worker): STT + LLM + tools + TTS

The original class is kept for `--single-process` debugging.
"""
import time
import os
import sys
import random
import numpy as np
import threading
import tempfile
import wave
from queue import Queue, Empty
from typing import Optional, List, Any, Dict
from wyzer.core.config import Config
from wyzer.core.logger import get_logger
from wyzer.core.state import AssistantState, RuntimeState
from wyzer.core.followup_manager import FollowupManager, is_exit_sentinel
from wyzer.audio.mic_stream import MicStream
from wyzer.audio.vad import VadDetector
from wyzer.audio.hotword import HotwordDetector
from wyzer.audio.audio_utils import concat_audio_frames
from wyzer.audio.audio_utils import audio_to_int16
from wyzer.stt.stt_router import STTRouter
from wyzer.brain.llm_engine import LLMEngine
from wyzer.tts.tts_router import TTSRouter
from wyzer.core.ipc import new_id, safe_put
from wyzer.core.process_manager import start_brain_process, stop_brain_process


class WyzerAssistant:
    """Main Wyzer Assistant coordinator"""
    
    def __init__(
        self,
        enable_hotword: bool = True,
        whisper_model: str = "small",
        whisper_device: str = "cpu",
        audio_device: Optional[int] = None,
        llm_mode: str = "ollama",
        ollama_model: str = "llama3.1:latest",
        ollama_url: str = "http://127.0.0.1:11434",
        llm_timeout: int = 30,
        tts_enabled: bool = True,
        tts_engine: str = "piper",
        piper_exe_path: str = "./wyzer/assets/piper/piper.exe",
        piper_model_path: str = "./wyzer/assets/piper/en_US-lessac-medium.onnx",
        piper_speaker_id: Optional[int] = None,
        tts_output_device: Optional[int] = None,
        speak_hotword_interrupt: bool = True
    ):
        """
        Initialize Wyzer Assistant
        
        Args:
            enable_hotword: Enable hotword detection
            whisper_model: Whisper model size
            whisper_device: Device for Whisper
            audio_device: Optional audio device index
            llm_mode: LLM mode ("ollama" or "off")
            ollama_model: Ollama model name
            ollama_url: Ollama API base URL
            llm_timeout: LLM request timeout in seconds
            tts_enabled: Enable TTS
            tts_engine: TTS engine ("piper")
            piper_exe_path: Path to Piper executable
            piper_model_path: Path to Piper model
            piper_speaker_id: Optional Piper speaker ID
            tts_output_device: Optional TTS output device
            speak_hotword_interrupt: Enable barge-in during speaking
        """
        self.logger = get_logger()
        self.enable_hotword = enable_hotword
        self.speak_hotword_interrupt = speak_hotword_interrupt
        
        # Initialize state
        self.state = RuntimeState()
        self.running = False
        
        # FOLLOWUP listening window manager
        self.followup_manager = FollowupManager()
        
        # Hotword cooldown tracking
        self.last_hotword_time: float = 0.0
        
        # Track if current speaking response is from FOLLOWUP
        self._is_followup_response: bool = False
        
        # Track if last response ended with a question (for conditional follow-up)
        self._last_response_is_question: bool = False

        # Post-barge-in speech gating
        self._bargein_pending_speech: bool = False
        self._bargein_wait_speech_deadline_ts: float = 0.0
        self._speaking_start_ts: float = 0.0
        
        # No-speech-start timeout tracking: deadline for speech to begin after LISTENING starts
        self._no_speech_start_deadline_ts: float = 0.0
        
        # Background transcription
        self.stt_thread: Optional[threading.Thread] = None
        self.stt_result: Optional[str] = None
        
        # Background thinking (Phase 4)
        self.thinking_thread: Optional[threading.Thread] = None
        self.thinking_result: Optional[dict] = None
        
        # Background speaking (Phase 5)
        self.speaking_thread: Optional[threading.Thread] = None
        self.tts_stop_event: threading.Event = threading.Event()
        
        # Audio buffer for recording
        self.audio_buffer: List[np.ndarray] = []
        
        # Initialize components
        self.logger.info("Initializing Wyzer Assistant...")
        
        # Audio stream
        self.audio_queue: Queue = Queue(maxsize=Config.AUDIO_QUEUE_MAX_SIZE)
        self.mic_stream = MicStream(
            audio_queue=self.audio_queue,
            device=audio_device
        )
        
        # VAD
        self.vad = VadDetector()
        
        # Hotword detector (if enabled)
        self.hotword: Optional[HotwordDetector] = None
        if self.enable_hotword:
            try:
                self.hotword = HotwordDetector()
                self.logger.info(f"Hotword detection enabled for: {Config.HOTWORD_KEYWORDS}")
            except Exception as e:
                self.logger.error(f"Failed to initialize hotword detector: {e}")
                self.logger.warning("Continuing without hotword detection")
                self.enable_hotword = False
        
        # STT router
        self.stt = STTRouter(
            whisper_model=whisper_model,
            whisper_device=whisper_device
        )
        
        # LLM Brain (Phase 4)
        # LLM Brain (Phase 4)
        self.brain: Optional[LLMEngine] = None
        if llm_mode == "ollama":
            self.brain = LLMEngine(
                base_url=ollama_url,
                model=ollama_model,
                timeout=llm_timeout,
                enabled=True
            )
        else:
            self.logger.info("LLM brain disabled (STT-only mode)")
        
        # TTS (Phase 5)
        self.tts: Optional[TTSRouter] = None
        if tts_enabled:
            try:
                self.tts = TTSRouter(
                    engine=tts_engine,
                    piper_exe_path=piper_exe_path,
                    piper_model_path=piper_model_path,
                    piper_speaker_id=piper_speaker_id,
                    output_device=tts_output_device,
                    enabled=True
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize TTS: {e}")
                self.logger.warning("TTS will be disabled")
                self.tts = None
        else:
            self.logger.info("TTS disabled")
        
        self.logger.info("Wyzer Assistant initialized successfully")
    
    def start(self) -> None:
        """Start the assistant"""
        if self.running:
            self.logger.warning("Assistant already running")
            return
        
        self.logger.info("Starting Wyzer Assistant...")
        self.running = True
        
        # Start audio stream
        self.mic_stream.start()
        
        # If hotword disabled, immediately start listening
        if not self.enable_hotword:
            self.logger.info("No-hotword mode: Starting immediate listening")
            self.state.transition_to(AssistantState.LISTENING)
            self.logger.info("Listening... (speak now)")
        else:
            self.logger.info(f"Listening for hotword: {Config.HOTWORD_KEYWORDS}")
        
        # Run main loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the assistant"""
        if not self.running:
            return
        
        self.logger.info("Stopping Wyzer Assistant...")
        self.running = False
        
        # Stop TTS if speaking
        if self.tts_stop_event:
            self.tts_stop_event.set()
        
        # Stop audio stream
        self.mic_stream.stop()
        
        # Wait briefly for threads to finish
        if self.speaking_thread and self.speaking_thread.is_alive():
            self.speaking_thread.join(timeout=0.5)
        
        self.logger.info("Wyzer Assistant stopped")
    
    def interrupt_current_process(self) -> None:
        """
        Interrupt the current process cleanly.
        Handles interruptions during all states without breaking the system.
        """
        self.logger.info(f"Interrupt requested from state: {self.state.current_state.value}")
        
        # Mark interrupt in state
        self.state.request_interrupt()
        
        # Stop TTS if speaking
        if self.tts_stop_event:
            self.tts_stop_event.set()
        
        # Handle based on current state
        if self.state.is_in_state(AssistantState.SPEAKING):
            # Already handled by hotword interrupt mechanism
            self.logger.info("Interrupting speech")
            # Wait for speaking thread to stop
            if self.speaking_thread:
                self.speaking_thread.join(timeout=0.3)
            self.speaking_thread = None
        
        elif self.state.is_in_state(AssistantState.LISTENING):
            # Stop listening and return to idle
            self.logger.info("Interrupting listening session")
            self._drain_audio_queue(0.05)
        
        elif self.state.is_in_state(AssistantState.THINKING):
            # Send interrupt to brain worker
            self.logger.info("Interrupting thinking process")
            if hasattr(self, 'core_to_brain_q') and self.core_to_brain_q:
                try:
                    safe_put(self.core_to_brain_q, {"type": "INTERRUPT"}, timeout=0.1)
                except Exception as e:
                    self.logger.warning(f"Could not send interrupt to brain: {e}")
        
        elif self.state.is_in_state(AssistantState.TRANSCRIBING):
            # Wait for transcription to finish (can't really interrupt STT)
            self.logger.info("Interrupting transcription")
            if self.stt_thread and self.stt_thread.is_alive():
                self.stt_thread.join(timeout=1.0)
        
        # Reset to idle state
        self._reset_to_idle()
        self.logger.info("Process interrupted successfully")
    
    
    def _main_loop(self) -> None:
        """Main processing loop"""
        while self.running:
            # Get audio frame from queue (with timeout)
            try:
                audio_frame = self.audio_queue.get(timeout=0.1)
            except Empty:
                # Check if transcription thread finished
                if self.state.is_in_state(AssistantState.TRANSCRIBING):
                    self._check_transcription_complete()
                # Check if thinking thread finished
                elif self.state.is_in_state(AssistantState.THINKING):
                    self._check_thinking_complete()
                # Check if speaking thread finished
                elif self.state.is_in_state(AssistantState.SPEAKING):
                    self._check_speaking_complete()
                # Check FOLLOWUP timeout
                elif self.state.is_in_state(AssistantState.FOLLOWUP):
                    if self.followup_manager.check_timeout():
                        self._reset_to_idle()
                continue
            
            # Process frame based on current state
            if self.state.is_in_state(AssistantState.IDLE):
                self._process_idle(audio_frame)
            
            elif self.state.is_in_state(AssistantState.LISTENING):
                self._process_listening(audio_frame)
            
            elif self.state.is_in_state(AssistantState.FOLLOWUP):
                self._process_followup(audio_frame)
            
            elif self.state.is_in_state(AssistantState.TRANSCRIBING):
                # In transcribing state, drain frames to prevent queue overflow
                # Don't process them, just discard
                # Also check if transcription is complete
                self._check_transcription_complete()
            
            elif self.state.is_in_state(AssistantState.THINKING):
                # In thinking state, drain frames and check if thinking is complete
                self._check_thinking_complete()
            
            elif self.state.is_in_state(AssistantState.SPEAKING):
                # In speaking state, drain frames BUT check for hotword interrupt
                self._process_speaking(audio_frame)
                self._check_speaking_complete()
    
    def _process_idle(self, audio_frame: np.ndarray) -> None:
        """
        Process audio frame in IDLE state
        
        Args:
            audio_frame: Audio frame to process
        """
        # In IDLE, check for hotword (if enabled)
        if self.enable_hotword and self.hotword:
            current_time = time.time()
            detected_keyword, score = self.hotword.detect(audio_frame)
            
            if detected_keyword:
                # Check cooldown period
                time_since_last = current_time - self.last_hotword_time
                
                if time_since_last < Config.HOTWORD_COOLDOWN_SEC:
                    self.logger.debug(
                        f"Hotword cooldown active: {time_since_last:.2f}s < {Config.HOTWORD_COOLDOWN_SEC}s (score: {score:.3f})"
                    )
                    return
                
                self.logger.info(f"Hotword '{detected_keyword}' accepted after cooldown")
                self.last_hotword_time = current_time
                
                # Drain a bit more to remove residual hotword audio
                self._drain_audio_queue(Config.POST_IDLE_DRAIN_SEC)
                
                # Transition to LISTENING
                self.state.transition_to(AssistantState.LISTENING)
                self.audio_buffer = []
                
                # Set no-speech-start deadline for quick abort if user stays silent
                self._no_speech_start_deadline_ts = time.time() + Config.NO_SPEECH_START_TIMEOUT_SEC
                
                self.logger.info("Listening... (speak now)")
    
    def _process_listening(self, audio_frame: np.ndarray) -> None:
        """
        Process audio frame in LISTENING state
        
        Args:
            audio_frame: Audio frame to process
        """
        # Check if we're in post-barge-in waiting for speech mode
        if self._bargein_pending_speech:
            current_time = time.time()
            
            # Check if wait deadline has expired
            if current_time > self._bargein_wait_speech_deadline_ts:
                self.logger.info("Post-barge-in speech wait timeout - returning to IDLE")
                self._clear_bargein_flags()
                self._reset_to_idle()
                return
        
        # Add frame to buffer
        self.audio_buffer.append(audio_frame)
        self.state.total_frames_recorded += 1
        
        # Check for speech using VAD
        is_speech = self.vad.is_speech(audio_frame)
        
        if is_speech:
            # Speech detected
            if not self.state.speech_detected:
                self.logger.debug("Speech started")
                self.state.speech_detected = True
                
                # Clear no-speech-start deadline since speech has begun
                self._no_speech_start_deadline_ts = 0.0
                
                # Clear barge-in pending flag when speech actually starts
                if self._bargein_pending_speech:
                    self.logger.debug("Speech started after barge-in - clearing pending flag")
                    self._bargein_pending_speech = False
            
            self.state.speech_frames_count += 1
            self.state.silence_frames = 0
        else:
            # No speech in this frame
            if self.state.speech_detected:
                # We've detected speech before, so this is silence after speech
                self.state.silence_frames += 1
        
        # Check stop conditions
        should_stop = False
        stop_reason = ""
        
        # 0. No-speech-start timeout: abort early if user never begins speaking
        #    This provides a fast exit when user triggers hotword but stays silent.
        #    Skip this check if barge-in is pending (use barge-in grace period instead).
        if (not self.state.speech_detected 
            and not self._bargein_pending_speech
            and self._no_speech_start_deadline_ts > 0
            and time.time() > self._no_speech_start_deadline_ts):
            self.logger.info(f"[LISTENING] No speech start within {Config.NO_SPEECH_START_TIMEOUT_SEC}s -> abort")
            self._no_speech_start_deadline_ts = 0.0  # Reset deadline
            self._reset_to_idle()
            return
        
        # 1. Silence timeout after speech
        if (self.state.speech_detected and 
            self.state.silence_frames >= Config.get_silence_timeout_frames()):
            should_stop = True
            stop_reason = "silence timeout"
        
        # 2. Maximum recording duration
        if self.state.total_frames_recorded >= Config.get_max_record_frames():
            should_stop = True
            stop_reason = "max duration"
        
        # 3. In no-hotword mode, stop after one utterance
        if (not self.enable_hotword and 
            self.state.speech_detected and 
            self.state.silence_frames >= Config.get_silence_timeout_frames()):
            should_stop = True
            stop_reason = "utterance complete (no-hotword mode)"
        
        if should_stop:
            self.logger.info(f"Recording stopped: {stop_reason}")
            
            # Only transcribe if we detected some speech
            if self.state.speech_frames_count > 0:
                self._transcribe_and_reset()
            else:
                self.logger.warning("No speech detected in recording")
                self._reset_to_idle()
    
    def _process_followup(self, audio_frame: np.ndarray) -> None:
        """
        Process audio frame in FOLLOWUP state (no hotword, follow-up listening).
        Similar to LISTENING but with different timeout and exit criteria.
        
        Args:
            audio_frame: Audio frame to process
        """
        # Add frame to buffer
        self.audio_buffer.append(audio_frame)
        self.state.total_frames_recorded += 1
        
        # Check for speech using VAD
        is_speech = self.vad.is_speech(audio_frame)
        
        if is_speech:
            # Speech detected - reset the silence timer
            if not self.state.speech_detected:
                self.logger.debug("FOLLOWUP: Speech started")
                self.state.speech_detected = True
            
            # Reset followup timeout on new speech
            self.followup_manager.reset_speech_timer()
            
            self.state.speech_frames_count += 1
            self.state.silence_frames = 0
        else:
            # No speech in this frame
            if self.state.speech_detected:
                # We've detected speech before, so this is silence after speech
                self.state.silence_frames += 1
        
        # Check stop conditions for FOLLOWUP
        should_stop = False
        stop_reason = ""
        
        # 1. Silence timeout after speech (uses FOLLOWUP_TIMEOUT_SEC)
        if (self.state.speech_detected and 
            self.followup_manager.check_timeout()):
            should_stop = True
            stop_reason = "followup silence timeout"
        
        # 2. Max FOLLOWUP chain depth reached
        if (self.state.speech_detected and 
            self.state.silence_frames >= Config.get_silence_timeout_frames()):
            if not self.followup_manager.increment_chain():
                should_stop = True
                stop_reason = "followup max chain depth"
        
        if should_stop:
            self.logger.info(f"FOLLOWUP stopped: {stop_reason}")
            
            # Only transcribe if we detected some speech
            if self.state.speech_frames_count > 0:
                self._transcribe_followup()
            else:
                self.logger.info("No speech in FOLLOWUP window")
                self.followup_manager.end_followup_window()
                self._reset_to_idle()
    
    def _transcribe_followup(self) -> None:
        """Start background transcription for FOLLOWUP response"""
        # Transition to TRANSCRIBING
        self.state.transition_to(AssistantState.TRANSCRIBING)
        
        # Concatenate audio buffer
        audio_data = concat_audio_frames(self.audio_buffer)
        
        self.logger.info(
            f"FOLLOWUP: Starting transcription of {len(audio_data)/Config.SAMPLE_RATE:.2f}s audio..."
        )
        
        # Start transcription in background thread
        self.stt_result = None
        self.stt_thread = threading.Thread(
            target=self._background_transcribe_followup,
            args=(audio_data,),
            daemon=True
        )
        self.stt_thread.start()
    
    def _background_transcribe_followup(self, audio_data: np.ndarray) -> None:
        """Background thread for STT processing in FOLLOWUP mode"""
        try:
            # Use mode="followup" for STT (can be used for future mode-specific processing)
            transcript = self.stt.transcribe(audio_data, mode="followup")
            self.stt_result = transcript
        except Exception as e:
            self.logger.error(f"Error in FOLLOWUP transcription: {e}")
            self.stt_result = None
    
    def _transcribe_and_reset(self) -> None:
        """Start background transcription and transition to TRANSCRIBING"""
        # Clear barge-in flags when completing listening
        self._clear_bargein_flags()
        
        # Transition to TRANSCRIBING
        self.state.transition_to(AssistantState.TRANSCRIBING)
        
        # Concatenate audio buffer
        audio_data = concat_audio_frames(self.audio_buffer)
        
        self.logger.info(
            f"Starting transcription of {len(audio_data)/Config.SAMPLE_RATE:.2f}s audio in background..."
        )
        
        # Start transcription in background thread
        self.stt_result = None
        self.stt_thread = threading.Thread(
            target=self._background_transcribe,
            args=(audio_data,),
            daemon=True
        )
        self.stt_thread.start()
    
    def _background_transcribe(self, audio_data: np.ndarray) -> None:
        """Background thread for STT processing"""
        try:
            transcript = self.stt.transcribe(audio_data)
            self.stt_result = transcript
        except Exception as e:
            self.logger.error(f"Error in background transcription: {e}")
            self.stt_result = None
    
    def _check_transcription_complete(self) -> None:
        """Check if background transcription is complete and handle result"""
        if self.stt_thread and not self.stt_thread.is_alive():
            # Thread finished
            transcript = self.stt_result
            
            # Determine if this is a FOLLOWUP transcription
            was_followup = self.followup_manager.is_followup_active()
            
            # Display transcript
            if transcript:
                self.logger.info(f"Transcript: {transcript}")
                print(f"\nYou: {transcript}")
                
                # Check if this is an exit phrase using sentinel pattern
                # This is the SINGLE source of truth for exit detection (single-process mode)
                exit_sentinel = self.followup_manager.check_exit_phrase(transcript)
                if exit_sentinel:
                    # Exit phrase detected - skip LLM processing, return to idle silently
                    if was_followup:
                        self.followup_manager.end_followup_window()
                    self._reset_to_idle()
                    return
                
                # Pass to LLM brain if enabled
                if self.brain:
                    self._think_and_respond(transcript, is_followup=was_followup)
                else:
                    # No brain, just show transcript and return to idle/followup
                    if not self.enable_hotword:
                        self.logger.info("No-hotword mode: Exiting after transcription")
                        self.running = False
                    elif was_followup:
                        # Go back to FOLLOWUP listening
                        self._re_enter_followup()
                    else:
                        self._reset_to_idle()
            else:
                self.logger.warning("No valid transcript (empty or filtered as garbage)")
                
                # In no-hotword mode, exit even if no transcript
                if not self.enable_hotword:
                    self.logger.info("No-hotword mode: Exiting")
                    self.running = False
                elif was_followup:
                    # Go back to FOLLOWUP listening
                    self._re_enter_followup()
                else:
                    self._reset_to_idle()
    
    def _reset_to_idle(self) -> None:
        """Reset state to IDLE with queue draining"""
        # Clear barge-in flags when returning to idle
        self._clear_bargein_flags()
        
        self.audio_buffer = []
        
        # Drain audio queue for POST_IDLE_DRAIN_SEC to remove residual wake audio
        drain_frames = int(Config.POST_IDLE_DRAIN_SEC * Config.SAMPLE_RATE / Config.CHUNK_SAMPLES)
        drained_count = 0
        
        self.logger.debug(f"Draining audio queue for {Config.POST_IDLE_DRAIN_SEC}s ({drain_frames} frames)...")
        
        for _ in range(drain_frames):
            try:
                frame = self.audio_queue.get_nowait()
                drained_count += 1
                
                # Process frame through hotword detector to update prev_scores
                # but ignore any detections during drain period
                if self.hotword:
                    self.hotword.detect(frame)
                    
            except Empty:
                break
        
        if drained_count > 0:
            self.logger.debug(f"Drained {drained_count} frames from queue")
        
        self.state.transition_to(AssistantState.IDLE)
        
        if self.enable_hotword:
            self.logger.info(f"Ready. Listening for hotword: {Config.HOTWORD_KEYWORDS}")
    
    def _think_and_respond(self, transcript: str, is_followup: bool = False) -> None:
        """
        Start background LLM thinking and transition to THINKING state
        
        Args:
            transcript: User's transcript
            is_followup: True if this came from FOLLOWUP listening
        """
        # Transition to THINKING
        self.state.transition_to(AssistantState.THINKING)
        
        self.logger.info("Thinking...")
        
        # Start thinking in background thread
        self.thinking_result = None
        self.thinking_thread = threading.Thread(
            target=self._background_think,
            args=(transcript, is_followup),
            daemon=True
        )
        self.thinking_thread.start()
    
    def _background_think(self, transcript: str, is_followup: bool = False) -> None:
        """Background thread for LLM processing"""
        try:
            # Use orchestrator for Phase 6 tool support
            from wyzer.core.orchestrator import handle_user_text
            result_dict = handle_user_text(transcript)
            
            # Convert orchestrator result to expected format
            self.thinking_result = {
                "reply": result_dict.get("reply", ""),
                "confidence": 0.8,
                "model": Config.OLLAMA_MODEL,
                "latency_ms": result_dict.get("latency_ms", 0),
                "is_followup": is_followup
            }
        except Exception as e:
            self.logger.error(f"Error in background thinking: {e}")
            self.thinking_result = {
                "reply": "I encountered an error processing your request.",
                "confidence": 0.3,
                "model": "error",
                "latency_ms": 0,
                "is_followup": is_followup
            }
    
    def _check_thinking_complete(self) -> None:
        """Check if background thinking is complete and handle result"""
        if self.thinking_thread and not self.thinking_thread.is_alive():
            # Thread finished
            result = self.thinking_result
            is_followup = result.get("is_followup", False) if result else False
            
            if result:
                # Display response
                reply = result.get("reply", "")
                latency = result.get("latency_ms", 0)
                
                self.logger.info(f"Response generated in {latency}ms")
                print(f"\nWyzer: {reply}\n")
                
                # Speak response if TTS enabled
                if self.tts:
                    self._speak_and_reset(reply, is_followup=is_followup)
                else:
                    # No TTS, just go to idle/followup or exit
                    if not self.enable_hotword:
                        self.logger.info("No-hotword mode: Exiting after response")
                        self.running = False
                    elif is_followup:
                        # Go back to FOLLOWUP listening
                        self._re_enter_followup()
                    else:
                        self._reset_to_idle()
            else:
                self.logger.warning("No response from LLM brain")
                
                # In no-hotword mode, exit even if no response
                if not self.enable_hotword:
                    self.logger.info("No-hotword mode: Exiting")
                    self.running = False
                elif is_followup:
                    # Go back to FOLLOWUP listening
                    self._re_enter_followup()
                else:
                    self._reset_to_idle()
    
    def _speak_and_reset(self, text: str, is_followup: bool = False) -> None:
        """
        Start background speaking and transition to SPEAKING state
        
        Args:
            text: Text to speak
            is_followup: True if we should re-enter FOLLOWUP after speaking
        """
        # Transition to SPEAKING
        self.state.transition_to(AssistantState.SPEAKING)
        
        self.logger.info("Speaking...")
        
        # Clear stop event
        self.tts_stop_event.clear()
        
        # Record speaking start time for cooldown check
        self._speaking_start_ts = time.time()
        self._is_followup_response = is_followup
        
        # Check if response ends with a question mark (for conditional follow-up)
        self._last_response_is_question = text.strip().endswith('?')

        # Start speaking in background thread
        self.speaking_thread = threading.Thread(
            target=self._background_speak,
            args=(text,),
            daemon=True
        )
        self.speaking_thread.start()
    
    def _background_speak(self, text: str) -> None:
        """Background thread for TTS processing"""
        try:
            self.tts.speak(text, self.tts_stop_event)
        except Exception as e:
            self.logger.error(f"Error in background speaking: {e}")
    
    def _process_speaking(self, audio_frame: np.ndarray) -> None:
        """
        Process audio frame in SPEAKING state
        Check for hotword interrupt (barge-in)
        
        Args:
            audio_frame: Audio frame to process
        """
        # Only check for hotword interrupt if enabled
        if not self.speak_hotword_interrupt or not self.enable_hotword or not self.hotword:
            return
        
        current_time = time.time()

        # Always run hotword detection to keep detector state up-to-date,
        # then decide whether to act on a detection.
        detected_keyword, score = self.hotword.detect(audio_frame)
        
        # Check if speaking just started (prevent immediate interrupt from residual hotword)
        time_since_speak_start = current_time - self._speaking_start_ts
        if time_since_speak_start <= Config.SPEAK_START_COOLDOWN_SEC:
            # Too soon after speaking started, ignore hotword triggers
            return
        
        # Additionally check if waiting for speech after barge-in
        if self._bargein_pending_speech:
            # Still waiting for speech start after previous barge-in
            return
        
        if detected_keyword:
            # Check cooldown period
            time_since_last = current_time - self.last_hotword_time
            
            if time_since_last < Config.HOTWORD_COOLDOWN_SEC:
                self.logger.debug(
                    f"Hotword cooldown active during speaking: {time_since_last:.2f}s < {Config.HOTWORD_COOLDOWN_SEC}s"
                )
                return
            
            self.logger.info(f"Hotword '{detected_keyword}' detected - interrupting speech (barge-in)")
            self.last_hotword_time = current_time
            
            # Set barge-in pending speech flags
            if Config.POST_BARGEIN_REQUIRE_SPEECH_START:
                self._bargein_pending_speech = True
                self._bargein_wait_speech_deadline_ts = current_time + Config.POST_BARGEIN_WAIT_FOR_SPEECH_SEC
                self.logger.info(
                    f"Post-barge-in speech gating enabled; "
                    f"waiting for speech start up to {Config.POST_BARGEIN_WAIT_FOR_SPEECH_SEC}s"
                )
            
            # Stop TTS immediately
            self.tts_stop_event.set()
            
            # Wait briefly for speaking thread to stop
            if self.speaking_thread:
                self.speaking_thread.join(timeout=0.3)
            
            # Drain audio queue to remove stale frames
            self._drain_audio_queue(Config.POST_SPEAK_DRAIN_SEC)
            
            # Transition directly to LISTENING
            self.state.transition_to(AssistantState.LISTENING)
            self.audio_buffer = []
            
            # Clear speaking thread reference to prevent _check_speaking_complete from triggering
            self.speaking_thread = None
            
            self.logger.info("Listening... (speak now)")
    
    def _check_speaking_complete(self) -> None:
        """Check if background speaking is complete"""
        if self.speaking_thread and not self.speaking_thread.is_alive():
            # If we're in LISTENING state, barge-in already handled the transition
            if self.state.is_in_state(AssistantState.LISTENING):
                self.speaking_thread = None
                return
            
            # Speaking finished normally
            self.logger.debug("Speaking completed")
            
            # In no-hotword mode, exit after speaking
            if not self.enable_hotword:
                self.logger.info("No-hotword mode: Exiting after speaking")
                self.running = False
            elif self._is_followup_response:
                # This response was in a FOLLOWUP chain, so re-enter FOLLOWUP
                self._is_followup_response = False
                self._drain_audio_queue(Config.POST_SPEAK_DRAIN_SEC)
                self.speaking_thread = None
                self._re_enter_followup()
            else:
                # Regular speaking complete
                self._drain_audio_queue(Config.POST_SPEAK_DRAIN_SEC)
                self.speaking_thread = None
                
                # Only enter follow-up if response ended with a question
                if self._last_response_is_question:
                    self._enter_followup_after_prompt()
                else:
                    # Go back to IDLE
                    self.state.transition_to(AssistantState.IDLE)
                    self.logger.info(f"Ready. Listening for hotword: {Config.HOTWORD_KEYWORDS}")
    
    def _start_followup_window(self) -> None:
        """
        Start a FOLLOWUP listening window after TTS completes.
        Speaks a prompt and transitions to FOLLOWUP state.
        """
        if not Config.FOLLOWUP_ENABLED:
            # FOLLOWUP disabled, go back to IDLE
            self.state.transition_to(AssistantState.IDLE)
            self.logger.info(f"Ready. Listening for hotword: {Config.HOTWORD_KEYWORDS}")
            return
        
        # Speak a prompt to signal FOLLOWUP window
        self._speak_followup_prompt()
    
    def _speak_followup_prompt(self) -> None:
        """Speak a follow-up prompt and prepare to transition to FOLLOWUP state"""
        # Start speaking the prompt
        self.state.transition_to(AssistantState.SPEAKING)
        self.logger.info("Speaking follow-up prompt...")
        
        # Clear stop event
        self.tts_stop_event.clear()
        
        # Record speaking start time for cooldown check
        self._speaking_start_ts = time.time()
        self._is_followup_response = False  # Don't re-enter FOLLOWUP on completion
        
        # Start speaking the prompt in background thread
        prompt = "Is there anything else?"
        self.speaking_thread = threading.Thread(
            target=self._background_speak,
            args=(prompt,),
            daemon=True
        )
        self.speaking_thread.start()
    
    def _enter_followup_after_prompt(self) -> None:
        """Enter FOLLOWUP state after the follow-up prompt has been spoken"""
        self.state.transition_to(AssistantState.FOLLOWUP)
        self.audio_buffer = []
        self.state.speech_detected = False
        self.state.silence_frames = 0
        self.state.speech_frames_count = 0
        self.state.total_frames_recorded = 0
        
        self.followup_manager.start_followup_window()
    
    def _re_enter_followup(self) -> None:
        """
        Re-enter FOLLOWUP window during a chain.
        Used when user's FOLLOWUP response gets processed and we respond again.
        """
        # Speak the follow-up prompt again
        self._speak_followup_prompt()
    
    def _drain_audio_queue(self, duration_sec: float) -> None:
        """Drain audio queue for specified duration"""
        drain_frames = int(duration_sec * Config.SAMPLE_RATE / Config.CHUNK_SAMPLES)
        drained_count = 0
        
        self.logger.debug(f"Draining audio queue for {duration_sec}s ({drain_frames} frames)...")
        
        for _ in range(drain_frames):
            try:
                frame = self.audio_queue.get_nowait()
                drained_count += 1
                
                # Process frame through hotword detector to update state
                # but ignore any detections during drain period
                if self.hotword:
                    self.hotword.detect(frame)
                    
            except Empty:
                break
        
        if drained_count > 0:
            self.logger.debug(f"Drained {drained_count} frames from queue")
    
    def _clear_bargein_flags(self) -> None:
        """Clear all barge-in related flags"""
        if self._bargein_pending_speech:
            self.logger.debug("Clearing barge-in pending speech flags")
        self._bargein_pending_speech = False
        self._bargein_wait_speech_deadline_ts = 0.0


class WyzerAssistantMultiprocess:
    """Realtime-core assistant that offloads heavy work to a Brain Worker process."""

    def __init__(
        self,
        enable_hotword: bool = True,
        whisper_model: str = "small",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        audio_device: Optional[int] = None,
        llm_mode: str = "ollama",
        ollama_model: str = "llama3.1:latest",
        ollama_url: str = "http://127.0.0.1:11434",
        llm_timeout: int = 30,
        # Llama.cpp embedded server settings (Phase 8)
        llamacpp_bin: str = "./wyzer/llm_bin/llama-server.exe",
        llamacpp_model: str = "./wyzer/llm_models/model.gguf",
        llamacpp_port: int = 8081,
        llamacpp_ctx: int = 2048,
        llamacpp_threads: int = 0,  # 0 = auto (recommended with auto-optimize)
        llamacpp_auto_optimize: bool = True,
        llamacpp_gpu_layers: int = -1,
        # TTS settings
        tts_enabled: bool = True,
        tts_engine: str = "piper",
        piper_exe_path: str = "./wyzer/assets/piper/piper.exe",
        piper_model_path: str = "./wyzer/assets/piper/en_US-lessac-medium.onnx",
        piper_speaker_id: Optional[int] = None,
        tts_output_device: Optional[int] = None,
        speak_hotword_interrupt: bool = True,
        log_level: str = "INFO",
        quiet_mode: bool = False,
        ipc_queue_maxsize: int = 50,
    ):
        self.logger = get_logger()
        self.quiet_mode = quiet_mode

        self.enable_hotword = enable_hotword
        self.speak_hotword_interrupt = speak_hotword_interrupt

        # State
        self.state = RuntimeState()
        self.running = False
        self.last_hotword_time: float = 0.0
        
        # FOLLOWUP listening window manager
        self.followup_manager = FollowupManager()

        # Post-barge-in speech gating
        self._bargein_pending_speech: bool = False
        self._bargein_wait_speech_deadline_ts: float = 0.0
        self._speaking_start_ts: float = 0.0
        
        # No-speech-start timeout tracking: deadline for speech to begin after LISTENING starts
        self._no_speech_start_deadline_ts: float = 0.0

        # Audio buffer
        self.audio_buffer: List[np.ndarray] = []

        # Audio stream
        self.audio_queue: Queue = Queue(maxsize=Config.AUDIO_QUEUE_MAX_SIZE)
        self.mic_stream = MicStream(audio_queue=self.audio_queue, device=audio_device)

        # VAD + hotword
        self.vad = VadDetector()
        self.hotword: Optional[HotwordDetector] = None
        if self.enable_hotword:
            try:
                self.hotword = HotwordDetector()
                self.logger.info(f"Hotword detection enabled for: {Config.HOTWORD_KEYWORDS}")
            except Exception as e:
                self.logger.error(f"Failed to initialize hotword detector: {e}")
                self.logger.warning("Continuing without hotword detection")
                self.enable_hotword = False

        # Brain worker process + IPC
        self._brain_proc = None
        self._core_to_brain_q = None
        self._brain_to_core_q = None

        # Brain speaking flag (driven by worker LOG events)
        self._brain_speaking: bool = False
        self._show_followup_prompt: bool = True  # Whether to show "Is there anything else?" prompt
        self._waiting_for_followup_tts: bool = False  # Track when we're waiting for followup TTS to finish

        self._exit_after_tts: bool = False

        self._brain_config: Dict[str, Any] = {
            "log_level": log_level,
            "quiet_mode": quiet_mode,
            "ipc_queue_maxsize": ipc_queue_maxsize,
            "whisper_model": whisper_model,
            "whisper_device": whisper_device,
            "whisper_compute_type": whisper_compute_type,
            "llm_mode": llm_mode,
            "ollama_model": ollama_model,
            "ollama_url": ollama_url,
            "llm_timeout": llm_timeout,
            # Llama.cpp settings (Phase 8)
            "llamacpp_bin": llamacpp_bin,
            "llamacpp_model": llamacpp_model,
            "llamacpp_port": llamacpp_port,
            "llamacpp_ctx": llamacpp_ctx,
            "llamacpp_threads": llamacpp_threads,
            "llamacpp_auto_optimize": llamacpp_auto_optimize,
            "llamacpp_gpu_layers": llamacpp_gpu_layers,
            # TTS settings
            "tts_enabled": bool(tts_enabled),
            "tts_engine": tts_engine,
            "piper_exe_path": piper_exe_path,
            "piper_model_path": piper_model_path,
            "piper_speaker_id": piper_speaker_id,
            "tts_output_device": tts_output_device,
        }

    def start(self) -> None:
        if self.running:
            self.logger.warning("Assistant already running")
            return

        import time
        
        # In quiet mode, show simple Loading... message
        if self.quiet_mode:
            print("Loading...", flush=True)
        
        # Log Main/Parent process role for architecture verification
        _role_log = {
            "role": "Main/Parent (orchestrator)",
            "responsibilities": "Process spawning, state machine, audio I/O coordination",
            "pid": os.getpid(),
            "ppid": os.getppid() if hasattr(os, 'getppid') else "N/A",
            "python_executable": sys.executable,
            "start_time": time.time(),
        }
        self.logger.info(f"[ROLE] Main orchestrator startup: pid={_role_log['pid']} exec={sys.executable}")
        self.logger.info(f"[ROLE] Main responsibilities: {_role_log['responsibilities']}")

        self.logger.info("Starting Wyzer Assistant (multiprocess)...")
        self.running = True

        self._brain_proc, self._core_to_brain_q, self._brain_to_core_q = start_brain_process(self._brain_config)
        
        # Log Core process info
        if self._brain_proc:
            # Note: Core is the main thread of this process; log after Brain starts
            _core_log = {
                "role": "Core (realtime reactive)",
                "responsibilities": "Microphone stream, VAD, hotword detection, state machine, barge-in handling",
                "pid": os.getpid(),
                "python_executable": sys.executable,
                "start_time": time.time(),
            }
            self.logger.info(f"[ROLE] Core (main thread) pid={_core_log['pid']} responsibilities=realtime audio/hotword/state")
            self.logger.info(f"[ROLE] Brain (worker process) spawned: pid={self._brain_proc.pid}")

        self.mic_stream.start()

        if not self.enable_hotword:
            self.logger.info("No-hotword mode: Starting immediate listening")
            self.state.transition_to(AssistantState.LISTENING)
            self.logger.info("Listening... (speak now)")
        else:
            self.logger.info(f"Listening for hotword: {Config.HOTWORD_KEYWORDS}")

        # In quiet mode, show simple Ready. message
        if self.quiet_mode:
            print("Ready.", flush=True)

        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        if not self.running:
            return

        self.logger.info("Stopping Wyzer Assistant...")
        self.running = False

        try:
            self.mic_stream.stop()
        except Exception:
            pass

        if self._brain_proc and self._core_to_brain_q:
            stop_brain_process(self._brain_proc, self._core_to_brain_q)

        self.logger.info("Wyzer Assistant stopped")

    def interrupt_current_process(self) -> None:
        """
        Interrupt the current process cleanly.
        Sends interrupt signal to brain worker and transitions to IDLE state.
        """
        self.logger.info(f"Interrupt requested from state: {self.state.current_state.value}")
        
        # Mark interrupt in state
        self.state.request_interrupt()
        
        # Send interrupt to brain worker if it's processing
        if self._core_to_brain_q:
            try:
                safe_put(self._core_to_brain_q, {"type": "INTERRUPT", "reason": "user_interrupt"}, timeout=0.1)
            except Exception as e:
                self.logger.warning(f"Could not send interrupt to brain: {e}")
        
        # Drain audio queue to clear stale frames
        self._drain_audio_queue(0.05)
        
        # Reset to idle state
        self._reset_to_idle()
        self.logger.info("Process interrupted successfully")

    def _main_loop(self) -> None:
        # Heartbeat tracking for runtime verification
        last_heartbeat = time.time()
        
        while self.running:
            # Emit heartbeat every ~10s (configurable)
            current_time = time.time()
            if current_time - last_heartbeat >= Config.HEARTBEAT_INTERVAL_SEC:
                self._emit_core_heartbeat()
                last_heartbeat = current_time
            
            # Drain brain->core messages first to keep UI/logging snappy
            self._poll_brain_messages()

            try:
                audio_frame = self.audio_queue.get(timeout=0.05)
            except Empty:
                # Also allow hotword gating timeouts in LISTENING
                if self.state.is_in_state(AssistantState.LISTENING):
                    self._process_listening_timeout_tick()
                # Allow FOLLOWUP timeout checks
                elif self.state.is_in_state(AssistantState.FOLLOWUP):
                    if self.followup_manager.check_timeout():
                        self._reset_to_idle()
                continue

            if self.state.is_in_state(AssistantState.IDLE):
                self._process_idle(audio_frame)
            elif self.state.is_in_state(AssistantState.LISTENING):
                self._process_listening(audio_frame)
            elif self.state.is_in_state(AssistantState.FOLLOWUP):
                self._process_followup(audio_frame)
            else:
                # In multiprocess mode, treat non-listening states like IDLE for hotword/barge-in.
                self._process_idle(audio_frame)

    def _emit_core_heartbeat(self) -> None:
        """Emit Core process health heartbeat for runtime verification"""
        try:
            queue_size_in = self._core_to_brain_q.qsize() if self._core_to_brain_q else 0
            queue_size_out = self._brain_to_core_q.qsize() if self._brain_to_core_q else 0
        except Exception:
            queue_size_in = queue_size_out = -1
        
        self.logger.info(
            f"[HEARTBEAT] role=Core pid={os.getpid()} "
            f"state={self.state.current_state.value} "
            f"q_in={queue_size_in} q_out={queue_size_out} "
            f"time_in_state={self.state.get_time_in_current_state():.1f}s"
        )

    def _process_idle(self, audio_frame: np.ndarray) -> None:
        if not (self.enable_hotword and self.hotword):
            return

        current_time = time.time()
        detected_keyword, score = self.hotword.detect(audio_frame)

        if not detected_keyword:
            return

        time_since_last = current_time - self.last_hotword_time
        if time_since_last < Config.HOTWORD_COOLDOWN_SEC:
            self.logger.debug(
                f"Hotword cooldown active: {time_since_last:.2f}s < {Config.HOTWORD_COOLDOWN_SEC}s (score: {score:.3f})"
            )
            return

        self.logger.info(f"Hotword '{detected_keyword}' accepted after cooldown")
        self.last_hotword_time = current_time

        # Hotword must barge-in instantly if TTS is speaking.
        # Only send INTERRUPT when the worker reports it's speaking.
        is_bargein = self.speak_hotword_interrupt and self._brain_speaking and self._core_to_brain_q
        if is_bargein:
            safe_put(self._core_to_brain_q, {"type": "INTERRUPT", "reason": "hotword"})
            # Enable post-barge-in speech gating: require speech to start within a grace period
            # This prevents the VAD from immediately timing out on residual silence after interrupting TTS
            if Config.POST_BARGEIN_REQUIRE_SPEECH_START:
                self._bargein_pending_speech = True
                self._bargein_wait_speech_deadline_ts = current_time + Config.POST_BARGEIN_WAIT_FOR_SPEECH_SEC
                self.logger.debug(
                    f"Post-barge-in speech gating enabled; "
                    f"waiting for speech start up to {Config.POST_BARGEIN_WAIT_FOR_SPEECH_SEC}s"
                )

        # Drain a bit more to remove residual hotword audio
        self._drain_audio_queue(Config.POST_IDLE_DRAIN_SEC)

        self.state.transition_to(AssistantState.LISTENING)
        self.audio_buffer = []
        
        # Set no-speech-start deadline for quick abort if user stays silent
        # Skip if barge-in is pending (barge-in uses its own grace period)
        if not self._bargein_pending_speech:
            self._no_speech_start_deadline_ts = time.time() + Config.NO_SPEECH_START_TIMEOUT_SEC
        
        self.logger.info("Listening... (speak now)")

    def _process_listening_timeout_tick(self) -> None:
        """Check for timeout conditions when no audio frame is available"""
        current_time = time.time()
        
        # Check post-barge-in speech wait timeout
        if self._bargein_pending_speech:
            if current_time > self._bargein_wait_speech_deadline_ts:
                self.logger.info("Post-barge-in speech wait timeout - returning to IDLE")
                self._clear_bargein_flags()
                self._reset_to_idle()
            return
        
        # Check no-speech-start timeout (only if speech hasn't started yet)
        if (not self.state.speech_detected 
            and self._no_speech_start_deadline_ts > 0
            and current_time > self._no_speech_start_deadline_ts):
            self.logger.info(f"[LISTENING] No speech start within {Config.NO_SPEECH_START_TIMEOUT_SEC}s -> abort")
            self._no_speech_start_deadline_ts = 0.0  # Reset deadline
            self._reset_to_idle()

    def _process_listening(self, audio_frame: np.ndarray) -> None:
        # Post-barge-in speech gating
        if self._bargein_pending_speech:
            current_time = time.time()
            if current_time > self._bargein_wait_speech_deadline_ts:
                self.logger.info("Post-barge-in speech wait timeout - returning to IDLE")
                self._clear_bargein_flags()
                self._reset_to_idle()
                return

        self.audio_buffer.append(audio_frame)
        self.state.total_frames_recorded += 1

        is_speech = self.vad.is_speech(audio_frame)
        if is_speech:
            if not self.state.speech_detected:
                self.logger.debug("Speech started")
                self.state.speech_detected = True
                
                # Clear no-speech-start deadline since speech has begun
                self._no_speech_start_deadline_ts = 0.0
                
                if self._bargein_pending_speech:
                    self.logger.debug("Speech started after barge-in - clearing pending flag")
                    self._bargein_pending_speech = False
            self.state.speech_frames_count += 1
            self.state.silence_frames = 0
        else:
            if self.state.speech_detected:
                self.state.silence_frames += 1

        should_stop = False
        stop_reason = ""
        
        # 0. No-speech-start timeout: abort early if user never begins speaking
        #    This provides a fast exit when user triggers hotword but stays silent.
        #    Skip this check if barge-in is pending (use barge-in grace period instead).
        if (not self.state.speech_detected 
            and not self._bargein_pending_speech
            and self._no_speech_start_deadline_ts > 0
            and time.time() > self._no_speech_start_deadline_ts):
            self.logger.info(f"[LISTENING] No speech start within {Config.NO_SPEECH_START_TIMEOUT_SEC}s -> abort")
            self._no_speech_start_deadline_ts = 0.0  # Reset deadline
            self._reset_to_idle()
            return
        
        if (
            self.state.speech_detected
            and self.state.silence_frames >= Config.get_silence_timeout_frames()
        ):
            should_stop = True
            stop_reason = "silence timeout"
        if self.state.total_frames_recorded >= Config.get_max_record_frames():
            should_stop = True
            stop_reason = "max duration"
        if (
            not self.enable_hotword
            and self.state.speech_detected
            and self.state.silence_frames >= Config.get_silence_timeout_frames()
        ):
            should_stop = True
            stop_reason = "utterance complete (no-hotword mode)"

        if not should_stop:
            return

        self.logger.info(f"Recording stopped: {stop_reason}")

        if self.state.speech_frames_count <= 0:
            self.logger.warning("No speech detected in recording")
            if not self.enable_hotword:
                self.logger.info("No-hotword mode: Exiting")
                self.running = False
            else:
                self._reset_to_idle()
            return

        self._send_audio_to_brain()

        # Return to IDLE immediately so hotword stays responsive.
        if self.enable_hotword:
            self._reset_to_idle()
        else:
            # In no-hotword mode, wait for RESULT then exit after TTS completes.
            self.state.transition_to(AssistantState.IDLE)

    def _process_followup(self, audio_frame: np.ndarray) -> None:
        """
        Process audio frame in FOLLOWUP state (multiprocess).
        Similar to LISTENING but with different timeout and exit criteria.
        
        Args:
            audio_frame: Audio frame to process
        """
        # Post-barge-in speech gating
        if self._bargein_pending_speech:
            current_time = time.time()
            if current_time > self._bargein_wait_speech_deadline_ts:
                self.logger.info("Post-barge-in speech wait timeout - returning to IDLE")
                self._clear_bargein_flags()
                self._reset_to_idle()
                return

        self.audio_buffer.append(audio_frame)
        self.state.total_frames_recorded += 1

        # During grace period (TTS prompt playing), don't process speech
        # This prevents picking up the prompt audio as user speech
        if self.followup_manager.is_in_grace_period():
            return

        is_speech = self.vad.is_speech(audio_frame)
        if is_speech:
            if not self.state.speech_detected:
                self.logger.debug("FOLLOWUP: Speech started")
                self.state.speech_detected = True
                # Reset followup timeout on new speech
                self.followup_manager.reset_speech_timer()
                if self._bargein_pending_speech:
                    self.logger.debug("Speech started after barge-in - clearing pending flag")
                    self._bargein_pending_speech = False
            else:
                # Continuous speech, keep resetting timer
                self.followup_manager.reset_speech_timer()
            self.state.speech_frames_count += 1
            self.state.silence_frames = 0
        else:
            if self.state.speech_detected:
                self.state.silence_frames += 1

        should_stop = False
        stop_reason = ""
        
        # Send audio early if we detect end of speech (short silence after speech ends)
        # This avoids waiting for the full 3-second timeout
        if self.state.speech_detected and self.state.silence_frames > 0:
            silence_duration = self.state.silence_frames * (Config.CHUNK_SAMPLES / Config.SAMPLE_RATE)
            # If we've had ~500ms of silence after speech, send for processing
            if silence_duration >= 0.5:
                should_stop = True
                stop_reason = "speech ended (early send)"
        
        # Check FOLLOWUP timeout
        if not should_stop and self.state.speech_detected and self.followup_manager.check_timeout():
            should_stop = True
            stop_reason = "followup silence timeout"
        
        # Max duration limit
        if self.state.total_frames_recorded >= Config.get_max_record_frames():
            should_stop = True
            stop_reason = "max duration"

        if not should_stop:
            return

        self.logger.info(f"FOLLOWUP stopped: {stop_reason}")

        if self.state.speech_frames_count <= 0:
            self.logger.info("No speech in FOLLOWUP window")
            self.followup_manager.end_followup_window()
            self._reset_to_idle()
            return

        self._send_audio_to_brain_followup()
        
        # Return to IDLE immediately so hotword stays responsive.
        if self.enable_hotword:
            self._reset_to_idle()
        else:
            # In no-hotword mode, wait for RESULT then exit after TTS completes.
            self.state.transition_to(AssistantState.IDLE)

    def _send_audio_to_brain_followup(self) -> None:
        """Send audio to brain worker in FOLLOWUP mode"""
        if not self._core_to_brain_q:
            self.logger.error("Brain worker queue not available")
            return

        self._clear_bargein_flags()
        audio_data = concat_audio_frames(self.audio_buffer)
        self.audio_buffer = []

        # Save to temp WAV (keeps IPC lightweight on slow PCs)
        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(prefix="wyzer_", suffix=".wav")
            try:
                os.close(fd)
            except Exception:
                pass
            int16_audio = audio_to_int16(audio_data)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(Config.SAMPLE_RATE)
                wf.writeframes(int16_audio.tobytes())
        except Exception as e:
            self.logger.error(f"Failed to write temp WAV: {e}")
            try:
                if wav_path:
                    os.unlink(wav_path)
            except Exception:
                pass
            return

        req_id = new_id()
        # Mark as followup so orchestrator can handle exit phrases
        safe_put(
            self._core_to_brain_q,
            {
                "type": "AUDIO",
                "id": req_id,
                "wav_path": wav_path,
                "pcm_bytes": None,
                "sample_rate": Config.SAMPLE_RATE,
                "meta": {"is_followup": True, "followup_chain": self.followup_manager.get_chain_count()},
            },
        )

    def _send_audio_to_brain(self) -> None:
        if not self._core_to_brain_q:
            self.logger.error("Brain worker queue not available")
            return

        self._clear_bargein_flags()
        audio_data = concat_audio_frames(self.audio_buffer)
        self.audio_buffer = []

        # Save to temp WAV (keeps IPC lightweight on slow PCs)
        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(prefix="wyzer_", suffix=".wav")
            try:
                os.close(fd)
            except Exception:
                pass
            int16_audio = audio_to_int16(audio_data)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(Config.SAMPLE_RATE)
                wf.writeframes(int16_audio.tobytes())
        except Exception as e:
            self.logger.error(f"Failed to write temp WAV: {e}")
            try:
                if wav_path:
                    os.unlink(wav_path)
            except Exception:
                pass
            return

        req_id = new_id()
        safe_put(
            self._core_to_brain_q,
            {
                "type": "AUDIO",
                "id": req_id,
                "wav_path": wav_path,
                "pcm_bytes": None,
                "sample_rate": Config.SAMPLE_RATE,
                "meta": {},
            },
        )

    def _poll_brain_messages(self) -> None:
        if not self._brain_to_core_q:
            return

        while True:
            try:
                msg = self._brain_to_core_q.get_nowait()
            except Exception:
                return

            mtype = (msg or {}).get("type")
            if mtype == "LOG":
                level = str(msg.get("level", "INFO")).upper()
                text = str(msg.get("msg", ""))

                if text == "tts_started":
                    self._brain_speaking = True
                elif text in {"tts_finished", "tts_interrupted"}:
                    self._brain_speaking = False
                    # Get show_followup_prompt from the TTS message metadata
                    tts_meta = msg.get("meta") or {}
                    show_followup = tts_meta.get("show_followup_prompt", False)
                    
                    # When TTS finishes, check if this was a followup prompt that needs listening
                    if self._waiting_for_followup_tts:
                        self._waiting_for_followup_tts = False
                        # Drain any audio that was captured during TTS playback
                        self._drain_audio_queue(0.2)
                        # Now that the followup TTS is done, start listening for user response
                        self.state.transition_to(AssistantState.FOLLOWUP)
                        self.audio_buffer = []
                        self.state.speech_detected = False
                        self.state.silence_frames = 0
                        self.state.speech_frames_count = 0
                        self.state.total_frames_recorded = 0
                        self.followup_manager.start_followup_window()
                    # When TTS finishes for non-followup response, enter FOLLOWUP ONLY if response was a question
                    elif Config.FOLLOWUP_ENABLED and self.enable_hotword and show_followup:
                        self._start_followup_window()
                    # Response was not a question - reset to IDLE and require hotword
                    elif self.enable_hotword:
                        self._reset_to_idle()

                if level == "DEBUG":
                    self.logger.debug(text)
                elif level == "WARNING":
                    self.logger.warning(text)
                elif level == "ERROR":
                    self.logger.error(text)
                else:
                    self.logger.info(text)

                # No-hotword mode: exit after speech finishes
                if not self.enable_hotword and self._exit_after_tts and text in {"tts_finished", "tts_interrupted"}:
                    self.running = False
                continue

            if mtype == "RESULT":
                meta = msg.get("meta") or {}
                user_text = str(meta.get("user_text") or "").strip()
                is_followup = meta.get("is_followup", False)
                
                # Check if this was a valid capture (not empty/minimal transcript)
                # If capture_valid is False, we should NOT enter follow-up mode even
                # if the response would normally trigger it. This prevents the case
                # where user activates hotword but stays silent, then says "never mind"
                # which would incorrectly be treated as a follow-up exit.
                capture_valid = meta.get("capture_valid", True)  # Default True for backward compat
                
                # Capture whether to show follow-up prompt (set by brain worker)
                # Override to False if capture was invalid
                self._show_followup_prompt = meta.get("show_followup_prompt", True) and capture_valid
                
                # Check if brain worker already detected exit phrase (sentinel in meta)
                # This avoids double-detection: brain worker is the single source of truth
                # for exit phrase detection in multiprocess mode.
                exit_sentinel = meta.get("exit_sentinel")
                if is_exit_sentinel(exit_sentinel):
                    # Exit phrase already detected by brain worker - handle silently
                    # NOTE: We do NOT send INTERRUPT here because:
                    # 1. Brain worker already returned empty tts_text (no TTS to interrupt)
                    # 2. Sending INTERRUPT would clear TTS queue, including important
                    #    announcements like timer alarms that may have just been enqueued
                    self.logger.debug(f"[EXIT] Handling exit sentinel from brain: {exit_sentinel.get('phrase')}")
                    if is_followup:
                        self.followup_manager.end_followup_window()
                    
                    # Silently return to idle (no INTERRUPT needed - brain handled it)
                    self._reset_to_idle()
                    continue
                
                # Fallback: check for exit phrase in user_text if NOT already handled
                # This catches edge cases where exit phrase was spoken outside followup
                # context (e.g., "never mind" as initial utterance without hotword mode).
                # Only check if:
                # 1. No sentinel was set AND user_text is present
                # 2. capture_valid is True (don't check empty/noise captures for exit phrases)
                if user_text and not exit_sentinel and capture_valid:
                    fallback_sentinel = self.followup_manager.check_exit_phrase(user_text, log_detection=True)
                    if fallback_sentinel:
                        self.logger.info("[EXIT] Fallback exit phrase detection - silently returning to idle")
                        if is_followup:
                            self.followup_manager.end_followup_window()
                        
                        # Interrupt any TTS
                        if self._core_to_brain_q:
                            safe_put(self._core_to_brain_q, {"type": "INTERRUPT", "reason": "exit_phrase"})
                        
                        self._reset_to_idle()
                        continue
                
                # For invalid captures, skip printing and just reset quietly
                if not capture_valid:
                    self.logger.debug(f"[CAPTURE] Invalid capture (empty/minimal transcript), skipping follow-up")
                    self._reset_to_idle()
                    continue
                
                if user_text:
                    print(f"\nYou: {user_text}")

                reply = str(msg.get("reply", ""))
                print(f"\nWyzer: {reply}\n")

                tts_text = msg.get("tts_text")
                
                # Handle FOLLOWUP continuation
                if is_followup and tts_text:
                    # Spoken response in FOLLOWUP context
                    # TTS completion will re-enter FOLLOWUP
                    pass
                elif not self.enable_hotword:
                    # Exit after TTS completes (if any)
                    self._exit_after_tts = bool(tts_text)
                    if not tts_text:
                        self.running = False
                elif not is_followup and tts_text:
                    # Normal response - will start FOLLOWUP window after TTS
                    pass
                continue

            # Unknown message types are ignored

    def _start_followup_window(self) -> None:
        """
        Start a FOLLOWUP listening window after TTS completes (multiprocess).
        Only enters FOLLOWUP if the response was a question (ends with ?).
        """
        if not Config.FOLLOWUP_ENABLED:
            # FOLLOWUP disabled, go back to IDLE
            self.state.transition_to(AssistantState.IDLE)
            self.logger.info(f"Ready. Listening for hotword: {Config.HOTWORD_KEYWORDS}")
            return
        
        # Only enter follow-up if the response was a question
        if self._show_followup_prompt:
            # Response was a question - go directly to FOLLOWUP listening (no extra prompt)
            self.state.transition_to(AssistantState.FOLLOWUP)
            self.audio_buffer = []
            self.state.speech_detected = False
            self.state.silence_frames = 0
            self.state.speech_frames_count = 0
            self.state.total_frames_recorded = 0
            self.followup_manager.start_followup_window()
        else:
            # Response was not a question - go back to IDLE
            self._reset_to_idle()

    def _reset_to_idle(self) -> None:
        self._clear_bargein_flags()
        self.audio_buffer = []
        self._drain_audio_queue(Config.POST_IDLE_DRAIN_SEC)
        self.state.transition_to(AssistantState.IDLE)
        if self.enable_hotword:
            self.logger.info(f"Ready. Listening for hotword: {Config.HOTWORD_KEYWORDS}")

    def _drain_audio_queue(self, duration_sec: float) -> None:
        drain_frames = int(duration_sec * Config.SAMPLE_RATE / Config.CHUNK_SAMPLES)
        for _ in range(drain_frames):
            try:
                frame = self.audio_queue.get_nowait()
                if self.hotword:
                    self.hotword.detect(frame)
            except Empty:
                break

    def _clear_bargein_flags(self) -> None:
        if self._bargein_pending_speech:
            self.logger.debug("Clearing barge-in pending speech flags")
        self._bargein_pending_speech = False
        self._bargein_wait_speech_deadline_ts = 0.0


