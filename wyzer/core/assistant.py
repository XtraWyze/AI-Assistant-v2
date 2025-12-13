"""
Main assistant coordinator for Wyzer AI Assistant.
Manages state machine and orchestrates audio pipeline and STT.
"""
import time
import numpy as np
import threading
from queue import Queue, Empty
from typing import Optional, List
from wyzer.core.config import Config
from wyzer.core.logger import get_logger
from wyzer.core.state import AssistantState, RuntimeState
from wyzer.audio.mic_stream import MicStream
from wyzer.audio.vad import VadDetector
from wyzer.audio.hotword import HotwordDetector
from wyzer.audio.audio_utils import concat_audio_frames
from wyzer.stt.stt_router import STTRouter
from wyzer.brain.llm_engine import LLMEngine
from wyzer.tts.tts_router import TTSRouter


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
        piper_exe_path: str = "./assets/piper/piper.exe",
        piper_model_path: str = "./assets/piper/en_US-voice.onnx",
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
        
        # Hotword cooldown tracking
        self.last_hotword_time: float = 0.0
        
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
                continue
            
            # Process frame based on current state
            if self.state.is_in_state(AssistantState.IDLE):
                self._process_idle(audio_frame)
            
            elif self.state.is_in_state(AssistantState.LISTENING):
                self._process_listening(audio_frame)
            
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
            detected_keyword, score = self.hotword.detect(audio_frame)
            
            if detected_keyword:
                # Check cooldown period
                current_time = time.time()
                time_since_last = current_time - self.last_hotword_time
                
                if time_since_last < Config.HOTWORD_COOLDOWN_SEC:
                    self.logger.debug(
                        f"Hotword cooldown active: {time_since_last:.2f}s < {Config.HOTWORD_COOLDOWN_SEC}s (score: {score:.3f})"
                    )
                    return
                
                self.logger.info(f"Hotword '{detected_keyword}' accepted after cooldown")
                self.last_hotword_time = current_time
                
                # Transition to LISTENING
                self.state.transition_to(AssistantState.LISTENING)
                self.audio_buffer = []
                self.logger.info("Listening... (speak now)")
    
    def _process_listening(self, audio_frame: np.ndarray) -> None:
        """
        Process audio frame in LISTENING state
        
        Args:
            audio_frame: Audio frame to process
        """
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
    
    def _transcribe_and_reset(self) -> None:
        """Start background transcription and transition to TRANSCRIBING"""
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
            
            # Display transcript
            if transcript:
                self.logger.info(f"Transcript: {transcript}")
                print(f"\nYou: {transcript}")
                
                # Pass to LLM brain if enabled
                if self.brain:
                    self._think_and_respond(transcript)
                else:
                    # No brain, just show transcript and return to idle
                    if not self.enable_hotword:
                        self.logger.info("No-hotword mode: Exiting after transcription")
                        self.running = False
                    else:
                        self._reset_to_idle()
            else:
                self.logger.warning("No valid transcript (empty or filtered as garbage)")
                
                # In no-hotword mode, exit even if no transcript
                if not self.enable_hotword:
                    self.logger.info("No-hotword mode: Exiting")
                    self.running = False
                else:
                    self._reset_to_idle()
    
    def _reset_to_idle(self) -> None:
        """Reset state to IDLE with queue draining"""
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
    
    def _think_and_respond(self, transcript: str) -> None:
        """Start background LLM thinking and transition to THINKING state"""
        # Transition to THINKING
        self.state.transition_to(AssistantState.THINKING)
        
        self.logger.info("Thinking...")
        
        # Start thinking in background thread
        self.thinking_result = None
        self.thinking_thread = threading.Thread(
            target=self._background_think,
            args=(transcript,),
            daemon=True
        )
        self.thinking_thread.start()
    
    def _background_think(self, transcript: str) -> None:
        """Background thread for LLM processing"""
        try:
            result = self.brain.think(transcript)
            self.thinking_result = result
        except Exception as e:
            self.logger.error(f"Error in background thinking: {e}")
            self.thinking_result = {
                "reply": "I encountered an error processing your request.",
                "confidence": 0.3,
                "model": "error",
                "latency_ms": 0
            }
    
    def _check_thinking_complete(self) -> None:
        """Check if background thinking is complete and handle result"""
        if self.thinking_thread and not self.thinking_thread.is_alive():
            # Thread finished
            result = self.thinking_result
            
            if result:
                # Display response
                reply = result.get("reply", "")
                latency = result.get("latency_ms", 0)
                
                self.logger.info(f"Response generated in {latency}ms")
                print(f"\nWyzer: {reply}\n")
            else:
                self.logger.warning("No response from LLM brain")
            
            # In no-hotword mode, exit after response
            if not self.enable_hotword:
                self.logger.info("No-hotword mode: Exiting after response")
                self.running = False
            else:
                # Reset to IDLE for next interaction
                self._reset_to_idle()


