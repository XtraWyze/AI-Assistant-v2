"""
TTS router for managing text-to-speech engines.
Routes TTS requests to appropriate engine (currently Piper).
"""
import os
import threading
from typing import Optional
from wyzer.core.logger import get_logger
from wyzer.tts.piper_engine import PiperTTSEngine
from wyzer.tts.audio_player import AudioPlayer


class TTSRouter:
    """TTS router for speech synthesis and playback"""
    
    def __init__(
        self,
        engine: str = "piper",
        piper_exe_path: str = "./assets/piper/piper.exe",
        piper_model_path: str = "./assets/piper/en_US-voice.onnx",
        piper_speaker_id: Optional[int] = None,
        output_device: Optional[int] = None,
        enabled: bool = True
    ):
        """
        Initialize TTS router
        
        Args:
            engine: TTS engine to use (currently only "piper")
            piper_exe_path: Path to Piper executable
            piper_model_path: Path to Piper model
            piper_speaker_id: Optional Piper speaker ID
            output_device: Optional audio output device
            enabled: Whether TTS is enabled
        """
        self.logger = get_logger()
        self.engine_name = engine
        self.enabled = enabled
        
        if not self.enabled:
            self.logger.info("TTS disabled")
            self.engine = None
            self.player = None
            return
        
        # Initialize engine
        if engine == "piper":
            try:
                self.engine = PiperTTSEngine(
                    exe_path=piper_exe_path,
                    model_path=piper_model_path,
                    speaker_id=piper_speaker_id
                )
                self.logger.info(f"TTS engine initialized: {engine}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Piper TTS: {e}")
                self.logger.warning("TTS will be disabled")
                self.enabled = False
                self.engine = None
                self.player = None
                return
        else:
            self.logger.error(f"Unknown TTS engine: {engine}")
            self.enabled = False
            self.engine = None
            self.player = None
            return
        
        # Initialize audio player
        self.player = AudioPlayer(device=output_device)
    
    def speak(self, text: str, stop_event: threading.Event) -> bool:
        """
        Synthesize and speak text
        
        Args:
            text: Text to speak
            stop_event: Event to signal stop
            
        Returns:
            True if completed, False if interrupted or error
        """
        if not self.enabled or not self.engine or not self.player:
            self.logger.debug("TTS not available")
            return False
        
        if not text or not text.strip():
            self.logger.debug("Empty text for TTS")
            return False
        
        # Synthesize to WAV
        self.logger.debug(f"Synthesizing: {text[:50]}...")
        wav_path = self.engine.synthesize_to_wav(text)
        
        if not wav_path:
            self.logger.error("Synthesis failed")
            return False
        
        # Play audio
        try:
            completed = self.player.play_wav(wav_path, stop_event)
            return completed
        finally:
            # Clean up temp WAV file
            try:
                os.unlink(wav_path)
            except:
                pass
