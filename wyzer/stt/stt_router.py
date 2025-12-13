"""
STT router module.
Routes transcription requests to appropriate STT engine.
Currently only supports Whisper, but designed for future expansion.
"""
from typing import Optional
import numpy as np
from wyzer.core.logger import get_logger
from wyzer.stt.whisper_engine import WhisperEngine


class STTRouter:
    """STT engine router"""
    
    def __init__(
        self,
        whisper_model: str = "small",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8"
    ):
        """
        Initialize STT router
        
        Args:
            whisper_model: Whisper model size
            whisper_device: Device for Whisper
            whisper_compute_type: Compute type for Whisper
        """
        self.logger = get_logger()
        self.whisper_engine: Optional[WhisperEngine] = None
        
        # Initialize Whisper engine
        self._init_whisper(whisper_model, whisper_device, whisper_compute_type)
    
    def _init_whisper(
        self,
        model: str,
        device: str,
        compute_type: str
    ) -> None:
        """Initialize Whisper engine"""
        try:
            self.whisper_engine = WhisperEngine(
                model_size=model,
                device=device,
                compute_type=compute_type
            )
            self.logger.info("STT Router: Whisper engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper engine: {e}")
            raise
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        engine: str = "whisper"
    ) -> str:
        """
        Transcribe audio using specified engine
        
        Args:
            audio: Audio data as float32 mono at 16kHz
            language: Language code
            engine: Engine to use (currently only "whisper")
            
        Returns:
            Transcribed text, or empty string if no speech
        """
        if engine == "whisper":
            if self.whisper_engine is None:
                self.logger.error("Whisper engine not initialized")
                return ""
            
            return self.whisper_engine.transcribe(audio, language)
        else:
            self.logger.error(f"Unknown STT engine: {engine}")
            return ""
    
    def get_available_engines(self) -> list:
        """Get list of available STT engines"""
        engines = []
        if self.whisper_engine is not None:
            engines.append("whisper")
        return engines
