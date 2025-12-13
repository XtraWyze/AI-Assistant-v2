"""
Voice Activity Detection (VAD) module.
Primary: silero-vad (if available)
Fallback: Energy-based VAD
"""
import numpy as np
from typing import Optional
from wyzer.core.config import Config
from wyzer.core.logger import get_logger
from wyzer.audio.audio_utils import get_rms_energy

# Try to import silero-vad
SILERO_AVAILABLE = False
try:
    import torch
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    SILERO_AVAILABLE = True
except ImportError:
    pass


class VadDetector:
    """Voice Activity Detection"""
    
    def __init__(
        self,
        sample_rate: int = Config.SAMPLE_RATE,
        threshold: float = Config.VAD_THRESHOLD,
        aggressiveness: int = 3
    ):
        """
        Initialize VAD detector
        
        Args:
            sample_rate: Audio sample rate
            threshold: Detection threshold (0.0-1.0)
            aggressiveness: VAD aggressiveness (higher = more aggressive)
        """
        self.logger = get_logger()
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.aggressiveness = aggressiveness
        self.use_silero = False
        self.model = None
        
        # Silero requires minimum 512 samples (32ms at 16kHz)
        self.silero_min_samples = 512
        self.frame_buffer = []
        
        # Try to load Silero VAD
        if SILERO_AVAILABLE:
            try:
                self._init_silero()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Silero VAD: {e}")
                self.logger.warning("Falling back to energy-based VAD")
        else:
            self.logger.warning(
                "Silero VAD not available (install with: pip install silero-vad torch). "
                "Using energy-based VAD fallback."
            )
    
    def _init_silero(self) -> None:
        """Initialize Silero VAD model"""
        try:
            self.logger.info("Loading Silero VAD model...")
            self.model = load_silero_vad()
            self.use_silero = True
            self.logger.info("Silero VAD initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD: {e}")
    
    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """
        Detect if audio frame contains speech
        
        Args:
            audio_frame: Audio data as float32 mono at sample_rate
            
        Returns:
            True if speech detected
        """
        if len(audio_frame) == 0:
            return False
        
        if self.use_silero and self.model is not None:
            # Buffer frames until we have enough for Silero (exactly 512 samples)
            self.frame_buffer.append(audio_frame)
            buffered = np.concatenate(self.frame_buffer)
            
            if len(buffered) >= self.silero_min_samples:
                # Use exactly 512 samples for Silero
                chunk = buffered[:self.silero_min_samples]
                result = self._is_speech_silero(chunk)
                # Keep remaining samples for next iteration
                self.frame_buffer = [buffered[self.silero_min_samples:]] if len(buffered) > self.silero_min_samples else []
                return result
            else:
                # Not enough buffered yet, use energy fallback
                return self._is_speech_energy(audio_frame)
        else:
            return self._is_speech_energy(audio_frame)
    
    def _is_speech_silero(self, audio_frame: np.ndarray) -> bool:
        """
        Speech detection using Silero VAD
        
        Args:
            audio_frame: Audio frame as float32
            
        Returns:
            True if speech detected
        """
        try:
            # Silero expects torch tensor
            audio_tensor = torch.from_numpy(audio_frame)
            
            # Get speech probability
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            return speech_prob > self.threshold
            
        except Exception as e:
            self.logger.warning(f"Silero VAD error: {e}, falling back to energy")
            return self._is_speech_energy(audio_frame)
    
    def _is_speech_energy(self, audio_frame: np.ndarray) -> bool:
        """
        Simple energy-based speech detection (fallback)
        
        Args:
            audio_frame: Audio frame as float32
            
        Returns:
            True if speech detected
        """
        # TODO: This is a simple fallback. Consider implementing a more
        # sophisticated energy-based VAD with adaptive thresholding and
        # spectral features for better performance when Silero is unavailable.
        
        energy = get_rms_energy(audio_frame)
        
        # Adjust threshold based on aggressiveness
        # Higher aggressiveness = higher threshold = less sensitive
        adjusted_threshold = self.threshold * (self.aggressiveness / 3.0)
        
        return energy > adjusted_threshold
    
    def reset(self) -> None:
        """Reset VAD state"""
        # Clear frame buffer
        self.frame_buffer = []
