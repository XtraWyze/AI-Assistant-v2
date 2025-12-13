"""
Whisper STT engine using faster-whisper.
Transcribes audio with repetition/garbage filtering.
"""
import numpy as np
from typing import Optional
from collections import Counter
from wyzer.core.config import Config
from wyzer.core.logger import get_logger

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None


class WhisperEngine:
    """Faster-Whisper STT engine"""
    
    def __init__(
        self,
        model_size: str = Config.WHISPER_MODEL,
        device: str = Config.WHISPER_DEVICE,
        compute_type: str = Config.WHISPER_COMPUTE_TYPE
    ):
        """
        Initialize Whisper engine
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda)
            compute_type: Compute type (int8, int16, float16, float32)
        """
        self.logger = get_logger()
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None
        
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper not available. Install with: pip install faster-whisper"
            )
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Whisper model"""
        try:
            self.logger.info(
                f"Loading Whisper model: {self.model_size} "
                f"(device={self.device}, compute_type={self.compute_type})"
            )
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            
            self.logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio data as float32 mono at 16kHz
            language: Language code (default: en)
            
        Returns:
            Transcribed text, or empty string if no speech/garbage
        """
        if self.model is None:
            self.logger.error("Model not loaded")
            return ""
        
        if len(audio) == 0:
            return ""
        
        try:
            # Transcribe
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=False,  # We already did VAD
                word_timestamps=False
            )
            
            # Collect all segment texts
            texts = []
            for segment in segments:
                texts.append(segment.text.strip())
            
            # Combine segments
            full_text = " ".join(texts).strip()
            
            # Apply filters
            if not self._is_valid_transcript(full_text):
                return ""
            
            return full_text
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""
    
    def _is_valid_transcript(self, text: str) -> bool:
        """
        Check if transcript is valid (not garbage/spam)
        
        Args:
            text: Transcript text
            
        Returns:
            True if valid
        """
        # Check minimum length
        if len(text) < Config.MIN_TRANSCRIPT_LENGTH:
            self.logger.debug(f"Transcript too short: '{text}'")
            return False
        
        # Check for repetition spam
        if self._is_repetition_spam(text):
            self.logger.warning(f"Repetition spam detected: '{text}'")
            return False
        
        # Check if mostly non-alphabetic (garbage)
        alpha_count = sum(c.isalpha() for c in text)
        if len(text) > 0 and alpha_count / len(text) < 0.3:
            self.logger.debug(f"Too few alphabetic characters: '{text}'")
            return False
        
        return True
    
    def _is_repetition_spam(self, text: str) -> bool:
        """
        Detect if text contains excessive token repetition
        
        Args:
            text: Text to check
            
        Returns:
            True if repetition spam detected
        """
        # Split into tokens (simple whitespace split)
        tokens = text.lower().split()
        
        if len(tokens) == 0:
            return False
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Check if any token repeats more than threshold
        max_repeats = max(token_counts.values())
        
        if max_repeats > Config.MAX_TOKEN_REPEATS:
            self.logger.debug(
                f"Token repetition detected: max={max_repeats}, "
                f"threshold={Config.MAX_TOKEN_REPEATS}"
            )
            return True
        
        # Check for character-level repetition (e.g., "aaaaaaa")
        for token in tokens:
            if len(token) > 3:
                # Count consecutive character repeats
                max_char_repeat = self._max_consecutive_chars(token)
                if max_char_repeat > 4:
                    self.logger.debug(f"Character repetition in token: '{token}'")
                    return True
        
        return False
    
    def _max_consecutive_chars(self, s: str) -> int:
        """
        Find maximum consecutive character repetition
        
        Args:
            s: String to check
            
        Returns:
            Maximum consecutive count
        """
        if len(s) == 0:
            return 0
        
        max_count = 1
        current_count = 1
        prev_char = s[0]
        
        for char in s[1:]:
            if char == prev_char:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 1
                prev_char = char
        
        return max_count
