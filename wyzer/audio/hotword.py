"""
Hotword detection module using openWakeWord.
Detects wake phrases like "hey wyzer" and "wyzer".
"""
import numpy as np
import os
import urllib.request
from pathlib import Path
from typing import Optional, List
from wyzer.core.config import Config
from wyzer.core.logger import get_logger

try:
    from openwakeword.model import Model as WakeWordModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    WakeWordModel = None


class HotwordDetector:
    """Hotword/wake word detection"""
    
    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        threshold: float = Config.HOTWORD_THRESHOLD,
        model_path: Optional[str] = None,
        sample_rate: int = Config.SAMPLE_RATE
    ):
        """
        Initialize hotword detector
        
        Args:
            keywords: List of wake words to detect
            threshold: Detection threshold (0.0-1.0)
            model_path: Optional custom model path
            sample_rate: Audio sample rate
        """
        self.logger = get_logger()
        self.keywords = keywords or Config.HOTWORD_KEYWORDS
        self.threshold = threshold
        self.model_path = model_path or Config.HOTWORD_MODEL_PATH
        self.sample_rate = sample_rate
        self.model = None
        
        # Rising-edge detection: track previous scores per model
        self.prev_scores: dict = {}
        
        if not OPENWAKEWORD_AVAILABLE:
            raise RuntimeError(
                "openWakeWord not available. Install with: pip install openwakeword"
            )
        
        # Ensure ONNX models are available
        self._ensure_onnx_models()
        
        self._init_model()
    
    def _ensure_onnx_models(self) -> None:
        """Ensure ONNX models are downloaded"""
        try:
            # Get openwakeword package location
            import openwakeword
            pkg_dir = Path(openwakeword.__file__).parent
            models_dir = pkg_dir / "resources" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # List of ONNX models to download from the correct GitHub location
            model_urls = {
                "hey_jarvis_v0.1.onnx": "https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.onnx",
            }
            
            downloaded_any = False
            # Download missing ONNX models
            for model_name, url in model_urls.items():
                model_path = models_dir / model_name
                if not model_path.exists():
                    self.logger.info(f"Downloading ONNX model: {model_name}...")
                    try:
                        urllib.request.urlretrieve(url, model_path)
                        self.logger.info(f"Downloaded {model_name}")
                        downloaded_any = True
                    except Exception as e:
                        self.logger.debug(f"Failed to download {model_name}: {e}")
            
            # If downloads failed, check if any ONNX models already exist
            existing_onnx = list(models_dir.glob("*.onnx"))
            if not downloaded_any and not existing_onnx:
                self.logger.warning(
                    "Could not download ONNX models. Hotword detection may not work properly. "
                    "You can manually download models from: "
                    "https://github.com/dscripka/openWakeWord/tree/main/openwakeword/resources/models"
                )
        
        except Exception as e:
            self.logger.warning(f"Could not ensure ONNX models: {e}")
    
    def _init_model(self) -> None:
        """Initialize openWakeWord model"""
        try:
            self.logger.info("Loading openWakeWord model...")
            
            # Use custom model path if provided, otherwise use downloaded ONNX model
            if self.model_path:
                model_paths = [self.model_path]
            else:
                # Find any available ONNX models
                import openwakeword
                pkg_dir = Path(openwakeword.__file__).parent
                models_dir = pkg_dir / "resources" / "models"
                
                # Look for hey_jarvis model first
                jarvis_model = models_dir / "hey_jarvis_v0.1.onnx"
                if jarvis_model.exists():
                    model_paths = [str(jarvis_model)]
                    self.logger.info(f"Using ONNX model: {jarvis_model.name}")
                else:
                    # Fall back to any available .onnx files
                    onnx_models = list(models_dir.glob("*.onnx"))
                    
                    if onnx_models:
                        model_paths = [str(onnx_models[0])]
                        self.logger.info(f"Using ONNX model: {onnx_models[0].name}")
                    else:
                        raise FileNotFoundError(
                            f"No ONNX models found in {models_dir}. "
                            f"Please download models manually from: "
                            f"https://github.com/dscripka/openWakeWord/tree/main/openwakeword/resources/models"
                        )
            
            # Initialize with ONNX models
            self.model = WakeWordModel(
                wakeword_models=model_paths,
                inference_framework="onnx"
            )
            
            self.logger.info(f"openWakeWord initialized for keywords: {self.keywords}")
            self.logger.info(f"Available models: {list(self.model.models.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize openWakeWord: {e}")
            raise
    
    def detect(self, audio_frame: np.ndarray) -> tuple[Optional[str], float]:
        """
        Detect hotword in audio frame with rising-edge detection
        
        Args:
            audio_frame: Audio data as float32 mono at sample_rate
            
        Returns:
            Tuple of (detected keyword string or None, max score)
        """
        if self.model is None:
            return None, 0.0
        
        if len(audio_frame) == 0:
            return None, 0.0
        
        try:
            # openWakeWord expects int16
            # Convert float32 [-1.0, 1.0] to int16
            audio_int16 = (audio_frame * 32767).astype(np.int16)
            
            # Run prediction
            prediction = self.model.predict(audio_int16)
            
            max_score = 0.0
            detected_keyword = None
            
            # Check each keyword with rising-edge detection
            for keyword in self.keywords:
                # Try to find matching model
                model_key = self._find_model_key(keyword)
                
                if model_key and model_key in prediction:
                    score = prediction[model_key]
                    max_score = max(max_score, score)
                    
                    # Get previous score for this model (default to 0.0)
                    prev_score = self.prev_scores.get(model_key, 0.0)
                    
                    # Rising-edge detection: trigger only when crossing threshold
                    if prev_score < self.threshold and score >= self.threshold:
                        self.logger.info(f"Hotword detected: '{keyword}' (score: {score:.3f}, prev: {prev_score:.3f})")
                        detected_keyword = keyword
                    
                    # Update previous score
                    self.prev_scores[model_key] = score
            
            return detected_keyword, max_score
            
        except Exception as e:
            self.logger.error(f"Error in hotword detection: {e}")
            return None, 0.0
    
    def _find_model_key(self, keyword: str) -> Optional[str]:
        """
        Find the model key that matches the keyword
        
        Args:
            keyword: Keyword to find
            
        Returns:
            Model key or None
        """
        if self.model is None:
            return None
        
        available_keys = list(self.model.models.keys())
        
        # If only one model loaded, use it for any keyword
        if len(available_keys) == 1:
            return available_keys[0]
        
        # Normalize keyword
        keyword_normalized = keyword.lower().replace(" ", "_")
        
        # Check exact match first
        if keyword_normalized in self.model.models:
            return keyword_normalized
        
        # Check with variations
        variations = [
            keyword.lower(),
            keyword.replace(" ", ""),
            keyword.replace(" ", "_"),
            f"hey_{keyword.lower()}",
        ]
        
        for var in variations:
            if var in self.model.models:
                return var
        
        # Log available models for debugging
        self.logger.debug(
            f"No exact model found for '{keyword}'. "
            f"Available models: {available_keys}"
        )
        
        # Return first available model as fallback
        if available_keys:
            self.logger.debug(f"Using fallback model: {available_keys[0]}")
            return available_keys[0]
        
        return None
    
    def reset(self) -> None:
        """Reset detector state including previous scores"""
        self.prev_scores = {}
        if self.model:
            # openWakeWord model doesn't need explicit reset for frame-by-frame
            pass
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        if self.model:
            return list(self.model.models.keys())
        return []
