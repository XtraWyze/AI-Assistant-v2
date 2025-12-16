"""
Hotword detection module using openWakeWord.
Detects wake phrases like "hey wyzer" and "wyzer".
Supports multiple wakeword models with per-model thresholds and cooldowns.
"""
import numpy as np
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from wyzer.core.config import Config
from wyzer.core.logger import get_logger

try:
    from openwakeword.model import Model as WakeWordModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    WakeWordModel = None


@dataclass
class HotwordEvent:
    """Structured event emitted on hotword trigger"""
    wakeword: str
    confidence: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "wakeword": self.wakeword,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


@dataclass
class WakewordConfig:
    """Configuration for a single wakeword model"""
    name: str
    model_path: str
    threshold: float
    cooldown_ms: int
    
    @classmethod
    def from_dict(cls, d: dict) -> "WakewordConfig":
        return cls(
            name=d.get("name", "wakeword"),
            model_path=d.get("model_path", ""),
            threshold=d.get("threshold", 0.5),
            cooldown_ms=d.get("cooldown_ms", 1500)
        )


class HotwordDetector:
    """
    Hotword/wake word detection with multi-model support.
    
    Loads multiple ONNX wakeword models into a single openWakeWord instance.
    Each model can have its own threshold and cooldown.
    """
    
    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        threshold: float = Config.HOTWORD_THRESHOLD,
        model_path: Optional[str] = None,
        sample_rate: int = Config.SAMPLE_RATE,
        wakeword_configs: Optional[List[dict]] = None
    ):
        """
        Initialize hotword detector
        
        Args:
            keywords: List of wake words to detect (legacy, for backward compat)
            threshold: Detection threshold (0.0-1.0) (legacy, for backward compat)
            model_path: Optional custom model path (legacy, for backward compat)
            sample_rate: Audio sample rate
            wakeword_configs: List of wakeword config dicts with name, model_path, threshold, cooldown_ms
        """
        self.logger = get_logger()
        self.sample_rate = sample_rate
        self.model = None
        
        # Build wakeword configurations
        self.wakeword_configs: List[WakewordConfig] = []
        self._model_key_to_config: Dict[str, WakewordConfig] = {}
        
        if wakeword_configs:
            # Use provided multi-model configs
            for cfg in wakeword_configs:
                self.wakeword_configs.append(WakewordConfig.from_dict(cfg))
        else:
            # Check for multi-model config in Config class
            multi_configs = Config.get_hotword_models()
            if multi_configs:
                for cfg in multi_configs:
                    self.wakeword_configs.append(WakewordConfig.from_dict(cfg))
            else:
                # Fallback to legacy single-model config
                self.wakeword_configs.append(WakewordConfig(
                    name=keywords[0] if keywords else (Config.HOTWORD_KEYWORDS[0] if Config.HOTWORD_KEYWORDS else "wakeword"),
                    model_path=model_path or Config.HOTWORD_MODEL_PATH,
                    threshold=threshold,
                    cooldown_ms=int(Config.HOTWORD_COOLDOWN_SEC * 1000)
                ))
        
        # Legacy compatibility: populate keywords and threshold from first config
        self.keywords = keywords or Config.HOTWORD_KEYWORDS
        self.threshold = threshold
        self.model_path = model_path or Config.HOTWORD_MODEL_PATH
        
        # Rising-edge detection: track previous scores per model
        self.prev_scores: Dict[str, float] = {}

        # Consecutive-frame streak tracking (per model key)
        self.streak_counts: Dict[str, int] = {}
        
        # Per-model cooldown tracking (last trigger time)
        self._last_trigger_time: Dict[str, float] = {}
        
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
        """Initialize openWakeWord model with all configured wakeword models"""
        try:
            self.logger.info("Loading openWakeWord model...")
            
            # Get openwakeword models directory
            import openwakeword
            pkg_dir = Path(openwakeword.__file__).parent
            models_dir = pkg_dir / "resources" / "models"
            
            # Resolve all model paths from wakeword configs
            model_paths: List[str] = []
            loaded_names: List[str] = []
            
            for cfg in self.wakeword_configs:
                resolved_path = self._resolve_model_path(cfg.model_path, models_dir)
                if resolved_path:
                    model_paths.append(resolved_path)
                    loaded_names.append(cfg.name)
                    self.logger.info(f"Resolved model for '{cfg.name}': {Path(resolved_path).name}")
                else:
                    self.logger.warning(f"Could not resolve model path for '{cfg.name}': {cfg.model_path}")
            
            # Fallback if no models resolved
            if not model_paths:
                # Try hey_jarvis as default
                jarvis_model = models_dir / "hey_jarvis_v0.1.onnx"
                if jarvis_model.exists():
                    model_paths = [str(jarvis_model)]
                    self.logger.info(f"Using fallback ONNX model: {jarvis_model.name}")
                else:
                    # Fall back to any available .onnx files
                    onnx_models = list(models_dir.glob("*.onnx"))
                    if onnx_models:
                        model_paths = [str(onnx_models[0])]
                        self.logger.info(f"Using fallback ONNX model: {onnx_models[0].name}")
                    else:
                        raise FileNotFoundError(
                            f"No ONNX models found in {models_dir}. "
                            f"Please download models manually from: "
                            f"https://github.com/dscripka/openWakeWord/tree/main/openwakeword/resources/models"
                        )
            
            # Initialize with all ONNX models in a single instance
            self.model = WakeWordModel(
                wakeword_models=model_paths,
                inference_framework="onnx"
            )
            
            # Build model_key -> config mapping
            self._build_model_key_mapping()
            
            loaded_model_names = list(self.model.models.keys())
            self.logger.info(f"[HOTWORD] Loaded models: {loaded_model_names}")
            self.logger.info(f"[HOTWORD] Wakeword configs: {[c.name for c in self.wakeword_configs]}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize openWakeWord: {e}")
            raise
    
    def _resolve_model_path(self, model_path: str, models_dir: Path) -> Optional[str]:
        """Resolve a model path to an absolute path"""
        if not model_path:
            return None
        
        path = Path(model_path)
        
        # Already absolute and exists
        if path.is_absolute() and path.exists():
            return str(path)
        
        # Check in openwakeword models directory
        resolved = models_dir / model_path
        if resolved.exists():
            return str(resolved)
        
        # Check relative to cwd
        if path.exists():
            return str(path.resolve())
        
        return None
    
    def _build_model_key_mapping(self) -> None:
        """Build mapping from openWakeWord model keys to our wakeword configs"""
        if not self.model:
            return
        
        available_keys = list(self.model.models.keys())
        
        for cfg in self.wakeword_configs:
            # Try to find matching model key
            model_key = self._find_model_key_for_config(cfg, available_keys)
            if model_key:
                self._model_key_to_config[model_key] = cfg
                self.logger.debug(f"Mapped model key '{model_key}' -> config '{cfg.name}'")
    
    def _find_model_key_for_config(self, cfg: WakewordConfig, available_keys: List[str]) -> Optional[str]:
        """Find the openWakeWord model key that matches a config"""
        # Extract base name from model path
        model_basename = Path(cfg.model_path).stem.lower()
        
        # Try exact match on model basename
        for key in available_keys:
            if key.lower() == model_basename:
                return key
        
        # Try partial match
        for key in available_keys:
            if model_basename in key.lower() or key.lower() in model_basename:
                return key
        
        # Try name-based matching
        name_normalized = cfg.name.lower().replace(" ", "_")
        for key in available_keys:
            if name_normalized in key.lower() or key.lower() in name_normalized:
                return key
        
        # If only one model and one config, use it
        if len(available_keys) == 1 and len(self.wakeword_configs) == 1:
            return available_keys[0]
        
        return None
    
    def _get_config_for_model_key(self, model_key: str) -> Optional[WakewordConfig]:
        """Get the wakeword config for a model key"""
        if model_key in self._model_key_to_config:
            return self._model_key_to_config[model_key]
        # Fallback: return first config if only one exists
        if len(self.wakeword_configs) == 1:
            return self.wakeword_configs[0]
        return None
    
    def _is_in_cooldown(self, model_key: str, cfg: WakewordConfig) -> bool:
        """Check if a wakeword is in cooldown period"""
        last_trigger = self._last_trigger_time.get(model_key, 0.0)
        cooldown_sec = cfg.cooldown_ms / 1000.0
        return (time.time() - last_trigger) < cooldown_sec
    
    def detect(self, audio_frame: np.ndarray) -> tuple[Optional[str], float]:
        """
        Detect hotword in audio frame with rising-edge detection.
        
        Scores all loaded models once per frame. Selects the highest-confidence
        wakeword above its per-model threshold, respecting per-model cooldowns.
        
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
            
            # Run prediction once for all models
            prediction = self.model.predict(audio_int16)
            
            # Track candidates: (model_key, config, score, streak)
            candidates: List[tuple] = []
            max_score = 0.0
            
            # Process all model predictions
            for model_key, score in prediction.items():
                max_score = max(max_score, score)
                
                # Get config for this model
                cfg = self._get_config_for_model_key(model_key)
                if cfg is None:
                    # No config for this model, use legacy threshold
                    cfg = WakewordConfig(
                        name=model_key,
                        model_path="",
                        threshold=self.threshold,
                        cooldown_ms=int(Config.HOTWORD_COOLDOWN_SEC * 1000)
                    )
                
                # Get previous score for this model
                prev_score = self.prev_scores.get(model_key, 0.0)
                
                # Update streak count using per-model threshold
                prev_streak = self.streak_counts.get(model_key, 0)
                if score >= cfg.threshold:
                    streak = prev_streak + 1
                else:
                    streak = 0
                self.streak_counts[model_key] = streak
                
                # Check if this model should trigger
                required_streak = max(1, int(getattr(Config, "HOTWORD_TRIGGER_STREAK", 1)))
                should_trigger = False
                
                if required_streak == 1:
                    # Rising-edge: trigger when crossing threshold upward
                    if prev_score < cfg.threshold and score >= cfg.threshold:
                        should_trigger = True
                else:
                    # Streak-based: trigger when reaching required streak
                    if prev_streak < required_streak and streak >= required_streak:
                        should_trigger = True
                
                # Update previous score
                self.prev_scores[model_key] = score
                
                # Check cooldown before adding to candidates
                if should_trigger and not self._is_in_cooldown(model_key, cfg):
                    candidates.append((model_key, cfg, score, streak))
            
            # Select highest-confidence candidate
            if candidates:
                # Sort by score descending
                candidates.sort(key=lambda x: x[2], reverse=True)
                best_model_key, best_cfg, best_score, best_streak = candidates[0]
                
                # Record trigger time for cooldown
                self._last_trigger_time[best_model_key] = time.time()
                
                # Create structured event
                event = HotwordEvent(
                    wakeword=best_cfg.name,
                    confidence=best_score,
                    timestamp=time.time()
                )
                
                self.logger.info(
                    f"[HOTWORD] Triggered \"{best_cfg.name}\" confidence={best_score:.3f}"
                )
                
                # Return the wakeword name (for backward compat with state machine)
                return best_cfg.name, best_score
            
            return None, max_score
            
        except Exception as e:
            self.logger.error(f"Error in hotword detection: {e}")
            return None, 0.0
    
    def detect_with_event(self, audio_frame: np.ndarray) -> tuple[Optional[HotwordEvent], float]:
        """
        Detect hotword and return structured event.
        
        Args:
            audio_frame: Audio data as float32 mono at sample_rate
            
        Returns:
            Tuple of (HotwordEvent or None, max score)
        """
        if self.model is None:
            return None, 0.0
        
        if len(audio_frame) == 0:
            return None, 0.0
        
        try:
            audio_int16 = (audio_frame * 32767).astype(np.int16)
            prediction = self.model.predict(audio_int16)
            
            candidates: List[tuple] = []
            max_score = 0.0
            
            for model_key, score in prediction.items():
                max_score = max(max_score, score)
                
                cfg = self._get_config_for_model_key(model_key)
                if cfg is None:
                    cfg = WakewordConfig(
                        name=model_key,
                        model_path="",
                        threshold=self.threshold,
                        cooldown_ms=int(Config.HOTWORD_COOLDOWN_SEC * 1000)
                    )
                
                prev_score = self.prev_scores.get(model_key, 0.0)
                prev_streak = self.streak_counts.get(model_key, 0)
                
                if score >= cfg.threshold:
                    streak = prev_streak + 1
                else:
                    streak = 0
                self.streak_counts[model_key] = streak
                
                required_streak = max(1, int(getattr(Config, "HOTWORD_TRIGGER_STREAK", 1)))
                should_trigger = False
                
                if required_streak == 1:
                    if prev_score < cfg.threshold and score >= cfg.threshold:
                        should_trigger = True
                else:
                    if prev_streak < required_streak and streak >= required_streak:
                        should_trigger = True
                
                self.prev_scores[model_key] = score
                
                if should_trigger and not self._is_in_cooldown(model_key, cfg):
                    candidates.append((model_key, cfg, score))
            
            if candidates:
                candidates.sort(key=lambda x: x[2], reverse=True)
                best_model_key, best_cfg, best_score = candidates[0]
                
                self._last_trigger_time[best_model_key] = time.time()
                
                event = HotwordEvent(
                    wakeword=best_cfg.name,
                    confidence=best_score,
                    timestamp=time.time()
                )
                
                self.logger.info(
                    f"[HOTWORD] Triggered \"{best_cfg.name}\" confidence={best_score:.3f}"
                )
                
                return event, best_score
            
            return None, max_score
            
        except Exception as e:
            self.logger.error(f"Error in hotword detection: {e}")
            return None, 0.0
    
    def _find_model_key(self, keyword: str) -> Optional[str]:
        """
        Find the model key that matches the keyword (legacy compatibility).
        
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
        """Reset detector state including previous scores, streaks, and cooldowns"""
        self.prev_scores = {}
        self.streak_counts = {}
        self._last_trigger_time = {}
        if self.model:
            # openWakeWord model doesn't need explicit reset for frame-by-frame
            pass
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        if self.model:
            return list(self.model.models.keys())
        return []
    
    def get_wakeword_configs(self) -> List[WakewordConfig]:
        """Get list of configured wakeword configs"""
        return self.wakeword_configs.copy()
