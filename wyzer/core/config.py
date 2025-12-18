"""
Configuration module for Wyzer AI Assistant.
Centralizes all settings with environment variable overrides.
"""
import os
from typing import List, Optional


class Config:
    """Central configuration for Wyzer"""
    
    # Audio settings
    SAMPLE_RATE: int = int(os.environ.get("WYZER_SAMPLE_RATE", "16000"))
    CHANNELS: int = 1
    CHUNK_MS: int = int(os.environ.get("WYZER_CHUNK_MS", "20"))
    CHUNK_SAMPLES: int = SAMPLE_RATE * CHUNK_MS // 1000  # 320 samples for 20ms at 16kHz
    
    # Recording limits
    MAX_RECORD_SECONDS: float = float(os.environ.get("WYZER_MAX_RECORD_SECONDS", "10.0"))
    VAD_SILENCE_TIMEOUT: float = float(os.environ.get("WYZER_VAD_SILENCE_TIMEOUT", "1.2"))
    
    # No-speech-start timeout: abort listening if VAD never detects speech start within this window
    # This provides a fast exit when user triggers hotword but stays silent (instead of waiting for max duration).
    NO_SPEECH_START_TIMEOUT_SEC: float = float(os.environ.get("WYZER_NO_SPEECH_START_TIMEOUT_SEC", "2.5"))
    
    # VAD settings
    VAD_THRESHOLD: float = float(os.environ.get("WYZER_VAD_THRESHOLD", "0.5"))
    VAD_MIN_SPEECH_DURATION_MS: int = int(os.environ.get("WYZER_VAD_MIN_SPEECH_MS", "250"))
    
    # Hotword settings
    # Legacy single-wakeword settings (still supported for backward compatibility)
    HOTWORD_KEYWORDS: List[str] = os.environ.get("WYZER_HOTWORD_KEYWORDS", "hey wyzer,wyzer").split(",")
    HOTWORD_THRESHOLD: float = float(os.environ.get("WYZER_HOTWORD_THRESHOLD", "0.5"))
    # Require this many consecutive frames above threshold before triggering.
    # Helps prevent false triggers from noise / TTS bleed-through.
    HOTWORD_TRIGGER_STREAK: int = int(os.environ.get("WYZER_HOTWORD_TRIGGER_STREAK", "3"))
    HOTWORD_MODEL_PATH: str = os.environ.get("WYZER_HOTWORD_MODEL_PATH", "hey_Wyzer.onnx")
    HOTWORD_COOLDOWN_SEC: float = float(os.environ.get("WYZER_HOTWORD_COOLDOWN_SEC", "1.5"))
    POST_IDLE_DRAIN_SEC: float = float(os.environ.get("WYZER_POST_IDLE_DRAIN_SEC", "0.5"))
    
    # Multi-wakeword configuration (list of wakeword model configs)
    # Each entry: {"name": str, "model_path": str, "threshold": float, "cooldown_ms": int}
    # If empty, falls back to legacy single-model config above
    HOTWORD_MODELS: List[dict] = [
        {
            "name": "hey wyzer",
            "model_path": "hey_Wyzer.onnx",
            "threshold": 0.5,
            "cooldown_ms": 1500
        },
        {
            "name": "wyzer",
            "model_path": "wiser.onnx",
            "threshold": 0.75,
            "cooldown_ms": 1500
        }
    ]
    
    @classmethod
    def get_hotword_models(cls) -> List[dict]:
        """
        Get configured hotword models.
        Returns multi-model config if set, otherwise builds from legacy single-model config.
        """
        if cls.HOTWORD_MODELS:
            return cls.HOTWORD_MODELS
        # Fallback to legacy single-model config
        return [{
            "name": cls.HOTWORD_KEYWORDS[0] if cls.HOTWORD_KEYWORDS else "wakeword",
            "model_path": cls.HOTWORD_MODEL_PATH,
            "threshold": cls.HOTWORD_THRESHOLD,
            "cooldown_ms": int(cls.HOTWORD_COOLDOWN_SEC * 1000)
        }]
    
    # STT settings
    WHISPER_MODEL: str = os.environ.get("WYZER_WHISPER_MODEL", "small")
    WHISPER_DEVICE: str = os.environ.get("WYZER_WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE: str = os.environ.get("WYZER_WHISPER_COMPUTE_TYPE", "int8")
    
    # Repetition filter (token repeats > this threshold => garbage)
    MAX_TOKEN_REPEATS: int = int(os.environ.get("WYZER_MAX_TOKEN_REPEATS", "6"))
    MIN_TRANSCRIPT_LENGTH: int = int(os.environ.get("WYZER_MIN_TRANSCRIPT_LENGTH", "2"))
    
    # LLM Brain settings (Phase 4 - enhanced Phase 7)
    LLM_MODE: str = os.environ.get("WYZER_LLM_MODE", "ollama")  # "ollama" or "off"
    NO_OLLAMA: bool = os.environ.get("WYZER_NO_OLLAMA", "false").lower() in ("true", "1", "yes")  # Run without Ollama entirely
    OLLAMA_BASE_URL: str = os.environ.get("WYZER_OLLAMA_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = os.environ.get("WYZER_OLLAMA_MODEL", "llama3.1:latest")
    LLM_TIMEOUT: int = int(os.environ.get("WYZER_LLM_TIMEOUT", "30"))
    OLLAMA_STREAM: bool = os.environ.get("WYZER_OLLAMA_STREAM", "true").lower() in ("true", "1", "yes")
    OLLAMA_TEMPERATURE: float = float(os.environ.get("WYZER_OLLAMA_TEMPERATURE", "0.4"))
    OLLAMA_TOP_P: float = float(os.environ.get("WYZER_OLLAMA_TOP_P", "0.9"))
    OLLAMA_NUM_CTX: int = int(os.environ.get("WYZER_OLLAMA_NUM_CTX", "4096"))
    OLLAMA_NUM_PREDICT: int = int(os.environ.get("WYZER_OLLAMA_NUM_PREDICT", "120"))
    LLM_MAX_PROMPT_CHARS: int = int(os.environ.get("WYZER_LLM_MAX_PROMPT_CHARS", "8000"))
    
    # TTS settings (Phase 5)
    TTS_ENABLED: bool = os.environ.get("WYZER_TTS_ENABLED", "true").lower() in ("true", "1", "yes")
    TTS_ENGINE: str = os.environ.get("WYZER_TTS_ENGINE", "piper")
    PIPER_EXE_PATH: str = os.environ.get("WYZER_PIPER_EXE_PATH", "./assets/piper/piper.exe")
    PIPER_MODEL_PATH: str = os.environ.get("WYZER_PIPER_MODEL_PATH", "./assets/piper/en_US-voice.onnx")
    PIPER_SPEAKER_ID: Optional[int] = None if not os.environ.get("WYZER_PIPER_SPEAKER_ID") else int(os.environ.get("WYZER_PIPER_SPEAKER_ID"))
    TTS_RATE: float = float(os.environ.get("WYZER_TTS_RATE", "1.0"))
    TTS_OUTPUT_DEVICE: Optional[int] = None if not os.environ.get("WYZER_TTS_OUTPUT_DEVICE") else int(os.environ.get("WYZER_TTS_OUTPUT_DEVICE"))
    SPEAK_HOTWORD_INTERRUPT: bool = os.environ.get("WYZER_SPEAK_HOTWORD_INTERRUPT", "true").lower() in ("true", "1", "yes")
    POST_SPEAK_DRAIN_SEC: float = float(os.environ.get("WYZER_POST_SPEAK_DRAIN_SEC", "0.35"))
    SPEAK_START_COOLDOWN_SEC: float = float(os.environ.get("WYZER_SPEAK_START_COOLDOWN_SEC", "1.8"))
    POST_BARGEIN_IGNORE_SEC: float = float(os.environ.get("WYZER_POST_BARGEIN_IGNORE_SEC", "3.0"))
    POST_BARGEIN_REQUIRE_SPEECH_START: bool = os.environ.get("WYZER_POST_BARGEIN_REQUIRE_SPEECH_START", "true").lower() in ("true", "1", "yes")
    POST_BARGEIN_WAIT_FOR_SPEECH_SEC: float = float(os.environ.get("WYZER_POST_BARGEIN_WAIT_FOR_SPEECH_SEC", "2.0"))
    
    # Queue settings
    AUDIO_QUEUE_MAX_SIZE: int = int(os.environ.get("WYZER_AUDIO_QUEUE_MAX_SIZE", "100"))
    
    # Logging
    LOG_LEVEL: str = os.environ.get("WYZER_LOG_LEVEL", "INFO")
    
    # Quiet Mode - hides debug info like heartbeats for cleaner user experience
    QUIET_MODE: bool = os.environ.get("WYZER_QUIET_MODE", "false").lower() in ("true", "1", "yes")
    
    # Tool Safety Settings (Phase 6)
    ENABLE_FORCE_CLOSE: bool = os.environ.get("WYZER_ENABLE_FORCE_CLOSE", "false").lower() in ("true", "1", "yes")
    ALLOWED_APPS_TO_LAUNCH: List[str] = os.environ.get(
        "WYZER_ALLOWED_APPS_TO_LAUNCH",
        "notepad,calc,calculator,paint,explorer,chrome,firefox,edge,vscode,cmd,powershell"
    ).split(",")
    ALLOWED_PROCESSES_TO_CLOSE: List[str] = os.environ.get(
        "WYZER_ALLOWED_PROCESSES_TO_CLOSE",
        ""
    ).split(",") if os.environ.get("WYZER_ALLOWED_PROCESSES_TO_CLOSE") else []
    REQUIRE_EXPLICIT_APP_MATCH: bool = os.environ.get("WYZER_REQUIRE_EXPLICIT_APP_MATCH", "true").lower() in ("true", "1", "yes")

    # LocalLibrary Auto-Alias (learn spoken phrases -> targets)
    AUTO_ALIAS_ENABLED: bool = os.environ.get("WYZER_AUTO_ALIAS_ENABLED", "true").lower() in ("true", "1", "yes")
    AUTO_ALIAS_MIN_CONFIDENCE: float = float(os.environ.get("WYZER_AUTO_ALIAS_MIN_CONFIDENCE", "0.85"))
    
    # FOLLOWUP listening window settings
    FOLLOWUP_ENABLED: bool = os.environ.get("WYZER_FOLLOWUP_ENABLED", "true").lower() in ("true", "1", "yes")
    FOLLOWUP_TIMEOUT_SEC: float = float(os.environ.get("WYZER_FOLLOWUP_TIMEOUT_SEC", "2.0"))
    FOLLOWUP_MAX_CHAIN: int = int(os.environ.get("WYZER_FOLLOWUP_MAX_CHAIN", "3"))
    
    # Tool Worker Pool settings (Runtime verification & warm workers)
    TOOL_POOL_ENABLED: bool = os.environ.get("WYZER_TOOL_POOL_ENABLED", "true").lower() in ("true", "1", "yes")
    TOOL_POOL_WORKERS: int = max(1, min(5, int(os.environ.get("WYZER_TOOL_POOL_WORKERS", "3"))))  # 1-5 workers
    TOOL_POOL_TIMEOUT_SEC: int = int(os.environ.get("WYZER_TOOL_POOL_TIMEOUT_SEC", "15"))
    
    # Heartbeat & verification settings
    HEARTBEAT_INTERVAL_SEC: float = float(os.environ.get("WYZER_HEARTBEAT_INTERVAL_SEC", "10.0"))
    VERIFY_MODE: bool = os.environ.get("WYZER_VERIFY_MODE", "false").lower() in ("true", "1", "yes")
    
    # Memory settings (Phase 7)
    SESSION_MEMORY_TURNS: int = int(os.environ.get("WYZER_SESSION_MEMORY_TURNS", "10"))
    MEMORY_FILE_PATH: str = os.environ.get("WYZER_MEMORY_FILE_PATH", "wyzer/data/memory.json")
    
    # Memory Injection flag - inject all long-term memories into LLM prompts
    # Default: True (can be disabled via --no-memories flag or WYZER_USE_MEMORIES=0 env var)
    USE_MEMORIES: bool = True
    
    @classmethod
    def get_frame_duration_ms(cls) -> float:
        """Get frame duration in milliseconds"""
        return cls.CHUNK_MS
    
    @classmethod
    def get_samples_per_frame(cls) -> int:
        """Get number of samples per frame"""
        return cls.CHUNK_SAMPLES
    
    @classmethod
    def get_max_record_frames(cls) -> int:
        """Get maximum number of frames to record"""
        return int(cls.MAX_RECORD_SECONDS * cls.SAMPLE_RATE / cls.CHUNK_SAMPLES)
    
    @classmethod
    def get_silence_timeout_frames(cls) -> int:
        """Get number of frames for silence timeout"""
        return int(cls.VAD_SILENCE_TIMEOUT * cls.SAMPLE_RATE / cls.CHUNK_SAMPLES)
    
    @classmethod
    def get_no_speech_start_timeout_frames(cls) -> int:
        """Get number of frames for no-speech-start timeout (abort if speech never begins)"""
        return int(cls.NO_SPEECH_START_TIMEOUT_SEC * cls.SAMPLE_RATE / cls.CHUNK_SAMPLES)
