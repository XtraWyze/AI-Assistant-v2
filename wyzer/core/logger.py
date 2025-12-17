"""
Logging module for Wyzer AI Assistant.
Simple, clean logging with optional rich formatting.
"""
import os
import re
import sys
from datetime import datetime
from typing import Optional, List

try:
    from rich.console import Console
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# Patterns to filter out in quiet mode (heartbeats, internal debug info, startup details)
QUIET_MODE_FILTERS: List[str] = [
    r"\[HEARTBEAT\]",           # Heartbeat messages
    r"\[POOL\]",                # Tool worker pool internals
    r"\[BRAIN\]",               # Brain process messages
    r"q_in=.*q_out=",           # Queue status messages
    r"jobs_processed=",         # Job processing stats
    r"\[VAD\]",                 # Voice activity detection internals
    r"\[HOTWORD\]",             # Hotword detection internals
    r"Drained \d+ frames",      # Audio frame draining
    r"Draining audio queue",    # Audio queue operations
    r"\[ROLE\]",                # Role/architecture logging
    r"Loading Silero VAD",      # VAD loading
    r"Silero VAD initialized",  # VAD init
    r"Loading openWakeWord",    # Wakeword loading
    r"Resolved model for",      # Model resolution
    r"Hotword detection enabled", # Hotword status
    r"Starting assistant",      # Startup message
    r"Main orchestrator startup", # Orchestrator startup
    r"Main responsibilities",   # Role descriptions
    r"Starting Wyzer Assistant", # Multiprocess startup
    r"Core \(main thread\)",    # Core thread info
    r"Brain \(worker process\)", # Brain process info
    r"Starting audio stream",   # Audio stream startup
    r"Audio stream started",    # Audio stream ready
    r"Listening for hotword",   # Hotword listening
    r"Brain worker startup",    # Brain worker startup
    r"Brain responsibilities",  # Brain role
    r"Loading Whisper model",   # Whisper loading
    r"Whisper model loaded",    # Whisper loaded
    r"STT Router.*initialized", # STT init
    r"Piper TTS initialized",   # TTS init
    r"TTS engine initialized",  # TTS engine
    r"brain_worker_started",    # Brain ready
    r"ToolWorker-\d+",          # Tool worker startup
    r"Running in quiet mode",   # Quiet mode message
    # Runtime operation messages
    r"Hotword.*accepted",       # Hotword accepted
    r"Recording stopped",       # Recording status
    r"\[HYBRID\]",              # Hybrid router decisions
    r"\[INTENT\]",              # Intent execution
    r"\[TOOLS\]",               # Tool execution details
    r"\[LLM\]",                 # LLM processing
    r"\[STT\]",                 # Speech-to-text
    r"\[TTS\]",                 # Text-to-speech internals
    r"FOLLOWUP",                # Followup handling
    r"Transcription:",          # Transcription logs
    r"silence timeout",         # Silence detection
    r"max duration",            # Recording limits
]

# Compiled patterns for efficient matching
_quiet_mode_patterns: Optional[List[re.Pattern]] = None


def _get_quiet_filters() -> List[re.Pattern]:
    """Get compiled regex patterns for quiet mode filtering"""
    global _quiet_mode_patterns
    if _quiet_mode_patterns is None:
        _quiet_mode_patterns = [re.compile(p, re.IGNORECASE) for p in QUIET_MODE_FILTERS]
    return _quiet_mode_patterns


def _should_filter_quiet(message: str) -> bool:
    """Check if message should be filtered in quiet mode"""
    for pattern in _get_quiet_filters():
        if pattern.search(message):
            return True
    return False


class LogLevel:
    """Log level constants"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Logger:
    """Simple logger with timestamps and optional rich formatting"""
    
    def __init__(self, level: str = "INFO", quiet_mode: bool = False):
        self.level = level
        self.quiet_mode = quiet_mode
        self.level_priority = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        self.use_rich = RICH_AVAILABLE
    
    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on level"""
        return self.level_priority.get(level, 0) >= self.level_priority.get(self.level, 0)
    
    def _should_filter_message(self, message: str) -> bool:
        """Check if message should be filtered (quiet mode)"""
        if not self.quiet_mode:
            return False
        return _should_filter_quiet(message)
    
    def _format_message(self, level: str, message: str) -> str:
        """Format log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"[{timestamp}] [{level:8}] {message}"
    
    def _get_level_color(self, level: str) -> str:
        """Get color for log level when using rich"""
        colors = {
            "DEBUG": "dim cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red"
        }
        return colors.get(level, "white")
    
    def log(self, level: str, message: str) -> None:
        """Log a message at the specified level"""
        if not self._should_log(level):
            return
        
        # Filter out noisy messages in quiet mode
        if self._should_filter_message(message):
            return
        
        formatted = self._format_message(level, message)
        
        if self.use_rich and console:
            color = self._get_level_color(level)
            console.print(formatted, style=color)
        else:
            print(formatted, flush=True)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.log(LogLevel.DEBUG, message)
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.log(LogLevel.INFO, message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.log(LogLevel.WARNING, message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.log(LogLevel.ERROR, message)
    
    def critical(self, message: str) -> None:
        """Log critical message"""
        self.log(LogLevel.CRITICAL, message)


# Global logger instance
_global_logger: Optional[Logger] = None


def init_logger(level: str = "INFO", quiet_mode: bool = False) -> Logger:
    """
    Initialize global logger
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        quiet_mode: If True, filter out noisy debug messages like heartbeats
    """
    global _global_logger
    _global_logger = Logger(level, quiet_mode=quiet_mode)
    return _global_logger


def get_logger() -> Logger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        # Check environment for quiet mode
        quiet = os.environ.get("WYZER_QUIET_MODE", "false").lower() in ("true", "1", "yes")
        _global_logger = Logger(quiet_mode=quiet)
    return _global_logger


def set_quiet_mode(enabled: bool) -> None:
    """Enable or disable quiet mode on the global logger"""
    global _global_logger
    if _global_logger:
        _global_logger.quiet_mode = enabled
