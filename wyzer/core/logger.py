"""
Logging module for Wyzer AI Assistant.
Simple, clean logging with optional rich formatting.
"""
import sys
from datetime import datetime
from typing import Optional

try:
    from rich.console import Console
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


class LogLevel:
    """Log level constants"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Logger:
    """Simple logger with timestamps and optional rich formatting"""
    
    def __init__(self, level: str = "INFO"):
        self.level = level
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


def init_logger(level: str = "INFO") -> Logger:
    """Initialize global logger"""
    global _global_logger
    _global_logger = Logger(level)
    return _global_logger


def get_logger() -> Logger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger
