"""
Media control tools - play/pause, next, previous, volume control.
"""
import time
import ctypes
from typing import Dict, Any
from wyzer.tools.tool_base import ToolBase

# Windows virtual key codes for media keys
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_VOLUME_UP = 0xAF
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_MUTE = 0xAD

# Key event flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

# Windows API
user32 = ctypes.windll.user32


def _send_key(vk_code: int) -> None:
    """
    Send a virtual key press and release.
    
    Args:
        vk_code: Virtual key code
    """
    # Key down
    user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
    # Key up
    user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


class MediaPlayPauseTool(ToolBase):
    """Tool to toggle media play/pause"""
    
    def __init__(self):
        super().__init__()
        self._name = "media_play_pause"
        self._description = "Toggle play/pause for currently playing media (music, video, etc.)"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Toggle play/pause.
        
        Returns:
            Dict with status or error
        """
        start_time = time.perf_counter()
        
        try:
            _send_key(VK_MEDIA_PLAY_PAUSE)
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "ok",
                "action": "play_pause",
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }


class MediaNextTool(ToolBase):
    """Tool to skip to next media track"""
    
    def __init__(self):
        super().__init__()
        self._name = "media_next"
        self._description = "Skip to next track/video in currently playing media"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Skip to next track.
        
        Returns:
            Dict with status or error
        """
        start_time = time.perf_counter()
        
        try:
            _send_key(VK_MEDIA_NEXT_TRACK)
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "ok",
                "action": "next",
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }


class MediaPreviousTool(ToolBase):
    """Tool to skip to previous media track"""
    
    def __init__(self):
        super().__init__()
        self._name = "media_previous"
        self._description = "Skip to previous track/video in currently playing media"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Skip to previous track.
        
        Returns:
            Dict with status or error
        """
        start_time = time.perf_counter()
        
        try:
            _send_key(VK_MEDIA_PREV_TRACK)
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "ok",
                "action": "previous",
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }


class VolumeUpTool(ToolBase):
    """Tool to increase volume"""
    
    def __init__(self):
        super().__init__()
        self._name = "volume_up"
        self._description = "Increase system volume by specified steps (default 2)"
        self._args_schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 2,
                    "description": "Number of volume increase steps (1-10)"
                }
            },
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Increase volume.
        
        Args:
            steps: Number of steps (default 2)
        
        Returns:
            Dict with status or error
        """
        start_time = time.perf_counter()
        
        steps = kwargs.get("steps", 2)
        
        try:
            for _ in range(steps):
                _send_key(VK_VOLUME_UP)
                time.sleep(0.05)  # Small delay between steps
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "ok",
                "action": "volume_up",
                "steps": steps,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }


class VolumeDownTool(ToolBase):
    """Tool to decrease volume"""
    
    def __init__(self):
        super().__init__()
        self._name = "volume_down"
        self._description = "Decrease system volume by specified steps (default 2)"
        self._args_schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 2,
                    "description": "Number of volume decrease steps (1-10)"
                }
            },
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Decrease volume.
        
        Args:
            steps: Number of steps (default 2)
        
        Returns:
            Dict with status or error
        """
        start_time = time.perf_counter()
        
        steps = kwargs.get("steps", 2)
        
        try:
            for _ in range(steps):
                _send_key(VK_VOLUME_DOWN)
                time.sleep(0.05)  # Small delay between steps
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "ok",
                "action": "volume_down",
                "steps": steps,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }


class VolumeMuteToggleTool(ToolBase):
    """Tool to toggle volume mute"""
    
    def __init__(self):
        super().__init__()
        self._name = "volume_mute_toggle"
        self._description = "Toggle system volume mute on/off"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Toggle mute.
        
        Returns:
            Dict with status or error
        """
        start_time = time.perf_counter()
        
        try:
            _send_key(VK_VOLUME_MUTE)
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "ok",
                "action": "mute_toggle",
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }
