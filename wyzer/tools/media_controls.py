"""
Media control tools - play/pause, next, previous, volume control.
"""
import time
import ctypes
import asyncio
from typing import Dict, Any, Optional
from wyzer.tools.tool_base import ToolBase

# Try to import Windows SDK for media session info
try:
    from winsdk.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as MediaManager,
        GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus
    )
    WINSDK_AVAILABLE = True
except ImportError:
    WINSDK_AVAILABLE = False

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


async def _get_media_info_async() -> Optional[Dict[str, Any]]:
    """
    Get current media session info asynchronously.
    
    Returns:
        Dict with media info or None if no media session
    """
    if not WINSDK_AVAILABLE:
        return None
    
    try:
        # Get the media session manager
        manager = await MediaManager.request_async()
        current_session = manager.get_current_session()
        
        if current_session is None:
            return None
        
        # Get media properties
        media_props = await current_session.try_get_media_properties_async()
        
        if media_props is None:
            return None
        
        # Get playback info
        playback_info = current_session.get_playback_info()
        
        # Map playback status
        status_map = {
            PlaybackStatus.CLOSED: "closed",
            PlaybackStatus.OPENED: "opened",
            PlaybackStatus.CHANGING: "changing",
            PlaybackStatus.STOPPED: "stopped",
            PlaybackStatus.PLAYING: "playing",
            PlaybackStatus.PAUSED: "paused",
        }
        
        playback_status = status_map.get(playback_info.playback_status, "unknown")
        
        # Get timeline info if available
        timeline = current_session.get_timeline_properties()
        position_seconds = None
        duration_seconds = None
        
        if timeline:
            # Position and duration are in 100-nanosecond units
            if timeline.position.total_seconds() >= 0:
                position_seconds = int(timeline.position.total_seconds())
            if timeline.end_time.total_seconds() > 0:
                duration_seconds = int(timeline.end_time.total_seconds())
        
        return {
            "title": media_props.title or None,
            "artist": media_props.artist or None,
            "album": media_props.album_title or None,
            "album_artist": media_props.album_artist or None,
            "track_number": media_props.track_number if media_props.track_number > 0 else None,
            "playback_status": playback_status,
            "source_app": current_session.source_app_user_model_id or None,
            "position_seconds": position_seconds,
            "duration_seconds": duration_seconds,
        }
        
    except Exception:
        return None


def _get_media_info() -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper to get media info.
    
    Returns:
        Dict with media info or None
    """
    try:
        # Create new event loop for sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_get_media_info_async())
            return result
        finally:
            loop.close()
    except Exception:
        return None


class GetNowPlayingTool(ToolBase):
    """Tool to get information about currently playing media"""
    
    def __init__(self):
        super().__init__()
        self._name = "get_now_playing"
        self._description = "Get information about currently playing media (song title, artist, album, playback status)"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Get now playing information.
        
        Returns:
            Dict with media info or error
        """
        start_time = time.perf_counter()
        
        if not WINSDK_AVAILABLE:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            return {
                "error": {
                    "type": "dependency_error",
                    "message": "winsdk package not installed. Run: pip install winsdk"
                },
                "latency_ms": latency_ms
            }
        
        try:
            media_info = _get_media_info()
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            if media_info is None:
                return {
                    "status": "no_media",
                    "message": "No media is currently playing",
                    "latency_ms": latency_ms
                }
            
            # Format position/duration as readable strings
            if media_info.get("position_seconds") is not None:
                pos = media_info["position_seconds"]
                media_info["position_formatted"] = f"{pos // 60}:{pos % 60:02d}"
            
            if media_info.get("duration_seconds") is not None:
                dur = media_info["duration_seconds"]
                media_info["duration_formatted"] = f"{dur // 60}:{dur % 60:02d}"
            
            return {
                "status": "ok",
                **media_info,
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
