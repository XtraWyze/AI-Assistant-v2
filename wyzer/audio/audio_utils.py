"""
Audio utility functions for Wyzer AI Assistant.
"""
import numpy as np
from typing import List


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to float32 range [-1.0, 1.0]
    
    Args:
        audio: Input audio array
        
    Returns:
        Normalized audio as float32
    """
    audio = audio.astype(np.float32)
    
    # If audio is int16, convert to float32 range
    if audio.dtype == np.int16 or np.abs(audio).max() > 1.0:
        audio = audio / 32768.0
    
    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio


def ensure_float32(audio: np.ndarray) -> np.ndarray:
    """
    Ensure audio is float32 type
    
    Args:
        audio: Input audio array
        
    Returns:
        Audio as float32
    """
    if audio.dtype != np.float32:
        return normalize_audio(audio)
    return audio


def concat_audio_frames(frames: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple audio frames into single array
    
    Args:
        frames: List of audio frame arrays
        
    Returns:
        Single concatenated audio array
    """
    if not frames:
        return np.array([], dtype=np.float32)
    
    return np.concatenate(frames, axis=0)


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Trim silence from beginning and end of audio
    
    Args:
        audio: Input audio array
        threshold: Amplitude threshold for silence
        
    Returns:
        Trimmed audio
    """
    if len(audio) == 0:
        return audio
    
    # Find first and last non-silent samples
    non_silent = np.abs(audio) > threshold
    if not np.any(non_silent):
        return audio
    
    indices = np.where(non_silent)[0]
    start_idx = indices[0]
    end_idx = indices[-1] + 1
    
    return audio[start_idx:end_idx]


def get_rms_energy(audio: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) energy of audio
    
    Args:
        audio: Input audio array
        
    Returns:
        RMS energy value
    """
    if len(audio) == 0:
        return 0.0
    
    return float(np.sqrt(np.mean(audio ** 2)))


def is_silence_energy_based(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Simple energy-based silence detection
    
    Args:
        audio: Input audio array
        threshold: Energy threshold
        
    Returns:
        True if audio is silence
    """
    return get_rms_energy(audio) < threshold


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Simple linear resampling (for basic use; scipy is better but not in deps)
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    
    # Simple linear interpolation
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    
    indices = np.linspace(0, len(audio) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(audio)), audio)
    
    return resampled.astype(np.float32)


def audio_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    Convert float32 audio to int16
    
    Args:
        audio: Input audio as float32 in range [-1.0, 1.0]
        
    Returns:
        Audio as int16
    """
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)
