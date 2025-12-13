"""
Audio player for TTS output.
Supports interruptible playback with stop events.
"""
import wave
import numpy as np
import sounddevice as sd
import threading
from typing import Optional
from wyzer.core.logger import get_logger


class AudioPlayer:
    """Interruptible audio player for TTS output"""
    
    def __init__(self, device: Optional[int] = None):
        """
        Initialize audio player
        
        Args:
            device: Optional sounddevice output device index
        """
        self.logger = get_logger()
        self.device = device
        self.current_stream: Optional[sd.OutputStream] = None
        self.stream_lock = threading.Lock()
    
    def play_wav(self, wav_path: str, stop_event: threading.Event) -> bool:
        """
        Play WAV file with interruptible streaming
        
        Args:
            wav_path: Path to WAV file
            stop_event: Event to signal stop
            
        Returns:
            True if played to completion, False if interrupted or error
        """
        try:
            # Open WAV file
            with wave.open(wav_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                
                # Read all audio data
                frames = wf.readframes(wf.getnframes())
                
                # Convert to numpy array
                if sample_width == 2:
                    # 16-bit PCM
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                elif sample_width == 1:
                    # 8-bit PCM
                    audio_data = np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128
                    audio_data = audio_data * 256  # Scale to 16-bit range
                elif sample_width == 4:
                    # 32-bit PCM
                    audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    self.logger.error(f"Unsupported sample width: {sample_width}")
                    return False
                
                # Convert to float32 in range [-1.0, 1.0]
                if sample_width != 4:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Reshape for channels
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels)
            
            # Play in chunks with stop checking
            chunk_size = int(sample_rate * 0.05)  # 50ms chunks
            total_samples = len(audio_data)
            position = 0
            
            with self.stream_lock:
                self.current_stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=channels,
                    dtype='float32',
                    device=self.device
                )
                self.current_stream.start()
            
            try:
                while position < total_samples:
                    # Check stop event
                    if stop_event.is_set():
                        self.logger.debug("Audio playback interrupted")
                        return False
                    
                    # Get next chunk
                    end_pos = min(position + chunk_size, total_samples)
                    chunk = audio_data[position:end_pos]
                    
                    # Write chunk
                    self.current_stream.write(chunk)
                    
                    position = end_pos
                
                # Finished successfully
                self.logger.debug("Audio playback completed")
                return True
                
            finally:
                with self.stream_lock:
                    if self.current_stream:
                        self.current_stream.stop()
                        self.current_stream.close()
                        self.current_stream = None
        
        except FileNotFoundError:
            self.logger.error(f"WAV file not found: {wav_path}")
            return False
        
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")
            return False
    
    def stop(self) -> None:
        """Stop current playback immediately"""
        with self.stream_lock:
            if self.current_stream:
                try:
                    self.current_stream.stop()
                    self.current_stream.close()
                except:
                    pass
                self.current_stream = None
