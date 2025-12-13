"""
Microphone stream module using sounddevice.
Captures audio in float32 mono at 16kHz and pushes to queue.
"""
import sounddevice as sd
import numpy as np
from queue import Queue, Full
from typing import Optional, Callable
from wyzer.core.config import Config
from wyzer.core.logger import get_logger


class MicStream:
    """Microphone audio stream manager"""
    
    def __init__(
        self,
        sample_rate: int = Config.SAMPLE_RATE,
        channels: int = Config.CHANNELS,
        chunk_samples: int = Config.CHUNK_SAMPLES,
        device: Optional[int] = None,
        audio_queue: Optional[Queue] = None
    ):
        """
        Initialize microphone stream
        
        Args:
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels (1 for mono)
            chunk_samples: Number of samples per chunk
            device: Optional specific device index
            audio_queue: Queue to push audio frames to
        """
        self.logger = get_logger()
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_samples = chunk_samples
        self.device = device
        self.audio_queue = audio_queue or Queue(maxsize=Config.AUDIO_QUEUE_MAX_SIZE)
        self.stream: Optional[sd.InputStream] = None
        self.is_running = False
        
        # Verify device supports requested sample rate
        if device is not None:
            self._verify_device()
    
    def _verify_device(self) -> None:
        """Verify device capabilities"""
        try:
            device_info = sd.query_devices(self.device, 'input')
            self.logger.debug(f"Using audio device: {device_info['name']}")
            
            # Check if sample rate is supported
            max_input_channels = device_info['max_input_channels']
            default_sr = device_info['default_samplerate']
            
            if max_input_channels < self.channels:
                self.logger.warning(
                    f"Device supports {max_input_channels} channels, requested {self.channels}"
                )
            
            if default_sr != self.sample_rate:
                self.logger.warning(
                    f"Device default sample rate is {default_sr}Hz, requested {self.sample_rate}Hz. "
                    f"Audio will be resampled if needed."
                )
        except Exception as e:
            self.logger.error(f"Error verifying device: {e}")
    
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status
    ) -> None:
        """
        Callback for audio input stream
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Stream status
        """
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        # Convert to mono if needed
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0]
        
        # Ensure float32
        audio_data = audio_data.astype(np.float32)
        
        # Push to queue (non-blocking, drop if full)
        try:
            self.audio_queue.put_nowait(audio_data.copy())
        except Full:
            self.logger.warning("Audio queue full, dropping frame")
    
    def start(self) -> None:
        """Start the audio stream"""
        if self.is_running:
            self.logger.warning("Stream already running")
            return
        
        try:
            self.logger.info(
                f"Starting audio stream: {self.sample_rate}Hz, "
                f"{self.channels}ch, {self.chunk_samples} samples/chunk"
            )
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_samples,
                device=self.device,
                callback=self._audio_callback
            )
            
            self.stream.start()
            self.is_running = True
            self.logger.info("Audio stream started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio stream: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the audio stream"""
        if not self.is_running:
            return
        
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.is_running = False
            self.logger.info("Audio stream stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping audio stream: {e}")
    
    def get_queue(self) -> Queue:
        """Get the audio queue"""
        return self.audio_queue
    
    @staticmethod
    def list_devices() -> None:
        """List all available audio devices"""
        print("\n=== Available Audio Devices ===")
        print(sd.query_devices())
        print()
