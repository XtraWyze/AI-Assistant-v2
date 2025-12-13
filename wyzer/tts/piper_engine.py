"""
Piper TTS engine for local, fast speech synthesis.
Uses Piper executable via subprocess for text-to-speech conversion.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from wyzer.core.logger import get_logger


class PiperTTSEngine:
    """Piper TTS engine for local speech synthesis"""
    
    def __init__(
        self,
        exe_path: str = "./assets/piper/piper.exe",
        model_path: str = "./assets/piper/en_US-voice.onnx",
        speaker_id: Optional[int] = None
    ):
        """
        Initialize Piper TTS engine
        
        Args:
            exe_path: Path to Piper executable
            model_path: Path to Piper ONNX model file
            speaker_id: Optional speaker ID for multi-speaker models
        """
        self.logger = get_logger()
        self.exe_path = exe_path
        self.model_path = model_path
        self.speaker_id = speaker_id
        
        # Validate paths
        self._validate_setup()
    
    def _validate_setup(self) -> None:
        """Validate that Piper executable and model exist"""
        # Check executable
        if self.exe_path and not os.path.exists(self.exe_path):
            # Try PATH if explicit path doesn't exist
            if self.exe_path != "piper":
                self.logger.warning(f"Piper executable not found at: {self.exe_path}")
                self.logger.warning("Will attempt to use 'piper' from PATH")
                self.exe_path = "piper"
        
        # Check model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Piper model not found at: {self.model_path}\n"
                f"Please download a Piper voice model and place it at the specified path.\n"
                f"Download models from: https://github.com/rhasspy/piper/releases"
            )
        
        self.logger.info(f"Piper TTS initialized with model: {self.model_path}")
    
    def synthesize_to_wav(self, text: str) -> str:
        """
        Synthesize text to WAV file
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to generated WAV file, or empty string on error
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for synthesis")
            return ""
        
        try:
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                mode='wb'
            )
            temp_wav.close()
            output_path = temp_wav.name
            
            # Build Piper command
            cmd = [self.exe_path, "-m", self.model_path, "-f", output_path]
            
            if self.speaker_id is not None:
                cmd.extend(["--speaker", str(self.speaker_id)])
            
            self.logger.debug(f"Running Piper: {' '.join(cmd)}")
            
            # Run Piper with text via stdin
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=10
            )
            
            # Check for errors
            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='ignore')
                self.logger.error(f"Piper synthesis failed: {stderr}")
                # Clean up temp file
                try:
                    os.unlink(output_path)
                except:
                    pass
                return ""
            
            # Verify output file was created
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                self.logger.error("Piper produced no output")
                try:
                    os.unlink(output_path)
                except:
                    pass
                return ""
            
            self.logger.debug(f"Synthesized to: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            self.logger.error("Piper synthesis timed out")
            try:
                os.unlink(output_path)
            except:
                pass
            return ""
        
        except FileNotFoundError:
            self.logger.error(
                f"Piper executable not found: {self.exe_path}\n"
                f"Please ensure Piper is installed and the path is correct.\n"
                f"Download from: https://github.com/rhasspy/piper/releases"
            )
            return ""
        
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            try:
                if 'output_path' in locals():
                    os.unlink(output_path)
            except:
                pass
            return ""
