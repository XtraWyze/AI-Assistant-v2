"""
State management for Wyzer AI Assistant.
Defines state machine and runtime state tracking.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import time


class AssistantState(Enum):
    """State machine states"""
    IDLE = "IDLE"
    HOTWORD_DETECTED = "HOTWORD_DETECTED"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


@dataclass
class RuntimeState:
    """Runtime state tracking"""
    current_state: AssistantState = AssistantState.IDLE
    last_hotword_time: float = 0.0
    last_state_change: float = field(default_factory=time.time)
    recording_started: float = 0.0
    speech_detected: bool = False
    silence_frames: int = 0
    speech_frames_count: int = 0
    total_frames_recorded: int = 0
    
    def transition_to(self, new_state: AssistantState) -> None:
        """Transition to a new state"""
        self.current_state = new_state
        self.last_state_change = time.time()
        
        # Reset state-specific counters on transition
        if new_state == AssistantState.LISTENING:
            self.recording_started = time.time()
            self.speech_detected = False
            self.silence_frames = 0
            self.speech_frames_count = 0
            self.total_frames_recorded = 0
        elif new_state == AssistantState.IDLE:
            self.recording_started = 0.0
            self.speech_detected = False
            self.silence_frames = 0
            self.speech_frames_count = 0
            self.total_frames_recorded = 0
    
    def is_in_state(self, state: AssistantState) -> bool:
        """Check if currently in given state"""
        return self.current_state == state
    
    def get_time_in_current_state(self) -> float:
        """Get time spent in current state (seconds)"""
        return time.time() - self.last_state_change
    
    def get_recording_duration(self) -> float:
        """Get current recording duration (seconds)"""
        if self.recording_started > 0:
            return time.time() - self.recording_started
        return 0.0
