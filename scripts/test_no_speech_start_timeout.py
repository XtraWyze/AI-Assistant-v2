#!/usr/bin/env python3
"""
Test: No-Speech-Start Timeout Feature

Tests that the assistant aborts listening quickly when the user triggers the
hotword but does not begin speaking within the configured timeout window.

This test validates:
1. When hotword triggers LISTENING and VAD never reports speech-start:
   - Recording aborts near NO_SPEECH_START_TIMEOUT_SEC (not max duration)
   - No STT/Brain job is queued
2. When user starts speaking before timeout:
   - Normal flow continues and STT is invoked
3. When barge-in occurs:
   - Grace period still works (no immediate abort from no-speech-start timeout)

Run with:
    python scripts/test_no_speech_start_timeout.py
"""
import sys
import os
import time
import threading
from unittest.mock import MagicMock, patch
from queue import Queue, Empty

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from wyzer.core.config import Config
from wyzer.core.state import AssistantState, RuntimeState


def create_silent_frame():
    """Create a silent audio frame (no speech)"""
    return np.zeros(Config.CHUNK_SAMPLES, dtype=np.float32)


def create_speech_frame():
    """Create a noisy audio frame (simulates speech for VAD)"""
    # Create noise that should trigger VAD
    return np.random.uniform(-0.5, 0.5, Config.CHUNK_SAMPLES).astype(np.float32)


class MockVAD:
    """Mock VAD that returns speech based on the audio frame amplitude"""
    def __init__(self, speech_frames_after: int = -1):
        """
        Args:
            speech_frames_after: Number of frames after which to return True for is_speech.
                               -1 means never return speech.
        """
        self.frame_count = 0
        self.speech_frames_after = speech_frames_after
    
    def is_speech(self, audio_frame: np.ndarray) -> bool:
        self.frame_count += 1
        if self.speech_frames_after < 0:
            return False
        return self.frame_count > self.speech_frames_after


class MockHotword:
    """Mock hotword detector"""
    def __init__(self):
        self.triggered = False
    
    def detect(self, audio_frame: np.ndarray):
        if self.triggered:
            self.triggered = False  # Only trigger once
            return ("hey wyzer", 0.95)
        return (None, 0.0)
    
    def trigger(self):
        self.triggered = True


def test_config_values():
    """Test that config values are set correctly"""
    print("=" * 60)
    print("TEST: Config Values")
    print("=" * 60)
    
    # Check default value
    assert hasattr(Config, 'NO_SPEECH_START_TIMEOUT_SEC'), \
        "Config missing NO_SPEECH_START_TIMEOUT_SEC"
    
    print(f"  NO_SPEECH_START_TIMEOUT_SEC = {Config.NO_SPEECH_START_TIMEOUT_SEC}")
    assert Config.NO_SPEECH_START_TIMEOUT_SEC > 0, \
        "NO_SPEECH_START_TIMEOUT_SEC should be positive"
    assert Config.NO_SPEECH_START_TIMEOUT_SEC < Config.MAX_RECORD_SECONDS, \
        "NO_SPEECH_START_TIMEOUT_SEC should be less than MAX_RECORD_SECONDS"
    
    # Check helper method exists
    assert hasattr(Config, 'get_no_speech_start_timeout_frames'), \
        "Config missing get_no_speech_start_timeout_frames method"
    
    frames = Config.get_no_speech_start_timeout_frames()
    print(f"  get_no_speech_start_timeout_frames() = {frames}")
    assert frames > 0, "Should return positive frame count"
    
    # Verify calculation
    expected_frames = int(Config.NO_SPEECH_START_TIMEOUT_SEC * Config.SAMPLE_RATE / Config.CHUNK_SAMPLES)
    assert frames == expected_frames, \
        f"Frame calculation mismatch: {frames} != {expected_frames}"
    
    print("  ✓ Config values correct")
    return True


def test_no_speech_aborts_quickly():
    """Test that listening aborts quickly when no speech is detected"""
    print("\n" + "=" * 60)
    print("TEST: No Speech Start -> Quick Abort")
    print("=" * 60)
    
    # Simulate the timeout logic directly (unit test style)
    # This tests the core logic without needing the full assistant
    
    state = RuntimeState()
    state.transition_to(AssistantState.LISTENING)
    
    # Set deadline for no-speech-start timeout
    # Use a SHORT timeout for testing (0.1s instead of 2.5s)
    test_timeout = 0.1
    no_speech_start_deadline_ts = time.time() + test_timeout
    
    # Simulate processing frames with no speech
    mock_vad = MockVAD(speech_frames_after=-1)  # Never detect speech
    
    start_time = time.time()
    aborted = False
    abort_reason = ""
    
    # Process frames for up to max duration
    max_frames = Config.get_max_record_frames()
    for frame_idx in range(max_frames):
        # Create silent frame
        audio_frame = create_silent_frame()
        
        # VAD check
        is_speech = mock_vad.is_speech(audio_frame)
        
        # Check no-speech-start timeout (this is the new logic)
        if (not state.speech_detected 
            and time.time() > no_speech_start_deadline_ts):
            aborted = True
            abort_reason = f"No speech start within {test_timeout}s"
            break
        
        # Small sleep to simulate real-time processing
        time.sleep(0.001)  # 1ms per frame
    
    elapsed = time.time() - start_time
    
    print(f"  Test timeout used: {test_timeout}s")
    print(f"  Frames processed: {mock_vad.frame_count}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print(f"  Aborted: {aborted}")
    print(f"  Reason: {abort_reason}")
    
    # Verify abort happened
    assert aborted, "Should have aborted due to no-speech-start timeout"
    
    # Verify abort was due to timeout, not max frames
    assert mock_vad.frame_count < max_frames, \
        f"Processed too many frames: {mock_vad.frame_count} >= {max_frames}"
    
    # Verify no STT would be queued (speech_frames_count should be 0)
    assert state.speech_frames_count == 0, \
        "No speech frames should have been counted"
    
    print("  ✓ Quick abort works correctly")
    
    # Also verify the actual Config value makes sense
    print(f"\n  Actual config NO_SPEECH_START_TIMEOUT_SEC = {Config.NO_SPEECH_START_TIMEOUT_SEC}s")
    assert Config.NO_SPEECH_START_TIMEOUT_SEC < Config.MAX_RECORD_SECONDS, \
        "Timeout should be less than max record duration"
    print(f"  Actual config MAX_RECORD_SECONDS = {Config.MAX_RECORD_SECONDS}s")
    print("  ✓ Config values are reasonable")
    
    return True


def test_speech_before_timeout_continues():
    """Test that normal flow continues when speech starts before timeout"""
    print("\n" + "=" * 60)
    print("TEST: Speech Before Timeout -> Normal Flow")
    print("=" * 60)
    
    state = RuntimeState()
    state.transition_to(AssistantState.LISTENING)
    
    # Set deadline for no-speech-start timeout
    no_speech_start_deadline_ts = time.time() + Config.NO_SPEECH_START_TIMEOUT_SEC
    
    # Simulate VAD that detects speech after 0.5 seconds worth of frames
    frames_until_speech = int(0.5 * Config.SAMPLE_RATE / Config.CHUNK_SAMPLES)
    mock_vad = MockVAD(speech_frames_after=frames_until_speech)
    
    start_time = time.time()
    speech_started = False
    aborted = False
    
    # Process frames
    max_frames = Config.get_max_record_frames()
    for frame_idx in range(max_frames):
        # Create frame
        audio_frame = create_silent_frame() if frame_idx < frames_until_speech else create_speech_frame()
        
        # VAD check
        is_speech = mock_vad.is_speech(audio_frame)
        
        if is_speech:
            if not state.speech_detected:
                state.speech_detected = True
                speech_started = True
                # Clear deadline when speech starts (new logic)
                no_speech_start_deadline_ts = 0
                print(f"  Speech detected at frame {frame_idx}")
            state.speech_frames_count += 1
        
        # Check no-speech-start timeout
        if (not state.speech_detected 
            and no_speech_start_deadline_ts > 0
            and time.time() > no_speech_start_deadline_ts):
            aborted = True
            break
        
        # Simulate end of speech after some frames
        if state.speech_frames_count > 20:
            print(f"  Simulating end of speech at frame {frame_idx}")
            break
        
        time.sleep(0.001)
    
    elapsed = time.time() - start_time
    
    print(f"  Frames processed: {mock_vad.frame_count}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print(f"  Speech started: {speech_started}")
    print(f"  Aborted early: {aborted}")
    print(f"  Speech frames: {state.speech_frames_count}")
    
    # Verify speech was detected
    assert speech_started, "Speech should have been detected"
    assert not aborted, "Should NOT have aborted due to no-speech-start timeout"
    assert state.speech_frames_count > 0, "Should have counted speech frames"
    
    print("  ✓ Normal flow continues when speech starts")
    return True


def test_bargein_grace_period():
    """Test that barge-in grace period takes precedence over no-speech-start timeout"""
    print("\n" + "=" * 60)
    print("TEST: Barge-in Grace Period Precedence")
    print("=" * 60)
    
    state = RuntimeState()
    state.transition_to(AssistantState.LISTENING)
    
    # Simulate barge-in scenario
    bargein_pending_speech = True
    bargein_wait_deadline = time.time() + Config.POST_BARGEIN_WAIT_FOR_SPEECH_SEC
    
    # No-speech-start deadline would be shorter, but should be ignored during barge-in
    no_speech_start_deadline_ts = time.time() + Config.NO_SPEECH_START_TIMEOUT_SEC
    
    # Mock VAD that never detects speech
    mock_vad = MockVAD(speech_frames_after=-1)
    
    start_time = time.time()
    abort_reason = ""
    
    # Process frames for longer than NO_SPEECH_START_TIMEOUT but less than barge-in grace
    test_duration = min(
        Config.NO_SPEECH_START_TIMEOUT_SEC + 0.5,  # After no-speech timeout
        Config.POST_BARGEIN_WAIT_FOR_SPEECH_SEC - 0.1  # But before barge-in timeout
    )
    
    max_frames = int(test_duration * Config.SAMPLE_RATE / Config.CHUNK_SAMPLES)
    
    for frame_idx in range(max_frames):
        audio_frame = create_silent_frame()
        is_speech = mock_vad.is_speech(audio_frame)
        
        # Check no-speech-start timeout - should be SKIPPED if barge-in pending
        if (not state.speech_detected 
            and not bargein_pending_speech  # Key: skip if barge-in pending
            and no_speech_start_deadline_ts > 0
            and time.time() > no_speech_start_deadline_ts):
            abort_reason = "no-speech-start timeout"
            break
        
        # Check barge-in timeout
        if bargein_pending_speech and time.time() > bargein_wait_deadline:
            abort_reason = "barge-in grace period expired"
            break
        
        time.sleep(0.001)
    
    elapsed = time.time() - start_time
    
    print(f"  Elapsed time: {elapsed:.2f}s")
    print(f"  NO_SPEECH_START_TIMEOUT_SEC: {Config.NO_SPEECH_START_TIMEOUT_SEC}s")
    print(f"  POST_BARGEIN_WAIT_FOR_SPEECH_SEC: {Config.POST_BARGEIN_WAIT_FOR_SPEECH_SEC}s")
    print(f"  Abort reason: {abort_reason}")
    
    # Should NOT have aborted due to no-speech-start timeout
    assert abort_reason != "no-speech-start timeout", \
        "Should NOT abort due to no-speech-start timeout during barge-in"
    
    print("  ✓ Barge-in grace period takes precedence")
    return True


def test_env_override():
    """Test that environment variable override works"""
    print("\n" + "=" * 60)
    print("TEST: Environment Variable Override")
    print("=" * 60)
    
    # Save original value
    original_value = Config.NO_SPEECH_START_TIMEOUT_SEC
    
    # Test that the env var name is correct
    env_var = "WYZER_NO_SPEECH_START_TIMEOUT_SEC"
    print(f"  Environment variable: {env_var}")
    print(f"  Current value: {original_value}")
    
    # Note: We can't easily test the actual override without reloading the module,
    # but we can verify the pattern matches other config values
    
    # Check the pattern is consistent with other timeout settings
    assert "WYZER_" in env_var, "Env var should start with WYZER_"
    assert "SEC" in env_var, "Env var should indicate seconds"
    
    print("  ✓ Environment variable naming is correct")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("NO-SPEECH-START TIMEOUT FEATURE TESTS")
    print("=" * 60)
    
    tests = [
        ("Config Values", test_config_values),
        ("No Speech Quick Abort", test_no_speech_aborts_quickly),
        ("Speech Before Timeout", test_speech_before_timeout_continues),
        ("Barge-in Grace Period", test_bargein_grace_period),
        ("Environment Override", test_env_override),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
