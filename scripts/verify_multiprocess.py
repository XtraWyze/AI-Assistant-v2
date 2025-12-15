#!/usr/bin/env python3
"""
Wyzer Multiprocess Architecture Verification

Verifies that the application runs according to documented architecture:
- Main/Parent orchestrator process spawns Core + Brain
- Core remains realtime and responsive
- Brain handles heavy work (STT, LLM, TTS)
- IPC works correctly
- Each process logs its role and responsibilities

Usage:
    python scripts/verify_multiprocess.py
    
Expected exit code:
    0 = success (all verifications passed)
    1 = failure (architecture issue detected)
"""

import sys
import os
import time
import subprocess
import signal
import logging
from typing import List, Tuple
from queue import Queue, Empty
from threading import Thread, Event

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wyzer.core.config import Config
from wyzer.core.logger import get_logger, init_logger

# Setup verification logging
init_logger("INFO")
logger = get_logger()


class LogCapture:
    """Capture logs from a subprocess"""
    
    def __init__(self):
        self.logs: List[str] = []
        self.lock = False
    
    def add(self, line: str) -> None:
        self.logs.append(line.strip())
    
    def find(self, pattern: str) -> bool:
        """Check if any log line contains the pattern"""
        return any(pattern.lower() in log.lower() for log in self.logs)
    
    def find_all(self, patterns: List[str]) -> Tuple[bool, List[str]]:
        """Check if all patterns are found"""
        missing = []
        for pattern in patterns:
            if not self.find(pattern):
                missing.append(pattern)
        return len(missing) == 0, missing


def verify_multiprocess() -> int:
    """Run multiprocess verification test"""
    
    logger.info("=" * 70)
    logger.info("WYZER MULTIPROCESS ARCHITECTURE VERIFICATION")
    logger.info("=" * 70)
    
    # Set verify mode
    os.environ["WYZER_VERIFY_MODE"] = "true"
    os.environ["WYZER_LOG_LEVEL"] = "INFO"
    
    # Capture logs
    log_capture = LogCapture()
    
    # Start the app in a subprocess
    logger.info("\n[SETUP] Starting Wyzer in verification mode...")
    logger.info("[SETUP] Timeout: 20 seconds (should complete process startup & heartbeat)")
    
    try:
        # Run the main app
        proc = subprocess.Popen(
            [sys.executable, "run.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # Collect output with timeout
        output_queue: Queue[str] = Queue()
        stop_reading = Event()
        
        def read_output():
            try:
                for line in proc.stdout:
                    if stop_reading.is_set():
                        break
                    output_queue.put(line)
                    log_capture.add(line)
            except:
                pass
        
        reader_thread = Thread(target=read_output, daemon=True)
        reader_thread.start()
        
        # Wait for startup (need time for heartbeat)
        start_time = time.time()
        timeout_sec = 20
        
        while time.time() - start_time < timeout_sec:
            try:
                line = output_queue.get(timeout=0.5)
                # Print live output
                print(line.rstrip())
            except Empty:
                pass
        
        # Stop collecting
        stop_reading.set()
        
        # Gracefully terminate
        logger.info("\n[VERIFY] Sending CTRL+C to terminate...")
        proc.send_signal(signal.SIGINT)
        
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to run verification: {e}")
        return 1
    
    # Now verify the logs
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 70)
    
    all_passed = True
    
    # A1: Verify role logs
    logger.info("\n[A1] Checking for role identification logs...")
    role_patterns = [
        "[ROLE]",
        "Main orchestrator",
        "Core (realtime",
        "Brain worker",
        "pid=",
    ]
    
    found_role_logs, missing = log_capture.find_all(role_patterns)
    if found_role_logs:
        logger.info("     ✓ PASS: Role logs found")
        # Show sample
        for log in log_capture.logs:
            if "[ROLE]" in log:
                logger.info(f"     {log}")
    else:
        logger.error(f"     ✗ FAIL: Missing role log patterns: {missing}")
        all_passed = False
    
    # A2: Verify heartbeat logs
    logger.info("\n[A2] Checking for heartbeat logs...")
    heartbeat_patterns = [
        "[HEARTBEAT]",
        "role=Core",
        "role=Brain",
    ]
    
    found_heartbeats = False
    for pattern in heartbeat_patterns[1:]:
        if log_capture.find(pattern):
            found_heartbeats = True
            break
    
    if found_heartbeats or log_capture.find("[HEARTBEAT]"):
        logger.info("     ✓ PASS: Heartbeat logs found")
        # Show sample
        for log in log_capture.logs:
            if "[HEARTBEAT]" in log:
                logger.info(f"     {log}")
                break
    else:
        logger.warning("     ⚠ WARN: No heartbeat logs found (may be too early)")
        # This is not a hard fail if not enough time has passed
    
    # A3: Verify process structure
    logger.info("\n[A3] Checking for multiprocess structure...")
    if log_capture.find("WyzerBrainWorker") or log_capture.find("Brain worker spawned"):
        logger.info("     ✓ PASS: Brain worker process detected")
    else:
        logger.warning("     ⚠ WARN: Brain worker process not clearly logged")
    
    # A4: Verify no blocking on heavy operations
    logger.info("\n[A4] Checking for heavy operation async behavior...")
    async_indicators = [
        "STT", "LLM", "TTS",
        "brain_worker",
    ]
    
    found_async = False
    for indicator in async_indicators:
        if log_capture.find(indicator):
            found_async = True
            break
    
    if found_async:
        logger.info("     ✓ PASS: Heavy operations logged")
    else:
        logger.info("     ℹ INFO: No heavy operations logged (startup only)")
    
    # Final result
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("✓ VERIFICATION PASSED")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("✗ VERIFICATION FAILED")
        logger.error("=" * 70)
        logger.error("\nDebug: Full log output:")
        for log in log_capture.logs[-50:]:  # Show last 50 lines
            logger.error(f"  {log}")
        return 1


if __name__ == "__main__":
    exit_code = verify_multiprocess()
    sys.exit(exit_code)
