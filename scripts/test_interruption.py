#!/usr/bin/env python3
"""
Test script for interruption handling.
Verifies that interrupt_current_process() works without breaking the system.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wyzer.core.state import RuntimeState, AssistantState
from wyzer.core.assistant import WyzerAssistant, WyzerAssistantMultiprocess


def test_state_interruption():
    """Test interruption flag in state management"""
    print("\n=== Testing State Interruption ===")
    
    state = RuntimeState()
    
    # Test initial state
    assert not state.is_interrupt_requested(), "Interrupt should not be requested initially"
    print("[PASS] Initial state has no interrupt")
    
    # Test request interrupt
    state.request_interrupt()
    assert state.is_interrupt_requested(), "Interrupt should be requested after request_interrupt()"
    print("[PASS] request_interrupt() works")
    
    # Test clear interrupt
    state.clear_interrupt()
    assert not state.is_interrupt_requested(), "Interrupt should be cleared after clear_interrupt()"
    print("[PASS] clear_interrupt() works")
    
    print("[PASS] State interruption tests passed!")
    return True


def test_single_process_interrupt_method():
    """Test that WyzerAssistant has interrupt_current_process method"""
    print("\n=== Testing Single-Process Interrupt Method ===")
    
    # Check method exists
    assert hasattr(WyzerAssistant, 'interrupt_current_process'), \
        "WyzerAssistant missing interrupt_current_process method"
    print("[PASS] WyzerAssistant has interrupt_current_process method")
    
    # Check it's callable
    assert callable(getattr(WyzerAssistant, 'interrupt_current_process')), \
        "interrupt_current_process is not callable"
    print("[PASS] interrupt_current_process is callable")
    
    print("[PASS] Single-process interrupt method tests passed!")
    return True


def test_multiprocess_interrupt_method():
    """Test that WyzerAssistantMultiprocess has interrupt_current_process method"""
    print("\n=== Testing Multi-Process Interrupt Method ===")
    
    # Check method exists
    assert hasattr(WyzerAssistantMultiprocess, 'interrupt_current_process'), \
        "WyzerAssistantMultiprocess missing interrupt_current_process method"
    print("[PASS] WyzerAssistantMultiprocess has interrupt_current_process method")
    
    # Check it's callable
    assert callable(getattr(WyzerAssistantMultiprocess, 'interrupt_current_process')), \
        "interrupt_current_process is not callable"
    print("[PASS] interrupt_current_process is callable")
    
    print("[PASS] Multi-process interrupt method tests passed!")
    return True


def test_state_transitions():
    """Test that state transitions work correctly with interruption"""
    print("\n=== Testing State Transitions ===")
    
    state = RuntimeState()
    
    # Test transitions
    state.transition_to(AssistantState.LISTENING)
    assert state.is_in_state(AssistantState.LISTENING), "Should be in LISTENING state"
    print("[PASS] Transitioned to LISTENING")
    
    state.transition_to(AssistantState.THINKING)
    assert state.is_in_state(AssistantState.THINKING), "Should be in THINKING state"
    print("[PASS] Transitioned to THINKING")
    
    # Interrupt flag should persist through transitions
    state.request_interrupt()
    state.transition_to(AssistantState.IDLE)
    assert state.is_interrupt_requested(), "Interrupt flag should persist through transition"
    print("[PASS] Interrupt flag persists through state transitions")
    
    state.clear_interrupt()
    assert not state.is_interrupt_requested(), "Interrupt flag should be cleared"
    print("[PASS] Interrupt flag cleared correctly")
    
    print("[PASS] State transition tests passed!")
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Interruption System")
    print("=" * 50)
    
    tests = [
        test_state_interruption,
        test_single_process_interrupt_method,
        test_multiprocess_interrupt_method,
        test_state_transitions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

