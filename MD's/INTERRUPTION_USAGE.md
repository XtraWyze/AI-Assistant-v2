# Interruption System Usage Examples

## Basic Usage

### Single-Process Mode
```python
from wyzer.core.assistant import WyzerAssistant

# Create assistant
assistant = WyzerAssistant(enable_hotword=True)

# Start assistant in a thread or subprocess
import threading
thread = threading.Thread(target=assistant.start)
thread.start()

# Later, interrupt the current process
assistant.interrupt_current_process()

# This will:
# 1. Stop any speaking (TTS)
# 2. Stop any listening
# 3. Cancel any thinking/processing
# 4. Return to IDLE state
```

### Multi-Process Mode
```python
from wyzer.core.assistant import WyzerAssistantMultiprocess

# Create assistant
assistant = WyzerAssistantMultiprocess(enable_hotword=True)

# Start assistant
assistant.start()  # This blocks until stopped

# From another thread, interrupt the current process
assistant.interrupt_current_process()

# This will:
# 1. Send interrupt to brain worker
# 2. Stop TTS in the brain process
# 3. Clear audio queue
# 4. Return both processes to IDLE state
```

## State-Based Interruption Handling

The system automatically handles interruption based on the current state:

```python
from wyzer.core.state import AssistantState

# Check current state
current_state = assistant.state.current_state

if current_state == AssistantState.SPEAKING:
    # Interruption will stop TTS immediately
    pass

elif current_state == AssistantState.LISTENING:
    # Interruption will stop recording and drain frames
    pass

elif current_state == AssistantState.THINKING:
    # Interruption will send INTERRUPT to brain worker
    pass

elif current_state == AssistantState.TRANSCRIBING:
    # Interruption will wait for transcription to finish
    pass

elif current_state == AssistantState.IDLE:
    # Already idle, nothing to interrupt
    pass

elif current_state == AssistantState.FOLLOWUP:
    # Interruption will stop followup listening and return to idle
    pass

# All states transition safely to IDLE after interruption
assistant.interrupt_current_process()
assert assistant.state.is_in_state(AssistantState.IDLE)
```

## Advanced Usage with Interrupt Flags

```python
# You can check if an interrupt was requested
if assistant.state.is_interrupt_requested():
    print("Interrupt was requested during this state")

# You can manually control the interrupt flag if needed
assistant.state.request_interrupt()   # Mark as interrupted
assistant.state.clear_interrupt()     # Clear the flag
```

## Safe Shutdown

```python
# The interruption system integrates with the shutdown sequence
assistant.interrupt_current_process()  # Stops current work
assistant.stop()                       # Properly shuts down the assistant

# Or use keyboard interrupt
try:
    assistant.start()
except KeyboardInterrupt:
    # This triggers the finally block which calls stop()
    print("Assistant stopped gracefully")
```

## Integration with External Systems

```python
# Example: External hotword detector triggers interruption
import pyaudio
from wyzer.core.assistant import WyzerAssistantMultiprocess

assistant = WyzerAssistantMultiprocess()

# Start in separate thread
import threading
main_thread = threading.Thread(target=assistant.start, daemon=True)
main_thread.start()

# Simulate external interruption trigger
import time
time.sleep(5)  # Let it run for 5 seconds
assistant.interrupt_current_process()  # External interrupt
print("Process interrupted")

# Wait for clean shutdown
main_thread.join(timeout=2.0)
```

## Error Handling

```python
try:
    assistant.interrupt_current_process()
except Exception as e:
    # Interruption is designed not to throw, but log any issues
    print(f"Interruption warning: {e}")

# The system continues to work even if interruption has issues
# because all state transitions are guarded
```

## Testing Interruption

Run the included test suite:
```bash
python scripts/test_interruption.py
```

This verifies:
- State interruption flag works correctly
- Both single-process and multi-process modes support interruption
- State transitions work with interruption flag
- No breaking changes to existing functionality
