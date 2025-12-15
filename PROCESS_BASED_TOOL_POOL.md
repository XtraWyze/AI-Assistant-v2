# Process-Based Tool Worker Pool

## Architecture Correction: From Threads to Processes ✅

The tool worker pool has been refactored from **thread-based** to **process-based** workers.

### Why This Matters

**Threads (Old):**
- ❌ Shared memory space with Brain process
- ❌ GIL (Global Interpreter Lock) limits parallel execution
- ❌ CPU-bound tools block each other

**Processes (New):**
- ✅ Separate memory space per worker
- ✅ True multicore parallelism without GIL
- ✅ CPU-bound tools execute in true parallel
- ✅ Better scalability for heavy operations

## Updated Architecture

```
Main Process (pid=PID1)
├─ Core Thread (realtime audio/hotword/state)
│  └─ Heartbeat: Every 10s with queue/state info
└─ Brain Process (pid=PID2, ppid=PID1, non-daemonic)
   ├─ STT Engine (Whisper)
   ├─ LLM Inference (Ollama)
   ├─ TTS Synthesis (Piper)
   ├─ Tool Worker Pool (3 processes)
   │  ├─ Worker Process 0 (non-daemonic)
   │  ├─ Worker Process 1 (non-daemonic)
   │  └─ Worker Process 2 (non-daemonic)
   └─ Heartbeat: Every 10s with queue/job info
```

## Implementation Changes

### 1. Tool Worker Class (`wyzer/core/tool_worker_pool.py`)

Changed from `threading.Thread` to `multiprocessing.Process`:

```python
class ToolWorker(mp.Process):
    """Worker process that executes tools from the task queue"""
    
    def __init__(self, worker_id: int, task_q: mp.Queue, result_q: mp.Queue):
        super().__init__(name=f"ToolWorker-{worker_id}", daemon=False)
        self.worker_id = worker_id
        self.task_q = task_q
        self.result_q = result_q
    
    def run(self) -> None:
        # Each process gets its own logger and registry
        logger = get_logger()
        registry = build_default_registry()
        
        logger.info(f"[POOL] Worker {self.worker_id} started")
        
        while True:
            try:
                job = self.task_q.get(timeout=1.0)
                if job is None:
                    break
                
                # Execute tool and return result
                result = self._execute_tool(registry, job.tool_name, job.tool_args)
                self.result_q.put(tool_result, timeout=2.0)
            except queue.Empty:
                continue
```

**Key differences:**
- Inherits from `mp.Process` instead of `threading.Thread`
- Creates own logger and registry in each process (`run()` method)
- Passes registry to `_execute_tool()` instead of using instance variable
- Uses poison pill (None) for clean shutdown instead of `_stop_event`
- Set `daemon=False` to allow spawning from daemonic Brain process

### 2. Pool Class Changes

```python
class ToolWorkerPool:
    def __init__(self, num_workers: int = 3):
        self.task_q: mp.Queue = mp.Queue(maxsize=50)  # mp.Queue instead of queue.Queue
        self.result_q: mp.Queue = mp.Queue(maxsize=100)
        self.workers: List[ToolWorker] = []
    
    def start(self) -> bool:
        for i in range(self.num_workers):
            worker = ToolWorker(i, self.task_q, self.result_q)
            worker.daemon = False  # Non-daemonic to allow spawning from Brain
            worker.start()
            self.workers.append(worker)
```

**Changes:**
- Uses `mp.Queue` for inter-process communication (automatically serializes data)
- Removed thread lock (`threading.Lock`) - not needed for process-based design
- Set worker processes to `daemon=False`

### 3. Brain Process Configuration (`wyzer/core/process_manager.py`)

Changed Brain process from daemonic to non-daemonic:

```python
proc = ctx.Process(
    target=run_brain_worker,
    args=(core_to_brain_q, brain_to_core_q, config),
    name="WyzerBrainWorker",
    daemon=False,  # Non-daemonic so it can spawn worker processes
)
```

**Why:**
- Python doesn't allow daemonic processes to spawn child processes
- Brain needs to spawn 3 worker processes
- Non-daemonic Brain means Main waits for Brain to complete on shutdown (proper cleanup)

### 4. Multiprocessing Context Setup

Added spawn context configuration for Windows compatibility:

```python
if mp.get_start_method(allow_none=True) is None:
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
```

**Why:**
- Windows only supports 'spawn' method (cannot fork)
- Ensures clean process state across platforms
- Prevents issues with inheriting thread state

## Verification

### Log Output Shows Process-Based Architecture

```
[ROLE] Main orchestrator startup: pid=4844
[ROLE] Core (main thread) pid=4844
[ROLE] Brain (worker process) spawned: pid=25888
[ROLE] Brain worker startup: pid=25888 ppid=4844
[POOL] Started pool with 3 workers
[POOL] Worker 0 started (process)
[POOL] Worker 1 started (process)
[POOL] Worker 2 started (process)
[HEARTBEAT] role=Core pid=4844 state=IDLE
[HEARTBEAT] role=Brain pid=25888 q_in=0 q_out=1 last_job=59f29dfb... interrupt_gen=0
```

### Performance Characteristics

**Tool Execution in Pool:**
- Worker 0 processes Tool A (CPU-intensive)
- Worker 1 processes Tool B (I/O-intensive)
- Worker 2 processes Tool C (Memory-intensive)
- **All three execute in parallel** without GIL contention ✅

**Example Result:**
```
[TOOLS] Pool result {'location': {...}, 'weather': 'Overcast', ...}
[POOL] Worker 0 stopped (processed=1, errors=0)
[POOL] Worker 1 stopped (processed=1, errors=0)  
[POOL] Worker 2 stopped (processed=1, errors=0)
```

## Configuration

Tools pool is enabled by default but configurable:

```python
# config.py
TOOL_POOL_ENABLED = True              # Enable process-based pool
TOOL_POOL_WORKERS = 3                 # Number of worker processes
TOOL_POOL_TIMEOUT_SEC = 15            # Max wait for result
```

Environment overrides:
```bash
export WYZER_TOOL_POOL_ENABLED=true
export WYZER_TOOL_POOL_WORKERS=4
export WYZER_TOOL_POOL_TIMEOUT_SEC=20
```

## Advantages of Process-Based Design

| Aspect | Threads | Processes ✅ |
|--------|---------|------------|
| **GIL Impact** | Blocks parallel execution | No GIL, true parallelism |
| **CPU Cores** | Single core due to GIL | Multiple cores utilized |
| **Memory** | Shared (lower overhead) | Isolated (better safety) |
| **Execution Model** | Concurrent (preemptive) | Parallel (true) |
| **Tool Isolation** | Shared state (errors propagate) | Isolated (errors contained) |
| **Scalability** | Limited by GIL | Scales with core count |

## Shutdown Sequence

Clean shutdown with proper process termination:

```
1. Main receives SHUTDOWN signal
2. Main sends SHUTDOWN message to Brain via core_to_brain_q
3. Brain receives SHUTDOWN in message handler
4. Brain calls orchestrator.shutdown_tool_pool()
5. Pool sends poison pills (None) to all workers
6. Workers finish current job and exit cleanly
7. Brain waits for all workers to join (timeout=5s)
8. Brain terminates remaining workers if needed
9. Brain exits normally
10. Main joins Brain process
```

## Files Modified

- `wyzer/core/tool_worker_pool.py` - Refactored to use processes
- `wyzer/core/process_manager.py` - Brain set to non-daemonic
- `wyzer/core/orchestrator.py` - No changes needed (pool interface unchanged)
- `wyzer/core/brain_worker.py` - No changes needed (pool init/shutdown works)

## Testing

**Direct pool test:**
```bash
python -c "
from wyzer.core.tool_worker_pool import ToolWorkerPool
pool = ToolWorkerPool(num_workers=3)
pool.start()
pool.submit_job('test-1', 'req-1', 'get_time', {})
result = pool.wait_for_result('test-1')
pool.shutdown()
"
```

**Architecture verification:**
```bash
python scripts/verify_multiprocess.py
```

**Full application:**
```bash
python run.py
```

## Status: ✅ COMPLETE

- ✅ Tool workers refactored from threads to processes
- ✅ True multicore parallelism enabled
- ✅ GIL eliminated from tool execution
- ✅ Process isolation for better error handling
- ✅ Verification logs confirm 3 worker processes spawned
- ✅ End-to-end testing passed with live interactions
- ✅ Backward compatible (pool optional via config)
