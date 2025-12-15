"""wyzer.core.tool_worker_pool

Persistent tool worker pool for Brain process.
Workers execute tools in background, keeping Brain lightweight and responsive.

Architecture:
- Main thread: submits tool jobs to task_q
- Worker processes: pull jobs, execute tools, return results to result_q
- All communication is JSON-serializable
- Each worker is a separate process for true parallelism (no GIL)
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import json
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from wyzer.core.logger import get_logger
from wyzer.tools.registry import build_default_registry
from wyzer.tools.validation import validate_args


# Set multiprocessing start method for Windows compatibility
# Use 'spawn' to avoid fork issues and ensure clean process state
if mp.get_start_method(allow_none=True) is None:
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


@dataclass
class ToolJob:
    """A tool execution job"""
    job_id: str
    request_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    timestamp: float


@dataclass
class ToolResult:
    """Result from tool execution"""
    job_id: str
    request_id: str
    tool_name: str
    result: Dict[str, Any]  # JSON-serializable result or error dict
    timestamp: float
    execution_time_ms: float


class ToolWorker(mp.Process):
    """Worker process that executes tools from the task queue"""
    
    def __init__(self, worker_id: int, task_q: mp.Queue, result_q: mp.Queue):
        super().__init__(name=f"ToolWorker-{worker_id}", daemon=True)
        self.worker_id = worker_id
        self.task_q = task_q
        self.result_q = result_q
        self.jobs_processed = 0
        self.errors = 0
    
    def run(self) -> None:
        """Main worker loop"""
        # Each process gets its own logger and registry
        logger = get_logger()
        registry = build_default_registry()
        
        logger.info(f"[POOL] Worker {self.worker_id} started")
        
        while True:
            try:
                # Wait for a job with timeout to allow graceful shutdown
                job: ToolJob = self.task_q.get(timeout=1.0)
                
                if job is None:  # Poison pill for shutdown
                    break
                
                start_time = time.perf_counter()
                
                try:
                    # Execute the tool
                    result = self._execute_tool(registry, job.tool_name, job.tool_args)
                    
                    execution_time_ms = (time.perf_counter() - start_time) * 1000
                    self.jobs_processed += 1
                    
                    # Return result
                    tool_result = ToolResult(
                        job_id=job.job_id,
                        request_id=job.request_id,
                        tool_name=job.tool_name,
                        result=result,
                        timestamp=time.time(),
                        execution_time_ms=execution_time_ms
                    )
                    
                    self.result_q.put(tool_result, timeout=2.0)
                
                except Exception as e:
                    self.errors += 1
                    execution_time_ms = (time.perf_counter() - start_time) * 1000
                    
                    error_result = ToolResult(
                        job_id=job.job_id,
                        request_id=job.request_id,
                        tool_name=job.tool_name,
                        result={
                            "error": {
                                "type": "worker_execution_error",
                                "message": str(e)
                            }
                        },
                        timestamp=time.time(),
                        execution_time_ms=execution_time_ms
                    )
                    
                    logger.error(f"[POOL] Worker {self.worker_id} error executing {job.tool_name}: {e}")
                    self.result_q.put(error_result, timeout=2.0)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[POOL] Worker {self.worker_id} unexpected error: {e}")
                self.errors += 1
        
        logger.info(f"[POOL] Worker {self.worker_id} stopped (processed={self.jobs_processed}, errors={self.errors})")
    
    def _execute_tool(self, registry: Any, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return JSON-serializable result"""
        
        # Get tool from registry
        tool = registry.get(tool_name)
        if tool is None:
            return {
                "error": {
                    "type": "tool_not_found",
                    "message": f"Tool '{tool_name}' does not exist"
                }
            }
        
        # Validate arguments
        is_valid, error = validate_args(tool.args_schema, tool_args)
        if not is_valid:
            return {"error": error}
        
        # Execute the tool
        result = tool.run(**tool_args)
        
        # Ensure result is JSON-serializable
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            return {
                "error": {
                    "type": "serialization_error",
                    "message": f"Tool result not JSON-serializable: {str(e)}"
                }
            }
        
        return result


class ToolWorkerPool:
    """Pool of worker processes for tool execution"""
    
    def __init__(self, num_workers: int = 3):
        """
        Initialize tool worker pool
        
        Args:
            num_workers: Number of worker processes (default 3, capped 1-5)
        """
        self.num_workers = max(1, min(5, num_workers))
        # Use multiprocessing queues for inter-process communication
        self.task_q: mp.Queue = mp.Queue(maxsize=50)
        self.result_q: mp.Queue = mp.Queue(maxsize=100)
        self.workers: List[ToolWorker] = []
        self.logger = get_logger()
        self._running = False
        self._pending_jobs: Dict[str, ToolJob] = {}
    
    def start(self) -> bool:
        """Start the worker pool"""
        if self._running:
            self.logger.warning("[POOL] Pool already running")
            return True
        
        try:
            for i in range(self.num_workers):
                worker = ToolWorker(i, self.task_q, self.result_q)
                worker.daemon = False  # Non-daemonic so Brain (daemonic) can spawn them
                worker.start()
                self.workers.append(worker)
            
            self._running = True
            self.logger.info(f"[POOL] Started pool with {self.num_workers} workers")
            return True
        
        except Exception as e:
            self.logger.error(f"[POOL] Failed to start pool: {e}")
            self._running = False
            return False
    
    def submit_job(self, job_id: str, request_id: str, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        """
        Submit a tool execution job
        
        Returns:
            True if job was queued, False if pool is unhealthy
        """
        if not self._running:
            self.logger.warning("[POOL] Pool not running, cannot submit job")
            return False
        
        job = ToolJob(
            job_id=job_id,
            request_id=request_id,
            tool_name=tool_name,
            tool_args=tool_args,
            timestamp=time.time()
        )
        
        try:
            self.task_q.put(job, timeout=1.0)
            self._pending_jobs[job_id] = job
            return True
        except queue.Full:
            self.logger.warning(f"[POOL] Task queue full, cannot submit job {job_id}")
            return False
    
    def poll_results(self) -> Optional[ToolResult]:
        """Poll for a completed job result (non-blocking)"""
        try:
            result: ToolResult = self.result_q.get_nowait()
            self._pending_jobs.pop(result.job_id, None)
            return result
        except queue.Empty:
            return None
    
    def wait_for_result(self, job_id: str, timeout: float = 15.0) -> Optional[ToolResult]:
        """Wait for a specific job result with timeout"""
        start_time = time.perf_counter()
        
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout:
                self.logger.warning(f"[POOL] Timeout waiting for job {job_id}")
                return None
            
            try:
                result: ToolResult = self.result_q.get(timeout=0.5)
                self._pending_jobs.pop(result.job_id, None)
                
                if result.job_id == job_id:
                    return result
                # Different job, put it back
                self.result_q.put(result)
            
            except queue.Empty:
                continue
    
    def get_status(self) -> Dict[str, Any]:
        """Get pool status"""
        return {
            "running": self._running,
            "num_workers": self.num_workers,
            "pending_jobs": len(self._pending_jobs),
            "task_q_size": self.task_q.qsize(),
            "result_q_size": self.result_q.qsize(),
        }
    
    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shut down the pool"""
        if not self._running:
            return
        
        self.logger.info("[POOL] Shutting down pool...")
        self._running = False
        
        # Send poison pills to workers
        for _ in self.workers:
            try:
                self.task_q.put(None, timeout=0.5)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
            if worker.is_alive():
                self.logger.warning(f"[POOL] Worker {worker.worker_id} did not stop gracefully")
                worker.terminate()
        
        self.logger.info("[POOL] Pool shut down complete")
