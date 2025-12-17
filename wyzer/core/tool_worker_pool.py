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
import os
import queue
import time
import json
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from wyzer.core.config import Config
from wyzer.core.logger import get_logger, init_logger
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


@dataclass
class WorkerHeartbeat:
    """Heartbeat from a tool worker process"""
    worker_id: int
    pid: int
    jobs_processed: int
    errors: int
    current_job: Optional[str]  # job_id if currently processing
    timestamp: float


class ToolWorker(mp.Process):
    """Worker process that executes tools from the task queue"""
    
    def __init__(self, worker_id: int, task_q: mp.Queue, result_q: mp.Queue, heartbeat_q: mp.Queue):
        super().__init__(name=f"ToolWorker-{worker_id}", daemon=True)
        self.worker_id = worker_id
        self.task_q = task_q
        self.result_q = result_q
        self.heartbeat_q = heartbeat_q
        self.jobs_processed = 0
        self.errors = 0
    
    def run(self) -> None:
        """Main worker loop"""
        # Each process gets its own logger and registry
        # Check environment for quiet mode
        quiet_mode = os.environ.get("WYZER_QUIET_MODE", "false").lower() in ("true", "1", "yes")
        log_level = os.environ.get("WYZER_LOG_LEVEL", "INFO")
        init_logger(log_level, quiet_mode=quiet_mode)
        logger = get_logger()
        registry = build_default_registry()
        pid = os.getpid()
        
        logger.info(f"[POOL] Worker {self.worker_id} started (pid={pid})")
        logger.info(f"[ROLE] ToolWorker-{self.worker_id} pid={pid}")
        
        last_heartbeat = time.time()
        current_job_id: Optional[str] = None
        jobs_processed = 0
        errors = 0
        
        while True:
            try:
                # Emit heartbeat periodically
                current_time = time.time()
                if current_time - last_heartbeat >= Config.HEARTBEAT_INTERVAL_SEC:
                    heartbeat = WorkerHeartbeat(
                        worker_id=self.worker_id,
                        pid=pid,
                        jobs_processed=jobs_processed,
                        errors=errors,
                        current_job=current_job_id,
                        timestamp=current_time
                    )
                    try:
                        self.heartbeat_q.put_nowait(heartbeat)
                        logger.info(
                            f"[HEARTBEAT] role=ToolWorker-{self.worker_id} pid={pid} "
                            f"jobs={jobs_processed} errors={errors} current_job={current_job_id}"
                        )
                    except queue.Full:
                        pass  # Don't block on heartbeat
                    last_heartbeat = current_time
                
                # Wait for a job with timeout to allow graceful shutdown and heartbeats
                try:
                    job: ToolJob = self.task_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if job is None:  # Poison pill for shutdown
                    break
                
                current_job_id = job.job_id
                start_time = time.perf_counter()
                
                try:
                    # Execute the tool
                    result = self._execute_tool(registry, job.tool_name, job.tool_args)
                    
                    execution_time_ms = (time.perf_counter() - start_time) * 1000
                    jobs_processed += 1
                    current_job_id = None
                    
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
                    errors += 1
                    current_job_id = None
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
                
            except Exception as e:
                logger.error(f"[POOL] Worker {self.worker_id} unexpected error: {e}")
                errors += 1
        
        logger.info(f"[POOL] Worker {self.worker_id} stopped (processed={jobs_processed}, errors={errors})")
    
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
        self.heartbeat_q: mp.Queue = mp.Queue(maxsize=50)  # Worker heartbeat queue
        self.workers: List[ToolWorker] = []
        self.logger = get_logger()
        self._running = False
        self._pending_jobs: Dict[str, ToolJob] = {}
        self._worker_heartbeats: Dict[int, WorkerHeartbeat] = {}  # Cache of latest heartbeats
    
    def start(self) -> bool:
        """Start the worker pool"""
        if self._running:
            self.logger.warning("[POOL] Pool already running")
            return True
        
        try:
            for i in range(self.num_workers):
                worker = ToolWorker(i, self.task_q, self.result_q, self.heartbeat_q)
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
        """Get pool status including worker heartbeats"""
        # Collect any pending heartbeats first
        self._collect_heartbeats()
        
        worker_status = []
        for worker_id, hb in self._worker_heartbeats.items():
            age = time.time() - hb.timestamp
            worker_status.append({
                "worker_id": hb.worker_id,
                "pid": hb.pid,
                "jobs_processed": hb.jobs_processed,
                "errors": hb.errors,
                "current_job": hb.current_job,
                "last_heartbeat_age_sec": round(age, 1),
                "healthy": age < (Config.HEARTBEAT_INTERVAL_SEC * 3)  # Healthy if < 3 intervals
            })
        
        return {
            "running": self._running,
            "num_workers": self.num_workers,
            "pending_jobs": len(self._pending_jobs),
            "task_q_size": self.task_q.qsize(),
            "result_q_size": self.result_q.qsize(),
            "workers": worker_status
        }
    
    def _collect_heartbeats(self) -> None:
        """Drain heartbeat queue and update cache"""
        while True:
            try:
                hb: WorkerHeartbeat = self.heartbeat_q.get_nowait()
                self._worker_heartbeats[hb.worker_id] = hb
            except queue.Empty:
                break
    
    def get_worker_heartbeats(self) -> List[Dict[str, Any]]:
        """Get summary of worker heartbeats for Brain heartbeat logging"""
        self._collect_heartbeats()
        
        summaries = []
        for worker_id, hb in sorted(self._worker_heartbeats.items()):
            age = time.time() - hb.timestamp
            summaries.append({
                "id": hb.worker_id,
                "pid": hb.pid,
                "jobs": hb.jobs_processed,
                "err": hb.errors,
                "age": round(age, 1)
            })
        return summaries
    
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
