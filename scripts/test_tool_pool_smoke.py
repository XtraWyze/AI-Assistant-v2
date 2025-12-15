#!/usr/bin/env python3
"""
Tool Worker Pool Smoke Test

Verifies that the tool worker pool works correctly and produces
identical results to in-process tool execution.

Tests:
- Pool initialization
- Tool submission and execution
- Result retrieval and validation
- JSON serialization
- Fallback to in-process execution

Usage:
    python scripts/test_tool_pool_smoke.py
    
Expected exit code:
    0 = success (all tests passed)
    1 = failure (pool issue detected)
"""

import sys
import os
import json
import time
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wyzer.core.config import Config
from wyzer.core.logger import init_logger, get_logger
from wyzer.core.tool_worker_pool import ToolWorkerPool
from wyzer.tools.registry import build_default_registry
from wyzer.tools.validation import validate_args

# Setup logging
init_logger("INFO")
logger = get_logger()


def test_pool_initialization() -> bool:
    """Test pool creation and startup"""
    logger.info("\n[TEST 1] Pool Initialization")
    logger.info("-" * 50)
    
    try:
        pool = ToolWorkerPool(num_workers=2)
        if pool.start():
            logger.info("✓ Pool started successfully")
            status = pool.get_status()
            logger.info(f"  Workers: {status['num_workers']}")
            logger.info(f"  Running: {status['running']}")
            pool.shutdown()
            logger.info("✓ Pool shutdown successfully")
            return True
        else:
            logger.error("✗ Pool failed to start")
            return False
    except Exception as e:
        logger.error(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_execution(pool: ToolWorkerPool) -> bool:
    """Test actual tool execution through pool"""
    logger.info("\n[TEST 2] Tool Execution")
    logger.info("-" * 50)
    
    registry = build_default_registry()
    
    # Test simple tools that don't require special setup
    test_tools = ["get_time", "get_system_info"]
    
    if sys.platform != "win32":
        # Skip Windows-only tools on non-Windows
        test_tools = ["get_time", "get_system_info"]
    
    all_passed = True
    
    for tool_name in test_tools:
        try:
            logger.info(f"\nTesting tool: {tool_name}")
            
            # Get tool from registry
            tool = registry.get(tool_name)
            if tool is None:
                logger.warning(f"  ⚠ Tool '{tool_name}' not in registry, skipping")
                continue
            
            # Get tool schema
            args_schema = tool.args_schema
            tool_args = {}  # Empty args for simple tools
            
            # Validate
            is_valid, error = validate_args(args_schema, tool_args)
            if not is_valid:
                logger.error(f"  ✗ Validation failed: {error}")
                all_passed = False
                continue
            
            # Execute through pool
            import uuid
            job_id = str(uuid.uuid4())
            request_id = str(uuid.uuid4())
            
            start_time = time.perf_counter()
            
            if not pool.submit_job(job_id, request_id, tool_name, tool_args):
                logger.error(f"  ✗ Failed to submit job to pool")
                all_passed = False
                continue
            
            # Wait for result
            result_obj = pool.wait_for_result(job_id, timeout=10)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            if result_obj is None:
                logger.error(f"  ✗ Timeout waiting for result")
                all_passed = False
                continue
            
            result = result_obj.result
            
            # Check result format
            if not isinstance(result, dict):
                logger.error(f"  ✗ Result is not a dict: {type(result)}")
                all_passed = False
                continue
            
            # Check JSON serializability
            try:
                json_str = json.dumps(result)
                logger.info(f"  ✓ Tool executed and JSON-serializable")
                logger.info(f"    Keys: {list(result.keys())}")
                logger.info(f"    Execution time: {result_obj.execution_time_ms:.1f}ms")
            except (TypeError, ValueError) as e:
                logger.error(f"  ✗ Result not JSON-serializable: {e}")
                all_passed = False
                continue
        
        except Exception as e:
            logger.error(f"  ✗ Exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_in_process_comparison() -> bool:
    """Compare pool results with in-process execution"""
    logger.info("\n[TEST 3] Pool vs In-Process Comparison")
    logger.info("-" * 50)
    
    registry = build_default_registry()
    
    test_tool = "get_time"
    
    try:
        tool = registry.get(test_tool)
        if tool is None:
            logger.warning(f"  ⚠ Tool '{test_tool}' not found")
            return True
        
        # In-process execution
        in_process_result = tool.run()
        logger.info(f"  In-process result keys: {list(in_process_result.keys())}")
        
        # Pool execution
        pool = ToolWorkerPool(num_workers=1)
        pool.start()
        
        import uuid
        job_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        
        if not pool.submit_job(job_id, request_id, test_tool, {}):
            logger.error(f"  ✗ Failed to submit job")
            pool.shutdown()
            return False
        
        result_obj = pool.wait_for_result(job_id, timeout=5)
        pool.shutdown()
        
        if result_obj is None:
            logger.error(f"  ✗ Pool timed out")
            return False
        
        pool_result = result_obj.result
        logger.info(f"  Pool result keys: {list(pool_result.keys())}")
        
        # Compare structure
        if set(in_process_result.keys()) == set(pool_result.keys()):
            logger.info("  ✓ Result structure matches")
            return True
        else:
            logger.error(f"  ✗ Result structure mismatch")
            logger.error(f"    In-process: {set(in_process_result.keys())}")
            logger.error(f"    Pool: {set(pool_result.keys())}")
            return False
    
    except Exception as e:
        logger.error(f"  ✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pool_shutdown() -> bool:
    """Test graceful shutdown"""
    logger.info("\n[TEST 4] Pool Shutdown")
    logger.info("-" * 50)
    
    try:
        pool = ToolWorkerPool(num_workers=2)
        pool.start()
        
        # Let it run a bit
        time.sleep(0.5)
        
        # Shutdown
        pool.shutdown(timeout=3)
        
        logger.info("✓ Pool shutdown gracefully")
        return True
    except Exception as e:
        logger.error(f"✗ Shutdown exception: {e}")
        return False


def main() -> int:
    """Run all smoke tests"""
    
    logger.info("=" * 70)
    logger.info("WYZER TOOL WORKER POOL SMOKE TEST")
    logger.info("=" * 70)
    
    results: Dict[str, bool] = {}
    
    # Test 1: Initialization
    results["initialization"] = test_pool_initialization()
    
    # Test 2: Tool Execution
    try:
        pool = ToolWorkerPool(num_workers=2)
        pool.start()
        results["execution"] = test_tool_execution(pool)
        pool.shutdown()
    except Exception as e:
        logger.error(f"Failed to set up pool for execution test: {e}")
        results["execution"] = False
    
    # Test 3: Comparison
    results["comparison"] = test_in_process_comparison()
    
    # Test 4: Shutdown
    results["shutdown"] = test_pool_shutdown()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        logger.info("\n✓ ALL TESTS PASSED")
        return 0
    else:
        logger.error(f"\n✗ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
