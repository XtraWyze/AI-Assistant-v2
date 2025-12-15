#!/usr/bin/env python3
"""
Hybrid Router Smoke Test

Verifies that the hybrid router correctly:
- Routes tool_plan commands with confidence >= 0.75
- Passes correct [HYBRID] route=tool_plan confidence=X format
- Maps tool parameters correctly to tool schemas
- Handles all major tool categories

Usage:
    python scripts/test_hybrid_router_smoke.py
    
Expected output:
    All test cases should pass with confidence >= 0.75 for tool_plan routes
    
Exit code:
    0 = success (all tests passed)
    1 = failure (route or confidence issues)
"""

import sys
import os
from typing import List, Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wyzer.core.hybrid_router import decide, HybridDecision
from wyzer.core.logger import init_logger, get_logger

# Setup logging
init_logger("INFO")
logger = get_logger()

# Test cases with expected outcomes
TEST_CASES: List[Tuple[str, str, float]] = [
    # Time queries
    ("what time is it", "tool_plan", 0.95),
    ("current time", "tool_plan", 0.95),
    ("what's the time", "tool_plan", 0.95),
    
    # Weather queries
    ("what's the weather", "tool_plan", 0.92),
    ("weather in new york", "tool_plan", 0.92),
    ("will it rain", "tool_plan", 0.92),
    ("what's the temperature", "tool_plan", 0.92),
    
    # System info
    ("tell me about my computer", "llm", 0.0),  # Goes to LLM for detailed info
    ("what are my system specs", "llm", 0.0),
    
    # Library refresh
    ("refresh library", "tool_plan", 0.93),
    ("rebuild library", "tool_plan", 0.93),
    ("scan my files", "tool_plan", 0.92),
    ("scan my apps", "tool_plan", 0.92),
    
    # System storage
    ("list drives", "tool_plan", 0.92),
    ("scan my drives", "tool_plan", 0.95),
    ("how much space on c drive", "tool_plan", 0.91),
    ("open d drive", "tool_plan", 0.93),
    
    # Open/launch
    ("open chrome", "tool_plan", 0.9),
    ("launch notepad", "tool_plan", 0.9),
    ("start spotify", "tool_plan", 0.9),
    
    # Close/quit
    ("close chrome", "tool_plan", 0.85),
    ("quit firefox", "tool_plan", 0.85),
    ("exit notepad", "tool_plan", 0.85),
    
    # Window management
    ("minimize chrome", "tool_plan", 0.85),
    ("maximize spotify", "tool_plan", 0.85),
    ("move chrome to monitor 2", "tool_plan", 0.85),
    
    # Audio device
    ("switch audio to headphones", "tool_plan", 0.9),
    ("set audio to speakers", "tool_plan", 0.9),
    ("list audio devices", "tool_plan", 0.92),
    
    # Volume control
    ("set volume to 50", "tool_plan", 0.9),
    ("increase volume", "tool_plan", 0.88),
    ("decrease volume", "tool_plan", 0.88),
    ("mute", "tool_plan", 0.93),
    ("unmute", "tool_plan", 0.93),
    ("get volume", "tool_plan", 0.92),
    ("what's the volume", "tool_plan", 0.92),
    
    # Media controls
    ("play the music", "tool_plan", 0.8),
    ("pause", "tool_plan", 0.8),
    ("next track", "tool_plan", 0.85),
    ("skip", "tool_plan", 0.85),
    ("previous track", "tool_plan", 0.85),
    ("go back", "tool_plan", 0.85),
]


def test_hybrid_router_routing() -> bool:
    """Test that hybrid router correctly routes commands."""
    logger.info("\n" + "=" * 70)
    logger.info("HYBRID ROUTER SMOKE TEST")
    logger.info("=" * 70)
    
    passed = 0
    failed = 0
    low_confidence = 0
    
    logger.info("\nTesting routing decisions and confidence scores...")
    logger.info("-" * 70)
    
    for query, expected_mode, expected_min_confidence in TEST_CASES:
        try:
            decision: HybridDecision = decide(query)
            
            # Check mode
            mode_ok = decision.mode == expected_mode
            
            # Check confidence threshold (0.75 to bypass LLM)
            confidence_ok = decision.confidence >= 0.75 if expected_mode == "tool_plan" else True
            bypass_llm = decision.confidence >= 0.75
            
            status = "[PASS]" if (mode_ok and confidence_ok) else "[FAIL]"
            
            if mode_ok and confidence_ok:
                passed += 1
                log_level = "info"
            elif expected_mode == "tool_plan" and not confidence_ok:
                low_confidence += 1
                log_level = "warning"
            else:
                failed += 1
                log_level = "error"
            
            msg = (
                f"{status} '{query}'\n"
                f"   Mode: {decision.mode} (expected {expected_mode})\n"
                f"   Confidence: {decision.confidence:.2f} (min: 0.75 to bypass LLM)\n"
                f"   [HYBRID] route={'tool_plan' if decision.mode == 'tool_plan' else 'llm'} "
                f"confidence={decision.confidence:.2f}"
            )
            
            if decision.mode == "tool_plan" and decision.intents:
                tools = [intent.get("tool", "?") for intent in decision.intents]
                msg += f"\n   Tools: {', '.join(tools)}"
                msg += f"\n   Bypass LLM: {bypass_llm}"
            
            if mode_ok and confidence_ok:
                logger.info(msg)
            elif expected_mode == "tool_plan" and not confidence_ok:
                logger.warning(msg)
            else:
                logger.error(msg)
            
        except Exception as e:
            logger.error(f"[FAIL] '{query}' - Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    logger.info("-" * 70)
    logger.info(f"\nRESULTS:")
    logger.info(f"  Passed:         {passed}")
    logger.info(f"  Low confidence: {low_confidence}")
    logger.info(f"  Failed:         {failed}")
    logger.info(f"  Total:          {len(TEST_CASES)}")
    
    success_rate = (passed / len(TEST_CASES)) * 100 if TEST_CASES else 0
    logger.info(f"\nSuccess rate: {success_rate:.1f}%")
    
    return failed == 0 and low_confidence == 0


def test_tool_plan_format() -> bool:
    """Test that tool_plan decisions have proper structure."""
    logger.info("\n" + "=" * 70)
    logger.info("TOOL PLAN FORMAT TEST")
    logger.info("=" * 70)
    
    logger.info("\nVerifying tool_plan structure for sample commands...")
    logger.info("-" * 70)
    
    sample_queries = [
        "what time is it",
        "open chrome",
        "set volume to 50",
        "next track",
    ]
    
    all_ok = True
    
    for query in sample_queries:
        decision = decide(query)
        
        if decision.mode == "tool_plan":
            # Check structure
            has_intents = decision.intents is not None and len(decision.intents) > 0
            all_intents_valid = True
            
            if has_intents:
                for intent in decision.intents:
                    has_tool = "tool" in intent and intent["tool"]
                    has_args = "args" in intent
                    has_continue = "continue_on_error" in intent
                    
                    if not (has_tool and has_args is not None and has_continue is not None):
                        all_intents_valid = False
                        logger.error(
                            f"[FAIL] Intent structure invalid:\n"
                            f"  Query: '{query}'\n"
                            f"  Intent: {intent}\n"
                            f"  has_tool: {has_tool}, has_args: {has_args}, has_continue: {has_continue}"
                        )
                        all_ok = False
            
            if has_intents and all_intents_valid:
                logger.info(
                    f"[PASS] '{query}'\n"
                    f"   Confidence: {decision.confidence:.2f}\n"
                    f"   Tools: {[i.get('tool') for i in decision.intents]}\n"
                    f"   Args: {[i.get('args') for i in decision.intents]}"
                )
    
    logger.info("-" * 70)
    return all_ok


def test_confidence_threshold() -> bool:
    """Test that all tool_plan routes have confidence >= 0.75."""
    logger.info("\n" + "=" * 70)
    logger.info("CONFIDENCE THRESHOLD TEST (>= 0.75)")
    logger.info("=" * 70)
    
    logger.info("\nChecking that all tool_plan routes meet 0.75 confidence threshold...")
    logger.info("-" * 70)
    
    below_threshold = []
    
    for query, expected_mode, _ in TEST_CASES:
        if expected_mode == "tool_plan":
            decision = decide(query)
            if decision.mode == "tool_plan" and decision.confidence < 0.75:
                below_threshold.append((query, decision.confidence))
                logger.warning(
                    f"⚠ '{query}' -> confidence {decision.confidence:.2f} < 0.75"
                )
    
    logger.info("-" * 70)
    
    if below_threshold:
        logger.error(f"\n{len(below_threshold)} routes below threshold:")
        for query, conf in below_threshold:
            logger.error(f"  - '{query}': {conf:.2f}")
        return False
    else:
        logger.info(f"\n[PASS] All {sum(1 for q, m, _ in TEST_CASES if m == 'tool_plan')} tool_plan routes meet threshold")
        return True


def main() -> int:
    """Run all smoke tests."""
    
    try:
        # Test 1: Routing decisions
        routing_ok = test_hybrid_router_routing()
        
        # Test 2: Format validation
        format_ok = test_tool_plan_format()
        
        # Test 3: Confidence threshold
        confidence_ok = test_confidence_threshold()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 70)
        
        if routing_ok and format_ok and confidence_ok:
            logger.info("[PASS] All smoke tests PASSED")
            logger.info("\nThe hybrid router is correctly:")
            logger.info("  - Routing commands with [HYBRID] route=tool_plan")
            logger.info("  - Maintaining confidence >= 0.75 for tool_plan routes")
            logger.info("  - Formatting tool_plan intents correctly")
            return 0
        else:
            logger.error("[FAIL] Some smoke tests FAILED")
            if not routing_ok:
                logger.error("  - Routing decisions issue")
            if not format_ok:
                logger.error("  - Tool plan format issue")
            if not confidence_ok:
                logger.error("  - Confidence threshold issue")
            return 1
        
    except Exception as e:
        logger.error(f"[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
