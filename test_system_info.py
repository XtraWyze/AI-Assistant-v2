#!/usr/bin/env python3
"""Test get_system_info routing."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wyzer.core.hybrid_router import decide

test_commands = [
    "get my system info",
    "tell me about my system",
    "what are my system specs",
    "how much ram do i have",
    "what's my cpu",
    "system information",
]

print("=== GET_SYSTEM_INFO ROUTING TEST ===\n")

for cmd in test_commands:
    d = decide(cmd)
    print(f"Command: '{cmd}'")
    print(f"  Route: {d.mode} (confidence: {d.confidence:.2f})")
    if d.mode == "tool_plan" and d.intents:
        print(f"  Tool: {d.intents[0].get('tool')}")
    print()
