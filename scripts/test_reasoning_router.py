"""Test script for reasoning/multi-intent routing to LLM."""

from wyzer.core.hybrid_router import decide, needs_reasoning, looks_multi_intent

# Test cases for reasoning questions (should route to LLM)
reasoning_tests = [
    'why is the sky blue',
    'how do I install Python',
    'explain how this works',
    'what is the difference between X and Y',
    'can you help me understand this',
    'should I use tabs or spaces',
    'what if I delete this file',
    'which one is better',
    'tell me about machine learning',
]

print('=== REASONING TESTS (should go to LLM) ===')
for test in reasoning_tests:
    decision = decide(test)
    status = '✓' if decision.mode == 'llm' else '✗'
    print(f'{status} "{test}" -> {decision.mode} (needs_reasoning={needs_reasoning(test)})')

# Test cases for multi-intent that need LLM splitting
multi_tests = [
    'open spotify and tell me about the weather',
    'close chrome and then explain what happened',
    'open steam but why is it slow',
]

print('\n=== MULTI-INTENT NEEDING LLM ===')
for test in multi_tests:
    decision = decide(test)
    status = '✓' if decision.mode == 'llm' else '✗'
    print(f'{status} "{test}" -> {decision.mode}')

# Test cases for simple multi-intent (should still work deterministically)
simple_multi = [
    'open spotify and chrome',
    'close discord then open steam',
    'mute and pause',
]

print('\n=== SIMPLE MULTI-INTENT (should be tool_plan) ===')
for test in simple_multi:
    decision = decide(test)
    status = '✓' if decision.mode == 'tool_plan' else '✗'
    print(f'{status} "{test}" -> {decision.mode}')

# Test simple tool commands still work
simple_tests = [
    'open spotify',
    'what time is it',
    'mute',
    'volume 50',
]

print('\n=== SIMPLE TOOL COMMANDS (should be tool_plan) ===')
for test in simple_tests:
    decision = decide(test)
    status = '✓' if decision.mode == 'tool_plan' else '✗'
    print(f'{status} "{test}" -> {decision.mode}')
