"""
Brain module for Wyzer AI Assistant - Phase 4
Local LLM integration via Ollama.

Phase 10: Added internal messages[] representation for structured prompt building.
Transport to Ollama remains a single prompt string.
"""
from wyzer.brain.llm_engine import LLMEngine
from wyzer.brain.messages import (
    Message,
    msg_system,
    msg_user,
    msg_assistant,
    flatten_messages,
    MessageBuilder
)

__all__ = [
    "LLMEngine",
    "Message",
    "msg_system",
    "msg_user", 
    "msg_assistant",
    "flatten_messages",
    "MessageBuilder"
]
