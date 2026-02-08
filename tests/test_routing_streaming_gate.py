"""
Tests for screen-state routing and streaming TTS gate.

Verifies:
- hybrid_router.decide() routes screen-state phrases to tool_plan / describe_screen.
- should_use_streaming_tts() returns False for screen-state phrases.
- should_use_streaming_tts() returns True for normal conversational chat
  (when streaming is globally enabled).
- Variants with wake words and trailing punctuation are handled correctly.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch


# ── Hybrid router: screen-state → tool_plan ────────────────────────────

class TestHybridRouterScreenState:
    """Screen-state queries must route to describe_screen via tool_plan."""

    def _decide(self, text: str):
        from wyzer.core.hybrid_router import decide
        return decide(text)

    @pytest.mark.parametrize("phrase", [
        "tell me what you see",
        "what do you see",
        "what's on screen",
        "what is on my screen",
        "describe the screen",
        "describe my screen",
        "describe what's in front of me",
        "what am i looking at",
    ])
    def test_screen_phrase_routes_to_tool_plan(self, phrase):
        decision = self._decide(phrase)
        assert decision.mode == "tool_plan", (
            f"Expected tool_plan for '{phrase}', got {decision.mode}"
        )
        assert decision.intents is not None and len(decision.intents) >= 1
        assert decision.intents[0]["tool"] == "describe_screen"

    def test_with_wake_word(self):
        decision = self._decide("hey wyzer, tell me what you see")
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "describe_screen"

    def test_with_trailing_punctuation(self):
        decision = self._decide("What do you see?")
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "describe_screen"

    def test_with_wake_word_and_punctuation(self):
        decision = self._decide("Wyzer, describe my screen!")
        assert decision.mode == "tool_plan"
        assert decision.intents[0]["tool"] == "describe_screen"

    def test_describe_screen_args_empty(self):
        """describe_screen must receive {} args to validate."""
        decision = self._decide("tell me what you see")
        assert decision.intents[0]["args"] == {}

    def test_conversational_see_not_captured(self):
        """'I see what you mean' must NOT route to describe_screen."""
        decision = self._decide("I see what you mean")
        assert decision.mode == "llm"


# ── Streaming TTS gate: screen-state → False ───────────────────────────

class TestStreamingGateScreenState:
    """should_use_streaming_tts must return False for screen-state phrases."""

    def _gate(self, text: str, stream_enabled: bool = True):
        """Call should_use_streaming_tts with stubbed config flags."""
        with patch("wyzer.core.orchestrator.Config") as MockCfg:
            MockCfg.OLLAMA_STREAM_TTS = stream_enabled
            MockCfg.NO_OLLAMA = False
            from wyzer.core.orchestrator import should_use_streaming_tts
            return should_use_streaming_tts(text)

    @pytest.mark.parametrize("phrase", [
        "tell me what you see",
        "what do you see",
        "what's on screen",
        "describe the screen",
        "what am i looking at",
    ])
    def test_screen_phrase_blocks_streaming(self, phrase):
        assert self._gate(phrase) is False, (
            f"Streaming should be blocked for '{phrase}'"
        )

    def test_wake_word_variant_blocks_streaming(self):
        assert self._gate("hey wyzer, tell me what you see?") is False

    def test_normal_chat_allows_streaming(self):
        """Normal conversational text should be allowed to stream."""
        assert self._gate("hey wyzer how are you") is True

    def test_streaming_disabled_globally(self):
        """When OLLAMA_STREAM_TTS is False, nothing streams."""
        assert self._gate("hey wyzer how are you", stream_enabled=False) is False
