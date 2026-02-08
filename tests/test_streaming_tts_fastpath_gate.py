import pytest


@pytest.fixture(autouse=True)
def _enable_streaming_gate(monkeypatch):
    """Force streaming feature on so should_use_streaming_tts logic is exercised."""
    from wyzer.core.config import Config

    monkeypatch.setattr(Config, "OLLAMA_STREAM_TTS", True, raising=False)
    monkeypatch.setattr(Config, "NO_OLLAMA", False, raising=False)
    yield


class TestStreamingTTSFastpathGate:
    def test_conversational_query_may_stream(self):
        from wyzer.core.orchestrator import should_use_streaming_tts

        assert should_use_streaming_tts("tell me a short story about cats") is True

    def test_polite_tool_command_blocks_streaming(self):
        from wyzer.core.orchestrator import should_use_streaming_tts

        assert should_use_streaming_tts("Could you open notepad please") is False

    def test_mixed_tool_plus_chat_blocks_streaming(self):
        from wyzer.core.orchestrator import should_use_streaming_tts

        assert should_use_streaming_tts("Can you open notepad and tell me a story") is False

    def test_fastpath_only_tool_blocks_streaming(self):
        from wyzer.core.orchestrator import should_use_streaming_tts

        # This is parsed by the fastpath audio-output clause parser.
        assert should_use_streaming_tts("set audio output to Realtek") is False

    def test_open_source_question_still_streams(self):
        from wyzer.core.orchestrator import should_use_streaming_tts

        assert should_use_streaming_tts("what is open source software") is True
