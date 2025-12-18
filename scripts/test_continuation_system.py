"""
Unit tests for the continuation/reply-only system in orchestrator.

These tests lock the behavior to prevent regressions in future phases.
Run with: python -m pytest scripts/test_continuation_system.py -v
Or standalone: python scripts/test_continuation_system.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wyzer.core.orchestrator import (
    is_continuation_phrase,
    is_explicit_continuation,
    is_informational_query,
    rewrite_continuation,
    set_last_topic,
    get_last_topic,
    reset_continuation_hops,
    increment_continuation_hops,
    get_continuation_hops,
    _extract_topic_from_query,
    _MAX_CONTINUATION_HOPS,
)


def reset_state():
    """Reset module state before each test."""
    # Clear topic by setting empty
    from wyzer.core import orchestrator
    orchestrator._last_topic = None
    orchestrator._continuation_hops = 0


class TestContinuationPhraseDetection:
    """Test vague continuation phrase detection."""
    
    def test_tell_me_more(self):
        reset_state()
        assert is_continuation_phrase("Tell me more.") == True
        assert is_continuation_phrase("tell me more") == True
        assert is_continuation_phrase("Tell me more?") == True
    
    def test_can_you_tell_me_more(self):
        reset_state()
        assert is_continuation_phrase("Can you tell me more?") == True
        assert is_continuation_phrase("can you tell me more") == True
    
    def test_go_on_continue(self):
        reset_state()
        assert is_continuation_phrase("Go on.") == True
        assert is_continuation_phrase("Continue.") == True
        assert is_continuation_phrase("keep going") == True
    
    def test_elaborate(self):
        reset_state()
        assert is_continuation_phrase("Elaborate.") == True
        assert is_continuation_phrase("Can you elaborate?") == True
        assert is_continuation_phrase("please elaborate") == True
    
    def test_affirmatives(self):
        reset_state()
        assert is_continuation_phrase("Yes.") == True
        assert is_continuation_phrase("yeah") == True
        assert is_continuation_phrase("sure") == True
        assert is_continuation_phrase("okay") == True
    
    def test_not_continuation_with_explicit_topic(self):
        reset_state()
        # These have explicit topics - should NOT match as vague continuation
        assert is_continuation_phrase("Tell me more about One Piece.") == False
        assert is_continuation_phrase("Tell me about Naruto.") == False
        assert is_continuation_phrase("What is Python?") == False


class TestExplicitContinuation:
    """Test explicit continuation detection (tell me more about X)."""
    
    def test_tell_me_more_about_x(self):
        reset_state()
        result = is_explicit_continuation("Tell me more about One Piece.")
        assert result == "One Piece"
    
    def test_more_about_x(self):
        reset_state()
        result = is_explicit_continuation("More about Naruto.")
        assert result == "Naruto"
    
    def test_elaborate_on_x(self):
        reset_state()
        result = is_explicit_continuation("Elaborate on the plot.")
        assert result == "the plot"
    
    def test_not_explicit_continuation(self):
        reset_state()
        assert is_explicit_continuation("Tell me more.") is None
        assert is_explicit_continuation("What time is it?") is None
        assert is_explicit_continuation("Open Chrome.") is None


class TestInformationalQuery:
    """Test informational query detection (should use reply-only)."""
    
    def test_tell_me_about(self):
        reset_state()
        assert is_informational_query("Tell me about Naruto.") == True
        assert is_informational_query("tell me about One Piece") == True
    
    def test_what_is(self):
        reset_state()
        assert is_informational_query("What is Python?") == True
        assert is_informational_query("What are neural networks?") == True
    
    def test_who_is(self):
        reset_state()
        assert is_informational_query("Who is Elon Musk?") == True
        assert is_informational_query("Who was Einstein?") == True
    
    def test_explain_describe(self):
        reset_state()
        assert is_informational_query("Explain quantum physics.") == True
        assert is_informational_query("Describe the solar system.") == True
    
    def test_not_informational_commands(self):
        reset_state()
        # Commands should NOT be informational
        assert is_informational_query("Open Chrome.") == False
        assert is_informational_query("Set a timer for 5 minutes.") == False
        assert is_informational_query("What time is it?") == False  # This is a tool request
        assert is_informational_query("Pause music.") == False


class TestTopicTracking:
    """Test topic extraction and tracking."""
    
    def test_set_and_get_topic(self):
        reset_state()
        set_last_topic("One Piece")
        assert get_last_topic() == "One Piece"
    
    def test_topic_hygiene_rejects_short(self):
        reset_state()
        set_last_topic("ab")  # Too short
        assert get_last_topic() is None
    
    def test_topic_hygiene_rejects_no_letters(self):
        reset_state()
        set_last_topic("123")  # No letters
        assert get_last_topic() is None
    
    def test_topic_hygiene_rejects_filler(self):
        reset_state()
        set_last_topic("the")  # Filler word
        assert get_last_topic() is None
        set_last_topic("it")
        assert get_last_topic() is None
    
    def test_topic_resets_continuation_hops(self):
        reset_state()
        increment_continuation_hops()
        increment_continuation_hops()
        assert get_continuation_hops() == 2
        set_last_topic("New Topic")  # Should reset hops
        assert get_continuation_hops() == 0
    
    def test_extract_topic_from_tell_me_about(self):
        reset_state()
        topic = _extract_topic_from_query("Tell me about One Piece.")
        assert topic == "One Piece"
    
    def test_extract_topic_from_what_is(self):
        reset_state()
        topic = _extract_topic_from_query("What is machine learning?")
        assert topic == "machine learning"


class TestContinuationRewrite:
    """Test continuation phrase rewriting."""
    
    def test_rewrite_with_topic(self):
        reset_state()
        set_last_topic("One Piece")
        result = rewrite_continuation("Tell me more.")
        assert result == "Continue explaining One Piece in more detail."
    
    def test_no_rewrite_without_topic(self):
        reset_state()
        # No topic set
        result = rewrite_continuation("Tell me more.")
        assert result == "Tell me more."  # Unchanged
    
    def test_no_rewrite_for_non_continuation(self):
        reset_state()
        set_last_topic("One Piece")
        result = rewrite_continuation("What time is it?")
        assert result == "What time is it?"  # Not a continuation, unchanged


class TestContinuationHops:
    """Test continuation depth limiting."""
    
    def test_hop_increment(self):
        reset_state()
        assert increment_continuation_hops() == 1
        assert increment_continuation_hops() == 2
        assert increment_continuation_hops() == 3
    
    def test_hop_reset(self):
        reset_state()
        increment_continuation_hops()
        increment_continuation_hops()
        reset_continuation_hops()
        assert get_continuation_hops() == 0
    
    def test_max_hops_constant(self):
        reset_state()
        assert _MAX_CONTINUATION_HOPS == 3


class TestIntegration:
    """Integration tests for the full flow."""
    
    def test_flow_tell_me_about_then_more(self):
        """User asks about topic, then says 'tell me more'."""
        reset_state()
        
        # Step 1: "Tell me about One Piece" - should be informational
        assert is_informational_query("Tell me about One Piece.") == True
        topic = _extract_topic_from_query("Tell me about One Piece.")
        assert topic == "One Piece"
        set_last_topic(topic)
        
        # Step 2: "Tell me more" - should rewrite
        assert is_continuation_phrase("Tell me more.") == True
        rewritten = rewrite_continuation("Tell me more.")
        assert rewritten == "Continue explaining One Piece in more detail."
    
    def test_flow_explicit_continuation(self):
        """User says 'tell me more about X' with explicit topic."""
        reset_state()
        
        # First establish a different topic
        set_last_topic("Naruto")
        
        # "Tell me more about One Piece" should update topic
        explicit = is_explicit_continuation("Tell me more about One Piece.")
        assert explicit == "One Piece"
        set_last_topic(explicit)
        assert get_last_topic() == "One Piece"
    
    def test_tool_query_not_reply_only(self):
        """Tool commands should NOT use reply-only mode."""
        reset_state()
        
        # These should all be False for informational query check
        assert is_informational_query("What time is it?") == False
        assert is_informational_query("Open Chrome.") == False
        assert is_informational_query("Set volume to 50.") == False
        assert is_informational_query("Pause music.") == False


def run_tests():
    """Run all tests and report results."""
    import traceback
    
    test_classes = [
        TestContinuationPhraseDetection,
        TestExplicitContinuation,
        TestInformationalQuery,
        TestTopicTracking,
        TestContinuationRewrite,
        TestContinuationHops,
        TestIntegration,
    ]
    
    total = 0
    passed = 0
    failed = 0
    
    for cls in test_classes:
        print(f"\n{'='*60}")
        print(f"  {cls.__name__}")
        print(f"{'='*60}")
        
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    failed += 1
    
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
