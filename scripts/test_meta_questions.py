"""Tests for Phase 12 - Meta-question deterministic answers.

Tests that meta-questions about Wyzer trigger deterministic responses
instead of LLM hallucinations.

Run with: python -m pytest scripts/test_meta_questions.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

from wyzer.policy.meta_answer import maybe_handle_meta_question


class TestMetaQuestionDetection:
    """Test meta-question pattern detection."""
    
    def test_how_did_you_do_that_detected(self):
        """'How'd you do that' should be detected as meta-question."""
        variants = [
            "how'd you do that",
            "how did you do that",
            "How did you do that?",
            "how'd you do it",
        ]
        
        for text in variants:
            handled, _ = maybe_handle_meta_question(text)
            assert handled, f"Expected '{text}' to be handled as meta-question"
    
    def test_how_does_it_work_detected(self):
        """'How does this work' should be detected as meta-question."""
        variants = [
            "how does this work",
            "how does that work",
            "how does it work",
            "how do you work",
        ]
        
        for text in variants:
            handled, _ = maybe_handle_meta_question(text)
            assert handled, f"Expected '{text}' to be handled as meta-question"
    
    def test_how_do_you_remember_detected(self):
        """'How do you remember' should be detected as meta-question."""
        variants = [
            "how do you remember",
            "how do you remember my name",
            "how do you know my name",
        ]
        
        for text in variants:
            handled, _ = maybe_handle_meta_question(text)
            assert handled, f"Expected '{text}' to be handled as meta-question"
    
    def test_what_are_you_doing_detected(self):
        """'What are you doing' should be detected as meta-question."""
        variants = [
            "what are you doing",
            "What are you doing?",
            "what were you doing",
        ]
        
        for text in variants:
            handled, _ = maybe_handle_meta_question(text)
            assert handled, f"Expected '{text}' to be handled as meta-question"
    
    def test_where_did_you_get_detected(self):
        """'Where did you get that' should be detected as meta-question."""
        variants = [
            "where did you get that",
            "where did you get this",
        ]
        
        for text in variants:
            handled, _ = maybe_handle_meta_question(text)
            assert handled, f"Expected '{text}' to be handled as meta-question"
    
    def test_how_did_you_know_detected(self):
        """'How did you know that' should be detected as meta-question."""
        variants = [
            "how did you know that",
            "how did you know this",
        ]
        
        for text in variants:
            handled, _ = maybe_handle_meta_question(text)
            assert handled, f"Expected '{text}' to be handled as meta-question"
    
    def test_non_meta_questions_not_detected(self):
        """Normal questions should NOT be detected as meta-questions."""
        non_meta = [
            "what's the weather",
            "open chrome",
            "close spotify",
            "how tall is mount everest",
            "what time is it",
            "tell me a story",
            "how do I install python",  # About external topic, not Wyzer
        ]
        
        for text in non_meta:
            handled, _ = maybe_handle_meta_question(text)
            assert not handled, f"Expected '{text}' to NOT be handled as meta-question"


class TestMetaAnswerContent:
    """Test that meta-answers are truthful and concise."""
    
    def test_answers_are_concise(self):
        """All meta-answers should be 1-2 sentences max."""
        test_cases = [
            "how'd you do that",
            "how does this work",
            "how do you remember",
            "what are you doing",
        ]
        
        for text in test_cases:
            handled, answer = maybe_handle_meta_question(text)
            assert handled
            assert len(answer) > 0
            
            # Count sentences (rough heuristic)
            sentence_count = answer.count('.') + answer.count('!') + answer.count('?')
            assert sentence_count <= 3, f"Answer too long ({sentence_count} sentences): {answer}"
    
    def test_no_generic_ml_claims(self):
        """Answers should not mention 'machine learning algorithms' generically."""
        test_cases = [
            "how'd you do that",
            "how does this work",
            "how do you remember",
        ]
        
        for text in test_cases:
            handled, answer = maybe_handle_meta_question(text)
            assert handled
            
            # Check for generic ML phrases
            answer_lower = answer.lower()
            assert "machine learning" not in answer_lower, f"Generic ML claim in: {answer}"
            assert "neural network" not in answer_lower, f"Generic ML claim in: {answer}"
            assert "ai algorithm" not in answer_lower, f"Generic ML claim in: {answer}"
    
    def test_memory_answer_mentions_explicit_storage(self):
        """Memory-related answers should mention explicit storage."""
        text = "how do you remember my name"
        handled, answer = maybe_handle_meta_question(text)
        
        assert handled
        answer_lower = answer.lower()
        
        # Should mention explicit storage/facts
        assert any(keyword in answer_lower for keyword in [
            "store", "save", "told", "inject", "fact", "memory"
        ]), f"Memory answer should mention explicit storage: {answer}"


class TestContextAwareAnswers:
    """Test that answers adapt to context."""
    
    def test_after_tool_execution(self):
        """'How'd you do that' after tool should mention deterministic routing."""
        execution_summary = {
            "ran": [{"tool": "close_window", "ok": True}]
        }
        
        handled, answer = maybe_handle_meta_question(
            "how'd you do that",
            last_execution_summary=execution_summary
        )
        
        assert handled
        answer_lower = answer.lower()
        assert any(keyword in answer_lower for keyword in [
            "deterministic", "routing", "tool", "match"
        ]), f"Should mention deterministic routing: {answer}"
    
    def test_after_memory_based_response(self):
        """'How did you know that' after memory-based response should mention memory."""
        handled, answer = maybe_handle_meta_question(
            "how did you know that",
            was_memory_based=True
        )
        
        assert handled
        answer_lower = answer.lower()
        assert any(keyword in answer_lower for keyword in [
            "memory", "store", "save", "told", "fact"
        ]), f"Should mention memory: {answer}"
    
    def test_after_identity_query(self):
        """'How did you know that' after identity query should mention memory."""
        handled, answer = maybe_handle_meta_question(
            "how did you know that",
            was_identity_query=True
        )
        
        assert handled
        answer_lower = answer.lower()
        assert any(keyword in answer_lower for keyword in [
            "memory", "store", "save", "told", "fact"
        ]), f"Should mention memory: {answer}"


class TestOrchestratorIntegration:
    """Test integration with orchestrator."""
    
    def test_meta_question_simple_detection(self):
        """Simple test that meta-question module works correctly."""
        from wyzer.policy.meta_answer import maybe_handle_meta_question
        
        # Test a simple meta-question
        handled, reply = maybe_handle_meta_question("how do you remember my name")
        
        assert handled, "Should handle meta-question"
        assert len(reply) > 0, "Should return non-empty reply"
        
        # Reply should mention memory
        assert any(keyword in reply.lower() for keyword in [
            "memory", "store", "save", "told", "fact"
        ]), f"Reply should mention memory: {reply}"


if __name__ == "__main__":
    print("Running meta-question tests...")
    pytest.main([__file__, "-v"])
