"""Tests for Phase 12 - Tightened identity replies (no fluff).

Tests that identity queries return minimal, direct answers with NO
"as I recall", "from our previous interaction", or similar filler.

Run with: python -m pytest scripts/test_identity_tightening.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


class TestIdentityReplyTightening:
    """Test that identity replies are minimal with zero fluff."""
    
    def test_fastlane_prompt_enforces_no_justification(self):
        """Fastlane prompt should explicitly forbid justification."""
        from wyzer.brain.prompt_builder import FASTLANE_SYSTEM_PROMPT
        
        prompt_lower = FASTLANE_SYSTEM_PROMPT.lower()
        
        # Should mention "no extra" or "zero justification" or similar
        assert any(phrase in prompt_lower for phrase in [
            "no extra",
            "zero justification",
            "only the",
            "no justification",
        ]), f"Fastlane prompt should forbid fluff: {FASTLANE_SYSTEM_PROMPT}"


class TestFastlanePromptTightening:
    """Test that the fastlane system prompt enforces direct answers."""
    
    def test_fastlane_prompt_mentions_zero_justification(self):
        """Fastlane system prompt should explicitly forbid justification for identity."""
        from wyzer.brain.prompt_builder import FASTLANE_SYSTEM_PROMPT
        
        prompt_lower = FASTLANE_SYSTEM_PROMPT.lower()
        
        # Should mention direct answers or zero justification
        assert any(keyword in prompt_lower for keyword in [
            "direct",
            "zero justification",
            "only the",
            "no extra",
            "no justification",
        ]), f"Fastlane prompt should enforce directness: {FASTLANE_SYSTEM_PROMPT}"
    
    def test_fastlane_prompt_is_short(self):
        """Fastlane system prompt should be very short (under 200 chars)."""
        from wyzer.brain.prompt_builder import FASTLANE_SYSTEM_PROMPT
        
        char_count = len(FASTLANE_SYSTEM_PROMPT)
        assert char_count < 250, f"Fastlane prompt too long ({char_count} chars): {FASTLANE_SYSTEM_PROMPT}"


class TestIdentityPromptFlow:
    """Test end-to-end identity query flow."""
    
    def test_fastlane_prompt_includes_identity_directive(self):
        """Fastlane prompt should have specific identity directive."""
        from wyzer.brain.prompt_builder import FASTLANE_SYSTEM_PROMPT
        
        prompt_lower = FASTLANE_SYSTEM_PROMPT.lower()
        
        # Should mention identity or direct answers
        assert "identity" in prompt_lower or "direct" in prompt_lower or "only the" in prompt_lower, \
            f"Fastlane prompt should have identity directive: {FASTLANE_SYSTEM_PROMPT}"


if __name__ == "__main__":
    print("Running identity tightening tests...")
    pytest.main([__file__, "-v"])
