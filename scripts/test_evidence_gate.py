"""
Acceptance tests for Phase 15: Global LLM Evidence Gate.

Tests the invariant:
  If a user query is "tool-relevant" and NO tool ran and NO deterministic
  WorldState fact exists, Wyzer MUST NOT produce an LLM-only answer.

These tests exercise:
  1. tool_relevance_gate.is_tool_relevant_query
  2. tool_relevance_gate.gate_decision
  3. evidence_envelope.EvidenceEnvelope construction
  4. Integration with prompt_builder (evidence block injection)
"""

import unittest


class TestToolRelevanceGate(unittest.TestCase):
    """Test the is_tool_relevant_query classifier."""

    def setUp(self):
        from wyzer.policy.tool_relevance_gate import is_tool_relevant_query
        self.is_tool_relevant = is_tool_relevant_query

    # ------- MUST be tool-relevant -------

    def test_tell_me_what_you_see(self):
        assert self.is_tool_relevant("Tell me what you see.")

    def test_what_do_you_see(self):
        assert self.is_tool_relevant("What do you see?")

    def test_whats_on_my_screen(self):
        assert self.is_tool_relevant("What's on my screen?")

    def test_what_did_i_open_recently(self):
        assert self.is_tool_relevant("What did I open most recently?")

    def test_did_that_work(self):
        assert self.is_tool_relevant("Did that work?")

    def test_did_it_succeed(self):
        assert self.is_tool_relevant("Did it succeed?")

    def test_what_just_happened(self):
        assert self.is_tool_relevant("What just happened?")

    def test_click_on_the_button(self):
        assert self.is_tool_relevant("Click on the button")

    def test_is_the_install_done(self):
        assert self.is_tool_relevant("Is the install done?")

    def test_what_error_did_you_get(self):
        assert self.is_tool_relevant("What error did you get?")

    def test_which_window_is_focused(self):
        assert self.is_tool_relevant("Which window is focused?")

    def test_describe_the_screen(self):
        assert self.is_tool_relevant("Describe the screen")

    def test_what_was_just_opened(self):
        assert self.is_tool_relevant("What was just opened?")

    def test_did_the_download_finish(self):
        assert self.is_tool_relevant("Did the download finish?")

    def test_scroll_down(self):
        assert self.is_tool_relevant("Scroll down")

    def test_what_did_you_click(self):
        assert self.is_tool_relevant("Did you click on it?")

    # ------- MUST NOT be tool-relevant (pure chat / informational) -------

    def test_what_is_a_vae(self):
        assert not self.is_tool_relevant("What is a VAE?")

    def test_tell_me_about_python(self):
        assert not self.is_tool_relevant("Tell me about Python")

    def test_explain_recursion(self):
        assert not self.is_tool_relevant("Explain what a recursive function is")

    def test_who_is_einstein(self):
        assert not self.is_tool_relevant("Who is Albert Einstein?")

    def test_tell_me_a_joke(self):
        assert not self.is_tool_relevant("Tell me a joke")

    def test_what_is_the_meaning_of_life(self):
        assert not self.is_tool_relevant("What is the meaning of life?")

    def test_how_does_a_cpu_work(self):
        assert not self.is_tool_relevant("How does a CPU work?")

    def test_define_an_error(self):
        """'What is an error' is informational, not checking for an error."""
        assert not self.is_tool_relevant("What is an error in programming?")


class TestGateDecision(unittest.TestCase):
    """Test the gate_decision function."""

    def setUp(self):
        from wyzer.policy.tool_relevance_gate import gate_decision
        self.gate_decision = gate_decision

    def test_pure_chat_allowed(self):
        """Pure chat question => gate returns None (LLM allowed)."""
        result = self.gate_decision("What is a VAE?", executed_any_tool=False)
        assert result is None, f"Expected None (LLM allowed), got: {result}"

    def test_tool_relevant_no_evidence_blocked(self):
        """Tool-relevant query + no tool ran => gate returns refusal string."""
        result = self.gate_decision("Tell me what you see.", executed_any_tool=False)
        assert isinstance(result, str), (
            f"Expected refusal string, got: {result}"
        )
        assert len(result) > 10  # Not empty

    def test_tool_relevant_with_tool_allowed(self):
        """Tool-relevant query + tool ran => gate returns None (LLM narration ok)."""
        result = self.gate_decision("Tell me what you see.", executed_any_tool=True)
        assert result is None, f"Expected None (narration allowed), got: {result}"

    def test_did_that_work_no_tool_blocked(self):
        """'Did that work?' with no tool => blocked."""
        result = self.gate_decision("Did that work?", executed_any_tool=False)
        assert isinstance(result, str)

    def test_did_that_work_after_tool_allowed(self):
        """'Did that work?' after tool ran => allowed."""
        result = self.gate_decision("Did that work?", executed_any_tool=True)
        assert result is None

    def test_what_did_i_open_no_tool_blocked(self):
        """'What did I open most recently?' with no tool => blocked."""
        result = self.gate_decision(
            "What did I open most recently?", executed_any_tool=False
        )
        assert isinstance(result, str)


class TestEvidenceEnvelope(unittest.TestCase):
    """Test the EvidenceEnvelope construction and rendering."""

    def test_empty_envelope(self):
        from wyzer.policy.evidence_envelope import build_empty_envelope, EvidenceEnvelope

        env = build_empty_envelope()
        assert isinstance(env, EvidenceEnvelope)
        assert not env.has_tool_evidence
        assert not env.any_tool_succeeded
        assert len(env.limitations) > 0

    def test_envelope_from_execution_summary(self):
        from wyzer.policy.evidence_envelope import (
            build_envelope_from_execution,
            EvidenceEnvelope,
        )
        from wyzer.core.intent_plan import ExecutionResult, ExecutionSummary

        summary = ExecutionSummary(
            ran=[
                ExecutionResult(
                    tool="get_time", ok=True, result={"time": "12:34 PM"}, error=None
                ),
            ],
            stopped_early=False,
        )
        env = build_envelope_from_execution(summary)
        assert isinstance(env, EvidenceEnvelope)
        assert env.has_tool_evidence
        assert env.any_tool_succeeded
        assert len(env.tools_executed) == 1
        assert env.tools_executed[0].name == "get_time"
        assert env.tools_executed[0].ok is True

    def test_prompt_block_rendering(self):
        from wyzer.policy.evidence_envelope import (
            build_envelope_from_execution,
        )
        from wyzer.core.intent_plan import ExecutionResult, ExecutionSummary

        summary = ExecutionSummary(
            ran=[
                ExecutionResult(
                    tool="describe_screen",
                    ok=True,
                    result={"description": "Chrome browser with Google open"},
                    error=None,
                ),
            ],
            stopped_early=False,
        )
        env = build_envelope_from_execution(summary)
        block = env.to_prompt_block()
        assert "VERIFIED_EVIDENCE:" in block
        assert "describe_screen" in block
        assert "OK" in block

    def test_envelope_dict_serializable(self):
        import json
        from wyzer.policy.evidence_envelope import build_empty_envelope

        env = build_empty_envelope()
        d = env.to_dict()
        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


class TestPromptBuilderEvidence(unittest.TestCase):
    """Test that PromptBuilder injects evidence envelope when provided."""

    def test_evidence_injected_into_prompt(self):
        from wyzer.brain.prompt_builder import PromptBuilder
        from wyzer.policy.evidence_envelope import (
            build_envelope_from_execution,
        )
        from wyzer.core.intent_plan import ExecutionResult, ExecutionSummary

        summary = ExecutionSummary(
            ran=[
                ExecutionResult(
                    tool="get_active_window",
                    ok=True,
                    result={"app": "Chrome", "title": "Google"},
                    error=None,
                ),
            ],
            stopped_early=False,
        )
        envelope = build_envelope_from_execution(summary)

        builder = PromptBuilder(
            user_text="What window is active?",
            evidence_envelope=envelope,
        )
        prompt, mode = builder.build()
        assert "VERIFIED_EVIDENCE:" in prompt
        assert "get_active_window" in prompt
        assert "EVIDENCE RULES" in prompt

    def test_no_evidence_no_injection(self):
        from wyzer.brain.prompt_builder import PromptBuilder

        builder = PromptBuilder(
            user_text="What is a VAE?",
            evidence_envelope=None,
        )
        prompt, mode = builder.build()
        assert "VERIFIED_EVIDENCE:" not in prompt


class TestAcceptanceScenarios(unittest.TestCase):
    """
    High-level acceptance scenarios from the spec.
    These test the gate + envelope at the policy level.
    """

    def test_scenario_1_see_without_perception(self):
        """
        Ask 'Tell me what you see.'
        No perception tool ran => must refuse.
        """
        from wyzer.policy.tool_relevance_gate import gate_decision

        refusal = gate_decision("Tell me what you see.", executed_any_tool=False)
        assert refusal is not None, "Must refuse when no perception tool ran"
        # Must NOT contain "I see"
        assert "I see" not in refusal

    def test_scenario_2_what_did_i_open(self):
        """
        Ask 'What did I open most recently?'
        No tool ran, no world state => must refuse.
        """
        from wyzer.policy.tool_relevance_gate import gate_decision

        refusal = gate_decision(
            "What did I open most recently?", executed_any_tool=False
        )
        assert refusal is not None

    def test_scenario_3_did_that_work_with_tool(self):
        """
        Ask 'Did that work?' right after a tool runs => must allow narration.
        """
        from wyzer.policy.tool_relevance_gate import gate_decision

        result = gate_decision("Did that work?", executed_any_tool=True)
        assert result is None, "LLM should be allowed when tool ran"

    def test_scenario_4_pure_chat_allowed(self):
        """
        Ask 'What is a VAE?' => LLM streaming is allowed.
        """
        from wyzer.policy.tool_relevance_gate import gate_decision

        result = gate_decision("What is a VAE?", executed_any_tool=False)
        assert result is None, "Pure chat should not be blocked"


if __name__ == "__main__":
    unittest.main()
