import pytest

from wyzer.core import hybrid_router


@pytest.mark.parametrize(
    "text",
    [
        "What time is it?",
        "Time.",
        "time",
        "TIME!!!",
        "time time time",
        "Time. What is it?",
        "Time. What is it? What time is it?",
        "uh time please",
        "hey can you tell me the current time right now",
    ],
)
def test_hybrid_router_time_fragments_route_to_get_time(text: str):
    decision = hybrid_router.decide(text)
    assert decision.mode == "tool_plan"
    assert decision.intents
    assert decision.intents[0]["tool"] == "get_time"
    assert decision.confidence >= 0.75


@pytest.mark.parametrize(
    "text",
    [
        "once upon a time",
        "tell me a story about time",
        "what is time",
    ],
)
def test_hybrid_router_does_not_hijack_non_time_queries(text: str):
    decision = hybrid_router.decide(text)
    # These should not become get_time tool calls.
    assert not (
        decision.mode == "tool_plan"
        and decision.intents
        and decision.intents[0].get("tool") == "get_time"
    )


@pytest.mark.parametrize(
    "text",
    [
        "What's the time and give me a short story?",
        "Give me a short story and what's the time?",
        "Tell me the current time then tell me a short story.",
    ],
)
def test_hybrid_router_time_plus_creative_returns_leftover(text: str):
    decision = hybrid_router.decide(text)
    assert decision.mode == "tool_plan"
    assert decision.intents
    assert decision.intents[0]["tool"] == "get_time"
    assert decision.confidence >= 0.75
    assert isinstance(decision.reply, str)
    assert decision.reply.startswith("__LEFTOVER__:")
    assert "story" in decision.reply.lower()


def test_hybrid_router_can_you_tell_me_the_time_and_story_leftover():
    text = "Can you tell me the time and tell me a short story?"
    decision = hybrid_router.decide(text)
    assert decision.mode == "tool_plan"
    assert decision.intents
    assert decision.intents[0]["tool"] == "get_time"
    assert decision.reply.startswith("__LEFTOVER__:")
    assert "short story" in decision.reply.lower()


def test_hybrid_router_open_notepad_plus_story_returns_leftover():
    # This is the normalized form brain_worker uses for routing prechecks.
    text = "open notepad and tell me a story"
    decision = hybrid_router.decide(text)
    assert decision.mode == "tool_plan"
    assert decision.intents
    assert decision.intents[0]["tool"] == "open_target"
    assert (decision.intents[0].get("args") or {}).get("query") == "notepad"
    assert decision.reply.startswith("__LEFTOVER__:")
    assert "story" in decision.reply.lower()
