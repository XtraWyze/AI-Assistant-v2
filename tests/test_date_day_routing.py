import pytest

from wyzer.core import hybrid_router


@pytest.mark.parametrize(
    "text",
    [
        "What is today?",
        "what day is it",
        "what's the date",
        "today's date",
        "current date",
        "day of the week",
        "hey wyzer, what is today",
        "time and date",
    ],
)
def test_hybrid_router_date_day_routes_to_get_time(text: str):
    decision = hybrid_router.decide(text)
    assert decision.mode == "tool_plan"
    assert decision.intents
    assert decision.intents[0]["tool"] == "get_time"
    assert decision.confidence >= 0.93
