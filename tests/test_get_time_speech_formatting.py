from wyzer.core.orchestrator import _format_fastpath_reply
from wyzer.core.intent_plan import Intent, ExecutionResult, ExecutionSummary


def _summary(time_str: str, date_str: str) -> ExecutionSummary:
    return ExecutionSummary(
        ran=[
            ExecutionResult(
                tool="get_time",
                ok=True,
                result={"time": time_str, "date": date_str},
                error=None,
            )
        ],
        stopped_early=False,
    )


def test_get_time_time_only_meta_format() -> None:
    intents = [Intent(tool="get_time", args={}, meta={"format": "time_only"})]
    reply = _format_fastpath_reply("what is the time", intents, _summary("18:52:00", "2026-02-07"))
    assert reply == "It is 6:52 PM."


def test_get_time_date_only_meta_format() -> None:
    intents = [Intent(tool="get_time", args={}, meta={"format": "date_only"})]
    reply = _format_fastpath_reply("what is today", intents, _summary("18:52:00", "2026-02-07"))
    assert reply == "Today is Saturday, February 7, 2026."


def test_get_time_time_and_date_meta_format() -> None:
    intents = [Intent(tool="get_time", args={}, meta={"format": "time_and_date"})]
    reply = _format_fastpath_reply("time and date", intents, _summary("18:52:00", "2026-02-07"))
    assert reply == "It is 6:52 PM on Saturday, February 7, 2026."


def test_get_time_date_only_keyword_fallback() -> None:
    intents = [Intent(tool="get_time", args={})]
    reply = _format_fastpath_reply("What is today?", intents, _summary("18:52:00", "2026-02-07"))
    assert reply == "Today is Saturday, February 7, 2026."


def test_get_time_time_and_date_keyword_fallback() -> None:
    intents = [Intent(tool="get_time", args={})]
    reply = _format_fastpath_reply("Can you tell me the time and date?", intents, _summary("18:52:00", "2026-02-07"))
    assert reply == "It is 6:52 PM on Saturday, February 7, 2026."
