import time

import pytest

from wyzer.core import hybrid_router
from wyzer.core import orchestrator
from wyzer.context.world_state import (
    clear_world_state,
    get_pending_action,
)


@pytest.fixture(autouse=True)
def _clean_state():
    clear_world_state()
    yield
    clear_world_state()


@pytest.mark.parametrize(
    "text, expected_tool",
    [
        ("what's on my screen", "describe_screen"),
        ("what is on my monitor?", "describe_screen"),
        ("what windows are open", "list_open_windows"),
        ("what's open?", "list_open_windows"),
    ],
)
def test_hybrid_router_window_perception_routes_to_tools(text: str, expected_tool: str):
    decision = hybrid_router.decide(text)
    assert decision.mode == "tool_plan"
    assert decision.intents
    assert decision.intents[0]["tool"] == expected_tool
    assert decision.confidence >= 0.9


def test_hybrid_router_guard_does_not_minimize_title_all_windows():
    decision = hybrid_router.decide("minimize all windows")

    if decision.mode == "tool_plan" and decision.intents:
        first = decision.intents[0]
        assert not (
            first.get("tool") == "minimize_window"
            and (first.get("args") or {}).get("title", "").strip().lower() == "all windows"
        )


def test_orchestrator_bulk_minimize_executes_each_window(monkeypatch):
    calls = []

    def _stub_get_registry():
        return {}

    def _stub_execute_tool(_registry, tool_name: str, tool_args: dict):
        calls.append((tool_name, dict(tool_args or {})))
        if tool_name == "list_open_windows":
            return {
                "windows": [
                    {"hwnd": 101, "id": 101, "title": "Alpha", "app": "alpha.exe", "pid": 1},
                    {"hwnd": 202, "id": 202, "title": "Beta", "app": "beta.exe", "pid": 2},
                ],
                "count": 2,
                "latency_ms": 1,
            }
        if tool_name == "minimize_window":
            hwnd = tool_args.get("hwnd")
            if hwnd == 101:
                return {"status": "minimized", "latency_ms": 1}
            return {"error": {"type": "window_not_found", "message": "No window"}, "latency_ms": 1}
        return {"error": {"type": "unexpected", "message": tool_name}}

    monkeypatch.setattr(orchestrator, "get_registry", _stub_get_registry)
    monkeypatch.setattr(orchestrator, "_execute_tool", _stub_execute_tool)

    out = orchestrator.handle_user_text("minimize all windows")
    assert "Minimized" in out["reply"]
    assert "1" in out["reply"]

    # Ensure we listed then attempted minimizations
    assert calls[0][0] == "list_open_windows"
    minimize_calls = [c for c in calls if c[0] == "minimize_window"]
    assert len(minimize_calls) == 2


def test_pending_action_question_then_do_it_executes(monkeypatch):
    calls = []

    def _stub_get_registry():
        return {}

    def _stub_execute_tool(_registry, tool_name: str, tool_args: dict):
        calls.append((tool_name, dict(tool_args or {})))
        if tool_name == "list_open_windows":
            return {
                "windows": [
                    {"hwnd": 111, "id": 111, "title": "One", "app": "one.exe", "pid": 1},
                ],
                "count": 1,
                "latency_ms": 1,
            }
        if tool_name == "minimize_window":
            return {"status": "minimized", "latency_ms": 1}
        return {"error": {"type": "unexpected", "message": tool_name}}

    monkeypatch.setattr(orchestrator, "get_registry", _stub_get_registry)
    monkeypatch.setattr(orchestrator, "_execute_tool", _stub_execute_tool)

    out1 = orchestrator.handle_user_text("can you minimize all windows?")
    assert "I can minimize" in out1["reply"]
    assert get_pending_action() is not None

    out2 = orchestrator.handle_user_text("Do it.")
    assert "Minimized" in out2["reply"]
    assert get_pending_action() is None
    assert any(name == "minimize_window" for name, _ in calls)
