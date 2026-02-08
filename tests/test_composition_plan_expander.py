import pytest

from wyzer.core.plan_expander import (
    MAX_COMPOSED_TOOL_CALLS,
    expand_foreach,
    validate_plan,
)
from wyzer.tools.registry import ToolRegistry
from wyzer.tools.tool_base import ToolBase


class _FakeTool(ToolBase):
    def __init__(self, name: str, schema: dict):
        super().__init__()
        self._name = name
        self._description = "fake"
        self._args_schema = schema

    def run(self, **kwargs):
        return {"ok": True, "args": kwargs}


def _registry_with(*tools: ToolBase) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def test_validate_plan_rejects_unknown_tool():
    reg = _registry_with(_FakeTool("get_time", {"type": "object", "properties": {}, "required": [], "additionalProperties": False}))
    ok, msg = validate_plan({"intents": [{"tool": "nope", "args": {}}]}, reg)
    assert ok is False
    assert "unknown tool" in msg.lower()


def test_foreach_expands_with_allowed_templates():
    reg = _registry_with(
        _FakeTool("list_open_windows", {"type": "object", "properties": {}, "required": [], "additionalProperties": False}),
        _FakeTool(
            "minimize_window",
            {
                "type": "object",
                "properties": {"hwnd": {"type": "integer"}},
                "required": ["hwnd"],
                "additionalProperties": False,
            },
        ),
    )

    plan = {
        "intents": [
            {"tool": "list_open_windows", "args": {}, "save_as": "windows"},
            {"foreach": "windows", "do": {"tool": "minimize_window", "args": {"hwnd": "{{item.hwnd}}"}}},
        ]
    }

    ok, msg = validate_plan(plan, reg)
    assert ok is True, msg

    saved = {
        "windows": {
            "windows": [
                {"id": 101, "hwnd": 101, "title": "A", "app": "chrome.exe"},
                {"id": 202, "hwnd": 202, "title": "B", "app": "notepad.exe"},
            ]
        }
    }

    expanded, stopped_early = expand_foreach(plan, saved, max_calls=MAX_COMPOSED_TOOL_CALLS)
    assert stopped_early is False
    # Includes the initial non-foreach intent
    assert expanded[0]["tool"] == "list_open_windows"
    assert expanded[1]["tool"] == "minimize_window"
    assert expanded[1]["args"]["hwnd"] == 101
    assert expanded[2]["args"]["hwnd"] == 202


def test_max_tool_call_limit_enforced():
    plan = {
        "intents": [
            {"tool": "list_open_windows", "args": {}, "save_as": "windows"},
            {"foreach": "windows", "do": {"tool": "minimize_window", "args": {"hwnd": "{{item.hwnd}}"}}},
        ]
    }

    saved = {
        "windows": {
            "windows": [{"id": i, "hwnd": i, "title": str(i), "app": "x"} for i in range(100)]
        }
    }

    expanded, stopped_early = expand_foreach(plan, saved, max_calls=25)
    assert stopped_early is True
    assert len(expanded) == 25
