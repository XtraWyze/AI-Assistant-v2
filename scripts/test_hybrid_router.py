"""Runnable verification for Hybrid Tool Calls.

This script monkeypatches LLM/tool execution to avoid:
- opening apps
- making network calls

It asserts the chosen hybrid route via orchestrator response metadata.

Run:
  python scripts/test_hybrid_router.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Allow running as a plain script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def main() -> int:
    from wyzer.core import orchestrator

    # --- Monkeypatch LLM calls (no network) ---
    llm_calls: List[Tuple[str, str]] = []

    def _stub_call_llm(user_text: str, registry) -> Dict[str, Any]:
        llm_calls.append(("_call_llm", user_text))
        # Reply-only: no tool calls.
        return {"reply": "(stubbed LLM reply)", "confidence": 0.1}

    def _stub_call_llm_reply_only(user_text: str) -> Dict[str, Any]:
        llm_calls.append(("_call_llm_reply_only", user_text))
        return {"reply": "(stubbed reply-only)", "confidence": 0.1}

    def _stub_call_llm_with_execution_summary(user_text: str, execution_summary, registry) -> Dict[str, Any]:
        llm_calls.append(("_call_llm_with_execution_summary", user_text))
        return {"reply": "(stubbed LLM final reply)", "confidence": 0.1}

    def _stub_call_llm_for_explicit_tool(user_text: str, tool_name: str, registry) -> Dict[str, Any]:
        llm_calls.append(("_call_llm_for_explicit_tool", user_text))
        return {"reply": "(stubbed explicit-tool)", "confidence": 0.1, "intents": []}

    # --- Monkeypatch tool execution (no side effects) ---
    tool_calls: List[Tuple[str, Dict[str, Any]]] = []

    def _stub_execute_tool(registry, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool_calls.append((tool_name, dict(args or {})))
        return {"status": "ok", "tool": tool_name, "args": dict(args or {})}

    # Apply monkeypatches.
    orchestrator._call_llm = _stub_call_llm
    orchestrator._call_llm_reply_only = _stub_call_llm_reply_only
    orchestrator._call_llm_with_execution_summary = _stub_call_llm_with_execution_summary
    orchestrator._call_llm_for_explicit_tool = _stub_call_llm_for_explicit_tool
    orchestrator._execute_tool = _stub_execute_tool

    # Helper runner.
    def run_case(text: str) -> Dict[str, Any]:
        tool_calls.clear()
        llm_calls.clear()
        out = orchestrator.handle_user_text(text)
        if "meta" not in out or "hybrid_route" not in out["meta"]:
            _fail(f"Missing meta.hybrid_route for input: {text!r}. Got keys: {list(out.keys())}")
        return out

    # --- Test cases ---
    cases = [
        {
            "text": "what time is it",
            "expect_route": "tool_plan",
            "expect_tool": "get_time",
            "expect_llm_calls": 0,
        },
        {
            "text": "open spotify",
            "expect_route": "tool_plan",
            "expect_tool": "open_target",
            "expect_llm_calls": 0,
        },
        {
            "text": "open spotify and play lofi",
            "expect_route": "tool_plan",
            "expect_tool": "open_target",
            "expect_llm_calls": 0,
        },
        {
            "text": "what time is it and open spotify",
            "expect_route": "tool_plan",
            "expect_tool": "get_time",
            "expect_llm_calls": 0,
        },
        {
            "text": "open https://google.com",
            "expect_route": "llm",
            "expect_tool": None,
            "expect_llm_calls": 1,
        },
        {
            "text": "open something",
            "expect_route": "llm",
            "expect_tool": None,
            "expect_llm_calls": 1,
        },
        {
            "text": "volume 35",
            "expect_route": "tool_plan",
            "expect_tool": "volume_control",
            "expect_llm_calls": 0,
            "expect_args": {"scope": "master", "action": "set", "level": 35},
        },
        {
            "text": "sound down a bit",
            "expect_route": "tool_plan",
            "expect_tool": "volume_control",
            "expect_llm_calls": 0,
            "expect_args": {"scope": "master", "action": "change", "delta": -5},
        },
        {
            "text": "mute",
            "expect_route": "tool_plan",
            "expect_tool": "volume_control",
            "expect_llm_calls": 0,
            "expect_args": {"scope": "master", "action": "mute"},
        },
        {
            "text": "unmute",
            "expect_route": "tool_plan",
            "expect_tool": "volume_control",
            "expect_llm_calls": 0,
            "expect_args": {"scope": "master", "action": "unmute"},
        },
        {
            "text": "what is the volume",
            "expect_route": "tool_plan",
            "expect_tool": "volume_control",
            "expect_llm_calls": 0,
            "expect_args": {"scope": "master", "action": "get"},
        },
        {
            "text": "spotify volume 30",
            "expect_route": "tool_plan",
            "expect_tool": "volume_control",
            "expect_llm_calls": 0,
            "expect_args": {"scope": "app", "process": "spotify", "action": "set", "level": 30},
        },
        {
            "text": "mute discord",
            "expect_route": "tool_plan",
            "expect_tool": "volume_control",
            "expect_llm_calls": 0,
            "expect_args": {"scope": "app", "process": "discord", "action": "mute"},
        },
        {
            "text": "what is spotify volume",
            "expect_route": "tool_plan",
            "expect_tool": "volume_control",
            "expect_llm_calls": 0,
            "expect_args": {"scope": "app", "process": "spotify", "action": "get"},
        },
        {
            "text": "what's the weather",
            "expect_route": "tool_plan",
            "expect_tool": "get_weather_forecast",
            "expect_llm_calls": 0,
            "expect_args": {},
        },
        {
            "text": "weather in new york",
            "expect_route": "tool_plan",
            "expect_tool": "get_weather_forecast",
            "expect_llm_calls": 0,
            "expect_args": {"location": "new york"},
        },
        {
            "text": "temperature in london",
            "expect_route": "tool_plan",
            "expect_tool": "get_weather_forecast",
            "expect_llm_calls": 0,
            "expect_args": {"location": "london"},
        },
        {
            "text": "forecast for tomorrow",
            "expect_route": "tool_plan",
            "expect_tool": "get_weather_forecast",
            "expect_llm_calls": 0,
            "expect_args": {"day_offset": 1, "days": 2},
        },
        {
            "text": "is it cold outside",
            "expect_route": "tool_plan",
            "expect_tool": "get_weather_forecast",
            "expect_llm_calls": 0,
            "expect_args": {},
        },
        {
            "text": "will it rain",
            "expect_route": "tool_plan",
            "expect_tool": "get_weather_forecast",
            "expect_llm_calls": 0,
            "expect_args": {},
        },
        {
            "text": "minimize chrome",
            "expect_route": "tool_plan",
            "expect_tool": "minimize_window",
            "expect_llm_calls": 0,
            "expect_args": {"title": "chrome"},
        },
        {
            "text": "shrink spotify",
            "expect_route": "tool_plan",
            "expect_tool": "minimize_window",
            "expect_llm_calls": 0,
            "expect_args": {"title": "spotify"},
        },
        {
            "text": "maximize notepad",
            "expect_route": "tool_plan",
            "expect_tool": "maximize_window",
            "expect_llm_calls": 0,
            "expect_args": {"title": "notepad"},
        },
        {
            "text": "fullscreen chrome",
            "expect_route": "tool_plan",
            "expect_tool": "maximize_window",
            "expect_llm_calls": 0,
            "expect_args": {"title": "chrome"},
        },
        {
            "text": "expand spotify",
            "expect_route": "tool_plan",
            "expect_tool": "maximize_window",
            "expect_llm_calls": 0,
            "expect_args": {"title": "spotify"},
        },
        {
            "text": "full screen notepad",
            "expect_route": "tool_plan",
            "expect_tool": "maximize_window",
            "expect_llm_calls": 0,
            "expect_args": {"title": "notepad"},
        },
        {
            "text": "move chrome to monitor 2",
            "expect_route": "tool_plan",
            "expect_tool": "move_window_to_monitor",
            "expect_llm_calls": 0,
            "expect_args": {"title": "chrome", "monitor": "2"},
        },
        {
            "text": "send spotify to monitor 1",
            "expect_route": "tool_plan",
            "expect_tool": "move_window_to_monitor",
            "expect_llm_calls": 0,
            "expect_args": {"title": "spotify", "monitor": "1"},
        },
        {
            "text": "move notepad to monitor next",
            "expect_route": "tool_plan",
            "expect_tool": "move_window_to_monitor",
            "expect_llm_calls": 0,
            "expect_args": {"title": "notepad", "monitor": "next"},
        },
        # Multi-intent commands: Time + Weather
        {
            "text": "what time is it and what's the weather",
            "expect_route": "tool_plan",
            "expect_tool": "get_time",  # First tool in multi-intent
            "expect_llm_calls": 0,
        },
        {
            "text": "what time is it and weather in new york",
            "expect_route": "tool_plan",
            "expect_tool": "get_time",
            "expect_llm_calls": 0,
        },
        # Multi-intent: Weather + Window
        {
            "text": "what's the weather and close chrome",
            "expect_route": "tool_plan",
            "expect_tool": "get_weather_forecast",
            "expect_llm_calls": 0,
        },
        {
            "text": "weather in london and minimize spotify",
            "expect_route": "tool_plan",
            "expect_tool": "get_weather_forecast",
            "expect_llm_calls": 0,
        },
        # Multi-intent: Time + Window
        {
            "text": "what time is it and open spotify",
            "expect_route": "tool_plan",
            "expect_tool": "get_time",
            "expect_llm_calls": 0,
        },
        {
            "text": "what time is it and minimize chrome",
            "expect_route": "tool_plan",
            "expect_tool": "get_time",
            "expect_llm_calls": 0,
        },
        # Multi-intent: Window + Window
        {
            "text": "open spotify and close chrome",
            "expect_route": "tool_plan",
            "expect_tool": "open_target",
            "expect_llm_calls": 0,
        },
        {
            "text": "minimize chrome and maximize notepad",
            "expect_route": "tool_plan",
            "expect_tool": "minimize_window",
            "expect_llm_calls": 0,
        },
        {
            "text": "close discord and move chrome to monitor 2",
            "expect_route": "tool_plan",
            "expect_tool": "close_window",
            "expect_llm_calls": 0,
        },
        # Multi-intent: Three commands
        {
            "text": "what time is it, weather in paris, and open spotify",
            "expect_route": "tool_plan",
            "expect_tool": "get_time",
            "expect_llm_calls": 0,
        },
        {
            "text": "open chrome, minimize spotify, and move notepad to monitor 2",
            "expect_route": "tool_plan",
            "expect_tool": "open_target",
            "expect_llm_calls": 0,
        },
        # Multi-intent: Sequential execution
        {
            "text": "what time is it then weather",
            "expect_route": "tool_plan",
            "expect_tool": "get_time",
            "expect_llm_calls": 0,
        },
        {
            "text": "open spotify then maximize it",
            "expect_route": "tool_plan",
            "expect_tool": "open_target",
            "expect_llm_calls": 0,
        },
        # Multi-intent: Implicit (no explicit separators like "and", ",", "then")
        # These test verb boundary detection: "close chrome open spotify" -> 2 intents
        {
            "text": "close chrome open spotify",
            "expect_route": "tool_plan",
            "expect_tool": "close_window",
            "expect_llm_calls": 0,
        },
        {
            "text": "minimize chrome maximize notepad",
            "expect_route": "tool_plan",
            "expect_tool": "minimize_window",
            "expect_llm_calls": 0,
        },
        {
            "text": "open chrome close spotify",
            "expect_route": "tool_plan",
            "expect_tool": "open_target",
            "expect_llm_calls": 0,
        },
        {
            "text": "move chrome to monitor 2 open spotify",
            "expect_route": "tool_plan",
            "expect_tool": "move_window_to_monitor",
            "expect_llm_calls": 0,
        },
        {
            "text": "close all open settings",
            "expect_route": "tool_plan",
            "expect_tool": "close_window",
            "expect_llm_calls": 0,
        },
        {
            "text": "minimize window maximize screen",
            "expect_route": "tool_plan",
            "expect_tool": "minimize_window",
            "expect_llm_calls": 0,
        },
    ]

    print("Hybrid router verification:")

    for c in cases:
        text = c["text"]
        out = run_case(text)

        route = out["meta"].get("hybrid_route")
        if route != c["expect_route"]:
            _fail(f"Route mismatch for {text!r}: expected {c['expect_route']!r}, got {route!r}")

        if c["expect_llm_calls"] is not None:
            if len(llm_calls) != c["expect_llm_calls"]:
                _fail(
                    f"LLM call count mismatch for {text!r}: expected {c['expect_llm_calls']}, got {len(llm_calls)} ({llm_calls})"
                )

        if c["expect_tool"]:
            if not tool_calls:
                _fail(f"Expected tool execution for {text!r}, but no tool calls happened")
            first_tool = tool_calls[0][0]
            if first_tool != c["expect_tool"]:
                _fail(f"Tool mismatch for {text!r}: expected {c['expect_tool']!r}, got {first_tool!r}")

            if "expect_args" in c and c["expect_args"] is not None:
                got_args = tool_calls[0][1]
                for k, v in (c["expect_args"] or {}).items():
                    if got_args.get(k) != v:
                        _fail(
                            f"Arg mismatch for {text!r}: expected {k}={v!r}, got {k}={got_args.get(k)!r}. Full args: {got_args}"
                        )

        print(
            f"- OK: {text!r} -> route={route}, llm_calls={len(llm_calls)}, tool_calls={[t for t,_ in tool_calls]}"
        )

    print("All hybrid routing checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as e:
        print(f"FAILED: {e}")
        raise
