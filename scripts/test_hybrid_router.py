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
            "expect_route": "llm",
            "expect_tool": None,
            "expect_llm_calls": 1,
        },
        {
            "text": "what time is it and open spotify",
            "expect_route": "llm",
            "expect_tool": None,
            "expect_llm_calls": 1,
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
