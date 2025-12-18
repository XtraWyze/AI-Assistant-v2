"""Tests for the google_search_open tool and hybrid router integration.

Run:
  python scripts/test_google_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch


# Allow running as a plain script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_regex_patterns():
    """Test that the regex patterns correctly extract queries."""
    from wyzer.core.hybrid_router import _GOOGLE_SEARCH_RE, _SEARCH_GOOGLE_RE
    
    print("Testing regex patterns...")
    
    # Test _GOOGLE_SEARCH_RE
    test_cases_google = [
        ("google cats", "cats"),
        ("Google cats", "cats"),  # Case insensitive
        ("GOOGLE cats", "cats"),
        ("google this cats", "cats"),
        ("google this: cats", "cats"),
        ("google this cute cats", "cute cats"),
        ("google how to make pasta", "how to make pasta"),
        ("google   multiple   spaces", "multiple   spaces"),  # Whitespace in query preserved
    ]
    
    for text, expected_query in test_cases_google:
        m = _GOOGLE_SEARCH_RE.match(text)
        if not m:
            _fail(f"_GOOGLE_SEARCH_RE did not match: {text!r}")
        extracted = m.group("q").strip()
        if extracted != expected_query:
            _fail(f"_GOOGLE_SEARCH_RE extracted {extracted!r}, expected {expected_query!r} from {text!r}")
        print(f"  ✓ '{text}' -> '{extracted}'")
    
    # Test _SEARCH_GOOGLE_RE
    test_cases_search = [
        ("search google for cats", "cats"),
        ("Search google for cats", "cats"),  # Case insensitive
        ("search google for cute cats", "cute cats"),
        ("search google for how to fix my computer", "how to fix my computer"),
    ]
    
    for text, expected_query in test_cases_search:
        m = _SEARCH_GOOGLE_RE.match(text)
        if not m:
            _fail(f"_SEARCH_GOOGLE_RE did not match: {text!r}")
        extracted = m.group("q").strip()
        if extracted != expected_query:
            _fail(f"_SEARCH_GOOGLE_RE extracted {extracted!r}, expected {expected_query!r} from {text!r}")
        print(f"  ✓ '{text}' -> '{extracted}'")
    
    # Test non-matches (should NOT match)
    non_matches = [
        "google",  # No query
        "open google",  # Different command
        "googles cats",  # Wrong verb
        "search cats",  # Missing "google for"
    ]
    
    for text in non_matches:
        m = _GOOGLE_SEARCH_RE.match(text)
        if m and m.group("q").strip():
            _fail(f"_GOOGLE_SEARCH_RE should NOT match: {text!r}, but got: {m.group('q')!r}")
        print(f"  ✓ '{text}' correctly not matched")
    
    print("All regex tests passed!\n")


def test_tool_url_formation():
    """Test that the tool correctly forms Google search URLs."""
    from wyzer.tools.google_search_open import GoogleSearchOpenTool
    
    print("Testing URL formation...")
    
    tool = GoogleSearchOpenTool()
    
    # Test basic properties
    assert tool.name == "google_search_open", f"Tool name should be 'google_search_open', got {tool.name}"
    print(f"  ✓ Tool name: {tool.name}")
    
    # Test URL encoding with mock browser open
    test_cases = [
        ("cats", "https://www.google.com/search?q=cats"),
        ("cute cats", "https://www.google.com/search?q=cute+cats"),
        ("how to cook pasta", "https://www.google.com/search?q=how+to+cook+pasta"),
        ("C++ programming", "https://www.google.com/search?q=C%2B%2B+programming"),
        ("what is 2+2?", "https://www.google.com/search?q=what+is+2%2B2%3F"),
    ]
    
    for query, expected_url in test_cases:
        with patch("webbrowser.open") as mock_open:
            result = tool.run(query=query)
            
            assert "ok" in result and result["ok"] is True, f"Expected ok=True, got {result}"
            assert "url" in result, f"Expected 'url' in result, got {result}"
            assert result["url"] == expected_url, f"Expected URL {expected_url!r}, got {result['url']!r}"
            
            # Verify webbrowser.open was called with correct args
            mock_open.assert_called_once_with(expected_url, new=2)
            print(f"  ✓ '{query}' -> {result['url']}")
    
    # Test empty query handling
    result = tool.run(query="")
    assert "error" in result, "Empty query should return error"
    print("  ✓ Empty query correctly returns error")
    
    result = tool.run(query="   ")
    assert "error" in result, "Whitespace-only query should return error"
    print("  ✓ Whitespace query correctly returns error")
    
    print("All URL formation tests passed!\n")


def test_hybrid_router_integration():
    """Test that the hybrid router correctly routes Google search commands."""
    from wyzer.core.hybrid_router import decide, _decide_single_clause
    
    print("Testing hybrid router integration...")
    
    test_cases = [
        ("google cats", "google_search_open", "cats"),
        ("Google cats", "google_search_open", "cats"),
        ("google this cats", "google_search_open", "cats"),
        ("google how to make pasta", "google_search_open", "how to make pasta"),
        ("search google for cats", "google_search_open", "cats"),
    ]
    
    for text, expected_tool, expected_query in test_cases:
        decision = decide(text)
        
        if decision.mode != "tool_plan":
            _fail(f"Expected mode 'tool_plan' for {text!r}, got {decision.mode!r}")
        
        if not decision.intents or len(decision.intents) == 0:
            _fail(f"Expected intents for {text!r}, got none")
        
        intent = decision.intents[0]
        if intent["tool"] != expected_tool:
            _fail(f"Expected tool {expected_tool!r} for {text!r}, got {intent['tool']!r}")
        
        if intent["args"].get("query") != expected_query:
            _fail(f"Expected query {expected_query!r} for {text!r}, got {intent['args'].get('query')!r}")
        
        # Check spoken response
        if not decision.reply:
            _fail(f"Expected a reply for {text!r}, got empty")
        if "Opening Google for:" not in decision.reply:
            _fail(f"Expected 'Opening Google for:' in reply, got {decision.reply!r}")
        
        print(f"  ✓ '{text}' -> tool={intent['tool']}, query={intent['args']['query']!r}")
    
    # Test that LLM is NOT invoked (confidence should be high, mode should be tool_plan)
    for text, _, _ in test_cases:
        decision = decide(text)
        assert decision.mode == "tool_plan", f"Google search should NOT use LLM for {text!r}"
        assert decision.confidence >= 0.9, f"Confidence should be >= 0.9 for {text!r}, got {decision.confidence}"
    
    print("  ✓ All commands bypass LLM (confidence >= 0.9)")
    print("All hybrid router integration tests passed!\n")


def test_no_llm_invocation():
    """Test that Google search commands do NOT invoke the LLM via orchestrator."""
    from wyzer.core import orchestrator
    
    print("Testing that LLM is not invoked...")
    
    # --- Monkeypatch LLM calls (no network) ---
    llm_calls: List[Tuple[str, str]] = []

    def _stub_call_llm(user_text: str, registry) -> Dict[str, Any]:
        llm_calls.append(("_call_llm", user_text))
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
        return {"ok": True, "url": "https://www.google.com/search?q=test"}

    # Apply monkeypatches.
    orchestrator._call_llm = _stub_call_llm
    orchestrator._call_llm_reply_only = _stub_call_llm_reply_only
    orchestrator._call_llm_with_execution_summary = _stub_call_llm_with_execution_summary
    orchestrator._call_llm_for_explicit_tool = _stub_call_llm_for_explicit_tool
    orchestrator._execute_tool = _stub_execute_tool

    # Test cases
    test_inputs = [
        "google cats",
        "google this cute kittens",
        "search google for python tutorials",
    ]
    
    for text in test_inputs:
        tool_calls.clear()
        llm_calls.clear()
        
        result = orchestrator.handle_user_text(text)
        
        # Verify no LLM calls were made
        if llm_calls:
            _fail(f"LLM was invoked for {text!r}: {llm_calls}")
        
        # Verify tool was called
        if not tool_calls:
            _fail(f"No tool was called for {text!r}")
        
        if tool_calls[0][0] != "google_search_open":
            _fail(f"Expected 'google_search_open' tool for {text!r}, got {tool_calls[0][0]!r}")
        
        print(f"  ✓ '{text}' -> No LLM calls, tool called: {tool_calls[0]}")
    
    print("All no-LLM tests passed!\n")


def main() -> int:
    print("=" * 60)
    print("Google Search Tool Tests")
    print("=" * 60 + "\n")
    
    try:
        test_regex_patterns()
        test_tool_url_formation()
        test_hybrid_router_integration()
        test_no_llm_invocation()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
