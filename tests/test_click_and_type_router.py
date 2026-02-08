"""
Tests for the click-and-type pattern in the hybrid router.

Verifies that:
    - "click on ask anything and type hello" matches __CLICK_AND_TYPE__
    - "click ask anything and type hello world" matches
    - "press search box and type test" matches
    - "click maximize" does NOT match click_and_type (no "and type")
    - "click the button" does NOT match
    - "type hello" does NOT match
"""

from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wyzer.core.hybrid_router import decide, _match_click_and_type


class TestClickAndTypeRouter(unittest.TestCase):
    def test_basic_match(self):
        d = _match_click_and_type("click on ask anything and type hello")
        self.assertIsNotNone(d)
        self.assertEqual(d.mode, "tool_plan")
        self.assertEqual(d.intents[0]["tool"], "__CLICK_AND_TYPE__")
        self.assertEqual(d.intents[0]["args"]["target"], "ask anything")
        self.assertEqual(d.intents[0]["args"]["text"], "hello")

    def test_no_on(self):
        d = _match_click_and_type("click ask anything and type hello world")
        self.assertIsNotNone(d)
        self.assertEqual(d.intents[0]["args"]["target"], "ask anything")
        self.assertEqual(d.intents[0]["args"]["text"], "hello world")

    def test_press_variant(self):
        d = _match_click_and_type("press the search box and type test query")
        self.assertIsNotNone(d)
        self.assertEqual(d.intents[0]["args"]["target"], "search box")
        self.assertEqual(d.intents[0]["args"]["text"], "test query")

    def test_hit_variant(self):
        d = _match_click_and_type("hit the input field and type something")
        self.assertIsNotNone(d)
        self.assertEqual(d.intents[0]["args"]["target"], "input field")

    def test_no_match_plain_click(self):
        d = _match_click_and_type("click the button")
        self.assertIsNone(d)

    def test_no_match_type_only(self):
        d = _match_click_and_type("type hello world")
        self.assertIsNone(d)

    def test_no_match_maximize(self):
        d = _match_click_and_type("click maximize")
        self.assertIsNone(d)

    def test_decide_routes_to_click_and_type(self):
        """Full decide() function routes click-and-type correctly."""
        d = decide("click on ask anything and type hello")
        self.assertEqual(d.mode, "tool_plan")
        self.assertEqual(d.intents[0]["tool"], "__CLICK_AND_TYPE__")
        self.assertGreaterEqual(d.confidence, 0.95)

    def test_decide_still_routes_plain_click(self):
        """Plain 'click X' now routes through __CLICK_AND_TYPE__ with text=''."""
        d = decide("click the submit button")
        self.assertEqual(d.mode, "tool_plan")
        self.assertEqual(d.intents[0]["tool"], "__CLICK_AND_TYPE__")
        self.assertEqual(d.intents[0]["args"]["text"], "")

    def test_case_insensitive(self):
        d = _match_click_and_type("CLICK ON Ask Anything AND TYPE Hello")
        self.assertIsNotNone(d)
        self.assertEqual(d.intents[0]["args"]["target"], "Ask Anything")
        self.assertEqual(d.intents[0]["args"]["text"], "Hello")

    def test_trailing_punctuation_stripped(self):
        d = _match_click_and_type("click on search and type hello.")
        self.assertIsNotNone(d)
        self.assertEqual(d.intents[0]["args"]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
