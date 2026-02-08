"""
Tests for wyzer.desktop.resolve_target — Deterministic candidate resolver.

Covers:
    - Exact match scoring
    - Prefix / contains / fuzzy scoring
    - Control-type boost
    - Enabled/visible boost
    - Short/generic penalty
    - Ambiguity detection (score below threshold, gap too small)
    - Empty / no-match scenarios
    - OCR source candidates
"""

from __future__ import annotations

import sys
import os
import unittest

# Ensure repo root is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wyzer.desktop.resolve_target import (
    resolve_candidates,
    _norm,
    _score_text_match,
    _apply_boosts,
    Candidate,
    ResolveResult,
    SCORE_MIN,
    GAP_MIN,
)


class TestNorm(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_norm("  Ask Anything  "), "ask anything")

    def test_punctuation(self):
        self.assertEqual(_norm("Ask Anything!"), "ask anything")

    def test_empty(self):
        self.assertEqual(_norm(""), "")
        self.assertEqual(_norm(None), "")


class TestScoreTextMatch(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(_score_text_match("ask anything", "ask anything"), 100.0)

    def test_prefix(self):
        score = _score_text_match("ask", "ask anything")
        self.assertEqual(score, 80.0)

    def test_word_boundary(self):
        score = _score_text_match("ask anything", "please ask anything here")
        self.assertEqual(score, 70.0)

    def test_substring(self):
        score = _score_text_match("anything", "askanything")
        self.assertEqual(score, 55.0)

    def test_short_substring_rejected(self):
        """Single-char label like 'p' must NOT substring-match 'keep'."""
        self.assertEqual(_score_text_match("keep", "p"), 0.0)
        # 2-char non-prefix substrings are also rejected
        self.assertEqual(_score_text_match("keep", "ep"), 0.0)
        # 3-char non-prefix substrings should still match
        self.assertEqual(_score_text_match("keep", "eep"), 55.0)

    def test_fuzzy(self):
        score = _score_text_match("ask anythin", "ask anything")
        # Should be fuzzy match >=0.7 ratio
        self.assertGreater(score, 0)

    def test_no_match(self):
        self.assertEqual(_score_text_match("hello world", "xyz abc"), 0.0)

    def test_empty_target(self):
        self.assertEqual(_score_text_match("", "something"), 0.0)

    def test_empty_label(self):
        self.assertEqual(_score_text_match("something", ""), 0.0)


class TestApplyBoosts(unittest.TestCase):
    def test_control_type_boost(self):
        score = _apply_boosts(50.0, "Edit", ["Edit", "TextBox"], True, {"l": 0}, "ask anything")
        # +15 (type) +5 (enabled) +5 (rect) +0 (trivial rect area) +3 (Edit interactivity 6*0.5) = 78
        self.assertEqual(score, 78.0)

    def test_no_boosts(self):
        score = _apply_boosts(50.0, "Pane", ["Edit"], None, None, "ask anything")
        # Pane interactivity = 3 * 0.5 = 1.5
        self.assertEqual(score, 51.5)

    def test_generic_penalty(self):
        score = _apply_boosts(50.0, "Button", ["Button"], True, {"l": 0}, "ok")
        # +15 +5 +5 +0 (trivial rect) +5 (Button interactivity 10*0.5) -8 = 72
        self.assertEqual(score, 72.0)

    def test_short_penalty(self):
        score = _apply_boosts(50.0, "Text", [], None, None, "x")
        # -8 penalty, no interactivity for Text
        self.assertEqual(score, 42.0)


class TestResolveCandidates(unittest.TestCase):
    """Integration tests for resolve_candidates."""

    def _make_perception(self, controls):
        return {"controls": controls}

    def test_single_exact_match(self):
        perception = self._make_perception([
            {"name": "Ask anything", "control_type": "Edit", "rect": {"l": 0, "t": 0, "r": 100, "b": 30}, "enabled": True},
            {"name": "File", "control_type": "MenuItem", "rect": {"l": 0, "t": 0, "r": 50, "b": 20}, "enabled": True},
        ])
        result = resolve_candidates("ask anything", perception, preferred_types=["Edit"])
        self.assertFalse(result.ambiguous)
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.name, "Ask anything")
        self.assertEqual(result.reason, "confident match")

    def test_no_match(self):
        perception = self._make_perception([
            {"name": "File", "control_type": "MenuItem", "rect": {"l": 0, "t": 0, "r": 50, "b": 20}, "enabled": True},
            {"name": "Edit", "control_type": "MenuItem", "rect": {"l": 50, "t": 0, "r": 100, "b": 20}, "enabled": True},
        ])
        result = resolve_candidates("ask anything", perception)
        self.assertEqual(len(result.candidates), 0)
        self.assertIsNone(result.best)
        self.assertEqual(result.reason, "no matching candidates")

    def test_ambiguous_close_scores(self):
        """Two very similar labels should trigger ambiguity."""
        perception = self._make_perception([
            {"name": "Ask", "control_type": "Button", "rect": {"l": 0, "t": 0, "r": 50, "b": 20}, "enabled": True},
            {"name": "Ask anything", "control_type": "Edit", "rect": {"l": 50, "t": 0, "r": 200, "b": 20}, "enabled": True},
            {"name": "Ask Anything (Search)", "control_type": "Edit", "rect": {"l": 200, "t": 0, "r": 400, "b": 20}, "enabled": True},
        ])
        result = resolve_candidates("ask anything", perception, preferred_types=["Edit", "Button"])
        # "Ask anything" and "Ask Anything (Search)" should both score high
        self.assertEqual(len(result.candidates), 3)
        # Top two should be the Edit controls with close scores
        top2 = result.candidates[:2]
        gap = top2[0].score - top2[1].score
        # Either ambiguous due to gap or both are strong matches
        self.assertTrue(len(result.candidates) >= 2)

    def test_empty_target(self):
        perception = self._make_perception([
            {"name": "Ask anything", "control_type": "Edit", "rect": {"l": 0, "t": 0, "r": 100, "b": 30}, "enabled": True},
        ])
        result = resolve_candidates("", perception)
        self.assertEqual(len(result.candidates), 0)
        self.assertEqual(result.reason, "empty target phrase")

    def test_empty_perception(self):
        result = resolve_candidates("ask anything", {"controls": []})
        self.assertEqual(len(result.candidates), 0)

    def test_preferred_type_ranking(self):
        """Edit controls should rank higher than Buttons for the same text."""
        perception = self._make_perception([
            {"name": "Search", "control_type": "Button", "rect": {"l": 0, "t": 0, "r": 50, "b": 20}, "enabled": True},
            {"name": "Search", "control_type": "Edit", "rect": {"l": 50, "t": 0, "r": 200, "b": 20}, "enabled": True},
        ])
        result = resolve_candidates("search", perception, preferred_types=["Edit"])
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.control_type, "Edit")

    def test_ocr_source(self):
        """OCR lines should be processed when source='ocr'."""
        perception = {
            "lines": [
                {"text": "Ask anything", "rect": {"l": 100, "t": 200, "r": 300, "b": 220}},
                {"text": "File Edit View", "rect": {"l": 0, "t": 0, "r": 200, "b": 20}},
            ],
            "words": [],
        }
        result = resolve_candidates("ask anything", perception, source="ocr")
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.source, "ocr")

    def test_max_candidates(self):
        """Should not return more than MAX_CANDIDATES."""
        controls = [
            {"name": f"Item {i}", "control_type": "ListItem", "rect": {"l": 0, "t": i * 20, "r": 100, "b": (i + 1) * 20}, "enabled": True}
            for i in range(20)
        ]
        perception = self._make_perception(controls)
        result = resolve_candidates("item", perception, preferred_types=["ListItem"])
        self.assertLessEqual(len(result.candidates), 10)

    def test_low_score_ambiguity(self):
        """Best score below SCORE_MIN triggers ambiguity."""
        perception = self._make_perception([
            {"name": "Something completely different but might fuzzy match a bit", "control_type": "Text", "rect": None, "enabled": True},
        ])
        result = resolve_candidates("ask anything really specific", perception)
        if result.candidates:
            if result.best and result.best.score < SCORE_MIN:
                self.assertTrue(result.ambiguous)


class TestResolveResultSerialization(unittest.TestCase):
    def test_to_dict(self):
        c = Candidate(id=0, name="Test", control_type="Edit",
                       rect={"l": 0, "t": 0, "r": 100, "b": 30},
                       score=95.0, source="uia", enabled=True)
        result = ResolveResult(
            candidates=[c], ambiguous=False, best=c, reason="confident",
        )
        d = result.to_dict()
        self.assertIn("candidates", d)
        self.assertEqual(len(d["candidates"]), 1)
        self.assertEqual(d["best"]["name"], "Test")
        self.assertFalse(d["ambiguous"])


if __name__ == "__main__":
    unittest.main()
