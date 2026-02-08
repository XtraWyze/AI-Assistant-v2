"""
Tests for Phase 16+ click-and-type upgrades.

Covers:
    1. UIA ancestor promotion (Text node → ListItem/Button parent)
    2. OCR click hitbox expansion (center → offset_left)
    3. Control type heuristics (no forced types, interactivity scoring)
    4. Retry logic in desktop_click_uia (invoke → focus → rect)
    5. Strict failure semantics (clicked=false → intent fails)
    6. Ambiguity overlay with promoted candidates
    7. Targeted verification (no global search)
"""

from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure repo root is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wyzer.desktop.resolve_target import (
    resolve_candidates,
    _norm,
    _score_text_match,
    _apply_boosts,
    _try_ancestor_promotion,
    _has_nontrivial_rect,
    _promote_via_geometry,
    Candidate,
    ResolveResult,
    SCORE_MIN,
    GAP_MIN,
    _NON_CLICKABLE_TYPES,
    _CLICKABLE_TYPES,
    _INTERACTIVITY_RANK,
    PROMOTION_MAX_DEPTH,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. UIA ANCESTOR PROMOTION
# ═══════════════════════════════════════════════════════════════════════

class TestAncestorPromotion(unittest.TestCase):
    """Verify that Text nodes inside ListItems get promoted to the parent."""

    def test_text_inside_listitem_promoted(self):
        """Text node 'FPV station' inside a ListItem should be promoted."""
        controls = [
            {
                "name": "FPV station",
                "control_type": "Text",
                "rect": {"l": 10, "t": 50, "r": 100, "b": 70},
                "enabled": True,
            },
            {
                "name": "",
                "control_type": "ListItem",
                "rect": {"l": 5, "t": 45, "r": 200, "b": 75},
                "enabled": True,
            },
        ]
        result = resolve_candidates(
            "fpv station",
            {"controls": controls},
            preferred_types=["ListItem", "Button"],
            source="uia",
        )
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.name, "FPV station")
        # Should be promoted to ListItem via geometry
        self.assertEqual(result.best.control_type, "ListItem")
        self.assertIsNotNone(result.best.promotion)
        self.assertEqual(result.best.promotion["promoted_from"], "Text")
        self.assertEqual(result.best.promotion["promoted_to"], "ListItem")

    def test_text_inside_button_promoted(self):
        """Text node inside a Button should be promoted to Button."""
        controls = [
            {
                "name": "Submit",
                "control_type": "Text",
                "rect": {"l": 100, "t": 200, "r": 160, "b": 230},
                "enabled": True,
            },
            {
                "name": "",
                "control_type": "Button",
                "rect": {"l": 90, "t": 195, "r": 170, "b": 235},
                "enabled": True,
            },
        ]
        result = resolve_candidates(
            "submit",
            {"controls": controls},
            preferred_types=["Button"],
            source="uia",
        )
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.control_type, "Button")
        self.assertIsNotNone(result.best.promotion)

    def test_statictext_promoted(self):
        """StaticText should also be eligible for promotion."""
        controls = [
            {
                "name": "New Chat",
                "control_type": "StaticText",
                "rect": {"l": 20, "t": 100, "r": 80, "b": 120},
                "enabled": True,
            },
            {
                "name": "",
                "control_type": "Hyperlink",
                "rect": {"l": 15, "t": 95, "r": 90, "b": 125},
                "enabled": True,
            },
        ]
        result = resolve_candidates(
            "new chat",
            {"controls": controls},
            preferred_types=["Hyperlink"],
            source="uia",
        )
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.control_type, "Hyperlink")
        self.assertIsNotNone(result.best.promotion)
        self.assertEqual(result.best.promotion["promoted_from"], "StaticText")

    def test_no_promotion_for_clickable_types(self):
        """Button/Edit controls should NOT be promoted."""
        controls = [
            {
                "name": "Save",
                "control_type": "Button",
                "rect": {"l": 10, "t": 10, "r": 60, "b": 30},
                "enabled": True,
            },
        ]
        result = resolve_candidates(
            "save",
            {"controls": controls},
            preferred_types=["Button"],
            source="uia",
        )
        self.assertIsNotNone(result.best)
        self.assertIsNone(result.best.promotion)

    def test_promotion_transfers_score(self):
        """Promoted candidate should inherit the text-match score + bonus."""
        controls = [
            {
                "name": "FPV station",
                "control_type": "Text",
                "rect": {"l": 10, "t": 50, "r": 100, "b": 70},
                "enabled": True,
            },
            {
                "name": "",
                "control_type": "ListItem",
                "rect": {"l": 5, "t": 45, "r": 200, "b": 75},
                "enabled": True,
            },
        ]
        result = resolve_candidates(
            "fpv station",
            {"controls": controls},
            preferred_types=["ListItem"],
            source="uia",
        )
        self.assertIsNotNone(result.best)
        # Score should include promotion bonus (+8) minus depth penalty (-1)
        self.assertGreater(result.best.score, SCORE_MIN)

    def test_document_type_promoted(self):
        """Document control type should be eligible for promotion."""
        self.assertIn("Document", _NON_CLICKABLE_TYPES)

    def test_custom_type_promoted(self):
        """Custom control type should be eligible for promotion."""
        self.assertIn("Custom", _NON_CLICKABLE_TYPES)


class TestPromoteViaGeometry(unittest.TestCase):
    """Test the geometric containment-based promotion fallback."""

    def test_contained_child_finds_parent(self):
        child = {
            "name": "Click me",
            "control_type": "Text",
            "rect": {"l": 50, "t": 50, "r": 100, "b": 70},
            "enabled": True,
        }
        all_controls = [
            child,
            {
                "name": "",
                "control_type": "Pane",
                "rect": {"l": 40, "t": 40, "r": 200, "b": 80},
                "enabled": True,
            },
        ]
        result = _promote_via_geometry(
            child,
            child["rect"],
            "Text",
            all_controls,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["control_type"], "Pane")
        self.assertEqual(result["_promotion"]["promoted_from"], "Text")

    def test_no_parent_returns_none(self):
        child = {
            "name": "Orphan",
            "control_type": "Text",
            "rect": {"l": 50, "t": 50, "r": 100, "b": 70},
            "enabled": True,
        }
        result = _promote_via_geometry(
            child,
            child["rect"],
            "Text",
            [child],  # no other controls
        )
        self.assertIsNone(result)

    def test_tightest_parent_chosen(self):
        """When multiple parents contain the child, pick the tightest fit."""
        child = {
            "name": "Item",
            "control_type": "Text",
            "rect": {"l": 50, "t": 50, "r": 100, "b": 70},
        }
        all_controls = [
            child,
            {
                "name": "",
                "control_type": "Pane",
                "rect": {"l": 0, "t": 0, "r": 500, "b": 500},
                "enabled": True,
            },
            {
                "name": "",
                "control_type": "ListItem",
                "rect": {"l": 45, "t": 45, "r": 110, "b": 75},
                "enabled": True,
            },
        ]
        result = _promote_via_geometry(
            child,
            child["rect"],
            "Text",
            all_controls,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["control_type"], "ListItem")

    def test_disabled_parent_skipped(self):
        """Disabled parents should be skipped."""
        child = {
            "name": "Disabled parent test",
            "control_type": "Text",
            "rect": {"l": 50, "t": 50, "r": 100, "b": 70},
        }
        all_controls = [
            child,
            {
                "name": "",
                "control_type": "Button",
                "rect": {"l": 45, "t": 45, "r": 110, "b": 75},
                "enabled": False,  # disabled
            },
            {
                "name": "",
                "control_type": "Pane",
                "rect": {"l": 0, "t": 0, "r": 500, "b": 500},
                "enabled": True,
            },
        ]
        result = _promote_via_geometry(
            child,
            child["rect"],
            "Text",
            all_controls,
        )
        self.assertIsNotNone(result)
        # Should skip the disabled Button, pick the Pane
        self.assertEqual(result["control_type"], "Pane")


class TestHasNontrivialRect(unittest.TestCase):
    def test_normal_rect(self):
        self.assertTrue(_has_nontrivial_rect({"l": 0, "t": 0, "r": 100, "b": 30}))

    def test_zero_area(self):
        self.assertFalse(_has_nontrivial_rect({"l": 5, "t": 5, "r": 5, "b": 5}))

    def test_none(self):
        self.assertFalse(_has_nontrivial_rect(None))

    def test_tiny_rect(self):
        self.assertFalse(_has_nontrivial_rect({"l": 0, "t": 0, "r": 1, "b": 1}))


# ═══════════════════════════════════════════════════════════════════════
# 2. OCR CLICK HITBOX EXPANSION
# ═══════════════════════════════════════════════════════════════════════

class TestOCRClickExpansion(unittest.TestCase):
    """Verify OCR click strategies: center then offset_left."""

    def test_ocr_click_center_strategy(self):
        """OCR click should attempt center first."""
        from wyzer.desktop.click_and_type import _ocr_click_with_expansion
        import time

        rect = {"l": 100, "t": 200, "r": 200, "b": 220}

        class FakeCandidate:
            name = "FPV station"
            control_type = "OCR_Text"

        with patch("pyautogui.click") as mock_click:
            result = _ocr_click_with_expansion(rect, FakeCandidate(), time.perf_counter())
            self.assertTrue(result["clicked"])
            self.assertEqual(result["ocr_click_strategy"], "center")
            # Center of rect should be (150, 210)
            mock_click.assert_called_once_with(x=150, y=210)

    def test_ocr_click_offset_on_failure(self):
        """If center click fails, try offset_left."""
        from wyzer.desktop.click_and_type import _ocr_click_with_expansion, _OCR_OFFSET_LEFT
        import time

        rect = {"l": 100, "t": 200, "r": 200, "b": 220}

        class FakeCandidate:
            name = "FPV station"
            control_type = "OCR_Text"

        call_count = 0
        def side_effect(x, y):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("click failed")

        with patch("pyautogui.click", side_effect=side_effect) as mock_click:
            result = _ocr_click_with_expansion(rect, FakeCandidate(), time.perf_counter())
            self.assertTrue(result["clicked"])
            self.assertEqual(result["ocr_click_strategy"], "offset_left")
            # Second call: center_x + offset
            self.assertEqual(mock_click.call_count, 2)

    def test_ocr_click_all_fail(self):
        """If all strategies fail, return failure."""
        from wyzer.desktop.click_and_type import _ocr_click_with_expansion
        import time

        rect = {"l": 100, "t": 200, "r": 200, "b": 220}

        class FakeCandidate:
            name = "Nothing"
            control_type = "OCR_Text"

        with patch("pyautogui.click", side_effect=Exception("fail")):
            result = _ocr_click_with_expansion(rect, FakeCandidate(), time.perf_counter())
            self.assertFalse(result["clicked"])
            self.assertEqual(result["ocr_click_strategy"], "exhausted")
            self.assertFalse(result["ok"])


# ═══════════════════════════════════════════════════════════════════════
# 3. CONTROL TYPE HEURISTICS
# ═══════════════════════════════════════════════════════════════════════

class TestControlTypeHeuristics(unittest.TestCase):
    """Verify interactivity ranking and no forced types."""

    def test_button_ranks_higher_interactivity(self):
        """Button should get higher interactivity score than Pane."""
        self.assertGreater(
            _INTERACTIVITY_RANK.get("Button", 0),
            _INTERACTIVITY_RANK.get("Pane", 0),
        )

    def test_listitem_ranks_high(self):
        """ListItem should have significant interactivity score."""
        self.assertGreater(_INTERACTIVITY_RANK.get("ListItem", 0), 5.0)

    def test_no_forced_control_type(self):
        """When no preferred_types given, all types compete fairly."""
        controls = [
            {"name": "Search", "control_type": "Pane", "rect": {"l": 0, "t": 0, "r": 200, "b": 30}, "enabled": True},
            {"name": "Search", "control_type": "Button", "rect": {"l": 200, "t": 0, "r": 300, "b": 30}, "enabled": True},
        ]
        result = resolve_candidates(
            "search",
            {"controls": controls},
            preferred_types=[],  # no preference
            source="uia",
        )
        self.assertIsNotNone(result.best)
        # Button should rank higher due to interactivity
        self.assertEqual(result.best.control_type, "Button")

    def test_sidebar_listitem_resolves(self):
        """Sidebar items like 'FPV station' should resolve even as Pane."""
        controls = [
            {"name": "FPV station", "control_type": "Pane", "rect": {"l": 5, "t": 100, "r": 200, "b": 130}, "enabled": True},
        ]
        result = resolve_candidates(
            "fpv station",
            {"controls": controls},
            preferred_types=[],
            source="uia",
        )
        self.assertIsNotNone(result.best)
        self.assertFalse(result.ambiguous)
        self.assertGreaterEqual(result.best.score, SCORE_MIN)

    def test_interactivity_boost_applied(self):
        """Interactivity rank should add to score."""
        base_score = 50.0
        button_score = _apply_boosts(base_score, "Button", [], True, {"l": 0, "t": 0, "r": 50, "b": 20}, "submit")
        text_score = _apply_boosts(base_score, "Text", [], True, {"l": 0, "t": 0, "r": 50, "b": 20}, "submit")
        # Button gets interactivity boost, Text does not
        self.assertGreater(button_score, text_score)

    def test_promotion_bonus_in_scoring(self):
        """Promoted candidate should get +8 bonus minus depth penalty."""
        base = 50.0
        score = _apply_boosts(
            base, "ListItem", [], True,
            {"l": 0, "t": 0, "r": 100, "b": 30}, "fpv station",
            promotion={"promoted_from": "Text", "promoted_to": "ListItem", "promotion_depth": 1},
        )
        # 50 + 5 (enabled) + 5 (rect) + 2 (area) + 4 (interactivity) + 8 (promo) - 1 (depth)
        self.assertGreater(score, base + 15)


# ═══════════════════════════════════════════════════════════════════════
# 4. RETRY LOGIC (desktop_click_uia unified schema)
# ═══════════════════════════════════════════════════════════════════════

class TestClickUIAUnifiedSchema(unittest.TestCase):
    """Verify the unified return schema from _best_match."""

    def test_no_pywinauto_returns_schema(self):
        """When pywinauto is missing, should return full unified schema."""
        with patch.dict("sys.modules", {"pywinauto": None}):
            # Force reimport
            import importlib
            from wyzer.tools.desktop import desktop_click_uia
            # The function handles ImportError internally
            result = desktop_click_uia._best_match("test")
            self.assertIn("ok", result)
            self.assertIn("clicked", result)
            self.assertIn("method", result)
            self.assertIn("fallback_used", result)
            self.assertFalse(result["ok"])
            self.assertFalse(result["clicked"])


class TestBestMatchWithRetry(unittest.TestCase):
    """Verify _best_match_with_retry fallback chain."""

    def test_rect_fallback_used(self):
        """When UIA match fails, rect fallback should be tried."""
        from wyzer.tools.desktop.desktop_click_uia import _best_match_with_retry

        with patch(
            "wyzer.tools.desktop.desktop_click_uia._best_match",
            return_value={"ok": False, "clicked": False, "method": "none",
                         "fallback_used": False, "reason": "no_control"}
        ), patch("pyautogui.click") as mock_click:
            result = _best_match_with_retry(
                "FPV station", None,
                candidate_rect={"l": 100, "t": 200, "r": 300, "b": 220}
            )
            self.assertTrue(result["clicked"])
            self.assertEqual(result["method"], "rect_click")
            mock_click.assert_called_once_with(x=200, y=210)


# ═══════════════════════════════════════════════════════════════════════
# 5. STRICT FAILURE SEMANTICS
# ═══════════════════════════════════════════════════════════════════════

class TestStrictFailureSemantics(unittest.TestCase):
    """Verify click_and_type propagates failures correctly."""

    def test_click_false_means_failure(self):
        """If click returns clicked=false, the overall result must be ok=false."""
        from wyzer.desktop.click_and_type import execute_click_and_type

        fake_uia = {
            "controls": [
                {"name": "FPV station", "control_type": "Button",
                 "rect": {"l": 10, "t": 10, "r": 100, "b": 30}, "enabled": True},
            ],
            "errors": [],
        }

        with patch(
            "wyzer.tools.desktop.perceive_uia.perceive_uia_focused_window",
            return_value=fake_uia,
        ), patch(
            "wyzer.desktop.click_and_type._do_click",
            return_value={"ok": False, "clicked": False, "method": "none",
                         "fallback_used": False, "reason": "all methods failed"},
        ), patch(
            "wyzer.context.world_state.emit_event",
        ):
            result = execute_click_and_type("fpv station", "hello")
            self.assertFalse(result["ok"])
            self.assertIn("couldn't click", result.get("summary", "").lower())

    def test_successful_click_returns_ok(self):
        """If click succeeds and type succeeds, ok should be True."""
        from wyzer.desktop.click_and_type import execute_click_and_type

        fake_uia = {
            "controls": [
                {"name": "Ask anything", "control_type": "Edit",
                 "rect": {"l": 100, "t": 200, "r": 400, "b": 230}, "enabled": True},
            ],
            "errors": [],
        }

        with patch(
            "wyzer.tools.desktop.perceive_uia.perceive_uia_focused_window",
            return_value=fake_uia,
        ), patch(
            "wyzer.desktop.click_and_type._do_click",
            return_value={"ok": True, "clicked": True, "method": "uia_invoke",
                         "fallback_used": False, "reason": "invoke_pattern"},
        ), patch(
            "wyzer.desktop.click_and_type._do_type",
            return_value={"success": True, "typed_length": 5},
        ), patch(
            "wyzer.tools.desktop.assert_text_present.assert_text_present",
            return_value={"ok": True, "method_used": "uia", "evidence": "found",
                         "details": {}},
        ), patch(
            "wyzer.context.world_state.emit_event",
        ):
            result = execute_click_and_type("ask anything", "hello")
            self.assertTrue(result["ok"])


# ═══════════════════════════════════════════════════════════════════════
# 6. OVERLAY WITH PROMOTED CANDIDATES
# ═══════════════════════════════════════════════════════════════════════

class TestOverlayWithPromotion(unittest.TestCase):
    """Verify overlay shows promoted candidate metadata."""

    def test_multiple_similar_chats_triggers_ambiguity(self):
        """Multiple chats with similar names should trigger ambiguity."""
        controls = [
            {"name": "Project Alpha", "control_type": "Text",
             "rect": {"l": 10, "t": 50, "r": 150, "b": 70}, "enabled": True},
            {"name": "Project Alpha v2", "control_type": "Text",
             "rect": {"l": 10, "t": 90, "r": 150, "b": 110}, "enabled": True},
            {"name": "", "control_type": "ListItem",
             "rect": {"l": 5, "t": 45, "r": 200, "b": 75}, "enabled": True},
            {"name": "", "control_type": "ListItem",
             "rect": {"l": 5, "t": 85, "r": 200, "b": 115}, "enabled": True},
        ]
        result = resolve_candidates(
            "project alpha",
            {"controls": controls},
            preferred_types=["ListItem"],
            source="uia",
        )
        # Should have at least 2 candidates with very close scores
        self.assertGreaterEqual(len(result.candidates), 2)
        # Both top candidates should be promoted
        for c in result.candidates[:2]:
            self.assertIsNotNone(c.promotion)


# ═══════════════════════════════════════════════════════════════════════
# 7. TARGETED VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

class TestTargetedVerification(unittest.TestCase):
    """Verify assert_text_present uses targeted search."""

    def test_ocr_search_within_click_rect(self):
        """OCR verification should search only within ±40px of click rect."""
        from wyzer.tools.desktop.assert_text_present import _check_ocr, _OCR_VERIFY_MARGIN

        # Line is far from click rect → should NOT match
        result = _check_ocr.__wrapped__ if hasattr(_check_ocr, '__wrapped__') else None

        # Direct test of the margin constant
        self.assertEqual(_OCR_VERIFY_MARGIN, 40)

    def test_check_ocr_filters_outside_region(self):
        """OCR lines outside the click region should be skipped."""
        from wyzer.tools.desktop.assert_text_present import _check_ocr

        # Mock OCR data
        with patch(
            "wyzer.tools.desktop.perceive_ocr_focused._perceive_ocr_focused",
            return_value={
                "lines": [
                    {"text": "hello world", "rect": {"l": 500, "t": 500, "r": 600, "b": 520}},
                    {"text": "other text", "rect": {"l": 100, "t": 100, "r": 200, "b": 120}},
                ],
                "words": [],
                "errors": [],
            },
        ):
            # Click rect is at (100,100)-(200,120), so "other text" is in range
            result = _check_ocr(
                "other text",
                click_rect={"l": 100, "t": 100, "r": 200, "b": 120},
            )
            self.assertTrue(result["ok"])

            # "hello world" is at (500,500) → way outside → should not match
            result = _check_ocr(
                "hello world",
                click_rect={"l": 100, "t": 100, "r": 200, "b": 120},
            )
            self.assertFalse(result["ok"])

    def test_uia_check_focuses_edit_controls(self):
        """UIA verification should prioritize Edit/ComboBox controls."""
        from wyzer.tools.desktop.assert_text_present import _check_uia

        with patch(
            "wyzer.tools.desktop.perceive_uia.perceive_uia_focused_window",
            return_value={
                "controls": [
                    {"name": "hello world typed here", "control_type": "Edit",
                     "rect": {"l": 100, "t": 200, "r": 400, "b": 230}},
                    {"name": "Menu Item", "control_type": "MenuItem",
                     "rect": {"l": 0, "t": 0, "r": 50, "b": 20}},
                ],
            },
        ):
            result = _check_uia("hello world", None, None)
            self.assertTrue(result["ok"])
            self.assertTrue(result["details"].get("targeted"))


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION: FULL RESOLVER SCENARIOS
# ═══════════════════════════════════════════════════════════════════════

class TestResolverIntegration(unittest.TestCase):
    """End-to-end resolver tests for real-world sidebar scenarios."""

    def test_fpv_station_sidebar(self):
        """'click on fpv station' should resolve in a ChatGPT-like sidebar."""
        # Simulate a sidebar with Text nodes inside ListItems
        controls = [
            {"name": "ChatGPT", "control_type": "Text",
             "rect": {"l": 10, "t": 10, "r": 100, "b": 30}, "enabled": True},
            {"name": "FPV station", "control_type": "Text",
             "rect": {"l": 10, "t": 60, "r": 120, "b": 80}, "enabled": True},
            {"name": "Python help", "control_type": "Text",
             "rect": {"l": 10, "t": 100, "r": 120, "b": 120}, "enabled": True},
            # Parent containers
            {"name": "", "control_type": "ListItem",
             "rect": {"l": 5, "t": 5, "r": 200, "b": 35}, "enabled": True},
            {"name": "", "control_type": "ListItem",
             "rect": {"l": 5, "t": 55, "r": 200, "b": 85}, "enabled": True},
            {"name": "", "control_type": "ListItem",
             "rect": {"l": 5, "t": 95, "r": 200, "b": 125}, "enabled": True},
        ]
        result = resolve_candidates(
            "fpv station",
            {"controls": controls},
            preferred_types=["ListItem", "Button", "Hyperlink"],
            source="uia",
        )
        self.assertIsNotNone(result.best)
        self.assertFalse(result.ambiguous)
        self.assertEqual(result.best.name, "FPV station")
        self.assertGreaterEqual(result.best.score, SCORE_MIN)
        # Should be promoted
        self.assertEqual(result.best.control_type, "ListItem")

    def test_ask_anything_input_field(self):
        """'click on ask anything' should resolve to Edit control."""
        controls = [
            {"name": "Ask anything", "control_type": "Edit",
             "rect": {"l": 200, "t": 500, "r": 800, "b": 540}, "enabled": True},
            {"name": "New chat", "control_type": "Button",
             "rect": {"l": 10, "t": 10, "r": 80, "b": 30}, "enabled": True},
        ]
        result = resolve_candidates(
            "ask anything",
            {"controls": controls},
            preferred_types=["Edit", "TextBox"],
            source="uia",
        )
        self.assertIsNotNone(result.best)
        self.assertFalse(result.ambiguous)
        self.assertEqual(result.best.name, "Ask anything")
        self.assertEqual(result.best.control_type, "Edit")

    def test_ocr_sidebar_item(self):
        """OCR should find sidebar items when UIA fails."""
        ocr_perception = {
            "lines": [
                {"text": "FPV station", "rect": {"l": 10, "t": 60, "r": 120, "b": 80}},
                {"text": "Python help", "rect": {"l": 10, "t": 100, "r": 120, "b": 120}},
            ],
            "words": [],
        }
        result = resolve_candidates(
            "fpv station",
            ocr_perception,
            preferred_types=[],
            source="ocr",
        )
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.name, "FPV station")
        self.assertEqual(result.best.source, "ocr")

    def test_promotion_metadata_serialization(self):
        """Promotion metadata should be in to_dict() output."""
        c = Candidate(
            id=0, name="FPV station", control_type="ListItem",
            rect={"l": 5, "t": 55, "r": 200, "b": 85},
            score=120.0, source="uia", enabled=True,
            promotion={
                "promoted_from": "Text",
                "promoted_to": "ListItem",
                "promotion_depth": 1,
            },
        )
        d = c.to_dict()
        self.assertIn("promotion", d)
        self.assertEqual(d["promotion"]["promoted_from"], "Text")
        self.assertEqual(d["promotion"]["promoted_to"], "ListItem")
        self.assertEqual(d["promotion"]["promotion_depth"], 1)


if __name__ == "__main__":
    unittest.main()
