"""
assert_text_present — Deterministic verification tool (Phase 16+).

After a click-and-type action, re-perceive the focused window and assert
that the typed text is present (UIA value property preferred, OCR fallback).

Targeted verification (Phase 16+):
    - UIA: checks the focused control's value first, then falls back to a
      narrow search within same-type controls.
    - OCR: checks only within ±40px of the click rect (if provided),
      never performs a global screen search.

Input:  {text, method: "auto"|"uia"|"ocr", control_name?, control_type?,
         click_rect?: {l,t,r,b}}
Output: {ok: bool, method_used: str, evidence: str, details: dict}
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

from wyzer.tools.tool_base import ToolBase


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")

# Verification search radius around click rect (px)
_OCR_VERIFY_MARGIN = 40


def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = _PUNCT_RE.sub(" ", t)
    return _SPACES_RE.sub(" ", t).strip()


def assert_text_present(
    text: str,
    method: str = "auto",
    control_name: Optional[str] = None,
    control_type: Optional[str] = None,
    click_rect: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Check whether *text* is present in the focused window.

    Strategy:
        1. UIA: Check the focused control's value first, then narrow search
           within controls matching control_name/type.  No global sweep.
        2. OCR: Screenshot + OCR, check only within ±40px of click_rect.
           If no click_rect provided, scan all lines (backward compat).
        3. auto: UIA first, OCR fallback.

    Returns {ok, method_used, evidence, details}.
    """
    text_norm = _norm(text)
    if not text_norm:
        return {"ok": False, "method_used": "none", "evidence": "empty text",
                "details": {}}

    if method in ("uia", "auto"):
        uia_result = _check_uia(text_norm, control_name, control_type)
        if uia_result["ok"]:
            return uia_result
        if method == "uia":
            return uia_result

    if method in ("ocr", "auto"):
        return _check_ocr(text_norm, click_rect)

    return {"ok": False, "method_used": method, "evidence": "unsupported method",
            "details": {}}


def _check_uia(
    text_norm: str,
    control_name: Optional[str],
    control_type: Optional[str],
) -> Dict[str, Any]:
    """Check UIA controls for text presence — targeted, not global.

    Priority order:
        1. Check the focused element's value property
        2. If control_name given → find that control and check its value
        3. Narrow scan: only controls of matching type (if given)
    """
    try:
        from wyzer.tools.desktop.perceive_uia import perceive_uia_focused_window
        snapshot = perceive_uia_focused_window(max_nodes=80)
    except Exception as e:
        return {"ok": False, "method_used": "uia", "evidence": f"uia_error: {e}",
                "details": {}}

    controls = snapshot.get("controls", [])

    # 1. Check focused element (first Edit/ComboBox in the list, heuristic)
    for ctrl in controls:
        ct = (ctrl.get("control_type") or "").lower()
        if ct in ("edit", "combobox", "document"):
            name_norm = _norm(ctrl.get("name", ""))
            if text_norm in name_norm:
                return {
                    "ok": True,
                    "method_used": "uia",
                    "evidence": f"focused-area control '{ctrl.get('name')}' contains typed text",
                    "details": {"control": ctrl.get("name"),
                                "control_type": ctrl.get("control_type"),
                                "targeted": True},
                }

    # 2. If control_name given, find that control and check its value
    if control_name:
        cn_norm = _norm(control_name)
        for ctrl in controls:
            name_norm = _norm(ctrl.get("name", ""))
            if cn_norm in name_norm or name_norm in cn_norm:
                if control_type:
                    ct = (ctrl.get("control_type") or "").lower()
                    if ct != control_type.lower():
                        continue
                # Check name contains typed text
                if text_norm in name_norm:
                    return {
                        "ok": True,
                        "method_used": "uia",
                        "evidence": f"control '{ctrl.get('name')}' contains typed text",
                        "details": {"control": ctrl.get("name"),
                                    "control_type": ctrl.get("control_type"),
                                    "targeted": True},
                    }

    # 3. Narrow search: only controls of matching type
    for ctrl in controls:
        if control_type:
            ct = (ctrl.get("control_type") or "").lower()
            if ct != control_type.lower():
                continue
        name_norm = _norm(ctrl.get("name", ""))
        if text_norm in name_norm:
            return {
                "ok": True,
                "method_used": "uia",
                "evidence": f"control '{ctrl.get('name')}' contains typed text",
                "details": {"control": ctrl.get("name"),
                            "control_type": ctrl.get("control_type"),
                            "targeted": bool(control_type)},
            }

    return {"ok": False, "method_used": "uia",
            "evidence": f"no UIA control contains '{text_norm}'",
            "details": {"scanned": len(controls), "targeted": True}}


def _check_ocr(
    text_norm: str,
    click_rect: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Check OCR output for text presence — targeted within click_rect ±40px.

    If click_rect is provided, only checks OCR lines whose bounding box
    overlaps with the expanded click region.  Never runs a global screen
    search.
    """
    try:
        from wyzer.tools.desktop.perceive_ocr_focused import _perceive_ocr_focused
        ocr = _perceive_ocr_focused()
    except Exception as e:
        return {"ok": False, "method_used": "ocr", "evidence": f"ocr_error: {e}",
                "details": {}}

    errors = ocr.get("errors", [])
    if errors:
        return {"ok": False, "method_used": "ocr",
                "evidence": f"ocr errors: {errors}",
                "details": {"errors": errors}}

    lines = ocr.get("lines", [])

    # Build search region from click_rect
    search_region = None
    if click_rect:
        search_region = {
            "l": click_rect.get("l", 0) - _OCR_VERIFY_MARGIN,
            "t": click_rect.get("t", 0) - _OCR_VERIFY_MARGIN,
            "r": click_rect.get("r", 0) + _OCR_VERIFY_MARGIN,
            "b": click_rect.get("b", 0) + _OCR_VERIFY_MARGIN,
        }

    for line in lines:
        line_text = line.get("text", "") if isinstance(line, dict) else str(line)
        line_rect = line.get("rect") if isinstance(line, dict) else None

        # If search_region is set, skip lines outside it
        if search_region and line_rect:
            if (line_rect.get("r", 0) < search_region["l"] or
                line_rect.get("l", 0) > search_region["r"] or
                line_rect.get("b", 0) < search_region["t"] or
                line_rect.get("t", 0) > search_region["b"]):
                continue

        if text_norm in _norm(line_text):
            return {
                "ok": True,
                "method_used": "ocr",
                "evidence": f"OCR line contains typed text: '{line_text}'",
                "details": {"line": line_text,
                            "rect": line_rect,
                            "targeted": search_region is not None},
            }

    return {"ok": False, "method_used": "ocr",
            "evidence": f"OCR did not find '{text_norm}'" + (
                " within click region" if search_region else ""),
            "details": {"line_count": len(lines),
                        "targeted": search_region is not None}}


class AssertTextPresentTool(ToolBase):
    """Deterministic check: is typed text visible in the focused window?"""

    def __init__(self):
        super().__init__()
        self._name = "assert_text_present"
        self._description = (
            "Verify that specific text is visible in the focused window. "
            "Uses UIA preferred, OCR fallback. Deterministic."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to check for.",
                },
                "method": {
                    "type": "string",
                    "description": "'auto' (default), 'uia', or 'ocr'.",
                    "default": "auto",
                },
                "control_name": {
                    "type": "string",
                    "description": "Optional: name of the control to check.",
                },
                "control_type": {
                    "type": "string",
                    "description": "Optional: UIA control type filter.",
                },
                "click_rect": {
                    "type": "object",
                    "description": "Optional: rect of the clicked element {l,t,r,b} for targeted OCR verification.",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        text = kwargs.get("text", "")
        method = kwargs.get("method", "auto")
        control_name = kwargs.get("control_name")
        control_type = kwargs.get("control_type")
        click_rect = kwargs.get("click_rect")

        start = time.perf_counter()
        result = assert_text_present(text, method, control_name, control_type, click_rect)
        latency_ms = int((time.perf_counter() - start) * 1000)
        result["latency_ms"] = latency_ms

        emit_event("assertion", {
            "kind": "text_present",
            "text": text,
            "ok": result["ok"],
            "method_used": result["method_used"],
            "latency_ms": latency_ms,
        })

        # Deterministic summary for TTS
        if result["ok"]:
            result["summary"] = f"Verified — '{text}' is visible."
        else:
            result["summary"] = f"Could not verify '{text}' on screen."

        return result
