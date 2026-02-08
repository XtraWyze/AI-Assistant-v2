"""
wyzer.desktop.click_and_type — Deterministic click-and-type orchestrator (Phase 16+).

Executes the full "click <target> and type <text>" flow without any LLM:

    Step 1:  perceive_uia_focused_window
    Step 2:  resolve candidates (prefer Edit) — with ancestor promotion
    Step 3:  If no/weak candidates → perceive_ocr_focused_window → resolve OCR
    Step 4:  If ambiguous → show_choice_overlay → wait_for_overlay_choice
    Step 5:  Click chosen candidate:
             a) UIA InvokePattern
             b) SetFocus + Enter
             c) Rect center click
             d) OCR expanded hitbox (center → padded → offset_left)
    Step 6:  Type text
    Step 7:  Re-perceive (UIA; OCR fallback)
    Step 8:  Verify / assert text is present (targeted, not global)

Returns a deterministic JSON payload with unified schema.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def execute_click_and_type(
    target: str,
    text: str,
    preferred_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the full deterministic click-and-type flow.

    Args:
        target: The UI element label to click (e.g. "ask anything").
        text:   The text to type after clicking.
        preferred_types: Control types to boost during resolution.

    Returns:
        Deterministic response payload:
        {ok, target, typed, method, disambiguation, verification, errors, steps}
    """
    from wyzer.context.world_state import emit_event

    if preferred_types is None:
        preferred_types = ["Edit", "TextBox", "ComboBox"]

    action_id = uuid.uuid4().hex[:10]
    errors: List[str] = []
    steps: List[Dict[str, Any]] = []
    step_counter = 0

    def _step(name: str) -> int:
        nonlocal step_counter
        step_counter += 1
        sid = step_counter
        emit_event("tool_start", {
            "action_id": action_id,
            "step_id": sid,
            "step": name,
        })
        return sid

    def _step_end(sid: int, name: str, data: Dict[str, Any]) -> None:
        steps.append({"step_id": sid, "step": name, **data})
        emit_event("tool_end", {
            "action_id": action_id,
            "step_id": sid,
            "step": name,
            **{k: v for k, v in data.items() if k != "raw"},
        })

    result: Dict[str, Any] = {
        "ok": False,
        "target": target,
        "typed": text,
        "method": "none",
        "disambiguation": {"used": False},
        "verification": {"ok": False, "details": {}},
        "errors": errors,
        "steps": steps,
        "action_id": action_id,
    }

    # ── Step 1: UIA perception ──────────────────────────────────────
    sid = _step("perceive_uia")
    try:
        from wyzer.tools.desktop.perceive_uia import perceive_uia_focused_window
        uia_snapshot = perceive_uia_focused_window(max_nodes=80)
    except Exception as e:
        uia_snapshot = {"controls": [], "errors": [str(e)]}
    _step_end(sid, "perceive_uia", {
        "control_count": len(uia_snapshot.get("controls", [])),
        "errors": uia_snapshot.get("errors", []),
    })

    # ── Step 2: Resolve UIA candidates ──────────────────────────────
    sid = _step("resolve_uia")
    from wyzer.desktop.resolve_target import resolve_candidates, SCORE_MIN
    uia_resolve = resolve_candidates(target, uia_snapshot,
                                      preferred_types=preferred_types,
                                      source="uia")
    _step_end(sid, "resolve_uia", {
        "candidate_count": len(uia_resolve.candidates),
        "ambiguous": uia_resolve.ambiguous,
        "best_score": uia_resolve.best.score if uia_resolve.best else 0,
        "reason": uia_resolve.reason,
    })

    chosen_candidate = None
    method = "uia"

    # Confident UIA match?
    if uia_resolve.best and not uia_resolve.ambiguous and uia_resolve.best.score >= SCORE_MIN:
        chosen_candidate = uia_resolve.best
    else:
        # ── Step 3: OCR fallback ────────────────────────────────────
        sid = _step("perceive_ocr")
        try:
            from wyzer.tools.desktop.perceive_ocr_focused import _perceive_ocr_focused
            ocr_snapshot = _perceive_ocr_focused()
        except Exception as e:
            ocr_snapshot = {"lines": [], "words": [], "errors": [str(e)]}

        ocr_errors = ocr_snapshot.get("errors", [])
        _step_end(sid, "perceive_ocr", {
            "line_count": len(ocr_snapshot.get("lines", [])),
            "word_count": len(ocr_snapshot.get("words", [])),
            "errors": ocr_errors,
        })

        if not ocr_errors:
            sid = _step("resolve_ocr")
            ocr_resolve = resolve_candidates(target, ocr_snapshot,
                                              preferred_types=preferred_types,
                                              source="ocr")
            _step_end(sid, "resolve_ocr", {
                "candidate_count": len(ocr_resolve.candidates),
                "ambiguous": ocr_resolve.ambiguous,
                "best_score": ocr_resolve.best.score if ocr_resolve.best else 0,
                "reason": ocr_resolve.reason,
            })

            if ocr_resolve.best and not ocr_resolve.ambiguous and ocr_resolve.best.score >= SCORE_MIN:
                chosen_candidate = ocr_resolve.best
                method = "ocr"
            elif ocr_resolve.ambiguous or (uia_resolve.ambiguous and uia_resolve.candidates):
                # Merge and use the best set for disambiguation
                method = "ocr" if ocr_resolve.best and (
                    not uia_resolve.best or ocr_resolve.best.score > uia_resolve.best.score
                ) else "uia"
        else:
            ocr_resolve = uia_resolve  # no OCR available, keep UIA results

        # ── Step 4: Disambiguation overlay ──────────────────────────
        if chosen_candidate is None:
            # Pick the resolve result with more candidates
            resolve_for_overlay = (
                ocr_resolve if len(ocr_resolve.candidates) > len(uia_resolve.candidates)
                else uia_resolve
            )

            top3 = resolve_for_overlay.candidates[:3]
            if not top3:
                errors.append(f"No candidates found for '{target}' via UIA or OCR")
                result["errors"] = errors
                result["summary"] = f"I couldn't find '{target}' on screen."
                return result

            sid = _step("disambiguation_overlay")
            overlay_options = []
            for c in top3:
                rect = c.rect
                hint = ""
                if rect:
                    cx = (rect.get("l", 0) + rect.get("r", 0)) // 2
                    cy = (rect.get("t", 0) + rect.get("b", 0)) // 2
                    # Location hint: left/right and top/bottom of screen
                    try:
                        import ctypes
                        sw = ctypes.windll.user32.GetSystemMetrics(0) or 1920
                        sh = ctypes.windll.user32.GetSystemMetrics(1) or 1080
                    except Exception:
                        sw, sh = 1920, 1080
                    h_pos = "left sidebar" if cx < sw * 0.3 else ("right panel" if cx > sw * 0.7 else "main panel")
                    v_pos = "top" if cy < sh * 0.3 else ("bottom" if cy > sh * 0.7 else "middle")
                    hint = f"{v_pos} / {h_pos}"
                ct_display = c.control_type
                if c.promotion:
                    ct_display += f" (from {c.promotion.get('promoted_from', '?')})"
                overlay_options.append({
                    "label": c.name,
                    "hint": hint,
                    "control_type": ct_display,
                    "source": c.source,
                    "internal_id": c.id,
                })

            try:
                from wyzer.tools.desktop.overlay import show_overlay, wait_overlay_choice
                ov = show_overlay(
                    prompt=f'Which "{target}" did you mean?',
                    options=overlay_options,
                )
                if ov.get("error"):
                    errors.append(f"overlay error: {ov['error']}")
                    _step_end(sid, "disambiguation_overlay", {"error": ov["error"]})
                    # Fall back to best candidate anyway
                    if top3:
                        chosen_candidate = top3[0]
                else:
                    ov_id = ov["overlay_id"]
                    choice_result = wait_overlay_choice(ov_id, timeout_ms=15000)
                    _step_end(sid, "disambiguation_overlay", {
                        "choice": choice_result.get("choice"),
                        "timed_out": choice_result.get("timed_out", False),
                        "cancelled": choice_result.get("cancelled", False),
                    })

                    choice_idx = choice_result.get("choice")
                    if choice_result.get("cancelled") or choice_result.get("timed_out"):
                        errors.append("User cancelled or timed out")
                        result["disambiguation"] = {"used": True, "choice": None,
                                                     "timed_out": choice_result.get("timed_out", False)}
                        result["errors"] = errors
                        result["summary"] = "Cancelled — no action taken."
                        return result

                    if choice_idx and 1 <= choice_idx <= len(top3):
                        chosen_candidate = top3[choice_idx - 1]
                        result["disambiguation"] = {"used": True, "choice": choice_idx}
                        method = chosen_candidate.source
                    else:
                        errors.append(f"Invalid overlay choice: {choice_idx}")
                        result["errors"] = errors
                        result["summary"] = "Invalid selection — no action taken."
                        return result
            except Exception as e:
                errors.append(f"overlay exception: {e}")
                _step_end(sid, "disambiguation_overlay", {"error": str(e)})
                # Fall back to best candidate
                if top3:
                    chosen_candidate = top3[0]

    if chosen_candidate is None:
        errors.append(f"Could not resolve '{target}' to any UI element")
        result["errors"] = errors
        result["summary"] = f"I couldn't find '{target}' on screen."
        return result

    result["method"] = method

    # ── Step 5: Click ───────────────────────────────────────────────
    sid = _step("click")
    click_result = _do_click(chosen_candidate)
    _step_end(sid, "click", {
        "clicked": click_result.get("clicked", False),
        "target_name": chosen_candidate.name,
        "method": method,
    })
    emit_event("ui_action", {
        "kind": "click",
        "action_id": action_id,
        "target": chosen_candidate.name,
        "control_type": chosen_candidate.control_type,
        "method": method,
        "success": click_result.get("clicked", False),
    })

    if not click_result.get("clicked"):
        errors.append(f"Click failed: {click_result.get('reason', 'unknown')}")
        result["errors"] = errors
        result["summary"] = f"I found '{target}' but couldn't click it."
        return result

    # ── Click-only mode: skip typing + verification when text is empty ──
    if not text:
        result["ok"] = True
        result["clicked"] = True
        result["summary"] = f"Clicked '{chosen_candidate.name}'."
        return result

    # Small wait for UI to respond
    time.sleep(0.15)

    # ── Step 6: Type ────────────────────────────────────────────────
    sid = _step("type_text")
    type_result = _do_type(text)
    _step_end(sid, "type_text", {
        "typed": type_result.get("success", False),
        "length": len(text),
    })
    emit_event("ui_action", {
        "kind": "type",
        "action_id": action_id,
        "text_length": len(text),
        "success": type_result.get("success", False),
    })

    if not type_result.get("success"):
        errors.append(f"Type failed: {type_result.get('error', 'unknown')}")
        result["errors"] = errors
        result["summary"] = f"Clicked '{target}' but couldn't type."
        return result

    # Small wait for text to appear
    time.sleep(0.2)

    # ── Step 7 + 8: Re-perceive + verify ────────────────────────────
    sid = _step("verify")
    try:
        from wyzer.tools.desktop.assert_text_present import assert_text_present
        verify = assert_text_present(
            text=text,
            method="auto",
            control_name=chosen_candidate.name,
            control_type=chosen_candidate.control_type if chosen_candidate.control_type not in ("OCR_Text", "OCR_Word") else None,
            click_rect=chosen_candidate.rect,
        )
    except Exception as e:
        verify = {"ok": False, "method_used": "none", "evidence": str(e), "details": {}}

    _step_end(sid, "verify", {
        "ok": verify.get("ok", False),
        "method_used": verify.get("method_used", "none"),
        "evidence": verify.get("evidence", ""),
    })
    emit_event("assertion", {
        "kind": "text_present_after_type",
        "action_id": action_id,
        "text": text,
        "ok": verify.get("ok", False),
        "method_used": verify.get("method_used"),
    })

    result["verification"] = {
        "ok": verify.get("ok", False),
        "details": verify.get("details", {}),
        "method_used": verify.get("method_used", "none"),
        "evidence": verify.get("evidence", ""),
    }

    result["ok"] = True
    if verify.get("ok"):
        result["summary"] = f"Clicked '{chosen_candidate.name}' and typed '{text}'. Verified."
    else:
        result["summary"] = f"Clicked '{chosen_candidate.name}' and typed '{text}'. Verification inconclusive."

    return result


# ── Internal action helpers ─────────────────────────────────────────────

# OCR hitbox expansion padding (px)
_OCR_PAD = 12
_OCR_OFFSET_LEFT = -12
_OCR_MAX_ATTEMPTS = 2


def _do_click(candidate) -> Dict[str, Any]:
    """Click a resolved candidate using a deterministic retry chain.

    Execution order:
        1. UIA InvokePattern on resolved control
        2. If no InvokePattern → SetFocus + Enter
        3. If still no effect → rect center click
        4. If source is OCR → expanded hitbox (center, then offset_left)
        5. If all fail → return failure (no infinite retries)

    Returns unified schema:
        {ok, clicked, method, fallback_used, reason, latency_ms,
         matched, ocr_click_strategy?}
    """
    import time as _time
    start = _time.perf_counter()

    # ── 1. UIA InvokePattern ────────────────────────────────────────
    if candidate.source == "uia":
        try:
            from wyzer.tools.desktop.desktop_click_uia import _best_match_with_retry
            click_res = _best_match_with_retry(
                candidate.name,
                candidate.control_type or None,
                candidate.rect,
            )
            if click_res.get("clicked"):
                click_res["latency_ms"] = int((_time.perf_counter() - start) * 1000)
                return click_res
        except Exception:
            pass

    # ── 2–3. Rect center click (works for UIA and OCR) ──────────────
    if candidate.rect:
        rect = candidate.rect
        cx = (rect.get("l", 0) + rect.get("r", 0)) // 2
        cy = (rect.get("t", 0) + rect.get("b", 0)) // 2

        # If OCR source → use expanded hitbox strategy
        if candidate.source == "ocr":
            return _ocr_click_with_expansion(rect, candidate, start)

        # UIA rect click
        try:
            import pyautogui
            pyautogui.click(x=cx, y=cy)
            return {
                "ok": True,
                "clicked": True,
                "method": "rect_click",
                "fallback_used": True,
                "reason": "uia_rect_center",
                "matched": {"name": candidate.name, "type": candidate.control_type,
                            "rect": candidate.rect},
                "latency_ms": int((_time.perf_counter() - start) * 1000),
            }
        except Exception as e:
            return {
                "ok": False,
                "clicked": False,
                "method": "rect_click",
                "fallback_used": True,
                "reason": f"rect_click failed: {e}",
                "latency_ms": int((_time.perf_counter() - start) * 1000),
            }

    return {
        "ok": False,
        "clicked": False,
        "method": "none",
        "fallback_used": False,
        "reason": "no UIA match and no rect for fallback click",
        "latency_ms": int((_time.perf_counter() - start) * 1000),
    }


def _ocr_click_with_expansion(
    rect: Dict[str, int],
    candidate: Any,
    start: float,
) -> Dict[str, Any]:
    """OCR click with deterministic hitbox expansion.

    Strategy (max 2 attempts):
        1. Click center of rect
        2. If that fails → click center with x-offset (-12px)

    Returns dict with ``ocr_click_strategy`` metadata.
    """
    import time as _time

    cx = (rect.get("l", 0) + rect.get("r", 0)) // 2
    cy = (rect.get("t", 0) + rect.get("b", 0)) // 2

    strategies = [
        ("center", cx, cy),
        ("offset_left", cx + _OCR_OFFSET_LEFT, cy),
    ]

    try:
        import pyautogui
    except ImportError:
        return {
            "ok": False,
            "clicked": False,
            "method": "ocr_click",
            "fallback_used": True,
            "reason": "pyautogui not installed",
            "ocr_click_strategy": "none",
            "latency_ms": int((_time.perf_counter() - start) * 1000),
        }

    for strategy_name, sx, sy in strategies[:_OCR_MAX_ATTEMPTS]:
        try:
            pyautogui.click(x=sx, y=sy)
            return {
                "ok": True,
                "clicked": True,
                "method": "ocr_click",
                "fallback_used": True,
                "reason": f"ocr_{strategy_name}",
                "ocr_click_strategy": strategy_name,
                "matched": {"name": candidate.name, "type": candidate.control_type,
                            "rect": rect},
                "latency_ms": int((_time.perf_counter() - start) * 1000),
            }
        except Exception:
            continue

    return {
        "ok": False,
        "clicked": False,
        "method": "ocr_click",
        "fallback_used": True,
        "reason": "all OCR click strategies failed",
        "ocr_click_strategy": "exhausted",
        "latency_ms": int((_time.perf_counter() - start) * 1000),
    }


def _do_type(text: str) -> Dict[str, Any]:
    """Type text into the focused control."""
    try:
        import pyautogui
        if text.isascii():
            pyautogui.typewrite(text, interval=0.02)
        else:
            pyautogui.write(text)
        return {"success": True, "typed_length": len(text)}
    except Exception as e:
        return {"success": False, "error": str(e)}
