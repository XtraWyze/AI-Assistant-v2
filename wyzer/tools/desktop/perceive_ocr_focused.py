"""
perceive_ocr_focused_window — Desktop Ground Truth Tool (Phase 16).

Captures a screenshot of the focused window and runs OCR on it.
Returns structured output:
    {lines: [{text, rect}], words: [{text, rect}], timestamp, errors: []}

Uses the existing screenshot + OCR infrastructure.  If OCR is not
available, returns errors: ["ocr_not_available"].

This is a separate tool from ``ocr_region`` — it always targets the
focused window, returns both line-level and word-level results, and
includes bounding rectangles for each element.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from wyzer.tools.tool_base import ToolBase


def _perceive_ocr_focused(
    max_lines: int = 100,
) -> Dict[str, Any]:
    """Capture the focused window and run OCR.

    Returns {lines, words, window_rect, timestamp, errors}.
    """
    result: Dict[str, Any] = {
        "lines": [],
        "words": [],
        "window_rect": None,
        "timestamp": time.time(),
        "errors": [],
    }

    # 1. Take screenshot of focused window
    try:
        from wyzer.tools.desktop.screenshot_tool import _capture_focused_window
        shot = _capture_focused_window()
        if "error" in shot:
            result["errors"].append(f"screenshot: {shot['error']}")
            return result
        image_path = shot.get("image_path")
        result["window_rect"] = shot.get("rect")
    except Exception as e:
        result["errors"].append(f"screenshot_failed: {e}")
        return result

    if not image_path:
        result["errors"].append("no_image_path")
        return result

    # 2. Run OCR via pytesseract
    try:
        import pytesseract
        from PIL import Image
    except ImportError as ie:
        result["errors"].append("ocr_not_available")
        return result

    try:
        img = Image.open(image_path)
    except Exception as e:
        result["errors"].append(f"image_open: {e}")
        return result

    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as e:
        err_s = str(e).lower()
        if "tesseract" in err_s and ("not found" in err_s or "not installed" in err_s):
            result["errors"].append("ocr_not_available")
        else:
            result["errors"].append(f"ocr_error: {e}")
        return result

    # 3. Build word-level and line-level results
    words: List[Dict[str, Any]] = []
    line_buckets: Dict[int, List[Dict[str, Any]]] = {}

    win_rect = result.get("window_rect") or {}
    win_left = win_rect.get("l", 0)
    win_top = win_rect.get("t", 0)

    for i, text in enumerate(data.get("text", [])):
        text = (text or "").strip()
        conf = int(data["conf"][i]) if data["conf"][i] != -1 else -1
        if not text or conf < 20:
            continue

        # Build absolute-screen-coordinate rect
        x = data["left"][i] + win_left
        y = data["top"][i] + win_top
        w = data["width"][i]
        h = data["height"][i]
        word_rect = {"l": x, "t": y, "r": x + w, "b": y + h}

        word_entry = {"text": text, "rect": word_rect, "conf": conf}
        words.append(word_entry)

        line_num = data["line_num"][i]
        block_num = data["block_num"][i]
        bucket_key = block_num * 10000 + line_num
        line_buckets.setdefault(bucket_key, []).append(word_entry)

    # Build line-level
    lines: List[Dict[str, Any]] = []
    for _key in sorted(line_buckets):
        bucket = line_buckets[_key]
        line_text = " ".join(w["text"] for w in bucket)
        if not line_text.strip():
            continue
        rects = [w["rect"] for w in bucket if w.get("rect")]
        if rects:
            line_rect = {
                "l": min(r["l"] for r in rects),
                "t": min(r["t"] for r in rects),
                "r": max(r["r"] for r in rects),
                "b": max(r["b"] for r in rects),
            }
        else:
            line_rect = None
        lines.append({"text": line_text, "rect": line_rect})
        if len(lines) >= max_lines:
            break

    result["lines"] = lines
    result["words"] = words
    return result


class PerceiveOCRFocusedWindowTool(ToolBase):
    """OCR perception of the focused window — deterministic, no LLM."""

    def __init__(self):
        super().__init__()
        self._name = "perceive_ocr_focused_window"
        self._description = (
            "Capture the focused window and run OCR. Returns lines + words "
            "with bounding rects. Deterministic fallback when UIA is insufficient."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "max_lines": {
                    "type": "integer",
                    "description": "Max OCR lines to return (default 100).",
                    "default": 100,
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event, update_last_perception
        from wyzer.tools.desktop.truth_contract import normalize_perception

        max_lines = kwargs.get("max_lines", 100)

        start = time.perf_counter()
        info = _perceive_ocr_focused(max_lines=max_lines)
        latency_ms = int((time.perf_counter() - start) * 1000)
        info["latency_ms"] = latency_ms

        errors = info.get("errors", [])
        if errors:
            emit_event("perception", {
                "source": "ocr_focused_window",
                "errors": errors,
                "latency_ms": latency_ms,
            })
        else:
            emit_event("perception", {
                "source": "ocr_focused_window",
                "line_count": len(info.get("lines", [])),
                "word_count": len(info.get("words", [])),
                "latency_ms": latency_ms,
            })
            normalized = normalize_perception(info)
            update_last_perception(normalized)

        return info
