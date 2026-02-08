"""
ocr_region — Desktop Ground Truth Tool (Phase 14, OPTIONAL)

OCR a screenshot or region using pytesseract (if installed).
Input:  {image_path?, rect?}
Output: {lines:[{text, conf?, bbox?}], full_text, source_image, missing_dependency?}

OPTIONAL: If pytesseract / Tesseract is not installed, returns
missing_dependency=true with guidance. Does NOT add heavy deps.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from wyzer.tools.tool_base import ToolBase


def _run_ocr(image_path: Optional[str] = None, rect: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Run OCR on an image file, optionally cropping to a rect first.

    Returns structured OCR result or missing_dependency indicator.
    """
    # Check pytesseract availability
    try:
        import pytesseract
    except ImportError:
        return {
            "lines": [],
            "full_text": "",
            "source_image": image_path,
            "missing_dependency": True,
            "guidance": (
                "pytesseract is installed but Tesseract OCR engine may not be. "
                "Install Tesseract: https://github.com/tesseract-ocr/tesseract "
                "and ensure 'tesseract' is on PATH."
            ),
        }

    # Need an image
    if not image_path:
        # Try screenshot of focused window first
        try:
            from wyzer.tools.desktop.screenshot_tool import _capture_focused_window
            shot = _capture_focused_window()
            if "error" in shot:
                return {"lines": [], "full_text": "", "error": shot["error"]}
            image_path = shot["image_path"]
        except Exception as e:
            return {"lines": [], "full_text": "", "error": f"no_image: {e}"}

    try:
        from PIL import Image
        img = Image.open(image_path)

        # Crop to rect if specified
        if rect:
            l, t, r, b = rect.get("l", 0), rect.get("t", 0), rect.get("r", img.width), rect.get("b", img.height)
            img = img.crop((l, t, r, b))

        # Run OCR
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError:
            return {
                "lines": [],
                "full_text": "",
                "source_image": image_path,
                "missing_dependency": True,
                "guidance": (
                    "Tesseract OCR engine is not installed or not on PATH. "
                    "Install: https://github.com/tesseract-ocr/tesseract"
                ),
            }

        lines: List[Dict[str, Any]] = []
        current_line: List[str] = []
        current_line_num = -1

        for i, text in enumerate(data.get("text", [])):
            text = (text or "").strip()
            line_num = data["line_num"][i]
            conf = data["conf"][i]

            if line_num != current_line_num:
                if current_line:
                    lines.append({"text": " ".join(current_line)})
                current_line = []
                current_line_num = line_num

            if text and int(conf) > 20:  # skip very low confidence junk
                current_line.append(text)

        if current_line:
            lines.append({"text": " ".join(current_line)})

        full_text = "\n".join(ln["text"] for ln in lines)

        return {
            "lines": lines,
            "full_text": full_text,
            "source_image": image_path,
        }

    except ImportError as ie:
        return {
            "lines": [],
            "full_text": "",
            "source_image": image_path,
            "missing_dependency": True,
            "guidance": f"Missing: {ie}. Install Pillow and pytesseract.",
        }
    except Exception as exc:
        return {
            "lines": [],
            "full_text": "",
            "source_image": image_path,
            "error": str(exc),
        }


class OCRRegionTool(ToolBase):
    """OCR a screenshot or region — OPTIONAL, requires Tesseract."""

    def __init__(self):
        super().__init__()
        self._name = "ocr_region"
        self._description = (
            "Run OCR on a screenshot (or capture one). Returns extracted text lines. "
            "OPTIONAL: requires Tesseract OCR engine installed."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to image file. If omitted, captures focused window.",
                },
                "rect": {
                    "type": "object",
                    "description": "Crop region {l, t, r, b} in pixels.",
                    "properties": {
                        "l": {"type": "integer"},
                        "t": {"type": "integer"},
                        "r": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event, update_last_perception

        image_path = kwargs.get("image_path")
        rect = kwargs.get("rect")

        start = time.perf_counter()
        result = _run_ocr(image_path=image_path, rect=rect)
        latency_ms = int((time.perf_counter() - start) * 1000)
        result["latency_ms"] = latency_ms

        is_missing = result.get("missing_dependency", False)
        if is_missing:
            emit_event("warning", {
                "type": "ocr_missing_dependency",
                "guidance": result.get("guidance", ""),
            })
        else:
            emit_event("perception", {
                "source": "ocr",
                "line_count": len(result.get("lines", [])),
                "full_text_len": len(result.get("full_text", "")),
                "latency_ms": latency_ms,
            })
            update_last_perception(result)

        return result
