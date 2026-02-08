"""
truth_contract — Standard JSON schema for perception outputs (Phase 15).

All perception tools MUST return (or be normalizable to) this shape:

{
  "window": {"app": str?, "title": str?, "pid": int?, "hwnd": int?},
  "controls": [{"name": str, "type": str, "rect": dict?, "enabled": bool?}],
  "text_lines": [str],
  "dialogs": [{"title": str, "rect": dict?}],
  "progress": {"percent": float?, "text": str?},
  "errors": [str],
  "timestamp": float,
}

Missing fields are allowed, but top-level keys MUST exist so consumers
never crash on KeyError.

This module also provides ``normalize_perception()`` which cleans any
perception dict into the canonical shape.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


# Canonical empty perception snapshot — copy for defaults.
EMPTY_PERCEPTION: Dict[str, Any] = {
    "window": {"app": None, "title": None, "pid": None, "hwnd": None},
    "controls": [],
    "text_lines": [],
    "dialogs": [],
    "progress": None,
    "errors": [],
    "timestamp": 0.0,
}


def normalize_perception(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an arbitrary perception dict into the truth-contract schema.

    Rules:
    - Top-level keys always present (filled with defaults if missing).
    - ``window`` is normalized to ``{app, title, pid, hwnd}``.
    - ``controls`` keeps ``name/type/rect/enabled``, renames ``control_type`` → ``type``.
    - ``text_lines`` is populated from ``lines`` / ``full_text`` (OCR) if present.
    - ``progress`` is normalized to ``{percent, text}``.
    - ``errors`` is always a list of strings.
    - ``timestamp`` defaults to current time if absent.
    - Extra keys are preserved under ``_extra`` for debugging.
    """
    if not raw or not isinstance(raw, dict):
        return dict(EMPTY_PERCEPTION, timestamp=time.time())

    out: Dict[str, Any] = {}

    # ── window ───────────────────────────────────────────────────────
    w_raw = raw.get("window") or {}
    out["window"] = {
        "app": w_raw.get("app") or w_raw.get("exe"),
        "title": w_raw.get("title"),
        "pid": w_raw.get("pid"),
        "hwnd": w_raw.get("hwnd"),
    }

    # ── controls ─────────────────────────────────────────────────────
    controls_raw: List[Dict[str, Any]] = raw.get("controls") or []
    controls: List[Dict[str, Any]] = []
    for c in controls_raw:
        if not isinstance(c, dict):
            continue
        controls.append({
            "name": c.get("name", ""),
            "type": c.get("type") or c.get("control_type", ""),
            "rect": c.get("rect"),
            "enabled": c.get("enabled"),
        })
    out["controls"] = controls

    # ── text_lines (OCR or fallback from controls) ───────────────────
    text_lines: List[str] = []
    if "text_lines" in raw:
        text_lines = [str(t) for t in raw["text_lines"] if t]
    elif "lines" in raw:
        # OCR output: [{text: ...}, ...]
        for ln in raw["lines"]:
            if isinstance(ln, dict) and ln.get("text"):
                text_lines.append(str(ln["text"]))
            elif isinstance(ln, str) and ln.strip():
                text_lines.append(ln)
    elif "full_text" in raw and raw["full_text"]:
        text_lines = [l for l in str(raw["full_text"]).split("\n") if l.strip()]
    out["text_lines"] = text_lines

    # ── dialogs ──────────────────────────────────────────────────────
    out["dialogs"] = raw.get("dialogs") or []

    # ── progress ─────────────────────────────────────────────────────
    p_raw = raw.get("progress")
    if p_raw and isinstance(p_raw, dict):
        out["progress"] = {
            "percent": p_raw.get("percent") or p_raw.get("value"),
            "text": p_raw.get("text", ""),
        }
    else:
        out["progress"] = None

    # ── errors ───────────────────────────────────────────────────────
    errs_raw = raw.get("errors")
    if isinstance(errs_raw, list):
        out["errors"] = [str(e) for e in errs_raw]
    elif isinstance(errs_raw, str):
        out["errors"] = [errs_raw]
    else:
        out["errors"] = []

    # ── timestamp ────────────────────────────────────────────────────
    out["timestamp"] = raw.get("timestamp") or time.time()

    return out


def perception_to_prompt_block(p: Dict[str, Any], max_controls: int = 10) -> str:
    """
    Render a normalized perception dict into a compact text block suitable
    for LLM prompt injection.

    Example output:
        [PERCEPTION SNAPSHOT]
        Window: Chrome — "GitHub - Dashboard"
        Controls (6): New repository (Button), Issues (Hyperlink), ...
        Dialogs: none
        Progress: none
        Errors: none
    """
    if not p:
        return "[PERCEPTION SNAPSHOT]\nNo perception data available."

    parts: list[str] = ["[PERCEPTION SNAPSHOT]"]

    # Window
    w = p.get("window") or {}
    app = w.get("app") or "unknown"
    title = w.get("title") or ""
    if title:
        parts.append(f"Window: {app} — \"{title[:80]}\"")
    else:
        parts.append(f"Window: {app}")

    # Controls
    controls = p.get("controls") or []
    if controls:
        items = []
        for c in controls[:max_controls]:
            name = (c.get("name") or "").strip()
            ctype = (c.get("type") or "").strip()
            if name:
                items.append(f"{name} ({ctype})" if ctype else name)
        suffix = f" (+{len(controls) - max_controls} more)" if len(controls) > max_controls else ""
        parts.append(f"Controls ({len(controls)}): {', '.join(items)}{suffix}")
    else:
        parts.append("Controls: none found")

    # Text lines (OCR)
    text_lines = p.get("text_lines") or []
    if text_lines:
        preview = "; ".join(text_lines[:5])
        if len(preview) > 200:
            preview = preview[:200] + "..."
        parts.append(f"OCR text ({len(text_lines)} lines): {preview}")

    # Dialogs
    dialogs = p.get("dialogs") or []
    if dialogs:
        titles = [d.get("title", "?") for d in dialogs[:3]]
        parts.append(f"Dialogs: {', '.join(titles)}")
    else:
        parts.append("Dialogs: none")

    # Progress
    prog = p.get("progress")
    if prog:
        pct = prog.get("percent")
        ptxt = prog.get("text", "")
        if pct is not None:
            parts.append(f"Progress: {pct}%{' — ' + ptxt if ptxt else ''}")
        elif ptxt:
            parts.append(f"Progress: {ptxt}")
    else:
        parts.append("Progress: none")

    # Errors
    errs = p.get("errors") or []
    if errs:
        parts.append(f"Errors: {'; '.join(errs[:3])}")
    else:
        parts.append("Errors: none")

    return "\n".join(parts)
