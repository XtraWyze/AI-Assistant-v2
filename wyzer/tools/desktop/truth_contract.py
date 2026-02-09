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


# =========================================================================
# AGENT-GRADE: Multi-source canonical snapshot (Phase 17)
# =========================================================================

FATAL_PERCEPTION_ERRORS = frozenset({
    "pywinauto_not_installed",
    "no_foreground_window",
    "no_top_windows",
    "foreground_not_in_uia_list",
})

# Tools whose execution invalidates the current observation snapshot.
UI_AFFECTING_TOOLS = frozenset({
    "desktop_click_uia", "click", "type_text", "send_keys",
    "switch_app", "open_app", "open_target", "close_window",
    "minimize_window", "maximize_window", "focus_window",
    "__CLICK_AND_TYPE__",
})


def normalize_perception_multi(
    raw_uia: dict | None,
    raw_ocr: dict | None,
    active_window: dict | None,
    recent_events: list | None,
) -> dict:
    """Merge multiple perception sources into one canonical snapshot.

    Canonical shape (exact keys)::

        {
            "timestamp_ms": int,
            "foreground": {"app": str|None, "title": str|None},
            "windows": [{"title": str, "app": str|None}],
            "controls": [{"name": str, "type": str|None, "automation_id": str|None}],
            "ocr_text": [str],
            "errors": [str],
        }
    """
    import time as _time

    errors: list[str] = []

    # ── foreground window ────────────────────────────────────────────
    fg_app: str | None = None
    fg_title: str | None = None
    if active_window and isinstance(active_window, dict):
        fg_app = active_window.get("exe") or active_window.get("app")
        fg_title = active_window.get("title")
        if active_window.get("error"):
            errors.append(active_window["error"])
    elif raw_uia and isinstance(raw_uia, dict):
        w = raw_uia.get("window") or {}
        fg_app = w.get("exe") or w.get("app")
        fg_title = w.get("title")

    # ── windows list (from UIA top-windows if available) ─────────────
    windows_list: list[dict] = []
    if raw_uia and isinstance(raw_uia, dict):
        # The UIA snapshot only covers the focused window, but we may have
        # open_windows from world_state injected via recent_events.  For now
        # populate from the focused window; the orchestrator can enrich later.
        w = raw_uia.get("window") or {}
        wtitle = w.get("title")
        wapp = w.get("exe") or w.get("app")
        if wtitle:
            windows_list.append({"title": wtitle, "app": wapp})

    # ── controls ─────────────────────────────────────────────────────
    controls: list[dict] = []
    if raw_uia and isinstance(raw_uia, dict):
        for c in raw_uia.get("controls") or []:
            if not isinstance(c, dict):
                continue
            controls.append({
                "name": (c.get("name") or "").strip(),
                "type": c.get("control_type") or c.get("type") or None,
                "automation_id": c.get("automation_id"),
            })
        for e in raw_uia.get("errors") or []:
            if e not in errors:
                errors.append(str(e))

    # ── OCR text lines ───────────────────────────────────────────────
    ocr_text: list[str] = []
    if raw_ocr and isinstance(raw_ocr, dict):
        for ln in raw_ocr.get("lines") or []:
            if isinstance(ln, dict) and ln.get("text"):
                ocr_text.append(str(ln["text"]))
            elif isinstance(ln, str) and ln.strip():
                ocr_text.append(ln)
        for e in raw_ocr.get("errors") or []:
            if e not in errors:
                errors.append(str(e))

    return {
        "timestamp_ms": int(_time.time() * 1000),
        "foreground": {"app": fg_app, "title": fg_title},
        "windows": windows_list,
        "controls": controls,
        "ocr_text": ocr_text,
        "errors": errors,
    }


def canonical_to_prompt_block(canonical: dict, max_controls: int = 12) -> str:
    """Render the *agent-grade* canonical snapshot into a prompt block.

    This produces the ``[PERCEPTION SNAPSHOT]`` section that gets injected
    into the LLM prompt on every agent-loop iteration.
    """
    if not canonical:
        return "[PERCEPTION SNAPSHOT]\nNo perception data available."

    parts: list[str] = ["[PERCEPTION SNAPSHOT]"]

    # Foreground
    fg = canonical.get("foreground") or {}
    app = fg.get("app") or "unknown"
    title = fg.get("title") or ""
    if title:
        parts.append(f"Foreground: {app} — \"{title[:80]}\"")
    else:
        parts.append(f"Foreground: {app}")

    # Windows
    wins = canonical.get("windows") or []
    if wins:
        items = [f"{w.get('title','?')} ({w.get('app','')})" for w in wins[:6]]
        parts.append(f"Windows ({len(wins)}): {'; '.join(items)}")
    else:
        parts.append("Windows: none listed")

    # Controls
    ctrls = canonical.get("controls") or []
    if ctrls:
        items = []
        for c in ctrls[:max_controls]:
            name = (c.get("name") or "").strip()
            ctype = (c.get("type") or "").strip()
            if name:
                items.append(f"{name} ({ctype})" if ctype else name)
        suffix = f" (+{len(ctrls) - max_controls} more)" if len(ctrls) > max_controls else ""
        parts.append(f"Controls ({len(ctrls)}): {', '.join(items)}{suffix}")
    else:
        parts.append("Controls: none found")

    # OCR
    ocr = canonical.get("ocr_text") or []
    if ocr:
        preview = "; ".join(ocr[:5])
        if len(preview) > 200:
            preview = preview[:200] + "..."
        parts.append(f"OCR ({len(ocr)} lines): {preview}")

    # Errors
    errs = canonical.get("errors") or []
    if errs:
        parts.append(f"Errors: {'; '.join(errs[:3])}")
    else:
        parts.append("Errors: none")

    return "\n".join(parts)


def has_fatal_perception_error(canonical: dict) -> bool:
    """Return True if the canonical snapshot contains a fatal perception error."""
    for e in canonical.get("errors") or []:
        if e in FATAL_PERCEPTION_ERRORS:
            return True
    return False
