"""
assertions — Desktop Ground Truth Assertion Helpers (Phase 14)

Deterministic assertion helpers that combine perception tools to answer:
- ui_find_text: Is text X visible on screen? (UIA preferred, OCR fallback)
- install_succeeded_check: Did an install succeed? (deterministic evidence)

These are functions (not ToolBase subclasses) but also exposed as tools.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

from wyzer.tools.tool_base import ToolBase


# ── Text normalization ─────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")


def _normalize_label(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    t = text.strip().lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _SPACES_RE.sub(" ", t).strip()
    return t


def _text_matches(needle: str, haystack: str, mode: str) -> bool:
    """Check whether *needle* matches *haystack* under the given *mode*.

    Both strings must already be normalised via ``_normalize_label``.

    Modes:
        exact    – normalised equality
        word     – needle appears as a whole-word token sequence
        contains – substring (legacy behaviour)
    """
    if mode == "exact":
        return needle == haystack
    if mode == "word":
        # Whole-word boundary match
        return bool(re.search(r"\b" + re.escape(needle) + r"\b", haystack))
    # contains (default)
    return needle in haystack


# ── ui_find_text ───────────────────────────────────────────────────────────

def ui_find_text(
    text: str,
    method: str = "uia",
    scope: str = "focused_window",
    control_type: Optional[str] = None,
    match_mode: str = "contains",
) -> Dict[str, Any]:
    """
    Search for text in the focused window using UIA or OCR.

    Args:
        text: Text to search for.
        method: 'uia', 'ocr', or 'auto'.
        scope: Currently only 'focused_window'.
        control_type: Optional UIA control type filter (e.g. 'Button').
        match_mode: 'exact', 'word', or 'contains' (default).

    Returns {found: bool, evidence: str, method: str, matches: [...], summary: str}.
    """
    text_norm = _normalize_label(text)
    if not text_norm:
        return {"found": False, "evidence": "empty_query", "method": method}

    if method == "uia":
        return _find_text_uia(text_norm, control_type=control_type, match_mode=match_mode)
    elif method == "ocr":
        return _find_text_ocr(text_norm, match_mode=match_mode)
    else:
        # Try UIA first, fallback to OCR
        result = _find_text_uia(text_norm, control_type=control_type, match_mode=match_mode)
        if result["found"]:
            return result
        ocr_result = _find_text_ocr(text_norm, match_mode=match_mode)
        if ocr_result["found"]:
            return ocr_result
        return result  # return UIA result (no match) as primary


def _find_text_uia(
    text_norm: str,
    control_type: Optional[str] = None,
    match_mode: str = "contains",
) -> Dict[str, Any]:
    """Search UIA controls for text match."""
    try:
        from wyzer.tools.desktop.perceive_uia import perceive_uia_focused_window
        snapshot = perceive_uia_focused_window(max_nodes=80)
    except Exception as e:
        return {"found": False, "evidence": f"uia_error: {e}", "method": "uia"}

    ct_lower = (control_type or "").strip().lower() if control_type else None

    matches = []
    for ctrl in snapshot.get("controls", []):
        name_norm = _normalize_label(ctrl.get("name") or "")
        if not name_norm:
            continue
        # control_type filter
        if ct_lower:
            ctrl_type = (ctrl.get("control_type") or "").strip().lower()
            if ctrl_type != ct_lower:
                continue
        if _text_matches(text_norm, name_norm, match_mode):
            matches.append({
                "name": ctrl.get("name"),
                "control_type": ctrl.get("control_type"),
                "rect": ctrl.get("rect"),
            })

    mode_tag = f"mode={match_mode}"
    type_tag = f" type={control_type}" if control_type else ""

    if matches:
        names = [m["name"] for m in matches if m.get("name")][:3]
        n = len(matches)
        word = "match" if n == 1 else "matches"
        summary = f"Yes \u2014 I found {n} {word}: {', '.join(names)}." if names else f"Yes \u2014 I found {n} {word}."
        return {
            "found": True,
            "evidence": f"UIA({mode_tag}{type_tag}): found {len(matches)} control(s) matching '{text_norm}'",
            "method": "uia",
            "matches": matches,
            "summary": summary,
        }

    return {
        "found": False,
        "evidence": f"UIA({mode_tag}{type_tag}): no controls matching '{text_norm}' in {len(snapshot.get('controls', []))} scanned",
        "method": "uia",
        "matches": [],
        "summary": f"No \u2014 I didn't find any controls matching '{text_norm}'.",
    }


def _find_text_ocr(text_norm: str, match_mode: str = "contains") -> Dict[str, Any]:
    """Search OCR output for text match."""
    try:
        from wyzer.tools.desktop.ocr_tool import _run_ocr
        ocr = _run_ocr()
    except Exception as e:
        return {"found": False, "evidence": f"ocr_error: {e}", "method": "ocr"}

    if ocr.get("missing_dependency"):
        return {
            "found": False,
            "evidence": "OCR not available (Tesseract not installed)",
            "method": "ocr",
            "missing_dependency": True,
        }

    full_norm = _normalize_label(ocr.get("full_text") or "")
    if _text_matches(text_norm, full_norm, match_mode):
        matching_lines = [
            ln["text"] for ln in ocr.get("lines", [])
            if _text_matches(text_norm, _normalize_label(ln["text"]), match_mode)
        ]
        n = len(matching_lines)
        word = "match" if n == 1 else "matches"
        preview = matching_lines[:3]
        summary = f"Yes \u2014 I found {n} {word}: {', '.join(preview)}." if preview else f"Yes \u2014 I found '{text_norm}' on screen."
        return {
            "found": True,
            "evidence": f"OCR(mode={match_mode}): found '{text_norm}' in text",
            "method": "ocr",
            "matches": matching_lines,
            "summary": summary,
        }

    return {
        "found": False,
        "evidence": f"OCR(mode={match_mode}): '{text_norm}' not found in {len(ocr.get('lines', []))} lines",
        "method": "ocr",
        "matches": [],
        "summary": f"No \u2014 I didn't find any text matching '{text_norm}'.",
    }


# ── install_succeeded_check ───────────────────────────────────────────────

# Indicators that strongly signal a *completed install* when they appear as
# the full control label (exact match after normalisation).
_SUCCESS_EXACT_LABELS = frozenset({
    "play", "launch", "open", "run", "finish", "done",
    "close", "restart", "restart now", "reboot",
    "install complete", "installation complete",
    "successfully installed", "setup complete",
    "download complete",
})

# Indicators matched with word-boundary search (may appear inside labels
# like "Installation completed successfully").
_SUCCESS_WORD_INDICATORS = [
    re.compile(r"\b(?:installed|completed|succeeded|successful(?:ly)?|finished)\b"),
    re.compile(r"\b(?:setup\s+(?:is\s+)?(?:complete|finished|done))\b"),
    re.compile(r"\b(?:installation\s+(?:is\s+)?(?:complete|finished|done|successful))\b"),
    re.compile(r"\b(?:ready\s+to\s+(?:use|launch|play))\b"),
]

# Controls whose names contain these substrings are NEVER install evidence
# even if they also match a success keyword.
_SUCCESS_NOISE_WORDS = frozenset({
    "conversation", "options", "settings", "preferences",
    "menu", "history", "bookmark", "tab", "search",
    "autocomplete", "tooltip", "dropdown",
})

_FAILURE_INDICATORS = ["error", "failed", "retry", "cancel", "problem", "could not"]

# Window titles that look install-related (used for context gating).
_INSTALL_WINDOW_RE = re.compile(
    r"(?:install|setup|wizard|update|download|uninstall|microsoft\s+store|"
    r"software\s+center|winget|chocolatey|steam|epic\s+games|gog|"
    r"app\s+installer)",
    re.IGNORECASE,
)


def _is_noise_control(name_lower: str) -> bool:
    """Return True if the control name is clearly unrelated to installation."""
    return any(w in name_lower for w in _SUCCESS_NOISE_WORDS)


def install_succeeded_check() -> Dict[str, Any]:
    """
    Deterministic check: did an install succeed?

    Strategy (UIA-first):
    1. Scan controls for success indicators ("Play", "Installed", "Complete")
    2. Check for error dialogs
    3. Check progress bar (value=100)
    4. OCR fallback if UIA inconclusive

    Returns {status: "success"|"fail"|"unknown", evidence: str, details: dict}.
    """
    try:
        from wyzer.tools.desktop.perceive_uia import perceive_uia_focused_window
        snapshot = perceive_uia_focused_window(max_nodes=80)
    except Exception as e:
        return {"status": "unknown", "evidence": f"uia_error: {e}", "details": {}}

    if snapshot.get("uac_detected"):
        return {"status": "unknown", "evidence": "UAC/secure desktop detected", "details": {"uac": True}}

    controls = snapshot.get("controls", [])
    progress = snapshot.get("progress")
    errors_list = snapshot.get("errors", [])

    # Window-title context: if the focused window doesn't look install-related
    # we require stronger evidence before claiming success.
    window_info = snapshot.get("window") or {}
    window_title = (window_info.get("title") or "").strip()
    is_install_window = bool(_INSTALL_WINDOW_RE.search(window_title))

    # Collect evidence (deduplicated)
    success_matches: list[dict] = []
    fail_matches: list[dict] = []
    _seen_success: set[str] = set()
    _seen_fail: set[str] = set()

    for ctrl in controls:
        raw_name = (ctrl.get("name") or "").strip()
        name_lower = raw_name.lower()
        ctrl_type = (ctrl.get("control_type") or "").lower()

        if not name_lower or name_lower in _seen_success:
            continue

        # Skip controls that are clearly noise
        if _is_noise_control(name_lower):
            continue

        norm = _normalize_label(raw_name)

        # --- Exact-label match (button-specific) ---
        if ctrl_type == "button" and norm in _SUCCESS_EXACT_LABELS:
            # Extra gate: bare "open"/"play"/"launch" buttons only count
            # when the window itself looks install-related.
            if norm in ("open", "play", "launch", "run") and not is_install_window:
                continue
            _seen_success.add(name_lower)
            success_matches.append({
                "name": raw_name, "type": ctrl.get("control_type"),
                "indicator": norm, "match": "exact_label",
            })
            continue

        # --- Word-boundary match (text/label controls) ---
        for pat in _SUCCESS_WORD_INDICATORS:
            if pat.search(name_lower):
                _seen_success.add(name_lower)
                success_matches.append({
                    "name": raw_name, "type": ctrl.get("control_type"),
                    "indicator": pat.pattern, "match": "word_boundary",
                })
                break

        # --- Failure indicators (substring is fine — "error" is unambiguous) ---
        for indicator in _FAILURE_INDICATORS:
            if indicator in name_lower and name_lower not in _seen_fail:
                _seen_fail.add(name_lower)
                fail_matches.append({
                    "name": raw_name, "type": ctrl.get("control_type"),
                    "indicator": indicator,
                })

    # Check progress
    if progress and progress.get("value") is not None:
        try:
            val = float(progress["value"])
            if val >= 100:
                success_matches.append({"type": "progress", "value": val})
        except (ValueError, TypeError):
            pass

    # Check for error dialogs
    for dialog in snapshot.get("dialogs", []):
        title_lower = (dialog.get("title") or "").lower()
        if title_lower in _seen_fail:
            continue
        for indicator in _FAILURE_INDICATORS:
            if indicator in title_lower:
                _seen_fail.add(title_lower)
                fail_matches.append({"dialog": dialog.get("title"), "indicator": indicator})

    # Determine status
    details = {
        "success_evidence": success_matches,
        "fail_evidence": fail_matches,
        "controls_scanned": len(controls),
        "window_title": window_title,
        "is_install_window": is_install_window,
    }

    if fail_matches and not success_matches:
        highlights = list(dict.fromkeys(m.get('name') or m.get('dialog') for m in fail_matches))[:3]
        summary = f"It looks like the install failed. I see: {', '.join(str(h) for h in highlights)}."
        return {"status": "fail", "evidence": f"Error indicators found: {highlights}", "details": details, "summary": summary}
    if success_matches and not fail_matches:
        highlights = list(dict.fromkeys(m.get('name') or str(m) for m in success_matches))[:3]
        summary = f"The install succeeded. I see: {', '.join(str(h) for h in highlights)}."
        return {"status": "success", "evidence": f"Success indicators found: {highlights}", "details": details, "summary": summary}
    if success_matches and fail_matches:
        summary = "I'm not sure \u2014 I see both success and error indicators."
        return {"status": "unknown", "evidence": "Mixed signals: both success and error indicators found", "details": details, "summary": summary}

    summary = (
        "I can't verify download or install status right now. "
        "I didn't find any clear success or failure indicators on screen. "
        "If something is downloading, try telling me which app so I can check that window."
    )
    return {"status": "unknown", "evidence": f"No clear indicators in {len(controls)} controls", "details": details, "summary": summary}


# ── Tool wrappers ──────────────────────────────────────────────────────────

class UIFindTextTool(ToolBase):
    """Find text in the focused window (UIA or OCR)."""

    def __init__(self):
        super().__init__()
        self._name = "ui_find_text"
        self._description = (
            "Search for text in the focused window using UIA controls or OCR. "
            "Returns {found: bool, evidence, matches}."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to search for."},
                "method": {
                    "type": "string",
                    "description": "'uia', 'ocr', or 'auto' (default: 'uia').",
                    "default": "uia",
                },
                "control_type": {
                    "type": "string",
                    "description": "Optional UIA control type filter, e.g. 'Button'.",
                },
                "match_mode": {
                    "type": "string",
                    "description": "'exact', 'word', or 'contains' (default: 'contains').",
                    "default": "contains",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        text = kwargs.get("text", "")
        method = kwargs.get("method", "uia")
        control_type = kwargs.get("control_type")
        match_mode = kwargs.get("match_mode", "contains")

        start = time.perf_counter()
        result = ui_find_text(text, method=method, control_type=control_type, match_mode=match_mode)
        latency_ms = int((time.perf_counter() - start) * 1000)
        result["latency_ms"] = latency_ms

        emit_event("perception", {
            "source": "ui_find_text",
            "query": text,
            "found": result["found"],
            "method": result.get("method"),
            "match_mode": match_mode,
            "control_type": control_type,
            "latency_ms": latency_ms,
        })

        return result


class InstallSucceededCheckTool(ToolBase):
    """Check if an install/download succeeded (deterministic)."""

    def __init__(self):
        super().__init__()
        self._name = "install_succeeded_check"
        self._description = (
            "Deterministic check: did an install/download succeed? "
            "Looks for Play/Installed/Complete buttons and error dialogs."
        )
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        start = time.perf_counter()
        result = install_succeeded_check()
        latency_ms = int((time.perf_counter() - start) * 1000)
        result["latency_ms"] = latency_ms

        emit_event("perception", {
            "source": "install_check",
            "status": result["status"],
            "evidence": result["evidence"],
            "latency_ms": latency_ms,
        })

        return result
