"""
wyzer.desktop.resolve_target — Deterministic candidate resolver (Phase 16+).

Given a target phrase and a perception result (UIA or OCR), score and rank
UI-element candidates without any LLM involvement.

Inputs:
    target_phrase       – e.g. "ask anything"
    perception_result   – dict from perceive_uia or perceive_ocr
    preferred_types     – list of control types to boost (e.g. ["Edit", "Button"])

Output:
    ResolveResult with ranked candidates (max 10) + ambiguity flag.

Scoring rules (all deterministic):
    exact match           100
    prefix match           80
    word-boundary contain  70
    substring contain      55
    fuzzy (>=0.7 ratio)    45 * ratio
    control_type boost    +15 if in preferred list
    enabled boost         + 5
    visible-rect boost    + 5
    short/generic penalty  -8
    long-name penalty      -0.3 per char over 60 (max -20)
    promotion bonus       + 8 (promoted from text node to clickable ancestor)
    promotion depth pen.  -1 per level

UIA Ancestor Promotion (Phase 16+):
    When a match is on a non-clickable type (Text, StaticText, Document, Custom),
    walk up the UIA parent chain (max 5 levels) and promote to the nearest
    clickable ancestor (InvokePattern support or ListItem/Button/Hyperlink/Pane).
    Original score transfers to the promoted ancestor with metadata recorded.

Control Type Heuristics:
    Never force control_type unless the user explicitly says it.
    Preferred ordering by intent:
        Edit / Document  → typing intent
        ListItem / Button / Hyperlink → navigation intent
        Pane → fallback container
    Scoring based on: text match quality, control interactivity, rect size,
    promotion depth penalty (small).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


# ── Thresholds ──────────────────────────────────────────────────────────

SCORE_MIN: float = 40.0      # below this → no confident candidate
GAP_MIN: float = 10.0        # if gap(#1, #2) < this → ambiguous
MAX_CANDIDATES: int = 10
PROMOTION_MAX_DEPTH: int = 5  # max UIA parent chain walk depth

# Types that are NOT inherently clickable → candidates for promotion
_NON_CLICKABLE_TYPES = frozenset({"Text", "StaticText", "Document", "Custom"})

# Types considered interactive / clickable targets for promotion
_CLICKABLE_TYPES = frozenset({
    "Button", "ListItem", "Hyperlink", "Pane", "TabItem",
    "MenuItem", "TreeItem", "Edit", "ComboBox", "CheckBox",
    "RadioButton",
})

# Interactivity score: higher = more likely to be a click target
_INTERACTIVITY_RANK: Dict[str, float] = {
    "Button": 10.0,
    "Hyperlink": 9.0,
    "ListItem": 8.0,
    "TabItem": 8.0,
    "MenuItem": 8.0,
    "TreeItem": 7.0,
    "Edit": 6.0,
    "ComboBox": 6.0,
    "CheckBox": 5.0,
    "RadioButton": 5.0,
    "Pane": 3.0,
}


# ── Data classes ────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """A scored UI-element candidate."""
    id: int
    name: str
    control_type: str
    rect: Optional[Dict[str, int]]
    score: float
    source: str                     # "uia" | "ocr"
    enabled: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    promotion: Optional[Dict[str, Any]] = None  # ancestor promotion metadata

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "control_type": self.control_type,
            "rect": self.rect,
            "score": round(self.score, 2),
            "source": self.source,
            "enabled": self.enabled,
            "extra": self.extra,
        }
        if self.promotion:
            d["promotion"] = self.promotion
        return d


@dataclass
class ResolveResult:
    """Result of candidate resolution."""
    candidates: List[Candidate]
    ambiguous: bool
    best: Optional[Candidate]
    reason: str                     # human-readable explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "ambiguous": self.ambiguous,
            "best": self.best.to_dict() if self.best else None,
            "reason": self.reason,
        }


# ── Helpers ─────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")

# Short/generic labels that get a penalty
_GENERIC_LABELS = frozenset({
    "", "ok", "yes", "no", "cancel", "close", "back", "next",
    "menu", "more", "options", "settings", "help",
})


def _norm(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    t = (text or "").strip().lower()
    t = _PUNCT_RE.sub(" ", t)
    return _SPACES_RE.sub(" ", t).strip()


def _score_text_match(target: str, label: str) -> float:
    """Score how well *label* matches the *target* phrase.

    Both inputs must already be normalised via ``_norm``.
    Returns a raw score in [0, 100].
    """
    if not target or not label:
        return 0.0

    # 1. Exact
    if label == target:
        return 100.0

    # 2. Prefix (label starts with target or vice-versa)
    if label.startswith(target) or target.startswith(label):
        return 80.0

    # 3. Word-boundary containment (target words appear inside label)
    if re.search(r"\b" + re.escape(target) + r"\b", label):
        return 70.0

    # 4. Simple substring — require the shorter side to be >= 3 chars
    #    so single-letter controls like 'p' don't match 'keep'.
    shorter_len = min(len(target), len(label))
    if shorter_len >= 3 and (target in label or label in target):
        return 55.0

    # 5. Fuzzy (SequenceMatcher)
    ratio = SequenceMatcher(None, target, label).ratio()
    if ratio >= 0.70:
        return 45.0 * ratio   # 31.5 – 45

    return 0.0


def _apply_boosts(
    score: float,
    control_type: str,
    preferred_types: List[str],
    enabled: Optional[bool],
    rect: Optional[Dict[str, int]],
    label_norm: str,
    promotion: Optional[Dict[str, Any]] = None,
) -> float:
    """Apply deterministic boost/penalty adjustments to a raw score.

    Includes interactivity ranking, rect-size awareness, and
    promotion depth penalty.
    """
    if control_type in preferred_types:
        score += 15.0
    if enabled is True:
        score += 5.0
    if rect is not None:
        score += 5.0
        # Bonus for non-trivial rect (area > 100 px²)
        w = abs(rect.get("r", 0) - rect.get("l", 0))
        h = abs(rect.get("b", 0) - rect.get("t", 0))
        if w * h > 100:
            score += 2.0
    if label_norm in _GENERIC_LABELS or len(label_norm) <= 1:
        score -= 8.0

    # Long-name penalty: labels longer than ~60 chars are likely chat titles,
    # log entries, or paragraph text — not real UI control labels.
    # Penalise proportionally so short exact matches always win.
    if len(label_norm) > 60:
        excess = len(label_norm) - 60
        score -= min(excess * 0.3, 20.0)   # max −20 for very long names

    # Interactivity boost: prefer obviously clickable controls
    interactivity = _INTERACTIVITY_RANK.get(control_type, 0.0)
    score += interactivity * 0.5  # scaled: max +5 for Button

    # Promotion: bonus for being promoted (clickable ancestor found),
    # but small penalty per depth level
    if promotion:
        score += 8.0   # promoted candidate bonus
        depth = promotion.get("promotion_depth", 0)
        score -= depth * 1.0  # small penalty per level

    return score


def _has_nontrivial_rect(rect: Optional[Dict[str, int]]) -> bool:
    """Check if a rect has non-zero area."""
    if rect is None:
        return False
    w = abs(rect.get("r", 0) - rect.get("l", 0))
    h = abs(rect.get("b", 0) - rect.get("t", 0))
    return w > 2 and h > 2


def _try_ancestor_promotion(
    ctrl: Dict[str, Any],
    all_controls: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Attempt UIA ancestor promotion for non-clickable control types.

    When a candidate match is found on a Text/StaticText/Document/Custom
    control, walk up the parent chain (via the ``_parent`` key or the
    ``parent_handle`` key injected by perceive_uia) and promote to the
    nearest clickable ancestor.

    This is a perception-side promotion: it relies on the ``_uia_elem``
    back-reference injected during live UIA walks (not available in
    static test data). For static data, we fall back to a heuristic:
    scan all controls for a parent whose rect fully contains the child's
    rect and is of a clickable type.

    Returns a promoted control dict or None.
    """
    child_type = ctrl.get("control_type") or ctrl.get("type") or ""
    if child_type not in _NON_CLICKABLE_TYPES:
        return None  # already clickable, no promotion needed

    child_rect = ctrl.get("rect")

    # ── Live UIA element back-reference ─────────────────────────────
    uia_elem = ctrl.get("_uia_elem")
    if uia_elem is not None:
        return _promote_via_uia_parent(uia_elem, child_type)

    # ── Static / test fallback: geometric containment ───────────────
    if not child_rect:
        return None

    return _promote_via_geometry(ctrl, child_rect, child_type, all_controls)


def _promote_via_uia_parent(uia_elem: Any, child_type: str) -> Optional[Dict[str, Any]]:
    """Walk the live UIA parent chain to find a clickable ancestor."""
    current = uia_elem
    for depth in range(1, PROMOTION_MAX_DEPTH + 1):
        try:
            parent = current.parent()
            if parent is None:
                break
        except Exception:
            break

        try:
            parent_type = parent.element_info.control_type or ""
        except Exception:
            parent_type = ""

        # Check if parent is clickable
        is_clickable = parent_type in _CLICKABLE_TYPES

        # Also check InvokePattern
        if not is_clickable:
            try:
                iface = parent.iface_invoke
                if iface:
                    is_clickable = True
            except Exception:
                pass

        if not is_clickable:
            current = parent
            continue

        # Check enabled + non-trivial rect
        try:
            enabled = parent.is_enabled()
        except Exception:
            enabled = None

        try:
            r = parent.rectangle()
            parent_rect = {"l": r.left, "t": r.top, "r": r.right, "b": r.bottom}
        except Exception:
            parent_rect = None

        if enabled is False:
            current = parent
            continue
        if not _has_nontrivial_rect(parent_rect):
            current = parent
            continue

        # Promoted!
        try:
            parent_name = (parent.window_text() or "").strip()
        except Exception:
            parent_name = ""

        return {
            "name": parent_name,
            "control_type": parent_type,
            "rect": parent_rect,
            "enabled": enabled,
            "_uia_elem": parent,
            "_promotion": {
                "promoted_from": child_type,
                "promoted_to": parent_type,
                "promotion_depth": depth,
            },
        }

        current = parent

    return None


def _promote_via_geometry(
    child: Dict[str, Any],
    child_rect: Dict[str, int],
    child_type: str,
    all_controls: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Fallback: find a clickable ancestor by geometric rect containment.

    Scans all controls for one whose rect fully contains the child rect,
    is of a clickable type, and is enabled.  Picks the tightest fit
    (smallest area).
    """
    cl, ct_r, cr, cb = (child_rect.get("l", 0), child_rect.get("t", 0),
                          child_rect.get("r", 0), child_rect.get("b", 0))
    best_parent = None
    best_area = float("inf")

    for other in all_controls:
        if other is child:
            continue
        other_type = other.get("control_type") or other.get("type") or ""
        if other_type not in _CLICKABLE_TYPES:
            continue
        other_rect = other.get("rect")
        if not other_rect:
            continue
        ol, ot, or_, ob = (other_rect.get("l", 0), other_rect.get("t", 0),
                             other_rect.get("r", 0), other_rect.get("b", 0))
        # Check containment
        if ol <= cl and ot <= ct_r and or_ >= cr and ob >= cb:
            area = (or_ - ol) * (ob - ot)
            if area < best_area:
                if other.get("enabled") is not False:
                    best_area = area
                    best_parent = other

    if best_parent is None:
        return None

    return {
        "name": (best_parent.get("name") or "").strip(),
        "control_type": best_parent.get("control_type") or best_parent.get("type") or "",
        "rect": best_parent.get("rect"),
        "enabled": best_parent.get("enabled"),
        "_promotion": {
            "promoted_from": child_type,
            "promoted_to": best_parent.get("control_type") or "",
            "promotion_depth": 1,  # geometric = 1 level
        },
    }


# ── Public API ──────────────────────────────────────────────────────────

def resolve_candidates(
    target_phrase: str,
    perception: Dict[str, Any],
    preferred_types: Optional[List[str]] = None,
    source: str = "uia",
) -> ResolveResult:
    """Score + rank candidates from a perception result.

    Args:
        target_phrase:  The phrase the user wants to click (e.g. "ask anything").
        perception:     Dict from perceive_uia or perceive_ocr. Expected keys:
                        ``controls`` (list of dicts with name/control_type/rect/enabled)
                        or ``lines``/``words`` (OCR).
        preferred_types: Control types to boost (e.g. ["Edit", "Button"]).
        source:         "uia" or "ocr".

    Returns:
        ResolveResult with candidates sorted by descending score, plus
        ambiguity flag.
    """
    if preferred_types is None:
        preferred_types = ["Edit", "TextBox"]

    target_norm = _norm(target_phrase)
    if not target_norm:
        return ResolveResult(
            candidates=[],
            ambiguous=False,
            best=None,
            reason="empty target phrase",
        )

    raw_candidates: List[Candidate] = []
    cid = 0

    # ── UIA controls ────────────────────────────────────────────────
    controls = perception.get("controls") or []
    for ctrl in controls:
        name = ctrl.get("name") or ctrl.get("text") or ""
        if not name.strip():
            continue
        label_norm = _norm(name)
        raw_score = _score_text_match(target_norm, label_norm)
        if raw_score <= 0:
            continue

        ct = ctrl.get("control_type") or ctrl.get("type") or ""
        enabled = ctrl.get("enabled")
        rect = ctrl.get("rect")

        # ── Ancestor promotion for non-clickable types ──────────────
        promotion_meta = None
        actual_ct = ct
        actual_rect = rect
        actual_enabled = enabled

        if ct in _NON_CLICKABLE_TYPES:
            promoted = _try_ancestor_promotion(ctrl, controls)
            if promoted:
                promotion_meta = promoted.get("_promotion")
                actual_ct = promoted.get("control_type") or ct
                actual_rect = promoted.get("rect") or rect
                actual_enabled = promoted.get("enabled", enabled)
                # Keep original name (the matched text), but use parent's
                # clickable properties

        score = _apply_boosts(
            raw_score, actual_ct, preferred_types,
            actual_enabled, actual_rect, label_norm,
            promotion=promotion_meta,
        )

        extra = {k: v for k, v in ctrl.items()
                 if k not in ("name", "text", "control_type", "type",
                              "rect", "enabled", "_uia_elem")}
        if promotion_meta:
            extra["_promoted_rect"] = actual_rect

        raw_candidates.append(Candidate(
            id=cid,
            name=name.strip(),
            control_type=actual_ct,
            rect=actual_rect,
            score=score,
            source=source,
            enabled=actual_enabled,
            extra=extra,
            promotion=promotion_meta,
        ))
        cid += 1

    # ── OCR lines (when source == "ocr") ────────────────────────────
    if source == "ocr":
        for ln in perception.get("lines") or []:
            text = ln.get("text", "") if isinstance(ln, dict) else str(ln)
            if not text.strip():
                continue
            label_norm = _norm(text)
            raw_score = _score_text_match(target_norm, label_norm)
            if raw_score <= 0:
                continue
            rect = ln.get("rect") if isinstance(ln, dict) else None
            score = raw_score
            if rect:
                score += 5.0
            raw_candidates.append(Candidate(
                id=cid,
                name=text.strip(),
                control_type="OCR_Text",
                rect=rect,
                score=score,
                source="ocr",
                extra={},
            ))
            cid += 1

        for word in perception.get("words") or []:
            text = word.get("text", "") if isinstance(word, dict) else str(word)
            if not text.strip():
                continue
            label_norm = _norm(text)
            raw_score = _score_text_match(target_norm, label_norm)
            if raw_score <= 0:
                continue
            rect = word.get("rect") if isinstance(word, dict) else None
            score = raw_score
            if rect:
                score += 5.0
            raw_candidates.append(Candidate(
                id=cid,
                name=text.strip(),
                control_type="OCR_Word",
                rect=rect,
                score=score,
                source="ocr",
                extra={},
            ))
            cid += 1

    # ── Sort + trim ─────────────────────────────────────────────────
    raw_candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = raw_candidates[:MAX_CANDIDATES]

    # ── Ambiguity detection ─────────────────────────────────────────
    if not candidates:
        return ResolveResult(
            candidates=[],
            ambiguous=False,
            best=None,
            reason="no matching candidates",
        )

    best = candidates[0]

    if best.score < SCORE_MIN:
        return ResolveResult(
            candidates=candidates,
            ambiguous=True,
            best=best,
            reason=f"best score {best.score:.1f} < threshold {SCORE_MIN}",
        )

    if len(candidates) >= 2:
        gap = best.score - candidates[1].score
        if gap < GAP_MIN:
            return ResolveResult(
                candidates=candidates,
                ambiguous=True,
                best=best,
                reason=f"gap ({gap:.1f}) between top-2 < threshold {GAP_MIN}",
            )

    return ResolveResult(
        candidates=candidates,
        ambiguous=False,
        best=best,
        reason="confident match",
    )
