"""
wyzer.world.window_diff

Phase 12: Pure diff logic for window snapshots.

This module contains ONLY pure Python logic for comparing window snapshots
and detecting changes. It has NO OS dependencies, making it fully testable
in CI without Windows APIs.

Event Types:
- opened: Window appeared in new snapshot but not in old
- closed: Window in old snapshot but not in new
- moved: Same hwnd but monitor changed or significant rect change
- title_changed: Same hwnd but title differs
- focus_changed: Foreground hwnd differs

Usage:
    events = diff_snapshots(prev_by_hwnd, next_by_hwnd, prev_focus, next_focus)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

# Type aliases
WindowRecord = Dict[str, Any]
EventRecord = Dict[str, Any]


def diff_snapshots(
    prev_by_hwnd: Dict[int, WindowRecord],
    next_by_hwnd: Dict[int, WindowRecord],
    prev_focus: Optional[int],
    next_focus: Optional[int],
) -> List[EventRecord]:
    """
    Compute diff between two window snapshots.
    
    This is a pure function with no side effects. It compares two
    snapshots (keyed by hwnd) and returns a list of change events.
    
    Args:
        prev_by_hwnd: Previous snapshot as {hwnd: record} dict
        next_by_hwnd: Current snapshot as {hwnd: record} dict
        prev_focus: hwnd of previously focused window (or None)
        next_focus: hwnd of currently focused window (or None)
        
    Returns:
        List of event dicts describing changes
    """
    events: List[EventRecord] = []
    ts = time.time()
    
    prev_hwnds = set(prev_by_hwnd.keys())
    next_hwnds = set(next_by_hwnd.keys())
    
    # OPENED: in next but not in prev
    for hwnd in next_hwnds - prev_hwnds:
        rec = next_by_hwnd[hwnd]
        events.append({
            "ts": ts,
            "type": "opened",
            "hwnd": hwnd,
            "title": rec.get("title", ""),
            "process": rec.get("process", ""),
            "from_monitor": None,
            "to_monitor": rec.get("monitor"),
        })
    
    # CLOSED: in prev but not in next
    for hwnd in prev_hwnds - next_hwnds:
        rec = prev_by_hwnd[hwnd]
        events.append({
            "ts": ts,
            "type": "closed",
            "hwnd": hwnd,
            "title": rec.get("title", ""),
            "process": rec.get("process", ""),
            "from_monitor": rec.get("monitor"),
            "to_monitor": None,
        })
    
    # EXISTING: check for moves, title changes
    for hwnd in prev_hwnds & next_hwnds:
        prev_rec = prev_by_hwnd[hwnd]
        next_rec = next_by_hwnd[hwnd]
        
        # Check for monitor change (moved)
        prev_mon = prev_rec.get("monitor")
        next_mon = next_rec.get("monitor")
        if prev_mon != next_mon:
            events.append({
                "ts": ts,
                "type": "moved",
                "hwnd": hwnd,
                "title": next_rec.get("title", ""),
                "process": next_rec.get("process", ""),
                "from_monitor": prev_mon,
                "to_monitor": next_mon,
            })
            continue  # Don't also emit title_changed for same hwnd
        
        # Check for significant rect change (moved within monitor)
        if _rects_differ_significantly(
            prev_rec.get("rect"),
            next_rec.get("rect"),
            threshold=50,
        ):
            events.append({
                "ts": ts,
                "type": "moved",
                "hwnd": hwnd,
                "title": next_rec.get("title", ""),
                "process": next_rec.get("process", ""),
                "from_monitor": prev_mon,
                "to_monitor": next_mon,
            })
            continue
        
        # Check for title change
        prev_title = (prev_rec.get("title") or "").strip()
        next_title = (next_rec.get("title") or "").strip()
        if prev_title != next_title:
            events.append({
                "ts": ts,
                "type": "title_changed",
                "hwnd": hwnd,
                "title": next_title,
                "process": next_rec.get("process", ""),
                "from_monitor": next_mon,
                "to_monitor": next_mon,
            })
    
    # FOCUS CHANGED
    if prev_focus != next_focus and next_focus is not None:
        next_rec = next_by_hwnd.get(next_focus, {})
        events.append({
            "ts": ts,
            "type": "focus_changed",
            "hwnd": next_focus,
            "title": next_rec.get("title", ""),
            "process": next_rec.get("process", ""),
            "from_monitor": None,
            "to_monitor": next_rec.get("monitor"),
        })
    
    return events


def _rects_differ_significantly(
    rect1: Optional[List[int]],
    rect2: Optional[List[int]],
    threshold: int = 50,
) -> bool:
    """
    Check if two rects differ significantly.
    
    Rects are [left, top, right, bottom]. Returns True if any dimension
    has changed by more than threshold pixels.
    
    Args:
        rect1: First rect [l, t, r, b] or None
        rect2: Second rect [l, t, r, b] or None
        threshold: Pixel difference threshold
        
    Returns:
        True if rects differ significantly
    """
    if rect1 is None or rect2 is None:
        return rect1 != rect2
    
    if len(rect1) != 4 or len(rect2) != 4:
        return True
    
    try:
        for i in range(4):
            if abs(rect1[i] - rect2[i]) > threshold:
                return True
        return False
    except (TypeError, IndexError):
        return True


def build_hwnd_dict(windows: List[WindowRecord]) -> Dict[int, WindowRecord]:
    """
    Build a dict keyed by hwnd from a list of window records.
    
    Utility function for converting list snapshots to dict format
    needed by diff_snapshots().
    
    Args:
        windows: List of window record dicts
        
    Returns:
        Dict mapping hwnd -> record
    """
    result: Dict[int, WindowRecord] = {}
    for w in windows:
        hwnd = w.get("hwnd")
        if hwnd is not None:
            result[hwnd] = w
    return result
