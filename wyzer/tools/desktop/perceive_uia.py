"""
perceive_uia_focused_window — Desktop Ground Truth Tool (Phase 14)

Returns a minimal UIA tree snapshot of the focused window:
  {window:{title, exe, hwnd, rect},
   controls:[{name, control_type, rect, enabled}],
   dialogs:[{title, rect}],
   progress:{value?, text?}?,
   errors:[string]?}

Uses pywinauto (UIA backend). Keeps output small (max_nodes default 60).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from wyzer.tools.tool_base import ToolBase

# Useful UIA control types to keep (others are noise)
_KEEP_TYPES = frozenset({
    "Button", "Edit", "Text", "ComboBox", "CheckBox",
    "RadioButton", "ProgressBar", "Window", "Pane",
    "TabItem", "MenuItem", "ListItem", "TreeItem",
    "Hyperlink", "Image", "StatusBar", "Slider",
    "Dialog",
})

_DEFAULT_MAX_NODES = 60


def _try_pywinauto(max_nodes: int = _DEFAULT_MAX_NODES) -> Dict[str, Any]:
    """
    Walk the UIA tree of the focused window using pywinauto.

    Returns a dict with window, controls, dialogs, progress, errors.
    """
    try:
        from pywinauto import Desktop
    except ImportError:
        return {
            "window": {},
            "controls": [],
            "dialogs": [],
            "progress": None,
            "errors": ["pywinauto_not_installed"],
            "timestamp": time.time(),
        }

    result: Dict[str, Any] = {
        "window": {},
        "controls": [],
        "dialogs": [],
        "progress": None,
        "errors": [],
        "timestamp": time.time(),
    }

    try:
        desktop = Desktop(backend="uia")
        top_windows = desktop.windows()
        if not top_windows:
            result["errors"].append("no_top_windows")
            return result

        # Get the foreground window
        import ctypes
        hwnd_fg = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd_fg:
            result["errors"].append("no_foreground_window")
            result["uac_detected"] = True
            return result

        # Find matching pywinauto wrapper
        target = None
        for w in top_windows:
            try:
                if w.handle == hwnd_fg:
                    target = w
                    break
            except Exception:
                continue

        if target is None:
            # Fallback: use the first top-level window
            result["errors"].append("foreground_not_in_uia_list")
            target = top_windows[0]

        # Window info
        try:
            rect = target.rectangle()
            result["window"] = {
                "title": target.window_text() or "",
                "hwnd": target.handle,
                "rect": {"l": rect.left, "t": rect.top, "r": rect.right, "b": rect.bottom},
            }
        except Exception as e:
            result["errors"].append(f"window_info: {e}")

        # Get exe from PID
        try:
            pid = target.process_id()
            result["window"]["pid"] = pid
            import psutil
            try:
                result["window"]["exe"] = psutil.Process(pid).name()
            except Exception:
                pass
        except Exception:
            pass

        # Walk descendants
        controls: List[Dict[str, Any]] = []
        dialogs: List[Dict[str, Any]] = []
        progress_info: Optional[Dict[str, Any]] = None
        node_count = 0

        try:
            descendants = target.descendants()
        except Exception:
            descendants = []

        for elem in descendants:
            if node_count >= max_nodes:
                break
            try:
                ctrl_type = elem.element_info.control_type or ""
                name = elem.window_text() or ""

                # Filter: keep only useful types or elements with names
                if ctrl_type not in _KEEP_TYPES and not name.strip():
                    continue

                # ── Noise reduction ──────────────────────────────────
                # Skip Text elements whose name is very long (log dumps,
                # code blocks, data tables, etc.)
                stripped = name.strip()
                if ctrl_type == "Text" and len(stripped) > 200:
                    continue
                # Skip names that look like log lines / stack traces
                if stripped.count("\n") > 3:
                    continue

                # Build record
                try:
                    r = elem.rectangle()
                    elem_rect = {"l": r.left, "t": r.top, "r": r.right, "b": r.bottom}
                except Exception:
                    elem_rect = None

                try:
                    enabled = elem.is_enabled()
                except Exception:
                    enabled = None

                record = {
                    "name": name.strip(),
                    "control_type": ctrl_type,
                    "rect": elem_rect,
                    "enabled": enabled,
                }
                controls.append(record)
                node_count += 1

                # Track dialogs
                if ctrl_type in ("Window", "Dialog") and name.strip():
                    dialogs.append({
                        "title": name.strip(),
                        "rect": elem_rect,
                    })

                # Track progress bars
                if ctrl_type == "ProgressBar":
                    prog: Dict[str, Any] = {"text": name.strip()}
                    try:
                        from pywinauto.controls.uia_controls import ProgressBarWrapper
                        if isinstance(elem, ProgressBarWrapper):
                            prog["value"] = elem.get_value()
                    except Exception:
                        pass
                    if progress_info is None:
                        progress_info = prog

            except Exception:
                continue

        result["controls"] = controls
        result["dialogs"] = dialogs
        result["progress"] = progress_info

    except Exception as exc:
        result["errors"].append(str(exc))

    return result


def perceive_uia_focused_window(max_nodes: int = _DEFAULT_MAX_NODES) -> Dict[str, Any]:
    """Public API: return a UIA snapshot of the focused window.

    Always returns the documented schema:
    {window, controls, dialogs, progress, errors, timestamp}
    """
    return _try_pywinauto(max_nodes=max_nodes)


class PerceiveUIAFocusedWindowTool(ToolBase):
    """UIA perception of the focused window — deterministic, no LLM."""

    def __init__(self):
        super().__init__()
        self._name = "perceive_uia_focused_window"
        self._description = (
            "Return a minimal UIA tree snapshot of the focused window: "
            "controls, dialogs, progress bars. Deterministic perception."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "max_nodes": {
                    "type": "integer",
                    "description": "Maximum number of controls to return (default 60).",
                    "default": 60,
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event, update_last_perception
        from wyzer.tools.desktop.truth_contract import normalize_perception

        max_nodes = kwargs.get("max_nodes", _DEFAULT_MAX_NODES)
        start = time.perf_counter()
        info = _try_pywinauto(max_nodes=max_nodes)
        latency_ms = int((time.perf_counter() - start) * 1000)
        info["latency_ms"] = latency_ms

        # Emit event
        names_sample = [c["name"] for c in info.get("controls", [])[:5] if c.get("name")]
        emit_event("perception", {
            "source": "uia",
            "found_controls_count": len(info.get("controls", [])),
            "found_names_sample": names_sample,
            "progress": info.get("progress"),
            "latency_ms": latency_ms,
        })

        # Normalize to truth-contract schema before storing
        normalized = normalize_perception(info)
        update_last_perception(normalized)
        return info
