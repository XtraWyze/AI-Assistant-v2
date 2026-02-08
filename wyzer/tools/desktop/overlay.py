"""
show_choice_overlay / wait_for_overlay_choice — Phase 16 Overlay Tools.

Always-on-top small window presenting TOP-N candidates for user selection
(keyboard 1/2/3 or mouse click).

IMPORTANT: Overlay runs in the Core process (tkinter on the main thread or
a dedicated UI thread).  Tool workers must NOT create GUI windows.

Architecture:
    show_choice_overlay  → spawns overlay in a daemon thread, returns overlay_id
    wait_for_overlay_choice → blocks until user picks or timeout expires
    _dismiss_overlay     → cleanup helper

The overlay is intentionally minimal (tkinter only, no extra deps).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from wyzer.tools.tool_base import ToolBase

logger = logging.getLogger(__name__)

# ── In-memory overlay store (Core-process only) ────────────────────────

@dataclass
class _OverlayState:
    """Mutable state for one overlay instance."""
    overlay_id: str
    options: List[Dict[str, Any]]
    choice: Optional[int] = None      # 1-based index or None
    cancelled: bool = False
    timed_out: bool = False
    done_event: threading.Event = field(default_factory=threading.Event)
    _root: Any = None                 # tkinter.Tk reference (internal)


_overlays: Dict[str, _OverlayState] = {}
_overlays_lock = threading.Lock()


# ── Tkinter overlay (runs in a daemon thread) ──────────────────────────

def _show_tkinter_overlay(state: _OverlayState, prompt: str) -> None:
    """Create a small always-on-top tkinter window with numbered options."""
    try:
        import tkinter as tk
    except ImportError:
        logger.warning("tkinter not available — overlay cannot be shown")
        state.cancelled = True
        state.done_event.set()
        return

    try:
        root = tk.Tk()
        root.title("Wyzer — Choose")
        root.attributes("-topmost", True)
        root.resizable(False, False)

        # Centre on screen
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        win_w, win_h = 420, 60 + 52 * len(state.options)
        x = (sw - win_w) // 2
        y = (sh - win_h) // 2
        root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        state._root = root

        # Prompt label
        tk.Label(
            root, text=prompt or "Choose an option:",
            font=("Segoe UI", 11, "bold"), wraplength=400, justify="left",
        ).pack(padx=10, pady=(10, 5), anchor="w")

        def _select(idx: int) -> None:
            state.choice = idx
            state.done_event.set()
            root.after(50, root.destroy)

        def _cancel() -> None:
            state.cancelled = True
            state.done_event.set()
            root.after(50, root.destroy)

        for i, opt in enumerate(state.options, 1):
            label = opt.get("label", f"Option {i}")
            hint = opt.get("hint", "")
            ct = opt.get("control_type", "")
            display = f"  {i}.  {label}"
            if ct:
                display += f"  [{ct}]"
            if hint:
                display += f"  — {hint}"
            btn = tk.Button(
                root, text=display, anchor="w", font=("Segoe UI", 10),
                command=lambda idx=i: _select(idx),
            )
            btn.pack(fill="x", padx=10, pady=2)

        cancel_btn = tk.Button(
            root, text="Cancel", font=("Segoe UI", 10), command=_cancel,
        )
        cancel_btn.pack(padx=10, pady=(5, 10))

        # Keyboard bindings: 1,2,3 and Escape
        for i in range(1, len(state.options) + 1):
            root.bind(str(i), lambda e, idx=i: _select(idx))
        root.bind("<Escape>", lambda e: _cancel())

        root.protocol("WM_DELETE_WINDOW", _cancel)
        root.mainloop()
    except Exception:
        logger.exception("Overlay tkinter error")
        state.cancelled = True
        state.done_event.set()


def show_overlay(
    prompt: str,
    options: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Show an always-on-top overlay with up to 6 numbered options.

    Returns {overlay_id, shown} or {error}.
    """
    if not options:
        return {"error": {"code": "no_options", "message": "No options provided"}}

    overlay_id = uuid.uuid4().hex[:12]
    state = _OverlayState(overlay_id=overlay_id, options=options[:6])

    with _overlays_lock:
        _overlays[overlay_id] = state

    thread = threading.Thread(
        target=_show_tkinter_overlay,
        args=(state, prompt),
        daemon=True,
        name=f"overlay-{overlay_id}",
    )
    thread.start()
    # Give tkinter a moment to initialise
    time.sleep(0.15)

    return {"overlay_id": overlay_id, "shown": True}


def wait_overlay_choice(
    overlay_id: str,
    timeout_ms: int = 15000,
) -> Dict[str, Any]:
    """Block until the user picks an option or timeout expires.

    Returns {choice: int|null, timed_out: bool, cancelled: bool}.
    """
    with _overlays_lock:
        state = _overlays.get(overlay_id)

    if state is None:
        return {"error": {"code": "unknown_overlay", "message": f"No overlay {overlay_id}"}}

    waited = state.done_event.wait(timeout=timeout_ms / 1000.0)

    if not waited:
        state.timed_out = True
        # Destroy the overlay window if still open
        try:
            if state._root:
                state._root.after(0, state._root.destroy)
        except Exception:
            pass
        state.done_event.set()

    result: Dict[str, Any] = {
        "choice": state.choice,
        "timed_out": state.timed_out,
        "cancelled": state.cancelled,
    }

    # Cleanup
    with _overlays_lock:
        _overlays.pop(overlay_id, None)

    return result


# ── ToolBase wrappers ──────────────────────────────────────────────────

class ShowChoiceOverlayTool(ToolBase):
    """Show an always-on-top overlay for user disambiguation."""

    def __init__(self):
        super().__init__()
        self._name = "show_choice_overlay"
        self._description = (
            "Show a small always-on-top overlay listing options for disambiguation. "
            "Returns overlay_id to use with wait_for_overlay_choice."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Title / prompt shown at the top of the overlay.",
                },
                "options": {
                    "type": "array",
                    "description": "List of options [{label, hint?, control_type?, rect?, source?, internal_id?}].",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "hint": {"type": "string"},
                            "control_type": {"type": "string"},
                            "internal_id": {"type": "integer"},
                        },
                        "required": ["label"],
                    },
                },
            },
            "required": ["prompt", "options"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        prompt = kwargs.get("prompt", "Choose:")
        options = kwargs.get("options", [])

        result = show_overlay(prompt, options)
        emit_event("ui_action", {
            "kind": "show_choice_overlay",
            "option_count": len(options),
            "overlay_id": result.get("overlay_id"),
        })
        return result


class WaitForOverlayChoiceTool(ToolBase):
    """Wait for user to pick from an overlay."""

    def __init__(self):
        super().__init__()
        self._name = "wait_for_overlay_choice"
        self._description = (
            "Block until the user picks an option from a previously shown overlay "
            "or the timeout expires."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "overlay_id": {
                    "type": "string",
                    "description": "ID returned by show_choice_overlay.",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Max wait in milliseconds (default 15000).",
                    "default": 15000,
                },
            },
            "required": ["overlay_id"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import emit_event

        overlay_id = kwargs.get("overlay_id", "")
        timeout_ms = kwargs.get("timeout_ms", 15000)

        result = wait_overlay_choice(overlay_id, timeout_ms)
        emit_event("ui_action", {
            "kind": "wait_for_overlay_choice",
            "overlay_id": overlay_id,
            "choice": result.get("choice"),
            "timed_out": result.get("timed_out", False),
            "cancelled": result.get("cancelled", False),
        })
        return result
