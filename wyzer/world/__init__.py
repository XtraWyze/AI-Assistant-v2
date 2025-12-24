"""
wyzer.world - Phase 12: Multi-Monitor Window Awareness

This package provides always-on world state tracking for open windows
across monitors. NO OCR, NO screenshot analysis.

Modules:
- window_diff: Pure diff logic for window snapshots (testable without OS calls)
- window_watcher: Background watcher that polls windows and updates world_state
"""

from wyzer.world.window_diff import diff_snapshots
from wyzer.world.window_watcher import WindowWatcher

__all__ = ["diff_snapshots", "WindowWatcher"]
