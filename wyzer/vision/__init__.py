"""
wyzer.vision

Phase 9: Screen Awareness (READ-ONLY)

This module provides awareness of the user's current visual context.
NO OCR, NO screenshots, NO UI automation.
Read-only detection of foreground application and window title.
"""

from wyzer.vision.window_context import get_foreground_window

__all__ = ["get_foreground_window"]
