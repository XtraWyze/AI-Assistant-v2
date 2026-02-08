"""
wyzer.tools.demo_desktop — Desktop Ground Truth Demo (Phase 14)

Manual test / demo script: run via
    python -m wyzer.tools.demo_desktop

Steps:
  1. Print active window metadata
  2. List top UIA controls in focused window
  3. Check for a provided button name (default: "Close")
  4. Take screenshot
  5. Run OCR if available; otherwise print guidance
"""

from __future__ import annotations

import sys
import json
import time


def _pp(label: str, data: dict) -> None:
    """Pretty-print a section."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2, default=str))


def main() -> None:
    button_name = sys.argv[1] if len(sys.argv) > 1 else "Close"
    print(f"[demo_desktop] Desktop Ground Truth Demo — Phase 14")
    print(f"[demo_desktop] Button to search for: '{button_name}'")

    # 1. Active window
    print("\n[1/5] Getting active window...")
    from wyzer.tools.desktop.get_active_window import get_active_window_info
    win_info = get_active_window_info()
    _pp("Active Window", win_info)

    # 2. UIA perception
    print("\n[2/5] Perceiving UIA controls (max 30)...")
    from wyzer.tools.desktop.perceive_uia import perceive_uia_focused_window
    uia = perceive_uia_focused_window(max_nodes=30)
    # Print summary, not full dump
    summary = {
        "window": uia.get("window"),
        "controls_count": len(uia.get("controls", [])),
        "controls_sample": [
            {"name": c["name"], "type": c["control_type"]}
            for c in uia.get("controls", [])[:10]
            if c.get("name")
        ],
        "dialogs": uia.get("dialogs"),
        "progress": uia.get("progress"),
        "errors": uia.get("errors"),
    }
    _pp("UIA Snapshot (summary)", summary)

    # 3. Button check
    print(f"\n[3/5] Checking for button: '{button_name}'...")
    from wyzer.tools.desktop.assertions import ui_find_text
    find_result = ui_find_text(button_name, method="uia")
    _pp(f"ui_find_text('{button_name}')", find_result)

    # 4. Screenshot
    print("\n[4/5] Taking screenshot...")
    from wyzer.tools.desktop.screenshot_tool import _capture_focused_window
    shot = _capture_focused_window()
    _pp("Screenshot", shot)

    # 5. OCR
    print("\n[5/5] Running OCR (if available)...")
    from wyzer.tools.desktop.ocr_tool import _run_ocr
    image_path = shot.get("image_path")
    if image_path and "error" not in shot:
        ocr_result = _run_ocr(image_path=image_path)
        if ocr_result.get("missing_dependency"):
            print("  OCR OPTIONAL — not installed.")
            print(f"  Guidance: {ocr_result.get('guidance', '')}")
        else:
            _pp("OCR Result", {
                "line_count": len(ocr_result.get("lines", [])),
                "full_text_preview": (ocr_result.get("full_text") or "")[:500],
            })
    else:
        print("  Skipped (screenshot failed)")

    # Event log
    print(f"\n{'='*60}")
    print("  Event Log (last 10)")
    print(f"{'='*60}")
    from wyzer.context.world_state import get_event_log
    for ev in get_event_log(10):
        ts = time.strftime("%H:%M:%S", time.localtime(ev.get("ts", 0)))
        print(f"  [{ts}] {ev.get('event', '?')}: {json.dumps({k: v for k, v in ev.items() if k not in ('event', 'ts')}, default=str)}")

    print("\n[demo_desktop] Done.")


if __name__ == "__main__":
    main()
