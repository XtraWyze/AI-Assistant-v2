"""Sanity-check runner for the real `volume_control` tool (no stubs).

Examples:
  # Master volume
  python scripts/volume_control_sanity.py master get
  python scripts/volume_control_sanity.py master set --level 35
  python scripts/volume_control_sanity.py master change --delta -10
  python scripts/volume_control_sanity.py master mute
  python scripts/volume_control_sanity.py master unmute

  # Per-app/session volume (requires the app to have an active audio session)
  python scripts/volume_control_sanity.py app get --process spotify
  python scripts/volume_control_sanity.py app set --process spotify --level 30
  python scripts/volume_control_sanity.py app change --process chrome --delta -10
  python scripts/volume_control_sanity.py app mute --process discord
  python scripts/volume_control_sanity.py app unmute --process discord

    # List active audio sessions (helps find the right process hint)
    python scripts/volume_control_sanity.py sessions
    python scripts/volume_control_sanity.py sessions --filter spotify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run wyzer.tools.volume_control.VolumeControlTool directly")
    sub = p.add_subparsers(dest="scope", required=True)

    def add_common_actions(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "action",
            choices=["get", "set", "change", "mute", "unmute"],
            help="Action to perform",
        )
        sp.add_argument("--level", type=int, help="Volume level 0-100 (for action=set)")
        sp.add_argument("--delta", type=int, help="Delta -100..100 (for action=change)")

    master = sub.add_parser("master", help="Master/system volume")
    add_common_actions(master)

    app = sub.add_parser("app", help="Per-app/session volume")
    add_common_actions(app)
    app.add_argument("--process", required=True, help="Process/app hint, e.g. spotify, chrome, discord")

    sessions = sub.add_parser("sessions", help="List active audio sessions")
    sessions.add_argument(
        "--filter",
        dest="filter_text",
        default="",
        help="Optional substring filter applied to process/display",
    )

    return p


def main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)

    if args.scope == "sessions":
        from pycaw.pycaw import AudioUtilities

        out = []
        flt = (getattr(args, "filter_text", "") or "").strip().lower()

        for s in AudioUtilities.GetAllSessions() or []:
            if s is None:
                continue
            proc = getattr(s, "Process", None)
            proc_name = ""
            try:
                if proc is not None:
                    proc_name = (proc.name() or "").strip()
            except Exception:
                proc_name = ""

            display = ""
            try:
                display = str(getattr(s, "DisplayName", "") or "").strip()
            except Exception:
                display = ""

            sav = getattr(s, "SimpleAudioVolume", None)
            if sav is None:
                continue

            try:
                vol = int(round(float(sav.GetMasterVolume()) * 100))
            except Exception:
                vol = None
            try:
                mute = bool(sav.GetMute())
            except Exception:
                mute = None

            row = {"process": proc_name, "display": display, "volume_percent": vol, "mute": mute}
            hay = f"{proc_name} {display}".lower()
            if flt and flt not in hay:
                continue

            out.append(row)

        print(json.dumps({"status": "ok", "sessions": out}, indent=2, ensure_ascii=False))
        return 0

    from wyzer.tools.volume_control import VolumeControlTool

    tool = VolumeControlTool()

    payload: dict = {"scope": args.scope, "action": args.action}
    if args.scope == "app":
        payload["process"] = args.process

    if args.action == "set":
        if args.level is None:
            raise SystemExit("--level is required for action=set")
        payload["level"] = int(args.level)

    if args.action == "change":
        if args.delta is None:
            raise SystemExit("--delta is required for action=change")
        payload["delta"] = int(args.delta)

    result = tool.run(**payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if isinstance(result, dict) and result.get("error"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
