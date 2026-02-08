from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


_IGNORE_TITLES = {
    "program manager",
    "windows input experience",
}


_FRIENDLY_APP_NAMES = {
    "code.exe": "VS Code",
    "chrome.exe": "Chrome",
    "discord.exe": "Discord",
    "spotify.exe": "Spotify",
    "windowsterminal.exe": "Windows Terminal",
    "wt.exe": "Windows Terminal",
    "explorer.exe": "File Explorer",
    "chatgpt.exe": "ChatGPT",
    "xboxpcapp.exe": "Xbox",
    "applicationframehost.exe": "Xbox",
    "python.exe": "Python",
    "cmd.exe": "Command Prompt",
    "powershell.exe": "PowerShell",
    "powertrader.exe": "PowerTrader",
}


# Higher = more relevant. Keep deterministic.
_APP_PRIORITY = {
    "code.exe": 120,
    "windowsterminal.exe": 115,
    "wt.exe": 115,
    "powertrader.exe": 110,
    "python.exe": 100,
    "cmd.exe": 95,
    "powershell.exe": 90,
    "chrome.exe": 80,
    "discord.exe": 70,
    "spotify.exe": 65,
    "chatgpt.exe": 60,
    "xbox": 40,
    "xboxpcapp.exe": 40,
    "applicationframehost.exe": 40,
    "explorer.exe": 20,
}


def _norm_exe(app: Any) -> str:
    if not isinstance(app, str):
        return ""
    s = app.strip().lower()
    if not s:
        return ""
    # If a path sneaks in, keep only basename.
    if "/" in s or "\\" in s:
        s = s.replace("/", "\\").split("\\")[-1]
    return s


def _canonical_app_key(app_exe: str) -> str:
    # Group Xbox windows together, regardless of host process.
    if app_exe in ("xboxpcapp.exe", "applicationframehost.exe"):
        return "xbox"
    return app_exe


def _friendly_name(app_key: str, app_exe: str) -> str:
    # app_key can be canonical (e.g., xbox). app_exe is the original exe if known.
    if app_key == "xbox":
        return "Xbox"
    name = _FRIENDLY_APP_NAMES.get(app_exe) or _FRIENDLY_APP_NAMES.get(app_key)
    if name:
        return name
    # Fallback: title-case the bare process name.
    base = (app_exe or app_key).replace(".exe", "")
    return base[:1].upper() + base[1:] if base else "Unknown"


def _should_ignore_window(app_exe: str, title: str) -> bool:
    t = (title or "").strip()
    t_lower = t.lower()
    if t_lower in _IGNORE_TITLES:
        return True

    # Filter low-value explorer shell windows.
    if app_exe == "explorer.exe":
        if not t:
            return True
        # Keep the common meaningful label explicitly.
        if t == "This PC - File Explorer":
            return False
    return False


def _title_score(title: str) -> Tuple[int, int, str]:
    """Deterministic scoring tuple for picking a representative title."""
    t = (title or "").strip()
    if not t:
        return (0, 0, "")
    # Prefer non-empty titles; longer tends to be more specific.
    return (1, len(t), t.lower())


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    return s[: max_len - 3] + "..."


@dataclass(frozen=True)
class AppWindowSummary:
    app_key: str
    app_exe: str
    friendly: str
    count: int
    best_title: str

    @property
    def has_title(self) -> bool:
        return bool((self.best_title or "").strip())


def summarize_windows(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize tool output for list_open_windows into grouped, filtered app buckets.

    Deterministic and only uses fields present in tool output.
    """
    if not isinstance(windows, list):
        windows = []

    buckets: Dict[str, Dict[str, Any]] = {}

    for w in windows:
        if not isinstance(w, dict):
            continue
        app_exe = _norm_exe(w.get("app"))
        title = (w.get("title") or "") if isinstance(w.get("title"), str) else ""
        title = title.strip()

        if _should_ignore_window(app_exe, title):
            continue

        app_key = _canonical_app_key(app_exe)
        b = buckets.get(app_key)
        if not b:
            b = {
                "app_key": app_key,
                "app_exe": app_exe,
                "titles": [],
                "count": 0,
            }
            buckets[app_key] = b

        b["count"] += 1
        if title:
            b["titles"].append(title)

    apps: List[AppWindowSummary] = []
    for app_key, b in buckets.items():
        titles: List[str] = list(b.get("titles") or [])
        best_title = ""
        if titles:
            # Pick best title deterministically.
            best_title = max(titles, key=lambda t: _title_score(t))
        app_exe = str(b.get("app_exe") or "")
        friendly = _friendly_name(app_key, app_exe)
        apps.append(
            AppWindowSummary(
                app_key=app_key,
                app_exe=app_exe,
                friendly=friendly,
                count=int(b.get("count") or 0),
                best_title=best_title,
            )
        )

    def _relevance(a: AppWindowSummary) -> Tuple[int, int, int, str, str]:
        # 1) Meaningful title first
        has_title = 1 if a.has_title else 0
        # 2) Explicit app priority
        base = _APP_PRIORITY.get(a.app_key) or _APP_PRIORITY.get(a.app_exe) or (60 if has_title else 30)
        # 3) More windows can be more relevant
        count = a.count
        # 4) Deterministic tie-breakers
        return (has_title, base, count, a.friendly.lower(), a.app_key)

    apps.sort(key=_relevance, reverse=True)

    return {
        "apps": apps,
        "apps_total": len(apps),
        "windows_total": len(windows),
    }


def format_windows_summary(
    summary: Dict[str, Any],
    *,
    max_apps: int = 6,
    max_title_len: int = 64,
) -> str:
    apps: List[AppWindowSummary] = list(summary.get("apps") or [])
    if not apps:
        return "I don't see any open windows."

    # "Top ~6" is a heuristic. If only a handful of apps are open,
    # listing all reads better and avoids hiding important items.
    if len(apps) <= 8:
        shown = apps
    else:
        shown = apps[: max(1, int(max_apps))]
    remaining = max(0, len(apps) - len(shown))

    parts: List[str] = []
    for a in shown:
        label = a.friendly
        title = _truncate(a.best_title, int(max_title_len)) if a.best_title else ""
        if title and title.strip().lower() != a.friendly.strip().lower():
            label = f"{label} ({title})"
        if a.count > 1:
            label = f"{label} (x{a.count})"
        parts.append(label)

    tail = f" and {remaining} more apps" if remaining > 0 else ""
    return "Here's what's open right now: " + ", ".join(parts) + tail + "."


def format_windows_verbose_list(
    windows: List[Dict[str, Any]],
    *,
    count: Optional[int] = None,
    max_lines: int = 6,
) -> str:
    if not isinstance(windows, list):
        windows = []
    if not isinstance(count, int):
        count = len(windows)
    if count == 0:
        return "I don't see any open windows."

    lines: List[str] = []
    for w in windows[: int(max_lines)]:
        if not isinstance(w, dict):
            continue
        title = (w.get("title") or "")
        title = title.strip() if isinstance(title, str) else ""
        app = (w.get("app") or "")
        app = app.strip() if isinstance(app, str) else ""
        app = app.replace(".exe", "")
        label = title or app or "Untitled"
        label = _truncate(label, 55)
        lines.append(f"• {label}")

    extra = count - int(max_lines)
    if extra > 0:
        lines.append(f"...and {extra} more")

    return f"You have {count} open windows:\n" + "\n".join(lines)
