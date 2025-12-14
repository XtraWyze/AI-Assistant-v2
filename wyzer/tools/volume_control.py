"""True system and per-app volume control (Windows) via pycaw.

This tool supports:
- Master volume: get/set/change + mute/unmute
- Per-app (per audio session) volume: get/set/change + mute/unmute

It uses fuzzy matching so spoken names like "spotify", "chrome", "discord" work
reliably against the current audio sessions.
"""

from __future__ import annotations

import platform
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from wyzer.tools.tool_base import ToolBase


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _normalize_for_tokens(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _normalize_compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _singularize_token(token: str) -> str:
    tok = (token or "").strip().lower()
    if len(tok) > 3 and tok.endswith("s"):
        return tok[:-1]
    return tok


def _token_match_ratio(query: str, candidate: str) -> float:
    q = _normalize_for_tokens(query)
    c = _normalize_for_tokens(candidate)
    if not q or not c:
        return 0.0

    q_tokens = [_singularize_token(t) for t in q.split() if t.strip()]
    c_tokens = [_singularize_token(t) for t in c.split() if t.strip()]
    if not q_tokens or not c_tokens:
        return 0.0

    def matches(qt: str) -> bool:
        for ct in c_tokens:
            if qt == ct:
                return True
            if len(qt) >= 3 and (qt in ct or ct in qt):
                return True
        return False

    hit = sum(1 for qt in q_tokens if matches(qt))
    return hit / max(1, len(q_tokens))


def _fuzzy_score(query: str, candidate: str) -> int:
    """Return a 0-100 similarity score."""
    q = _normalize_for_tokens(query)
    c = _normalize_for_tokens(candidate)

    if not q or not c:
        return 0
    if q == c:
        return 100

    ratio = SequenceMatcher(None, q, c).ratio()

    q_tokens = {_singularize_token(t) for t in q.split() if t.strip()}
    c_tokens = {_singularize_token(t) for t in c.split() if t.strip()}
    union = q_tokens | c_tokens
    inter = q_tokens & c_tokens
    jaccard = (len(inter) / len(union)) if union else 0.0

    qc = _normalize_compact(query)
    cc = _normalize_compact(candidate)
    partial = 0.0
    if qc and cc:
        if qc in cc:
            partial = min(1.0, len(qc) / max(1, len(cc)))
        else:
            match = SequenceMatcher(None, qc, cc).find_longest_match()
            if match.size > 0:
                partial = min(1.0, match.size / max(1, len(qc)))

    token_hit = _token_match_ratio(query, candidate)

    score = max(ratio, jaccard, partial, token_hit)

    if token_hit >= 0.999:
        score = max(score, 0.90)
    elif token_hit >= 0.66:
        score = max(score, 0.78)

    return int(round(score * 100))


def _strip_trailing_punct(text: str) -> str:
    return (text or "").strip().rstrip(".?!,;:\"'")


def _clean_process_query(query: str) -> str:
    q = _strip_trailing_punct(query)
    ql = q.lower()

    # Remove common fillers.
    ql = re.sub(r"\b(?:the|my|this|that)\b", " ", ql)
    ql = re.sub(r"\b(?:app|application|process|program)\b", " ", ql)
    ql = re.sub(r"\b(?:volume|sound|audio)\b", " ", ql)
    ql = re.sub(r"\s+", " ", ql).strip()

    return ql


def _safe_proc_name(proc: Any) -> str:
    if proc is None:
        return ""
    try:
        name = proc.name()
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass

    try:
        name = getattr(proc, "name", None)
        if callable(name):
            name = name()
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass

    return ""


def _proc_base(proc_name: str) -> str:
    p = (proc_name or "").strip()
    if p.lower().endswith(".exe"):
        p = p[:-4]
    return p


def _resolve_process_hint_via_local_library(process_query: str) -> Optional[str]:
    """Best-effort: use LocalLibrary aliases to map spoken names to an EXE base name."""
    try:
        from wyzer.local_library.resolver import resolve_target

        res = resolve_target(process_query)
        if not isinstance(res, dict):
            return None
        if res.get("type") != "app":
            return None

        path = res.get("path")
        if not isinstance(path, str) or not path.strip():
            return None

        # Extract basename from a filesystem path.
        base = path.replace("/", "\\").split("\\")[-1]
        base = base.strip()
        if not base:
            return None
        if base.lower().endswith(".exe"):
            base = base[:-4]
        return base.lower().strip() or None
    except Exception:
        return None


def _get_endpoint_volume() -> Any:
    import warnings

    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        device = AudioUtilities.GetSpeakers()

    # Some pycaw versions return an AudioDevice wrapper that already exposes
    # a ready-to-use IAudioEndpointVolume pointer.
    try:
        ev = getattr(device, "EndpointVolume", None)
        if ev is not None and hasattr(ev, "GetMasterVolumeLevelScalar"):
            return ev
    except Exception:
        pass

    # pycaw return types vary by version:
    # - Some return an IMMDevice with .Activate
    # - Some return an AudioDevice wrapper with .device/.Device/._device
    activate_target = device
    for attr in ("device", "Device", "_device"):
        try:
            candidate = getattr(device, attr, None)
        except Exception:
            candidate = None
        if candidate is not None and hasattr(candidate, "Activate"):
            activate_target = candidate
            break

    if not hasattr(activate_target, "Activate"):
        raise RuntimeError(f"Unsupported pycaw device object type: {type(device)!r}")

    interface = activate_target.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    endpoint = cast(interface, POINTER(IAudioEndpointVolume))
    return endpoint


def _list_sessions() -> List[Any]:
    import warnings
    from pycaw.pycaw import AudioUtilities

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return list(AudioUtilities.GetAllSessions() or [])


def _pick_best_session(process_query: str, sessions: List[Any]) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    pq = _clean_process_query(process_query)
    scored: List[Dict[str, Any]] = []

    for s in sessions:
        if s is None:
            continue

        proc_name = _safe_proc_name(getattr(s, "Process", None))
        display_name = ""
        try:
            display_name = str(getattr(s, "DisplayName", "") or "").strip()
        except Exception:
            display_name = ""

        candidates = []
        if proc_name:
            candidates.append(proc_name)
            candidates.append(_proc_base(proc_name))
        if display_name:
            candidates.append(display_name)

        # Some sessions have neither proc nor display; skip.
        if not candidates:
            continue

        best = 0
        best_label = ""
        for c in candidates:
            score = _fuzzy_score(pq, c)
            if score > best:
                best = score
                best_label = c

        scored.append(
            {
                "session": s,
                "score": int(best),
                "process": proc_name,
                "display": display_name,
                "label": best_label,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    debug = [
        {
            "process": x["process"],
            "display": x["display"],
            "label": x["label"],
            "score": int(x["score"]),
        }
        for x in scored[:8]
    ]

    if not scored:
        return None, debug

    return scored[0]["session"], debug


def _session_volume_info(session: Any) -> Dict[str, Any]:
    sav = getattr(session, "SimpleAudioVolume", None)
    if sav is None:
        return {}

    vol = float(sav.GetMasterVolume())
    mute = bool(sav.GetMute())
    proc_name = _safe_proc_name(getattr(session, "Process", None))

    display_name = ""
    try:
        display_name = str(getattr(session, "DisplayName", "") or "").strip()
    except Exception:
        display_name = ""

    return {
        "process": proc_name,
        "display": display_name,
        "volume_percent": int(round(vol * 100)),
        "mute": bool(mute),
    }


class VolumeControlTool(ToolBase):
    def __init__(self):
        super().__init__()
        self._name = "volume_control"
        self._description = "Get/set master or per-app volume using Windows audio APIs (pycaw)"
        self._args_schema = {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["master", "app"],
                    "description": "master for system volume, app for per-process/session volume",
                },
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "change", "mute", "unmute"],
                },
                "level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Target volume 0-100 (for action=set)",
                },
                "delta": {
                    "type": "integer",
                    "minimum": -100,
                    "maximum": 100,
                    "description": "Delta in percent points (for action=change)",
                },
                "process": {
                    "type": "string",
                    "description": "Process/app name hint, e.g. 'spotify', 'chrome' (required when scope=app)",
                },
            },
            "required": ["scope", "action"],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # Tool is Windows-only.
        if platform.system().lower() != "windows":
            end_time = time.perf_counter()
            return {
                "error": {
                    "type": "unsupported_platform",
                    "message": "volume_control is only supported on Windows",
                },
                "latency_ms": int((end_time - start_time) * 1000),
            }

        scope = str(kwargs.get("scope") or "").strip().lower()
        action = str(kwargs.get("action") or "").strip().lower()
        level = kwargs.get("level")
        delta = kwargs.get("delta")
        process = kwargs.get("process")

        try:
            if scope not in {"master", "app"}:
                raise ValueError("scope must be 'master' or 'app'")
            if action not in {"get", "set", "change", "mute", "unmute"}:
                raise ValueError("action must be one of get/set/change/mute/unmute")

            if scope == "master":
                endpoint = _get_endpoint_volume()

                if action == "get":
                    vol = float(endpoint.GetMasterVolumeLevelScalar())
                    mute = bool(endpoint.GetMute())
                    lvl = int(round(vol * 100))
                    end_time = time.perf_counter()
                    return {
                        "status": "ok",
                        "scope": "master",
                        "action": "get",
                        "volume_percent": lvl,
                        "mute": bool(mute),
                        # Compatibility keys (used by orchestrator reply formatting)
                        "level": lvl,
                        "muted": bool(mute),
                        "latency_ms": int((end_time - start_time) * 1000),
                    }

                if action == "mute":
                    endpoint.SetMute(1, None)
                elif action == "unmute":
                    endpoint.SetMute(0, None)
                elif action == "set":
                    if not isinstance(level, int):
                        raise ValueError("level (0-100) is required for action=set")
                    pct = _clamp_int(level, 0, 100)
                    endpoint.SetMasterVolumeLevelScalar(pct / 100.0, None)
                elif action == "change":
                    if not isinstance(delta, int):
                        raise ValueError("delta (-100..100) is required for action=change")
                    cur = float(endpoint.GetMasterVolumeLevelScalar())
                    cur_pct = int(round(cur * 100))
                    pct = _clamp_int(cur_pct + int(delta), 0, 100)
                    endpoint.SetMasterVolumeLevelScalar(pct / 100.0, None)

                # Return updated state.
                vol = float(endpoint.GetMasterVolumeLevelScalar())
                mute = bool(endpoint.GetMute())
                lvl = int(round(vol * 100))
                end_time = time.perf_counter()
                return {
                    "status": "ok",
                    "scope": "master",
                    "action": action,
                    "volume_percent": lvl,
                    "mute": bool(mute),
                    # Compatibility keys
                    "new_level": lvl,
                    "muted": bool(mute),
                    "latency_ms": int((end_time - start_time) * 1000),
                }

            # scope == app
            if not isinstance(process, str) or not process.strip():
                raise ValueError("process is required when scope=app")

            sessions = _list_sessions()
            best_session, debug = _pick_best_session(process, sessions)

            # If we couldn't find a decent match, try resolving through LocalLibrary aliases.
            if best_session is None or (debug and int(debug[0].get("score") or 0) < 65):
                resolved = _resolve_process_hint_via_local_library(process)
                if resolved:
                    best_session, debug = _pick_best_session(resolved, sessions)

            if best_session is None:
                end_time = time.perf_counter()
                return {
                    "error": {
                        "type": "not_found",
                        "message": f"No active audio session matched process '{process}'",
                        "debug_candidates": debug,
                    },
                    "latency_ms": int((end_time - start_time) * 1000),
                }

            best_proc = _safe_proc_name(getattr(best_session, "Process", None))
            best_proc_base = _proc_base(best_proc).lower().strip() if best_proc else ""

            # Apply to all sessions for the same process name when possible.
            targets: List[Any] = []
            for s in sessions:
                if s is None:
                    continue
                pn = _safe_proc_name(getattr(s, "Process", None))
                if best_proc_base and _proc_base(pn).lower().strip() == best_proc_base:
                    targets.append(s)
            if not targets:
                targets = [best_session]

            if action == "get":
                info = _session_volume_info(best_session)
                end_time = time.perf_counter()
                return {
                    "status": "ok",
                    "scope": "app",
                    "action": "get",
                    "requested_process": process,
                    "matched": {
                        "process": best_proc,
                        "display": info.get("display", ""),
                        "score": int((debug[0].get("score") if debug else 0) or 0),
                    },
                    "volume_percent": int(info.get("volume_percent") or 0),
                    "mute": bool(info.get("mute") or False),
                    # Compatibility keys
                    "process": best_proc,
                    "display": info.get("display", ""),
                    "level": int(info.get("volume_percent") or 0),
                    "muted": bool(info.get("mute") or False),
                    "affected_sessions": len(targets),
                    "latency_ms": int((end_time - start_time) * 1000),
                    "debug_candidates": debug,
                }

            # Read current from primary.
            primary_sav = getattr(best_session, "SimpleAudioVolume", None)
            if primary_sav is None:
                raise RuntimeError("Selected session has no SimpleAudioVolume")

            cur_pct = int(round(float(primary_sav.GetMasterVolume()) * 100))

            target_pct: Optional[int] = None
            target_mute: Optional[bool] = None

            if action == "mute":
                target_mute = True
            elif action == "unmute":
                target_mute = False
            elif action == "set":
                if not isinstance(level, int):
                    raise ValueError("level (0-100) is required for action=set")
                target_pct = _clamp_int(level, 0, 100)
            elif action == "change":
                if not isinstance(delta, int):
                    raise ValueError("delta (-100..100) is required for action=change")
                target_pct = _clamp_int(cur_pct + int(delta), 0, 100)

            updated = 0
            for s in targets:
                sav = getattr(s, "SimpleAudioVolume", None)
                if sav is None:
                    continue
                if target_pct is not None:
                    sav.SetMasterVolume(target_pct / 100.0, None)
                if target_mute is not None:
                    sav.SetMute(1 if target_mute else 0, None)
                updated += 1

            info = _session_volume_info(best_session)
            end_time = time.perf_counter()
            lvl = int(info.get("volume_percent") or 0)
            muted = bool(info.get("mute") or False)
            return {
                "status": "ok",
                "scope": "app",
                "action": action,
                "requested_process": process,
                "matched": {
                    "process": best_proc,
                    "display": info.get("display", ""),
                    "score": int((debug[0].get("score") if debug else 0) or 0),
                },
                "volume_percent": lvl,
                "mute": muted,
                # Compatibility keys
                "process": best_proc,
                "display": info.get("display", ""),
                "new_level": lvl if action in {"set", "change"} else lvl,
                "muted": muted,
                "affected_sessions": int(updated),
                "latency_ms": int((end_time - start_time) * 1000),
                "debug_candidates": debug,
            }

        except Exception as e:
            end_time = time.perf_counter()
            return {
                "error": {"type": "execution_error", "message": str(e)},
                "latency_ms": int((end_time - start_time) * 1000),
            }
