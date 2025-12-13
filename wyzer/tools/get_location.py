"""\
Get approximate location tool.

This tool uses an IP-based geolocation provider, which is inherently approximate.
It requires an internet connection.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

from wyzer.tools.tool_base import ToolBase


def _fetch_json(url: str, timeout_sec: float = 6.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Wyzer/2 (location; +https://local)"
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    return json.loads(data)


def _normalize_location_payload(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    # ipapi.co
    if source == "ipapi":
        return {
            "ip": payload.get("ip"),
            "city": payload.get("city"),
            "region": payload.get("region"),
            "country": payload.get("country_name"),
            "postal": payload.get("postal"),
            "timezone": payload.get("timezone"),
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "approximate": True,
            "source": "ipapi.co",
        }

    # ipwho.is
    tz = payload.get("timezone") or {}
    return {
        "ip": payload.get("ip"),
        "city": payload.get("city"),
        "region": payload.get("region"),
        "country": payload.get("country"),
        "postal": payload.get("postal"),
        "timezone": tz.get("id") if isinstance(tz, dict) else tz,
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "approximate": True,
        "source": "ipwho.is",
    }


def _get_ip_location(timeout_sec: float = 6.0) -> Dict[str, Any]:
    errors = []

    # Provider 1: ipapi.co (HTTPS, no key)
    try:
        payload = _fetch_json("https://ipapi.co/json/", timeout_sec=timeout_sec)
        # ipapi returns {"error": true, "reason": ...} sometimes
        if payload.get("error"):
            raise RuntimeError(str(payload.get("reason") or payload.get("message") or "ipapi error"))
        normalized = _normalize_location_payload(payload, "ipapi")
        if normalized.get("latitude") is None or normalized.get("longitude") is None:
            raise RuntimeError("Missing latitude/longitude")
        return normalized
    except Exception as e:
        errors.append(f"ipapi.co: {e}")

    # Provider 2: ipwho.is (HTTPS, no key)
    try:
        payload = _fetch_json("https://ipwho.is/", timeout_sec=timeout_sec)
        if payload.get("success") is False:
            raise RuntimeError(str(payload.get("message") or "ipwho.is error"))
        normalized = _normalize_location_payload(payload, "ipwho")
        if normalized.get("latitude") is None or normalized.get("longitude") is None:
            raise RuntimeError("Missing latitude/longitude")
        return normalized
    except Exception as e:
        errors.append(f"ipwho.is: {e}")

    return {
        "error": {
            "type": "network_error",
            "message": "Unable to determine location from IP",
            "details": errors,
        }
    }


class GetLocationTool(ToolBase):
    """Tool to get the user's approximate location."""

    def __init__(self):
        super().__init__()
        self._name = "get_location"
        self._description = "Get approximate location (city/region/country + lat/lon) using IP-based geolocation; requires internet"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        try:
            return _get_ip_location()
        except urllib.error.URLError as e:
            return {
                "error": {
                    "type": "network_error",
                    "message": str(e),
                }
            }
        except Exception as e:
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e),
                }
            }
