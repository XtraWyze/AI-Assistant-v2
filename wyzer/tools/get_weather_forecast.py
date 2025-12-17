"""\
Weather + forecast tool.

Uses Open-Meteo (no API key) and optionally Open-Meteo Geocoding.
If no location is provided, it falls back to approximate IP-based location.
Requires internet.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple

from wyzer.tools.tool_base import ToolBase
from wyzer.tools.get_location import _get_ip_location


_WEATHER_CODE_TEXT = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _fetch_json(url: str, timeout_sec: float = 8.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Wyzer/2 (weather; +https://local)"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    return json.loads(data)


def _weather_code_to_text(code: Optional[int]) -> Optional[str]:
    if code is None:
        return None
    try:
        code_int = int(code)
    except Exception:
        return None
    return _WEATHER_CODE_TEXT.get(code_int)


def _geocode_location(name: str, timeout_sec: float = 8.0) -> Dict[str, Any]:
    q = urllib.parse.urlencode(
        {
            "name": name,
            "count": 1,
            "language": "en",
            "format": "json",
        }
    )
    url = f"https://geocoding-api.open-meteo.com/v1/search?{q}"
    payload = _fetch_json(url, timeout_sec=timeout_sec)
    results = payload.get("results") or []
    if not results:
        return {
            "error": {
                "type": "not_found",
                "message": f"Could not find location '{name}'",
            }
        }
    r0 = results[0]
    return {
        "name": r0.get("name"),
        "admin1": r0.get("admin1"),
        "country": r0.get("country"),
        "timezone": r0.get("timezone"),
        "latitude": r0.get("latitude"),
        "longitude": r0.get("longitude"),
        "source": "open-meteo-geocoding",
        "approximate": False,
    }


def _build_forecast_url(
    latitude: float,
    longitude: float,
    days: int,
    units: str,
) -> str:
    # Open-Meteo supports unit params; default is metric-like.
    params: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "auto",
        "forecast_days": days,
        "current": ",".join(
            [
                "temperature_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
            ]
        ),
    }

    if units == "imperial":
        params["temperature_unit"] = "fahrenheit"
        params["wind_speed_unit"] = "mph"
        params["precipitation_unit"] = "inch"

    q = urllib.parse.urlencode(params)
    return f"https://api.open-meteo.com/v1/forecast?{q}"


def _extract_daily_forecast(daily: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(daily, dict):
        return []

    times = daily.get("time") or []
    codes = daily.get("weather_code") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    precip_sum = daily.get("precipitation_sum") or []
    precip_prob = daily.get("precipitation_probability_max") or []

    n = min(len(times), len(codes), len(tmax), len(tmin), len(precip_sum), len(precip_prob))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append(
            {
                "date": times[i],
                "weather_code": codes[i],
                "weather": _weather_code_to_text(codes[i]),
                "temp_max": tmax[i],
                "temp_min": tmin[i],
                "precipitation_sum": precip_sum[i],
                "precipitation_probability_max": precip_prob[i],
            }
        )
    return out


class GetWeatherForecastTool(ToolBase):
    """Tool to get current weather and multi-day forecast."""

    def __init__(self):
        super().__init__()
        self._name = "get_weather_forecast"
        self._description = "Get current weather and multi-day forecast for a location; defaults to Fahrenheit/imperial units; if location is omitted uses approximate IP location (Open-Meteo, no API key)"
        self._args_schema = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Optional location name (e.g., 'Seattle', 'Paris, France'). If omitted, uses approximate IP-based location.",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of forecast days (1-14). Default 3.",
                },
                "units": {
                    "type": "string",
                    "pattern": "^(metric|imperial|celsius|fahrenheit|c|f)$",
                    "description": "Unit system. Accepted: metric/imperial or celsius/fahrenheit (c/f). Default fahrenheit/imperial.",
                },
                "day_offset": {
                    "type": "integer",
                    "description": "Day offset for response formatting (0=today, 1=tomorrow, etc.). Used by orchestrator, not the tool itself.",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        location_name = (kwargs.get("location") or "").strip()
        units_raw = (kwargs.get("units") or "fahrenheit").strip().lower()
        days_raw = kwargs.get("days")

        try:
            days = int(days_raw) if days_raw is not None else 3
        except Exception:
            days = 3

        if days < 1:
            days = 1
        if days > 14:
            days = 14

        if units_raw in {"fahrenheit", "f"}:
            units = "imperial"
        elif units_raw in {"celsius", "c"}:
            units = "metric"
        elif units_raw in {"metric", "imperial"}:
            units = units_raw
        else:
            units = "metric"

        # Resolve location -> lat/lon
        loc: Dict[str, Any]
        if location_name:
            loc = _geocode_location(location_name)
            if "error" in loc:
                return loc
        else:
            loc = _get_ip_location()
            if "error" in loc:
                return loc

        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is None or lon is None:
            return {
                "error": {
                    "type": "location_unavailable",
                    "message": "Location did not include latitude/longitude",
                    "details": loc,
                }
            }

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            return {
                "error": {
                    "type": "location_unavailable",
                    "message": "Invalid latitude/longitude values",
                    "details": {"latitude": lat, "longitude": lon},
                }
            }

        url = _build_forecast_url(lat_f, lon_f, days=days, units=units)

        try:
            payload = _fetch_json(url)
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

        current = payload.get("current") or {}
        current_out = {
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "precipitation": current.get("precipitation"),
            "weather_code": current.get("weather_code"),
            "weather": _weather_code_to_text(current.get("weather_code")),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
        }

        daily_out = _extract_daily_forecast(payload.get("daily") or {})

        # Friendly name for IP-derived location
        if not location_name:
            city = loc.get("city")
            region = loc.get("region")
            country = loc.get("country")
            display_name = ", ".join([p for p in [city, region, country] if p]) or "your area"
        else:
            display_name = ", ".join([p for p in [loc.get("name"), loc.get("admin1"), loc.get("country")] if p]) or location_name

        return {
            "location": {
                "name": display_name,
                "latitude": lat_f,
                "longitude": lon_f,
                "timezone": payload.get("timezone") or loc.get("timezone"),
                "approximate": bool(loc.get("approximate")),
                "source": loc.get("source"),
            },
            "units": units,
            "current": current_out,
            "forecast_daily": daily_out,
            "attribution": "Weather data by Open-Meteo.com",
        }
