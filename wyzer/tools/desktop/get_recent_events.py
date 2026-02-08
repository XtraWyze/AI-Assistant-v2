"""
get_recent_events — Queryable event log tool (Phase 15).

Returns the last N structured events from WorldState.event_log.
This lets the LLM (or tests) inspect what actually happened.
"""

from __future__ import annotations

from typing import Any, Dict

from wyzer.tools.tool_base import ToolBase


class GetRecentEventsTool(ToolBase):
    """Return the last N events from the world-state event log."""

    def __init__(self):
        super().__init__()
        self._name = "get_recent_events"
        self._description = (
            "Return the most recent N events from the world-state event log. "
            "Events include tool_start, tool_end, perception, ui_action, warning."
        )
        self._args_schema = {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max events to return (default 10, max 50).",
                    "default": 10,
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def run(self, **kwargs) -> Dict[str, Any]:
        from wyzer.context.world_state import get_event_log

        limit = min(max(kwargs.get("limit", 10), 1), 50)
        events = get_event_log(limit=limit)
        return {
            "events": events,
            "count": len(events),
        }
