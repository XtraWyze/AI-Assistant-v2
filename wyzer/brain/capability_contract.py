"""
Capability contract generation for Wyzer.

This module builds a deterministic, human-readable capability contract
from the registered tool registry. The contract is cached and reused
for every LLM request to ensure capabilities never drift.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from wyzer.tools.registry import ToolRegistry, build_default_registry


_CACHED_SIGNATURE: Optional[Tuple[Tuple[str, str], ...]] = None
_CACHED_CONTRACT: Optional[str] = None
_CACHED_TOOL_MANIFEST: Optional[str] = None


def get_capability_contract(registry: Optional[ToolRegistry] = None) -> str:
    """
    Get the cached capability contract, generating it if needed.

    Args:
        registry: ToolRegistry to derive capabilities from. If None, a
                  default registry is built.

    Returns:
        Capability contract block as a system prompt string.
    """
    global _CACHED_SIGNATURE, _CACHED_CONTRACT

    tool_items = _get_sorted_tool_items(registry)
    signature = tuple(tool_items)

    if _CACHED_SIGNATURE == signature and _CACHED_CONTRACT:
        return _CACHED_CONTRACT

    contract = _build_capability_contract(tool_items)
    _CACHED_SIGNATURE = signature
    _CACHED_CONTRACT = contract
    return contract


def get_tool_manifest(registry: Optional[ToolRegistry] = None) -> str:
    """
    Get the cached tool manifest block for prompt injection.

    Args:
        registry: ToolRegistry to derive tools from. If None, a
                  default registry is built.

    Returns:
        Tool manifest block as a system prompt string.
    """
    global _CACHED_SIGNATURE, _CACHED_TOOL_MANIFEST

    tool_items = _get_sorted_tool_items(registry)
    signature = tuple(tool_items)

    if _CACHED_SIGNATURE == signature and _CACHED_TOOL_MANIFEST:
        return _CACHED_TOOL_MANIFEST

    manifest = _build_tool_manifest(tool_items)
    _CACHED_SIGNATURE = signature
    _CACHED_TOOL_MANIFEST = manifest
    return manifest


def _get_sorted_tool_items(registry: Optional[ToolRegistry]) -> List[Tuple[str, str]]:
    """Return a sorted list of (tool_name, description) pairs."""
    if registry is None:
        registry = build_default_registry()

    tools = registry.list_tools()
    items = []
    for tool in tools:
        name = str(tool.get("name", "")).strip()
        desc = str(tool.get("description", "")).strip()
        if not name:
            continue
        items.append((name, desc))

    items.sort(key=lambda item: item[0])
    return items


def _build_capability_contract(tool_items: List[Tuple[str, str]]) -> str:
    """Build the capability contract text block."""
    grouped = _group_tools_by_capability(tool_items)

    lines = [
        "CAPABILITY CONTRACT (SYSTEM - IMMUTABLE):",
        "- These capabilities are authoritative and override memory and conversation.",
        "- Tools and world_state are the only sources of truth for system actions and state.",
        "- If a user asks whether you can do something covered by tools, answer YES and explain briefly.",
        "- Never claim inability to perform a registered tool action.",
        "- Never assume an action succeeded without tool confirmation.",
        "- Never invent system state or tool results.",
        "- If intent is unclear, ask ONE clarifying question.",
        "",
        "CAPABILITIES (from tool registry):",
    ]

    for category in sorted(grouped.keys()):
        tools = grouped[category]
        tools.sort()
        tools_str = ", ".join(tools)
        lines.append(f"- {category}: {tools_str}")

    return "\n".join(lines)


def _build_tool_manifest(tool_items: List[Tuple[str, str]]) -> str:
    """Build the tool manifest block used for tool name grounding."""
    lines = ["AVAILABLE TOOLS (use ONLY these exact names):"]

    for name, desc in tool_items:
        if desc:
            lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}")

    lines.extend([
        "",
        "CRITICAL TOOL RULES:",
        "- You MUST ONLY use tool names from the list above",
        "- NEVER invent new tool names or arguments",
        "- If no tool applies, respond with reply-only (no tools)",
        "- For questions, explanations, stories, or opinions: use reply-only",
    ])

    return "\n".join(lines)


def _group_tools_by_capability(tool_items: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Group tool names into human-readable capability buckets."""
    grouped: Dict[str, List[str]] = {}

    for name, _desc in tool_items:
        category = _category_for_tool(name)
        grouped.setdefault(category, []).append(name)

    return grouped


def _category_for_tool(name: str) -> str:
    """Assign a tool name to a capability category."""
    name_lower = name.lower()

    if name_lower.startswith("open_") or name_lower.endswith("_open") or name_lower == "open_target":
        return "Open apps, files, and websites"
    if "window" in name_lower or name_lower in ("switch_app",):
        return "Window and app focus"
    if name_lower.startswith("media_") or name_lower == "get_now_playing":
        return "Media controls"
    if "volume" in name_lower or "audio_output" in name_lower:
        return "Audio controls"
    if name_lower.startswith("system_storage"):
        return "Storage management"
    if name_lower == "timer":
        return "Timers"
    if name_lower == "local_library_refresh":
        return "App library refresh"
    if "search" in name_lower:
        return "Web search"
    if name_lower.startswith("get_") or name_lower in ("monitor_info",):
        return "System and context info"

    return "Other tools"
