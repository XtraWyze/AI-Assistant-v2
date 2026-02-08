"""wyzer.core.plan_expander

Deterministic validation and expansion for composition planner JSON plans.

The LLM may propose a plan, but deterministic code must:
- Validate tool names exist in the registry
- Validate args against tool schemas (after template substitution)
- Expand foreach macros safely with a strict template whitelist
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from wyzer.tools.validation import validate_args


MAX_COMPOSED_TOOL_CALLS = 25


_TEMPLATE_RE = re.compile(r"^\{\{item\.(id|hwnd|title|app)\}\}$")


def _is_safe_var_name(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    name = name.strip()
    if not name:
        return False
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,39}", name))


def _contains_any_template(value: Any) -> bool:
    if isinstance(value, str):
        return "{{" in value or "}}" in value
    if isinstance(value, dict):
        return any(_contains_any_template(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_any_template(v) for v in value)
    return False


def _extract_iterable(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Extract a foreach iterable from a saved var.

    Rules:
    - Prefer value["windows"] if present and a list
    - Else if value is already a list, use it
    - Only lists of dict-ish items are accepted
    """
    if isinstance(value, dict) and isinstance(value.get("windows"), list):
        raw = value.get("windows")
    elif isinstance(value, list):
        raw = value
    else:
        return None

    items: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            items.append(item)
        else:
            return None
    return items


def validate_plan(plan_json: Any, registry) -> Tuple[bool, str]:
    """Validate a composition plan JSON object.

    Notes:
    - Normal intents validate args fully using validate_args().
    - Foreach intents validate tool existence + template whitelist, but defer
      full args type validation until after substitution (execution time).
    """
    if not isinstance(plan_json, dict):
        return False, "Plan must be a JSON object"

    intents = plan_json.get("intents")
    if not isinstance(intents, list) or not intents:
        return False, "Plan must include a non-empty 'intents' list"

    saved_vars = set()

    for idx, raw in enumerate(intents):
        if not isinstance(raw, dict):
            return False, f"Intent {idx + 1} must be an object"

        is_foreach = "foreach" in raw

        if is_foreach:
            var_name = raw.get("foreach")
            do_obj = raw.get("do")

            if not _is_safe_var_name(var_name):
                return False, f"Intent {idx + 1}: foreach must be a valid variable name"

            if var_name not in saved_vars:
                return False, f"Intent {idx + 1}: foreach variable '{var_name}' was not saved by a prior intent"

            if not isinstance(do_obj, dict):
                return False, f"Intent {idx + 1}: foreach 'do' must be an object"

            tool_name = do_obj.get("tool")
            if not isinstance(tool_name, str) or not tool_name.strip():
                return False, f"Intent {idx + 1}: foreach do.tool must be a non-empty string"
            tool_name = tool_name.strip()

            if not registry.has_tool(tool_name):
                return False, f"Intent {idx + 1}: unknown tool '{tool_name}'"

            args = do_obj.get("args", {})
            if args is None:
                args = {}
            if not isinstance(args, dict):
                return False, f"Intent {idx + 1}: foreach do.args must be an object"

            # Strict template whitelist: only allow exact "{{item.<field>}}" strings.
            def _check_templates(v: Any) -> Optional[str]:
                if isinstance(v, str):
                    if "{{" in v or "}}" in v:
                        if not _TEMPLATE_RE.fullmatch(v.strip()):
                            return f"Invalid template '{v}'"
                    return None
                if isinstance(v, dict):
                    for vv in v.values():
                        err = _check_templates(vv)
                        if err:
                            return err
                    return None
                if isinstance(v, list):
                    for vv in v:
                        err = _check_templates(vv)
                        if err:
                            return err
                    return None
                return None

            template_err = _check_templates(args)
            if template_err:
                return False, f"Intent {idx + 1}: {template_err}"

            # Additional schema guard: disallow unknown arg keys when the schema is strict.
            tool = registry.get(tool_name)
            schema = getattr(tool, "args_schema", {}) if tool is not None else {}
            if isinstance(schema, dict) and schema.get("additionalProperties") is False:
                props = schema.get("properties", {})
                if isinstance(props, dict):
                    unknown_keys = [k for k in args.keys() if k not in props]
                    if unknown_keys:
                        return False, f"Intent {idx + 1}: foreach do.args has unknown field(s): {', '.join(unknown_keys)}"

            # Foreach 'do' full validate_args is deferred.
            continue

        # Normal intent
        tool_name = raw.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return False, f"Intent {idx + 1}: tool must be a non-empty string"
        tool_name = tool_name.strip()
        if not registry.has_tool(tool_name):
            return False, f"Intent {idx + 1}: unknown tool '{tool_name}'"

        args = raw.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return False, f"Intent {idx + 1}: args must be an object"

        if _contains_any_template(args):
            return False, f"Intent {idx + 1}: templates are only allowed inside foreach do.args"

        tool = registry.get(tool_name)
        schema = getattr(tool, "args_schema", {}) if tool is not None else {}
        is_valid, error = validate_args(schema, args)
        if not is_valid:
            msg = (error or {}).get("message") or str(error)
            return False, f"Intent {idx + 1}: invalid args for '{tool_name}': {msg}"

        save_as = raw.get("save_as")
        if save_as is not None:
            if not _is_safe_var_name(save_as):
                return False, f"Intent {idx + 1}: save_as must be a valid variable name"
            saved_vars.add(save_as)

    return True, ""


def apply_item_template(value: Any, item: Dict[str, Any]) -> Any:
    """Apply a single allowed template value against one foreach item."""
    if isinstance(value, str) and ("{{" in value or "}}" in value):
        m = _TEMPLATE_RE.fullmatch(value.strip())
        if not m:
            raise ValueError(f"Invalid template '{value}'")

        field = m.group(1)
        if field == "app":
            # Prefer common keys, but fall back to provided "app".
            return item.get("app") or item.get("process")
        return item.get(field)
    return value


def apply_templates(args: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, dict):
            out[k] = apply_templates(v, item)
        elif isinstance(v, list):
            out[k] = [apply_item_template(x, item) for x in v]
        else:
            out[k] = apply_item_template(v, item)
    return out


def expand_foreach(plan_json: Any, saved_vars: Dict[str, Any], *, max_calls: int = MAX_COMPOSED_TOOL_CALLS) -> Tuple[List[Dict[str, Any]], bool]:
    """Expand foreach macros into a flat list of tool-call intents.

    This is deterministic and side-effect free. It requires saved_vars to
    already contain the foreach source variable(s).
    """
    stopped_early = False
    expanded: List[Dict[str, Any]] = []

    if not isinstance(plan_json, dict):
        return [], True
    intents = plan_json.get("intents")
    if not isinstance(intents, list):
        return [], True

    for raw in intents:
        if len(expanded) >= max_calls:
            stopped_early = True
            break

        if not isinstance(raw, dict):
            continue

        if "foreach" in raw:
            var_name = raw.get("foreach")
            do_obj = raw.get("do")
            if not isinstance(var_name, str) or not isinstance(do_obj, dict):
                continue
            items = _extract_iterable(saved_vars.get(var_name))
            if items is None:
                continue

            tool_name = do_obj.get("tool")
            do_args = do_obj.get("args", {})
            if not isinstance(tool_name, str) or not tool_name.strip() or not isinstance(do_args, dict):
                continue

            for item in items:
                if len(expanded) >= max_calls:
                    stopped_early = True
                    break
                concrete_args = apply_templates(do_args, item)
                expanded.append({"tool": tool_name.strip(), "args": concrete_args})
            continue

        tool_name = raw.get("tool")
        args = raw.get("args", {})
        if isinstance(tool_name, str) and tool_name.strip() and isinstance(args, dict):
            expanded.append({"tool": tool_name.strip(), "args": args})

    return expanded[:max_calls], stopped_early
