"""wyzer.policy.evidence_envelope

Central "LLM Evidence Envelope" for each turn.

Every orchestrator turn that executes tools builds an EvidenceEnvelope
containing ONLY verified, JSON-serializable facts.  The prompt builder
uses this envelope to constrain LLM narration to reality.

HARD RULES:
- Only tool outputs and deterministic WorldState fields may populate the envelope.
- The LLM is NEVER allowed to add to or modify the envelope.
- If the envelope is empty (no tools ran, no deterministic state),
  the LLM must refuse to answer tool-relevant queries.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolEvidence:
    """Single tool execution record inside an envelope."""
    name: str
    args: Dict[str, Any]
    ok: bool
    result: Optional[Any] = None
    error: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "ok": self.ok}
        if self.ok and self.result is not None:
            d["result"] = _truncate(self.result, max_len=300)
        if not self.ok and self.error is not None:
            d["error"] = _truncate(self.error, max_len=200)
        return d


@dataclass
class EvidenceEnvelope:
    """
    Immutable evidence snapshot for a single orchestrator turn.

    Fields:
        turn_id:          Unique turn identifier.
        tools_executed:    Ordered list of tool executions with outcomes.
        world_facts:      Minimal deterministic fields from WorldState
                          (e.g., focused_window, last_tool, active_app).
        limitations:      Human-readable strings describing what Wyzer
                          CANNOT verify this turn (no perception tool, etc.).
    """
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tools_executed: List[ToolEvidence] = field(default_factory=list)
    world_facts: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)

    # ----- convenience helpers -----

    @property
    def has_tool_evidence(self) -> bool:
        return len(self.tools_executed) > 0

    @property
    def any_tool_succeeded(self) -> bool:
        return any(t.ok for t in self.tools_executed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "tools_executed": [t.to_dict() for t in self.tools_executed],
            "world_facts": self.world_facts,
            "limitations": self.limitations,
        }

    def to_prompt_block(self) -> str:
        """Render the envelope as a prompt section for injection into LLM prompts."""
        lines: List[str] = ["VERIFIED_EVIDENCE:"]

        if self.tools_executed:
            lines.append("  tools_executed:")
            for te in self.tools_executed:
                status = "OK" if te.ok else "FAILED"
                detail = ""
                if te.ok and te.result is not None:
                    detail = f" result={_truncate_str(te.result, 200)}"
                elif not te.ok and te.error is not None:
                    detail = f" error={_truncate_str(te.error, 150)}"
                lines.append(f"    - {te.name}: {status}{detail}")
        else:
            lines.append("  tools_executed: (none)")

        if self.world_facts:
            lines.append("  world_facts:")
            for k, v in self.world_facts.items():
                lines.append(f"    {k}: {_truncate_str(v, 120)}")
        else:
            lines.append("  world_facts: (none)")

        if self.limitations:
            lines.append("  limitations:")
            for lim in self.limitations:
                lines.append(f"    - {lim}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder helpers  (called from orchestrator after tool execution)
# ---------------------------------------------------------------------------

def build_envelope_from_execution(
    execution_summary,
    world_state=None,
) -> EvidenceEnvelope:
    """
    Build an EvidenceEnvelope from an ExecutionSummary and optional WorldState.

    Args:
        execution_summary: ExecutionSummary (from intent_plan) or a dict
                           with ``ran`` key containing tool result dicts.
        world_state:       Optional WorldState dataclass instance.

    Returns:
        Populated EvidenceEnvelope.
    """
    envelope = EvidenceEnvelope()

    # ---- tools_executed ----
    ran_list = []
    if hasattr(execution_summary, "ran"):
        ran_list = execution_summary.ran or []
    elif isinstance(execution_summary, dict):
        ran_list = execution_summary.get("ran", [])

    for r in ran_list:
        if hasattr(r, "tool"):
            # ExecutionResult dataclass
            envelope.tools_executed.append(ToolEvidence(
                name=r.tool,
                args={},  # args not stored on ExecutionResult
                ok=r.ok,
                result=r.result,
                error=r.error,
            ))
        elif isinstance(r, dict):
            envelope.tools_executed.append(ToolEvidence(
                name=r.get("tool", "unknown"),
                args=r.get("args", {}),
                ok=r.get("ok", False),
                result=r.get("result"),
                error=r.get("error"),
            ))

    # ---- world_facts (minimal, deterministic) ----
    if world_state is not None:
        wf: Dict[str, Any] = {}
        if getattr(world_state, "last_tool", None):
            wf["last_tool"] = world_state.last_tool
        if getattr(world_state, "last_target", None):
            wf["last_target"] = world_state.last_target
        if getattr(world_state, "active_app", None):
            wf["active_app"] = world_state.active_app
        if getattr(world_state, "active_window_title", None):
            wf["active_window_title"] = world_state.active_window_title
        envelope.world_facts = wf

    # ---- limitations ----
    tool_names = {te.name for te in envelope.tools_executed}
    perception_tools = {
        "describe_screen", "perceive_uia_focused_window",
        "screenshot_focused_window", "ocr_region", "get_window_context",
        "get_active_window",
    }
    if not tool_names & perception_tools:
        envelope.limitations.append(
            "No perception tool was executed this turn; "
            "Wyzer cannot describe what is visually on screen."
        )

    return envelope


def build_empty_envelope(world_state=None) -> EvidenceEnvelope:
    """Build an envelope with NO tool evidence (for reply-only turns)."""
    envelope = EvidenceEnvelope()
    if world_state is not None:
        wf: Dict[str, Any] = {}
        if getattr(world_state, "last_tool", None):
            wf["last_tool"] = world_state.last_tool
        if getattr(world_state, "last_target", None):
            wf["last_target"] = world_state.last_target
        envelope.world_facts = wf
    envelope.limitations.append("No tools were executed this turn.")
    return envelope


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(value: Any, max_len: int = 300) -> Any:
    """Truncate large values to keep envelope small."""
    if isinstance(value, str):
        return value[:max_len] + ("..." if len(value) > max_len else "")
    if isinstance(value, dict):
        import json
        try:
            s = json.dumps(value, ensure_ascii=False, default=str)
            if len(s) > max_len:
                return s[:max_len] + "..."
            return value
        except Exception:
            return str(value)[:max_len]
    if isinstance(value, (list, tuple)):
        import json
        try:
            s = json.dumps(value, ensure_ascii=False, default=str)
            if len(s) > max_len:
                return s[:max_len] + "..."
            return value
        except Exception:
            return str(value)[:max_len]
    return value


def _truncate_str(value: Any, max_len: int = 200) -> str:
    """Coerce to string and truncate."""
    if isinstance(value, str):
        s = value
    elif isinstance(value, dict):
        import json
        try:
            s = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            s = str(value)
    else:
        s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s
