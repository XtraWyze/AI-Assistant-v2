"""wyzer.policy

Phase 11 - Autonomy Policy Module.

This package provides:
- Risk classification for tool plans
- Autonomy policy assessment (execute/ask/deny decisions)
- Session state for autonomy mode and confirmations

HARD RULES:
- Autonomy is OFF by default
- All decisions are deterministic (no LLM involvement)
- Policy never modifies tool schemas or routing logic
"""

from wyzer.policy.risk import classify_plan
from wyzer.policy.autonomy_policy import (
    AutonomyDecision,
    assess,
    AUTONOMY_MODES,
)

__all__ = [
    "classify_plan",
    "AutonomyDecision",
    "assess",
    "AUTONOMY_MODES",
]
