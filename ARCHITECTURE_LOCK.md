Wyzer Architecture Lock (Phase 6)

- assistant.py: audio + state only
- orchestrator.py: intent + tool routing only
- tools/: stateless, JSON-only, no I/O
- local_library/: indexing + resolution only

Any change violating this requires explicit justification.
