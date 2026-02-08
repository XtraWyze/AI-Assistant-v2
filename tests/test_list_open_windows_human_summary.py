from wyzer.core import orchestrator


def test_list_open_windows_human_summary_groups_and_filters(monkeypatch):
    # Ensure verbose fallback isn't accidentally enabled by the environment.
    monkeypatch.delenv("WYZER_WINDOWS_VERBOSE", raising=False)
    monkeypatch.delenv("WYZER_LIST_OPEN_WINDOWS_VERBOSE", raising=False)

    calls = []

    class _StubRegistry:
        def has_tool(self, _name: str) -> bool:
            return True

    def _stub_get_registry():
        return _StubRegistry()

    sample = {
        "windows": [
            # Dev / work
            {"hwnd": 1, "id": 1, "title": "AI-Assistant-v2 — orchestrator.py", "app": "code.exe", "pid": 111},
            {"hwnd": 2, "id": 2, "title": "PowerShell", "app": "windowsterminal.exe", "pid": 222},
            {"hwnd": 3, "id": 3, "title": "pytest -q", "app": "windowsterminal.exe", "pid": 222},
            {"hwnd": 4, "id": 4, "title": "PowerTrader - Positions", "app": "powertrader.exe", "pid": 333},

            # Browsers / comms
            {"hwnd": 5, "id": 5, "title": "Wyzer - GitHub", "app": "chrome.exe", "pid": 444},
            {"hwnd": 6, "id": 6, "title": "Discord", "app": "discord.exe", "pid": 555},
            {"hwnd": 7, "id": 7, "title": "Spotify", "app": "spotify.exe", "pid": 666},

            # Xbox (grouped across host processes)
            {"hwnd": 8, "id": 8, "title": "Xbox", "app": "xboxpcapp.exe", "pid": 777},
            {"hwnd": 9, "id": 9, "title": "", "app": "applicationframehost.exe", "pid": 777},

            # Explorer spam (should be grouped + filtered)
            {"hwnd": 10, "id": 10, "title": "", "app": "explorer.exe", "pid": 888},
            {"hwnd": 11, "id": 11, "title": "", "app": "explorer.exe", "pid": 888},
            {"hwnd": 12, "id": 12, "title": "This PC - File Explorer", "app": "explorer.exe", "pid": 888},
            {"hwnd": 13, "id": 13, "title": "Program Manager", "app": "explorer.exe", "pid": 888},
            {"hwnd": 14, "id": 14, "title": "Windows Input Experience", "app": "explorer.exe", "pid": 888},
        ],
        "count": 14,
        "latency_ms": 1,
    }

    def _stub_execute_tool(_registry, tool_name: str, tool_args: dict):
        calls.append((tool_name, dict(tool_args or {})))
        if tool_name == "list_open_windows":
            return sample
        return {"error": {"type": "unexpected", "message": tool_name}, "latency_ms": 1}

    monkeypatch.setattr(orchestrator, "get_registry", _stub_get_registry)
    monkeypatch.setattr(orchestrator, "_execute_tool", _stub_execute_tool)

    out = orchestrator.handle_user_text("what windows are open")
    reply = out.get("reply") or ""

    # Helpful when running with -s; otherwise it appears in captured logs on failure.
    print("WINDOW SUMMARY REPLY:", reply)

    assert calls and calls[0][0] == "list_open_windows"

    # No old bullet list spam.
    assert "\n•" not in reply
    assert "explorer.exe" not in reply.lower()
    # Only one grouped File Explorer entry (title may include the words too).
    assert reply.lower().count("file explorer (") <= 1

    # Must include key apps.
    assert "VS Code" in reply
    assert "Chrome" in reply
    assert "Discord" in reply
    assert "Spotify" in reply
    assert "Xbox" in reply
    assert "File Explorer" in reply
    assert "Terminal" in reply  # Windows Terminal
    assert "PowerTrader" in reply

    # Keep it compact: ~2-3 lines and reasonable length.
    assert reply.count("\n") <= 2
    assert len(reply) <= 320
