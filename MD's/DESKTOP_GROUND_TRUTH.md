# Desktop Ground Truth Tools (Phase 14)

Wyzer can **see and control** the Windows desktop deterministically.
The LLM only narrates structured tool results — it never guesses.

## Architecture

```
Tools = truth.  LLM = narrator.
If not observed via structured perception, do not claim it.
```

**3-tier perception stack:**

| Tier | Method | Tool | Reliability |
|------|--------|------|-------------|
| A | Active window metadata | `get_active_window` | Highest (Win32 API) |
| B | UI Automation tree | `perceive_uia_focused_window` | High (pywinauto UIA) |
| C | Screenshot + OCR | `screenshot_focused_window` + `ocr_region` | Fallback (OCR optional) |

## Tools

### Perception
| Tool | Returns |
|------|---------|
| `get_active_window` | `{title, exe, pid, hwnd, rect, monitor, timestamp}` |
| `perceive_uia_focused_window` | `{window, controls[], dialogs[], progress?, errors[]}` |
| `describe_screen` | `{summary, highlights[], window:{title,exe}, evidence:{…}}` — **spoken reply** |
| `screenshot_focused_window` | `{image_path, hwnd, rect}` |
| `ocr_region` *(optional)* | `{lines[], full_text, missing_dependency?}` |

### Assertion
| Tool | Returns |
|------|---------|
| `ui_find_text` | `{found: bool, evidence, matches[]}` — search by text in UIA/OCR |
| `install_succeeded_check` | `{status: success\|fail\|unknown, evidence, details}` |

### Action
| Tool | Description |
|------|-------------|
| `desktop_click_uia` | Click a named control via UIA InvokePattern |
| `hotkey` | Press keyboard combos (e.g. `ctrl+c`) |
| `type_text` | Type text into focused control |
| `click_xy` | Click at screen coordinates |
| `scroll` | Mouse wheel scroll |
| `wait_ms` | Sleep for N milliseconds |

### Voice/Hybrid-Router Patterns
These phrases route deterministically (no LLM):

#### Anchored patterns (exact whole-utterance match)
- "what's on screen right now?" → `describe_screen`
- "describe the screen" → `describe_screen`
- "read the screen" → `describe_screen`
- "is there a button that says Install?" → `ui_find_text`
- "did install succeed?" → `install_succeeded_check`
- "what's on my screen?" → `get_window_context` (existing)
- "what windows are open?" → `list_open_windows` (existing)

#### Broad phrases (Phase 14b — matched via normalized text containment)
Screen-description intents (routes to `describe_screen`):
- "what's on my screen" / "what is on my screen"
- "describe my screen" / "describe the screen"
- "can you describe it"
- "what do you see"
- "what is on the screen" / "what's on screen"
- "screen right now"
- "read the screen" / "read my screen"
- "what can you see"
- "tell me what's on my screen"
- "what's currently on my screen"

`describe_screen` returns a short spoken summary that includes:
- Focused window title + exe name
- Up to 6 notable interactive controls (buttons, tabs, menu items, edits, etc.)
- Active dialog overlay title (if any)
- Progress bar value (if any)
- If no readable controls exist, it says so (never guesses)

The response is **never** the generic "Done." — always a human-readable description.

Example natural-speech phrases that now route correctly:
- "Oh, what's on my screen? Can you describe it?" → `describe_screen`
- "What do you see right now?" → `describe_screen`
- "Read the screen" → `describe_screen`

Element-verify intents (routes to `ui_find_text`):
Triggered when text contains "is there" / "do you see" / "can you see" / "you see" / "i see"
AND mentions "button" or a known label (install, play, close, ok, cancel, etc.)
- "You see an install button." → `ui_find_text(text="install")`
- "Is there a button that says Play?" → `ui_find_text(text="play")`
- "Do you see a close button?" → `ui_find_text(text="close")`
- "Can you see the submit button?" → `ui_find_text(text="submit")`

#### LLM Safety Guard
The LLM is **never** allowed to describe the screen without tool evidence.
If a screen-vision query somehow reaches the reply-only path, Wyzer refuses:
> "I can't verify what's on screen right now because the perception tools aren't available in reply-only mode. Try focusing the window and ask again."

#### Router Debug Logging
All routing decisions emit `[ROUTER]` log lines:
- `[ROUTER] matched=screen_describe tool=perceive_uia_focused_window reason=...`
- `[ROUTER] matched=verify_element tool=ui_find_text reason=...`
- `[ROUTER] no_match -> LLM (needs_reasoning=True)`

## Dependencies

### Required (already in repo)
- `pywin32` — Win32 API calls (window metadata)
- `psutil` — Process info
- `mss` — Screenshot capture
- `Pillow` — Image processing
- `pyautogui` — Input simulation (keyboard/mouse)
- `comtypes` — COM interface support

### Added (Phase 14)
- `pywinauto>=0.6.8,<0.7` — UI Automation backend (UIA)

### Optional (OCR)
- `pytesseract` — *(already in repo)* Python wrapper
- **Tesseract OCR engine** — must be installed separately:
  1. Download from https://github.com/tesseract-ocr/tesseract
  2. Install and ensure `tesseract` is on PATH
  3. OCR tools will auto-detect; if missing, they return `missing_dependency: true`

## Installation

```bash
# Standard install (includes pywinauto)
pip install -r requirements.txt

# Optional: Install Tesseract OCR engine for screenshot->text fallback
# Download from https://github.com/tesseract-ocr/tesseract
```

No existing pins were changed. Only `pywinauto>=0.6.8,<0.7` was added.

## Event Log

Every tool execution emits structured events into `WorldState.event_log` (ring buffer, maxlen=200):
- `tool_start` — before execution
- `tool_end` — after execution (with success/failure, duration_ms)
- `perception` — from perception tools
- `ui_action` — from action tools (click, hotkey, type, scroll)
- `warning` — UAC detected, missing deps, etc.

## Demo

```bash
python -m wyzer.tools.demo_desktop           # default: search for "Close" button
python -m wyzer.tools.demo_desktop "Install"  # search for "Install" button
```

Output:
1. Active window metadata
2. UIA controls summary (top 10 with names)
3. Button search result
4. Screenshot path
5. OCR result (or "optional not installed")
6. Event log (last 10 entries)
