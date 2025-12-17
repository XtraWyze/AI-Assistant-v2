# Quiet Mode - User Interface Configuration

## Overview

Quiet Mode provides a cleaner user experience by hiding internal debug information like heartbeats, worker pool status, and other technical logging that isn't relevant to end users.

## Usage

### Command Line Flag

Run with the `--quiet` flag to enable quiet mode:

```bash
python run.py --quiet
```

Or with the batch file:
```batch
run.bat --quiet
```

### Environment Variable

Set the environment variable to enable quiet mode:

```bash
# Windows (PowerShell)
$env:WYZER_QUIET_MODE = "true"
python run.py

# Windows (CMD)
set WYZER_QUIET_MODE=true
python run.py

# Linux/Mac
export WYZER_QUIET_MODE=true
python run.py
```

### Configuration

In `wyzer/core/config.py`:
```python
QUIET_MODE: bool = os.environ.get("WYZER_QUIET_MODE", "false").lower() in ("true", "1", "yes")
```

## What Gets Filtered

When quiet mode is enabled, the following types of messages are hidden:

| Pattern | Description |
|---------|-------------|
| `[HEARTBEAT]` | Worker and brain heartbeat status messages |
| `[POOL].*Worker` | Tool worker pool internal operations |
| `[BRAIN].*heartbeat` | Brain process heartbeat messages |
| `q_in=.*q_out=` | Queue status updates |
| `jobs_processed=` | Job processing statistics |
| `[VAD]` | Voice activity detection internals |
| `[HOTWORD].*score=` | Hotword detection confidence scores |
| `Drained \d+ frames` | Audio frame draining operations |
| `Draining audio queue` | Audio queue management |

## What Still Shows

Even in quiet mode, important messages are still displayed:

- ✅ User transcripts and assistant responses
- ✅ Errors and warnings
- ✅ Startup information
- ✅ Tool execution results
- ✅ Important state changes (listening, processing, etc.)

## Combining with Other Options

Quiet mode works alongside other configuration options:

```bash
# Quiet mode with different log level
python run.py --quiet --log-level WARNING

# Quiet mode without hotword
python run.py --quiet --no-hotword

# Full user-friendly experience
python run.py --quiet --tts on
```

## Programmatic Control

You can also toggle quiet mode programmatically:

```python
from wyzer.core.logger import set_quiet_mode, get_logger

# Enable quiet mode
set_quiet_mode(True)

# Disable quiet mode  
set_quiet_mode(False)

# Check current mode
logger = get_logger()
print(f"Quiet mode: {logger.quiet_mode}")
```

## Adding Custom Filters

To add more patterns to filter in quiet mode, edit `wyzer/core/logger.py`:

```python
QUIET_MODE_FILTERS: List[str] = [
    r"\[HEARTBEAT\]",
    r"\[POOL\].*Worker",
    # Add your patterns here
    r"\[YOUR_PATTERN\]",
]
```

Patterns are regular expressions matched case-insensitively.
