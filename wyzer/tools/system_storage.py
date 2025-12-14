"""
System storage scanning and drive management tools.

Provides:
- system_storage_scan: Deep scan with caching
- system_storage_list: Quick list of drives
- system_storage_open: Open a drive in file manager

Cache is stored in wyzer/data/system_storage_index.json
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from wyzer.tools.tool_base import ToolBase


# ============================================================================
# Internals: Drive scanning, caching, normalization
# ============================================================================

def _get_data_dir() -> Path:
    """Get or create the data directory for caching."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _get_cache_path() -> Path:
    """Get the path to the system storage cache file."""
    return _get_data_dir() / "system_storage_index.json"


def _round_gb(bytes_val: float) -> float:
    """Convert bytes to GB and round to 2 decimals."""
    gb = bytes_val / (1024 ** 3)
    return round(gb, 2)


def normalize_drive_token(text: str) -> Optional[str]:
    """Normalize a drive token to a canonical mountpoint.
    
    Examples:
        - "d" -> "D:\\"  (Windows)
        - "D:" -> "D:\\"  (Windows)
        - "d drive" -> "D:\\"  (Windows)
        - "D:\\" -> "D:\\"  (Windows)
        - "/mnt/storage" -> "/mnt/storage"  (Linux/macOS)
    
    Returns:
        Canonical mountpoint string or None if not recognized.
    """
    if not text:
        return None
    
    text = text.strip().lower()
    
    # Remove " drive" suffix (as a word, not character set)
    if text.endswith(" drive"):
        text = text[:-6].strip()  # Remove " drive"
    
    # Windows: single letter (a-z) optionally followed by colons/backslashes
    text_clean = text.rstrip(":\\").strip()
    if len(text_clean) == 1 and text_clean.isalpha():
        letter = text_clean.upper()
        # On Windows, return the drive letter format
        if platform.system() == "Windows":
            return f"{letter}:\\"
        # On POSIX, this won't match; let it fall through
    
    # Already looks like a full Windows path
    if text.startswith(("/mnt/", "/media/", "/")) and platform.system() != "Windows":
        return text
    
    # Attempt to match against actual mounted drives
    try:
        drives = _scan_drives_impl()
        for drive_info in drives:
            mp = drive_info.get("mountpoint", "").lower()
            name = drive_info.get("name", "").lower()
            
            if mp.lower() == text or mp.lower().rstrip("\\").lower() == text.rstrip("\\").lower():
                return drive_info.get("mountpoint")
            if name.lower() == text or name.lower().rstrip(":").lower() == text.rstrip(":").lower():
                return drive_info.get("mountpoint")
    except Exception:
        pass
    
    return None


def _scan_drives_impl() -> List[Dict[str, Any]]:
    """Internal implementation of drive scanning.
    
    Tries psutil first, falls back to platform-specific methods.
    """
    drives = []
    
    # Try psutil first (preferred)
    try:
        import psutil
        
        for partition in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                drives.append({
                    "name": partition.device.split("\\")[-1].split("/")[-1] or partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total_gb": _round_gb(usage.total),
                    "used_gb": _round_gb(usage.used),
                    "free_gb": _round_gb(usage.free),
                    "percent_used": round(usage.percent, 1),
                    "is_removable": None,
                    "label": None,
                })
            except Exception:
                # Skip drives we can't read
                pass
        
        if drives:
            return drives
    except ImportError:
        pass
    
    # Fallback: Windows
    if platform.system() == "Windows":
        import string
        
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                try:
                    import shutil
                    usage = shutil.disk_usage(drive)
                    drives.append({
                        "name": f"{letter}:",
                        "mountpoint": drive,
                        "fstype": "NTFS",  # Simplified; could detect
                        "total_gb": _round_gb(usage.total),
                        "used_gb": _round_gb(usage.used),
                        "free_gb": _round_gb(usage.free),
                        "percent_used": round((usage.used / usage.total * 100) if usage.total else 0, 1),
                        "is_removable": None,
                        "label": None,
                    })
                except Exception:
                    pass
    
    # Fallback: POSIX (Linux/macOS)
    elif platform.system() in ("Linux", "Darwin"):
        import shutil
        
        # Try /proc/mounts on Linux
        if os.path.exists("/proc/mounts"):
            try:
                with open("/proc/mounts") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            device, mountpoint = parts[0], parts[1]
                            if os.path.exists(mountpoint):
                                try:
                                    usage = shutil.disk_usage(mountpoint)
                                    drives.append({
                                        "name": device.split("/")[-1],
                                        "mountpoint": mountpoint,
                                        "fstype": parts[2] if len(parts) > 2 else "unknown",
                                        "total_gb": _round_gb(usage.total),
                                        "used_gb": _round_gb(usage.used),
                                        "free_gb": _round_gb(usage.free),
                                        "percent_used": round((usage.used / usage.total * 100) if usage.total else 0, 1),
                                        "is_removable": None,
                                        "label": None,
                                    })
                                except Exception:
                                    pass
            except Exception:
                pass
        
        # Fallback: check common mount points
        if not drives:
            common_mounts = ["/", "/home", "/mnt", "/media", "/Volumes"]
            for mp in common_mounts:
                if os.path.exists(mp):
                    try:
                        import shutil
                        usage = shutil.disk_usage(mp)
                        drives.append({
                            "name": mp.split("/")[-1] or "root",
                            "mountpoint": mp,
                            "fstype": "unknown",
                            "total_gb": _round_gb(usage.total),
                            "used_gb": _round_gb(usage.used),
                            "free_gb": _round_gb(usage.free),
                            "percent_used": round((usage.used / usage.total * 100) if usage.total else 0, 1),
                            "is_removable": None,
                            "label": None,
                        })
                    except Exception:
                        pass
    
    return drives


def _load_cache() -> Optional[Dict[str, Any]]:
    """Load drive index from cache if it exists and is recent."""
    cache_path = _get_cache_path()
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def _save_cache(drives: List[Dict[str, Any]]) -> None:
    """Save drive index to cache."""
    cache_path = _get_cache_path()
    try:
        cache_data = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "drives": drives,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception:
        # Silently fail; cache is optional
        pass


def scan_drives(refresh: bool = False) -> List[Dict[str, Any]]:
    """Scan drives with optional caching.
    
    Args:
        refresh: If True, always rescan; if False, use cache if available
        
    Returns:
        List of drive info dicts
    """
    if not refresh:
        cached = _load_cache()
        if cached:
            return cached.get("drives", [])
    
    drives = _scan_drives_impl()
    _save_cache(drives)
    return drives


# ============================================================================
# Tools
# ============================================================================

class SystemStorageScanTool(ToolBase):
    """Tool to scan all drives and create/update index."""
    
    def __init__(self):
        """Initialize system_storage_scan tool."""
        super().__init__()
        self._name = "system_storage_scan"
        self._description = "Scan all mounted drives and partitions, creating a cached index with detailed storage info"
        self._args_schema = {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "description": "Force refresh (default False to use cache)"
                }
            },
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run system storage scan.
        
        Returns:
            {
                "status": "ok",
                "refreshed": bool,
                "index_path": str,
                "drives": [...],
                "latency_ms": int
            }
        """
        start_time = time.perf_counter()
        
        try:
            refresh = kwargs.get("refresh", False)
            if not isinstance(refresh, bool):
                refresh = False
            
            drives = scan_drives(refresh=refresh)
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "ok",
                "refreshed": refresh,
                "index_path": str(_get_cache_path()),
                "drives": drives,
                "latency_ms": latency_ms
            }
        
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "scan_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }


class SystemStorageListTool(ToolBase):
    """Tool to quickly list drives."""
    
    def __init__(self):
        """Initialize system_storage_list tool."""
        super().__init__()
        self._name = "system_storage_list"
        self._description = "List all mounted drives with free/total space and usage percentage"
        self._args_schema = {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "description": "Force refresh (default False to use cache)"
                },
                "drive": {
                    "type": "string",
                    "description": "Optional: filter to specific drive (e.g., 'D', 'D:', or '/mnt/storage')"
                }
            },
            "required": [],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run system storage list.
        
        Returns:
            {
                "status": "ok",
                "drives": [
                    {
                        "name": str,
                        "mountpoint": str,
                        "free_gb": float,
                        "total_gb": float,
                        "percent_used": float
                    }, ...
                ]
            }
        """
        try:
            refresh = kwargs.get("refresh", False)
            if not isinstance(refresh, bool):
                refresh = False
            
            drives = scan_drives(refresh=refresh)
            
            # Optional filtering by drive
            drive_filter = kwargs.get("drive", "").strip()
            if drive_filter:
                canonical = normalize_drive_token(drive_filter)
                if canonical:
                    drives = [d for d in drives if d.get("mountpoint").lower() == canonical.lower()]
                    if not drives:
                        return {
                            "error": {
                                "type": "invalid_argument",
                                "message": f"Drive '{drive_filter}' not found"
                            }
                        }
                else:
                    return {
                        "error": {
                            "type": "invalid_argument",
                            "message": f"Could not parse drive '{drive_filter}'"
                        }
                    }
            
            return {
                "status": "ok",
                "drives": [
                    {
                        "name": d.get("name"),
                        "mountpoint": d.get("mountpoint"),
                        "free_gb": d.get("free_gb"),
                        "total_gb": d.get("total_gb"),
                        "percent_used": d.get("percent_used")
                    }
                    for d in drives
                ]
            }
        
        except Exception as e:
            return {
                "error": {
                    "type": "list_error",
                    "message": str(e)
                }
            }


class SystemStorageOpenTool(ToolBase):
    """Tool to open a drive in file explorer/manager."""
    
    def __init__(self):
        """Initialize system_storage_open tool."""
        super().__init__()
        self._name = "system_storage_open"
        self._description = "Open a drive or folder in the file manager"
        self._args_schema = {
            "type": "object",
            "properties": {
                "drive": {
                    "type": "string",
                    "description": "Drive or path to open (e.g., 'D', 'D:', 'D:\\\\', '/mnt/storage')"
                }
            },
            "required": ["drive"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Open a drive in file explorer/manager.
        
        Returns:
            {
                "status": "ok",
                "opened": str  (the mountpoint that was opened)
            }
        """
        try:
            drive_token = kwargs.get("drive", "").strip()
            if not drive_token:
                return {
                    "error": {
                        "type": "invalid_argument",
                        "message": "drive parameter is required"
                    }
                }
            
            mountpoint = normalize_drive_token(drive_token)
            if not mountpoint:
                return {
                    "error": {
                        "type": "invalid_argument",
                        "message": f"Could not parse drive '{drive_token}'"
                    }
                }
            
            if not os.path.exists(mountpoint):
                return {
                    "error": {
                        "type": "not_found",
                        "message": f"Drive '{mountpoint}' does not exist"
                    }
                }
            
            system = platform.system()
            
            try:
                if system == "Windows":
                    os.startfile(mountpoint)
                elif system == "Darwin":
                    subprocess.run(["open", mountpoint], check=True, timeout=5)
                elif system == "Linux":
                    subprocess.run(["xdg-open", mountpoint], check=True, timeout=5)
                else:
                    return {
                        "error": {
                            "type": "platform_error",
                            "message": f"Unsupported platform: {system}"
                        }
                    }
            except subprocess.TimeoutExpired:
                # Even if subprocess times out, it likely started
                pass
            except Exception as e:
                return {
                    "error": {
                        "type": "open_error",
                        "message": str(e)
                    }
                }
            
            return {
                "status": "ok",
                "opened": mountpoint
            }
        
        except Exception as e:
            return {
                "error": {
                    "type": "open_error",
                    "message": str(e)
                }
            }
