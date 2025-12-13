"""
Indexer for LocalLibrary - builds and maintains index of folders, files, and apps.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List


# Path to library.json (generated index)
LIBRARY_JSON_PATH = Path(__file__).parent / "library.json"


def refresh_index() -> Dict[str, Any]:
    """
    Refresh the local library index.
    
    Indexes:
    - Common folders (Desktop, Downloads, Documents, Pictures, Videos, Music)
    - Installed apps (Start Menu shortcuts + common install locations)
    - User-defined aliases (from aliases.json if present)
    
    Returns:
        {"status": "ok", "counts": {...}, "latency_ms": int}
    """
    start_time = time.perf_counter()
    
    try:
        index_data = {
            "version": "1.0",
            "timestamp": time.time(),
            "folders": {},
            "apps": {},
            "aliases": {}
        }
        
        # Index common folders
        folders = _index_common_folders()
        index_data["folders"] = folders
        
        # Index installed apps
        apps = _index_apps()
        index_data["apps"] = apps
        
        # Load user aliases if present
        aliases = _load_aliases()
        index_data["aliases"] = aliases
        
        # Write to library.json
        with open(LIBRARY_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        return {
            "status": "ok",
            "counts": {
                "folders": len(folders),
                "apps": len(apps),
                "aliases": len(aliases)
            },
            "latency_ms": latency_ms
        }
        
    except Exception as e:
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        return {
            "error": {
                "type": "index_error",
                "message": str(e)
            },
            "latency_ms": latency_ms
        }


def _index_common_folders() -> Dict[str, str]:
    """
    Index common user folders.
    
    Returns:
        Dict mapping folder name (lowercase) to absolute path
    """
    folders = {}
    
    # Get user home
    home = Path.home()
    
    # Common folder names
    common_folders = [
        "Desktop",
        "Downloads",
        "Documents",
        "Pictures",
        "Videos",
        "Music"
    ]
    
    for folder_name in common_folders:
        folder_path = home / folder_name
        if folder_path.exists():
            folders[folder_name.lower()] = str(folder_path)
    
    # Add some common aliases
    if "downloads" in folders:
        folders["download"] = folders["downloads"]
    if "pictures" in folders:
        folders["pics"] = folders["pictures"]
        folders["photos"] = folders["pictures"]
    if "videos" in folders:
        folders["vids"] = folders["videos"]
    if "documents" in folders:
        folders["docs"] = folders["documents"]
    
    return folders


def _index_apps() -> Dict[str, Dict[str, str]]:
    """
    Index installed applications from Start Menu and common locations.
    
    Returns:
        Dict mapping app name (lowercase) to {"path": "...", "type": "shortcut|exe"}
    """
    apps = {}
    
    # Index Start Menu shortcuts
    start_menu_paths = [
        Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    ]
    
    for start_menu in start_menu_paths:
        if start_menu.exists():
            _scan_start_menu(start_menu, apps)
    
    # Add common apps from known locations
    _add_common_apps(apps)
    
    return apps


def _scan_start_menu(path: Path, apps: Dict[str, Dict[str, str]], max_depth: int = 3, current_depth: int = 0) -> None:
    """
    Recursively scan Start Menu for .lnk shortcuts.
    
    Args:
        path: Path to scan
        apps: Dict to populate with found apps
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
    """
    if current_depth >= max_depth:
        return
    
    try:
        for entry in path.iterdir():
            if entry.is_dir():
                _scan_start_menu(entry, apps, max_depth, current_depth + 1)
            elif entry.is_file() and entry.suffix.lower() == ".lnk":
                # Extract app name from filename
                app_name = entry.stem.lower()
                
                # Skip duplicates, uninstallers, etc.
                skip_keywords = ["uninstall", "readme", "help", "website"]
                if any(kw in app_name for kw in skip_keywords):
                    continue
                
                # Store shortcut path
                if app_name not in apps:
                    apps[app_name] = {
                        "path": str(entry),
                        "type": "shortcut"
                    }
    except (PermissionError, OSError):
        # Skip inaccessible directories
        pass


def _add_common_apps(apps: Dict[str, Dict[str, str]]) -> None:
    """
    Add common applications from known install locations.
    
    Args:
        apps: Dict to populate with found apps
    """
    # Common app definitions: {name: [possible_paths]}
    common_apps = {
        "notepad": [r"C:\Windows\System32\notepad.exe"],
        "calc": [r"C:\Windows\System32\calc.exe"],
        "calculator": [r"C:\Windows\System32\calc.exe"],
        "paint": [r"C:\Windows\System32\mspaint.exe"],
        "explorer": [r"C:\Windows\explorer.exe"],
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
        ],
        "firefox": [
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"
        ],
        "edge": [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"],
        "vscode": [
            r"C:\Program Files\Microsoft VS Code\Code.exe",
            r"C:\Users\{}\AppData\Local\Programs\Microsoft VS Code\Code.exe".format(os.environ.get("USERNAME", ""))
        ],
        "cmd": [r"C:\Windows\System32\cmd.exe"],
        "powershell": [r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"]
    }
    
    for app_name, paths in common_apps.items():
        if app_name in apps:
            continue  # Already found in Start Menu
        
        for path_str in paths:
            path = Path(path_str)
            if path.exists():
                apps[app_name] = {
                    "path": str(path),
                    "type": "exe"
                }
                break


def _load_aliases() -> Dict[str, Dict[str, str]]:
    """
    Load user-defined aliases from aliases.json if present.
    
    Returns:
        Dict mapping alias name (lowercase) to {"target": "...", "type": "folder|file|app|url"}
    """
    aliases_path = Path(__file__).parent / "aliases.json"
    
    if not aliases_path.exists():
        return {}
    
    try:
        with open(aliases_path, 'r', encoding='utf-8') as f:
            aliases_data = json.load(f)
        
        # Normalize keys to lowercase
        normalized = {}
        for key, value in aliases_data.items():
            if isinstance(value, dict) and "target" in value:
                normalized[key.lower()] = value
        
        return normalized
        
    except Exception:
        return {}


def get_cached_index() -> Dict[str, Any]:
    """
    Get cached index from library.json.
    
    Returns:
        Index data dict, or empty structure if not found
    """
    if not LIBRARY_JSON_PATH.exists():
        # Return empty structure
        return {
            "version": "1.0",
            "timestamp": 0,
            "folders": {},
            "apps": {},
            "aliases": {}
        }
    
    try:
        with open(LIBRARY_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {
            "version": "1.0",
            "timestamp": 0,
            "folders": {},
            "apps": {},
            "aliases": {}
        }
