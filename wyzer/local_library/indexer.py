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


def refresh_index(mode: str = "normal") -> Dict[str, Any]:
    """
    Refresh the local library index.
    
    Indexes:
    - Common folders (Desktop, Downloads, Documents, Pictures, Videos, Music)
    - Installed apps (Start Menu shortcuts + common install locations)
    - User-defined aliases (from aliases.json if present)
    - Tier 2 apps (EXEs from Program Files and user install dirs) - if mode="full"
    - Tier 3 files (root drive scan) - if mode="tier3"
    - Games (Steam, Epic, shortcuts, folder scan, Xbox) - always refreshed
    
    Args:
        mode: "normal" (Start Menu only), "full" (includes Tier 2 EXE scanning), 
              or "tier3" (includes full file system scan)
    
    Returns:
        {"status": "ok", "counts": {...}, "latency_ms": int}
    """
    from wyzer.core.logger import get_logger
    logger = get_logger()
    
    start_time = time.perf_counter()
    
    try:
        # Log scan start
        logger.info("[SCAN] Scanning apps...")
        
        # Load existing index to preserve scan_meta
        existing_index = get_cached_index()
        existing_scan_meta = existing_index.get("scan_meta", {
            "last_refresh": "",
            "dirs": {}
        })
        
        index_data = {
            "version": "1.0",
            "timestamp": time.time(),
            "folders": {},
            "apps": {},
            "aliases": {},
            "tier2_apps": [],
            "tier3_files": [],
            "tier3_drives": [],
            "games": [],
            "games_scan_meta": {},
            "uwp_apps": [],
            "uwp_scan_meta": {},
            "scan_meta": existing_scan_meta
        }
        
        # Index common folders
        logger.info("[SCAN] Indexing folders...")
        folders = _index_common_folders()
        index_data["folders"] = folders
        
        # Index installed apps
        logger.info("[SCAN] Indexing installed apps...")
        apps = _index_apps()
        index_data["apps"] = apps
        
        # Load user aliases if present
        logger.info("[SCAN] Loading aliases...")
        aliases = _load_aliases()
        index_data["aliases"] = aliases
        
        # Tier 2 EXE scanning (if mode="full" or mode="tier3")
        tier2_apps = []
        tier2_errors = []
        if mode in ["full", "tier3"]:
            logger.info("[SCAN] Scanning Tier 2 applications (this may take a moment)...")
            tier2_apps, tier2_errors = _index_tier2_apps(existing_scan_meta, existing_index.get("tier2_apps", []))
            index_data["tier2_apps"] = tier2_apps
            index_data["scan_meta"]["last_refresh"] = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Preserve existing tier2 data in normal mode
            tier2_apps = existing_index.get("tier2_apps", [])
            index_data["tier2_apps"] = tier2_apps
        
        # Tier 3 file system scanning (if mode="tier3")
        tier3_files = []
        tier3_drives = []
        tier3_errors = []
        if mode == "tier3":
            logger.info("[SCAN] Detecting available drives...")
            tier3_drives = get_available_drives()
            index_data["tier3_drives"] = tier3_drives
            
            # Scan C: drive by default
            default_drive = None
            for drive in tier3_drives:
                if drive["letter"] == "C":
                    default_drive = drive
                    break
            
            if default_drive:
                logger.info(f"[SCAN] Scanning Tier 3 files on drive {default_drive['letter']}: (this may take several minutes)...")
                scan_result = scan_tier3_root(default_drive['letter'], max_results=5000)
                
                if scan_result.get("status") == "ok":
                    tier3_files = scan_result.get("files", [])
                    index_data["tier3_files"] = tier3_files
                    if scan_result.get("errors", 0) > 0:
                        tier3_errors.append(f"Tier 3 scan errors: {scan_result['errors']}")
                else:
                    tier3_errors.append(f"Tier 3 scan failed: {scan_result.get('message', 'Unknown error')}")
        else:
            # Preserve existing tier3 data in non-tier3 modes
            tier3_files = existing_index.get("tier3_files", [])
            tier3_drives = existing_index.get("tier3_drives", [])
            index_data["tier3_files"] = tier3_files
            index_data["tier3_drives"] = tier3_drives
        
        # Index games (always refresh)
        logger.info("[SCAN] Scanning games...")
        from wyzer.local_library.game_indexer import merge_games_into_library
        index_data = merge_games_into_library(index_data)
        
        # Index UWP apps (always refresh)
        logger.info("[SCAN] Scanning UWP apps...")
        from wyzer.local_library.uwp_indexer import refresh_uwp_index
        uwp_result = refresh_uwp_index()
        index_data["uwp_apps"] = uwp_result.get("apps", [])
        index_data["uwp_scan_meta"] = {
            "last_refresh": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": uwp_result.get("count", 0),
            "status": uwp_result.get("status", "error")
        }
        if "error" in uwp_result:
            index_data["uwp_scan_meta"]["error"] = uwp_result["error"]
        
        # Write to library.json
        logger.info("[SCAN] Finalizing scan...")
        save_library(index_data)
        
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        result = {
            "status": "ok",
            "counts": {
                "folders": len(folders),
                "apps": len(apps),
                "aliases": len(aliases),
                "tier2_apps": len(tier2_apps),
                "tier3_files": len(tier3_files),
                "tier3_drives": len(tier3_drives),
                "games": len(index_data.get("games", [])),
                "uwp_apps": len(index_data.get("uwp_apps", []))
            },
            "latency_ms": latency_ms
        }
        
        if tier2_errors:
            result["tier2_errors"] = tier2_errors
        
        if tier3_errors:
            result["tier3_errors"] = tier3_errors
        
        # Include games scan metadata
        if "games_scan_meta" in index_data:
            result["games_sources"] = index_data["games_scan_meta"].get("sources", {})
            if "errors" in index_data["games_scan_meta"]:
                result["games_errors"] = index_data["games_scan_meta"]["errors"]
        
        return result
        
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
            "aliases": {},
            "tier2_apps": [],
            "tier3_files": [],
            "tier3_drives": [],
            "games": [],
            "games_scan_meta": {},
            "uwp_apps": [],
            "uwp_scan_meta": {},
            "scan_meta": {
                "last_refresh": "",
                "dirs": {}
            }
        }
    
    try:
        with open(LIBRARY_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure backward compatibility
        if "tier2_apps" not in data:
            data["tier2_apps"] = []
        if "tier3_files" not in data:
            data["tier3_files"] = []
        if "tier3_drives" not in data:
            data["tier3_drives"] = []
        if "scan_meta" not in data:
            data["scan_meta"] = {"last_refresh": "", "dirs": {}}
        if "games" not in data:
            data["games"] = []
        if "games_scan_meta" not in data:
            data["games_scan_meta"] = {}
        if "uwp_apps" not in data:
            data["uwp_apps"] = []
        if "uwp_scan_meta" not in data:
            data["uwp_scan_meta"] = {}

        # Always overlay aliases from aliases.json so new aliases take effect
        # immediately without requiring a full refresh.
        data["aliases"] = _load_aliases()
        
        return data
    except Exception:
        return {
            "version": "1.0",
            "timestamp": 0,
            "folders": {},
            "apps": {},
            "aliases": {},
            "tier2_apps": [],
            "tier3_files": [],
            "tier3_drives": [],
            "games": [],
            "games_scan_meta": {},
            "uwp_apps": [],
            "uwp_scan_meta": {},
            "scan_meta": {
                "last_refresh": "",
                "dirs": {}
            }
        }


# ============================================================================
# TIER 2 EXE SCANNING
# ============================================================================

# Directories to EXCLUDE from scanning (case-insensitive)
EXCLUDE_DIRS = {
    "windows", "system32", "winsxs", "installer", "drivers", "dotnet",
    "common files", "runtime", "temp", "cache", "updates", "$recycle.bin",
    "windowsapps", "microsoft.net", "windows defender", "windows mail",
    "windows media player", "windows nt", "windows photo viewer",
    "windows security", "windowspowershell"
}

# EXE filenames to EXCLUDE (case-insensitive substring match)
EXCLUDE_EXE_KEYWORDS = {
    "uninstall", "update", "helper", "crash", "setup", "service",
    "daemon", "installer", "uninst", "updater", "crashreport",
    "crashhandler", "bootstrapper", "launcher.exe", "unins"
}


def _index_tier2_apps(scan_meta: Dict[str, Any], existing_apps: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Index user-facing executables from common install locations.
    
    Uses incremental scanning: only rescans directories that have changed.
    Preserves apps from unchanged directories.
    
    Args:
        scan_meta: Existing scan metadata with directory mtimes
        existing_apps: Previously indexed Tier 2 apps to preserve
        
    Returns:
        Tuple of (tier2_apps_list, error_messages)
    """
    newly_scanned_apps = []
    errors = []
    
    # Define scan locations
    scan_locations = _get_tier2_scan_locations()
    
    # Track which directories were rescanned
    rescanned_dirs = set()
    updated_dirs = {}
    
    for location in scan_locations:
        if not location["path"].exists():
            continue
        
        try:
            # Check if directory needs rescanning
            dir_key = str(location["path"])
            dir_mtime = location["path"].stat().st_mtime
            
            # Skip if unchanged (incremental scan optimization)
            if dir_key in scan_meta.get("dirs", {}) and scan_meta["dirs"][dir_key] == dir_mtime:
                # Directory hasn't changed, we'll keep existing apps from it
                continue
            
            # Mark this directory as rescanned
            rescanned_dirs.add(dir_key)
            
            # Scan this location
            found_apps = _scan_tier2_location(
                location["path"],
                location["source"],
                max_depth=3
            )
            
            newly_scanned_apps.extend(found_apps)
            updated_dirs[dir_key] = dir_mtime
            
        except Exception as e:
            errors.append(f"Error scanning {location['path']}: {str(e)}")
    
    # Merge: Keep existing apps from unchanged directories
    final_apps = []
    
    # Add newly scanned apps
    final_apps.extend(newly_scanned_apps)
    
    # Add existing apps from unchanged directories
    for app in existing_apps:
        # Check if this app's root directory was rescanned
        app_path = app["exe_path"]
        in_rescanned_dir = any(app_path.startswith(scan_dir) for scan_dir in rescanned_dirs)
        
        if not in_rescanned_dir:
            # This app is from an unchanged directory, keep it
            final_apps.append(app)
    
    # Update scan_meta with new mtimes
    scan_meta["dirs"].update(updated_dirs)
    
    # Deduplicate by exe_path
    seen_paths = set()
    unique_apps = []
    for app in final_apps:
        if app["exe_path"] not in seen_paths:
            seen_paths.add(app["exe_path"])
            unique_apps.append(app)
    
    return unique_apps, errors


def _get_tier2_scan_locations() -> List[Dict[str, Any]]:
    """
    Get list of Tier 2 scan locations for Windows.
    
    Returns:
        List of {"path": Path, "source": str}
    """
    locations = []
    
    # C:\Program Files
    pf = Path(r"C:\Program Files")
    if pf.exists():
        locations.append({"path": pf, "source": "program_files"})
    
    # C:\Program Files (x86)
    pf_x86 = Path(r"C:\Program Files (x86)")
    if pf_x86.exists():
        locations.append({"path": pf_x86, "source": "program_files"})
    
    # %LOCALAPPDATA%\Programs
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        local_programs = Path(localappdata) / "Programs"
        if local_programs.exists():
            locations.append({"path": local_programs, "source": "user_programs"})
    
    # %APPDATA%\Programs
    appdata = os.environ.get("APPDATA")
    if appdata:
        appdata_programs = Path(appdata) / "Programs"
        if appdata_programs.exists():
            locations.append({"path": appdata_programs, "source": "user_programs"})
    
    return locations


def _scan_tier2_location(
    root_path: Path,
    source: str,
    max_depth: int = 3,
    current_depth: int = 0
) -> List[Dict[str, Any]]:
    """
    Recursively scan a location for user-facing executables.
    
    Args:
        root_path: Root directory to scan
        source: Source type ("program_files" or "user_programs")
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        
    Returns:
        List of app metadata dicts
    """
    apps = []
    
    if current_depth >= max_depth:
        return apps
    
    try:
        for entry in root_path.iterdir():
            # Check if directory should be excluded
            if entry.is_dir():
                dir_name_lower = entry.name.lower()
                
                # Skip excluded directories
                if any(exclude in dir_name_lower for exclude in EXCLUDE_DIRS):
                    continue
                
                # Recurse into subdirectory
                apps.extend(_scan_tier2_location(
                    entry,
                    source,
                    max_depth,
                    current_depth + 1
                ))
            
            elif entry.is_file() and entry.suffix.lower() == ".exe":
                # Check if EXE should be excluded
                exe_name_lower = entry.name.lower()
                
                if any(keyword in exe_name_lower for keyword in EXCLUDE_EXE_KEYWORDS):
                    continue
                
                # Extract metadata
                try:
                    stat_info = entry.stat()
                    
                    # Generate friendly name
                    friendly_name = _generate_friendly_name(entry)
                    
                    apps.append({
                        "name": friendly_name,
                        "exe_path": str(entry),
                        "source": source,
                        "folder": entry.parent.name,
                        "mtime": stat_info.st_mtime
                    })
                except Exception:
                    # Skip files that can't be accessed
                    pass
    
    except (PermissionError, OSError):
        # Skip inaccessible directories
        pass
    
    return apps


def _generate_friendly_name(exe_path: Path) -> str:
    """
    Generate a friendly display name for an executable.
    
    Args:
        exe_path: Path to the executable
        
    Returns:
        Friendly name string
    """
    # Start with filename without extension
    name = exe_path.stem
    
    # Check if parent folder name is cleaner
    parent_name = exe_path.parent.name
    
    # Prefer parent folder if:
    # 1. EXE name is generic (like "app.exe", "main.exe")
    # 2. Parent folder name is more descriptive
    generic_names = {"app", "main", "launcher", "start", "run", "client", "program"}
    
    if name.lower() in generic_names and len(parent_name) > 3:
        name = parent_name
    
    # Clean up the name
    # Remove common suffixes
    suffixes_to_remove = ["_app", "_client", "_launcher", ".app", "-app", "-client"]
    for suffix in suffixes_to_remove:
        if name.lower().endswith(suffix):
            name = name[:- len(suffix)]
    
    # Replace underscores and hyphens with spaces
    name = name.replace("_", " ").replace("-", " ")
    
    # Title case
    name = name.title()
    
    return name


# ============================================================================
# TIER 3 FILE SYSTEM SCANNING
# ============================================================================

# Directories to EXCLUDE from tier 3 scanning (case-insensitive)
EXCLUDE_DIRS_TIER3 = {
    "$recycle.bin", "system volume information", "pagefile.sys", "hiberfil.sys",
    "windows", "program files", "program files (x86)", "programdata",
    "appdata", "application data", "temp", "tmp", "$temp", "cache",
    "recycler", "system32", "syswow64", "drivers", "winsxs",
    "perflogs", ".git", ".svn", "node_modules", "__pycache__",
    ".venv", "venv", "env", ".env"
}

# File extensions to prioritize in tier 3 scans
PRIORITIZED_EXTENSIONS = {
    ".exe", ".msi", ".iso", ".zip", ".7z", ".rar",
    ".doc", ".docx", ".xls", ".xlsx", ".pdf",
    ".jpg", ".jpeg", ".png", ".gif", ".mp3", ".mp4"
}


def get_available_drives() -> List[Dict[str, Any]]:
    """
    Get list of available drives on Windows.
    
    Returns:
        List of {"letter": "C", "path": "C:\\", "total_gb": float, "free_gb": float, "label": str}
    """
    import shutil
    drives = []
    
    # Check all drive letters A-Z
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive_path = f"{letter}:\\"
        
        try:
            # Check if drive exists by trying to access it
            if os.path.exists(drive_path):
                # Get drive space info
                stat = shutil.disk_usage(drive_path)
                total_gb = stat.total / (1024 ** 3)
                free_gb = stat.free / (1024 ** 3)
                
                # Get drive label (volume name)
                label = ""
                try:
                    # Try to get volume label from registry or ctypes
                    import ctypes
                    volume_label = ctypes.create_unicode_buffer(1024)
                    ctypes.windll.kernel32.GetVolumeInformationW(
                        ctypes.c_wchar_p(drive_path),
                        volume_label, ctypes.sizeof(volume_label),
                        None, None, None, None, 0
                    )
                    label = volume_label.value if volume_label.value else "Local Disk"
                except Exception:
                    label = "Local Disk"
                
                drives.append({
                    "letter": letter,
                    "path": drive_path,
                    "total_gb": round(total_gb, 2),
                    "free_gb": round(free_gb, 2),
                    "label": label
                })
        except (OSError, PermissionError):
            # Drive not accessible
            pass
    
    return drives


def scan_tier3_root(drive_letter: str = "C", max_results: int = 10000) -> Dict[str, Any]:
    """
    Perform a Tier 3 scan on a specific drive root.
    
    Recursively scans a drive and indexes:
    - All executable files
    - Common document types
    - Media files
    - Compressed archives
    
    Args:
        drive_letter: Drive letter to scan (e.g., "C", "D")
        max_results: Maximum files to index before stopping
        
    Returns:
        {"status": "ok", "count": int, "files": [...], "latency_ms": int}
    """
    from wyzer.core.logger import get_logger
    logger = get_logger()
    
    start_time = time.perf_counter()
    
    try:
        drive_path = Path(f"{drive_letter}:\\")
        
        if not drive_path.exists():
            return {
                "status": "error",
                "message": f"Drive {drive_letter}: not found"
            }
        
        logger.info(f"[SCAN] Starting Tier 3 scan on drive {drive_letter}:")
        
        files = []
        error_count = 0
        
        # Recursively scan the drive
        def _recursive_tier3_scan(path: Path, depth: int = 0, max_depth: int = 10) -> None:
            nonlocal files, error_count
            
            if len(files) >= max_results or depth >= max_depth:
                return
            
            try:
                for entry in path.iterdir():
                    if len(files) >= max_results:
                        return
                    
                    try:
                        if entry.is_dir():
                            dir_name_lower = entry.name.lower()
                            
                            # Skip excluded directories
                            if any(exclude in dir_name_lower for exclude in EXCLUDE_DIRS_TIER3):
                                continue
                            
                            # Recurse
                            _recursive_tier3_scan(entry, depth + 1, max_depth)
                        
                        elif entry.is_file():
                            suffix_lower = entry.suffix.lower()
                            
                            # Only index prioritized extensions
                            if suffix_lower in PRIORITIZED_EXTENSIONS:
                                try:
                                    stat_info = entry.stat()
                                    files.append({
                                        "name": entry.name,
                                        "path": str(entry),
                                        "type": suffix_lower,
                                        "size_mb": round(stat_info.st_size / (1024 ** 2), 2),
                                        "mtime": stat_info.st_mtime
                                    })
                                except Exception:
                                    error_count += 1
                    
                    except (PermissionError, OSError):
                        error_count += 1
            
            except (PermissionError, OSError):
                error_count += 1
        
        _recursive_tier3_scan(drive_path)
        
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        logger.info(f"[SCAN] Tier 3 scan complete: {len(files)} files indexed in {latency_ms}ms")
        
        return {
            "status": "ok",
            "count": len(files),
            "files": files,
            "errors": error_count,
            "latency_ms": latency_ms
        }
    
    except Exception as e:
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        return {
            "status": "error",
            "message": str(e),
            "latency_ms": latency_ms
        }


def scan_specific_drive(drive_path: str, max_results: int = 10000) -> Dict[str, Any]:
    """
    Scan a specific drive or folder path (USB, external HDD, network, etc).
    
    Can scan any accessible path, including:
    - USB drives (e.g., "G:\\")
    - External hard drives
    - Network shares
    - Custom folders
    
    Args:
        drive_path: Path to scan (e.g., "G:\\", "D:\\Games", "\\\\network\\share")
        max_results: Maximum files to index
        
    Returns:
        {"status": "ok", "count": int, "files": [...], "latency_ms": int}
    """
    from wyzer.core.logger import get_logger
    logger = get_logger()
    
    start_time = time.perf_counter()
    
    try:
        target_path = Path(drive_path)
        
        if not target_path.exists():
            return {
                "status": "error",
                "message": f"Path not found: {drive_path}"
            }
        
        logger.info(f"[SCAN] Scanning external drive/path: {drive_path}")
        
        files = []
        error_count = 0
        
        def _recursive_drive_scan(path: Path, depth: int = 0, max_depth: int = 10) -> None:
            nonlocal files, error_count
            
            if len(files) >= max_results or depth >= max_depth:
                return
            
            try:
                for entry in path.iterdir():
                    if len(files) >= max_results:
                        return
                    
                    try:
                        if entry.is_dir():
                            dir_name_lower = entry.name.lower()
                            
                            # Skip excluded directories
                            if any(exclude in dir_name_lower for exclude in EXCLUDE_DIRS_TIER3):
                                continue
                            
                            # Recurse
                            _recursive_drive_scan(entry, depth + 1, max_depth)
                        
                        elif entry.is_file():
                            suffix_lower = entry.suffix.lower()
                            
                            # Index prioritized extensions
                            if suffix_lower in PRIORITIZED_EXTENSIONS:
                                try:
                                    stat_info = entry.stat()
                                    files.append({
                                        "name": entry.name,
                                        "path": str(entry),
                                        "type": suffix_lower,
                                        "size_mb": round(stat_info.st_size / (1024 ** 2), 2),
                                        "mtime": stat_info.st_mtime
                                    })
                                except Exception:
                                    error_count += 1
                    
                    except (PermissionError, OSError):
                        error_count += 1
            
            except (PermissionError, OSError):
                error_count += 1
        
        _recursive_drive_scan(target_path)
        
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        logger.info(f"[SCAN] External scan complete: {len(files)} files indexed in {latency_ms}ms")
        
        return {
            "status": "ok",
            "count": len(files),
            "path": drive_path,
            "files": files,
            "errors": error_count,
            "latency_ms": latency_ms
        }
    
    except Exception as e:
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        return {
            "status": "error",
            "message": str(e),
            "latency_ms": latency_ms
        }


def ensure_library_exists() -> None:
    """
    Ensure library.json exists. Create empty structure if not.
    """
    if not LIBRARY_JSON_PATH.exists():
        empty_library = {
            "version": "1.0",
            "timestamp": 0,
            "folders": {},
            "apps": {},
            "aliases": {},
            "tier2_apps": [],
            "tier3_files": [],
            "tier3_drives": [],
            "games": [],
            "games_scan_meta": {},
            "uwp_apps": [],
            "uwp_scan_meta": {},
            "scan_meta": {
                "last_refresh": "",
                "dirs": {}
            }
        }
        save_library(empty_library)


def save_library(library: Dict[str, Any]) -> None:
    """
    Save library data to library.json.
    
    Args:
        library: Library data dict to save
    """
    with open(LIBRARY_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(library, f, indent=2)
