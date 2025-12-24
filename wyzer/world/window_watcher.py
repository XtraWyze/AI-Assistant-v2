"""
wyzer.world.window_watcher

Phase 12: Multi-Monitor Window Awareness (Always-On World State)

This module provides a lightweight background watcher that continuously
tracks open windows across all monitors. NO OCR, NO screenshots.

Features:
- Enumerates top-level windows (title/process/hwnd)
- Detects which monitor each window is on (screen index 1..N)
- Tracks focused/active window
- Maintains ring buffer of recent changes (opened/closed/moved/title_changed/focus_changed)

Usage:
    watcher = WindowWatcher(poll_ms=500)
    watcher.start()
    
    # Later, in your tick loop:
    snapshot, events = watcher.tick()
    
    # Or get latest without triggering a new poll:
    snapshot = watcher.get_latest_snapshot()
    
    watcher.stop()
"""

from __future__ import annotations

import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from wyzer.world.window_diff import diff_snapshots, build_hwnd_dict

# Lazy logger to avoid circular imports
_logger = None


def _get_logger():
    """Lazy-load logger."""
    global _logger
    if _logger is None:
        try:
            from wyzer.core.logger import get_logger
            _logger = get_logger()
        except Exception:
            _logger = None
    return _logger


# ============================================================================
# Windows API - try pywin32, fallback to ctypes
# ============================================================================
HAS_PYWIN32 = False
try:
    import win32gui
    import win32process
    import win32api
    import psutil
    HAS_PYWIN32 = True
except ImportError:
    pass

HAS_CTYPES = False
try:
    import ctypes
    from ctypes import wintypes
    HAS_CTYPES = True
except ImportError:
    pass


# ============================================================================
# DPI Awareness - Must be set ONCE at module load to get correct monitor rects
# ============================================================================
_DPI_AWARE_SET = False


def _ensure_dpi_awareness() -> bool:
    """
    Set DPI awareness to get accurate monitor coordinates on high-DPI displays.
    
    Must be called before any monitor enumeration. Safe to call multiple times.
    
    Returns:
        True if DPI awareness was successfully set (or already set)
    """
    global _DPI_AWARE_SET
    if _DPI_AWARE_SET:
        return True
    
    if not HAS_CTYPES:
        return False
    
    try:
        user32 = ctypes.windll.user32
        
        # Try Windows 10+ SetProcessDpiAwarenessContext (best option)
        # -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        try:
            result = user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            if result:
                _DPI_AWARE_SET = True
                logger = _get_logger()
                if logger:
                    logger.debug("[WORLD] DPI awareness: PER_MONITOR_AWARE_V2")
                return True
        except (AttributeError, OSError):
            pass
        
        # Try Windows 8.1+ SetProcessDpiAwareness
        # 2 = PROCESS_PER_MONITOR_DPI_AWARE
        try:
            shcore = ctypes.windll.shcore
            result = shcore.SetProcessDpiAwareness(2)
            if result == 0:  # S_OK
                _DPI_AWARE_SET = True
                logger = _get_logger()
                if logger:
                    logger.debug("[WORLD] DPI awareness: PROCESS_PER_MONITOR_DPI_AWARE")
                return True
        except (AttributeError, OSError):
            pass
        
        # Fall back to Windows Vista+ SetProcessDPIAware
        try:
            result = user32.SetProcessDPIAware()
            if result:
                _DPI_AWARE_SET = True
                logger = _get_logger()
                if logger:
                    logger.debug("[WORLD] DPI awareness: SetProcessDPIAware (legacy)")
                return True
        except (AttributeError, OSError):
            pass
        
        # No DPI awareness API available
        logger = _get_logger()
        if logger:
            logger.warning("[WORLD] Could not set DPI awareness - monitor detection may be inaccurate")
        return False
        
    except Exception as e:
        logger = _get_logger()
        if logger:
            logger.warning(f"[WORLD] DPI awareness error: {e}")
        return False


class WindowWatcher:
    """
    Background watcher for tracking windows across monitors.
    
    Designed to be lightweight and non-blocking. Uses efficient diffs
    to minimize allocations and CPU usage.
    """
    
    def __init__(
        self,
        poll_ms: int = 500,
        ignore_processes: Optional[List[str]] = None,
        ignore_titles: Optional[List[str]] = None,
        max_events: int = 25,
    ):
        """
        Initialize the window watcher.
        
        Args:
            poll_ms: Poll interval in milliseconds (minimum 100)
            ignore_processes: Process names to exclude (case-insensitive)
            ignore_titles: Title substrings to exclude (case-insensitive)
            max_events: Maximum events to keep in ring buffer
        """
        self._poll_ms = max(100, poll_ms)
        self._ignore_processes = set(
            (p.lower().strip() for p in (ignore_processes or []))
        )
        self._ignore_titles = [
            t.lower().strip() for t in (ignore_titles or [])
        ]
        self._max_events = max(1, max_events)
        
        # State
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Snapshots
        self._last_snapshot: List[Dict[str, Any]] = []
        self._last_snapshot_by_hwnd: Dict[int, Dict[str, Any]] = {}
        self._last_focus_hwnd: Optional[int] = None
        self._last_snapshot_ts: float = 0.0
        
        # Grouped by monitor
        self._windows_by_monitor: Dict[int, List[Dict[str, Any]]] = {}
        
        # Focused window record
        self._focused_window: Optional[Dict[str, Any]] = None
        
        # Recent events (ring buffer)
        self._recent_events: deque = deque(maxlen=self._max_events)
        
        # Monitor info cache (refreshed each poll)
        self._monitors: List[Dict[str, Any]] = []
    
    def start(self) -> None:
        """Start the watcher thread (optional - can also call tick() manually)."""
        with self._lock:
            if self._running:
                return
            self._running = True
        
        self._thread = threading.Thread(
            target=self._run_loop,
            name="WindowWatcher",
            daemon=True,
        )
        self._thread.start()
        
        # Do initial monitor enumeration and log results
        self._monitors = self._enumerate_monitors()
        logger = _get_logger()
        if logger:
            monitor_count = len(self._monitors)
            rects = [m.get("rect", []) for m in self._monitors]
            logger.info(f"[WORLD] detected_monitors={monitor_count} rects={rects}")
            
            if monitor_count == 1:
                logger.warning(
                    "[WORLD][WARN] Only 1 monitor detected. "
                    "DPI awareness / permissions may be affecting enumeration."
                )
            
            logger.info(f"[WORLD] WindowWatcher started (poll_ms={self._poll_ms})")
    
    def stop(self) -> None:
        """Stop the watcher thread."""
        with self._lock:
            self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        logger = _get_logger()
        if logger:
            logger.debug("[WORLD] WindowWatcher stopped")
    
    def _run_loop(self) -> None:
        """Background thread loop."""
        poll_sec = self._poll_ms / 1000.0
        
        while True:
            with self._lock:
                if not self._running:
                    break
            
            try:
                self.tick()
            except Exception as e:
                logger = _get_logger()
                if logger:
                    logger.error(f"[WORLD] tick error: {e}")
            
            time.sleep(poll_sec)
    
    def tick(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Perform one poll cycle: enumerate windows, compute diff, update state.
        
        Can be called manually (no thread) or runs in background thread.
        
        Returns:
            (snapshot, events) tuple
        """
        try:
            # Enumerate monitors
            self._monitors = self._enumerate_monitors()
            
            # Enumerate windows
            windows = self._enumerate_windows()
            
            # Apply filters
            windows = self._apply_filters(windows)
            
            # Assign monitor indices to each window
            for w in windows:
                w["monitor"] = self._get_monitor_for_rect(w.get("rect"))
            
            # Get focused window
            focus_hwnd = self._get_foreground_hwnd()
            
            # Build dict for diff
            next_by_hwnd = build_hwnd_dict(windows)
            
            # Compute diff
            with self._lock:
                prev_by_hwnd = self._last_snapshot_by_hwnd
                prev_focus = self._last_focus_hwnd
            
            events = diff_snapshots(prev_by_hwnd, next_by_hwnd, prev_focus, focus_hwnd)
            
            # Group by monitor
            by_monitor: Dict[int, List[Dict[str, Any]]] = {}
            for w in windows:
                mon = w.get("monitor", 1)
                if mon not in by_monitor:
                    by_monitor[mon] = []
                by_monitor[mon].append(w)
            
            # Find focused window record
            focused_rec = next_by_hwnd.get(focus_hwnd) if focus_hwnd else None
            
            # Update state
            with self._lock:
                self._last_snapshot = windows
                self._last_snapshot_by_hwnd = next_by_hwnd
                self._last_focus_hwnd = focus_hwnd
                self._last_snapshot_ts = time.time()
                self._windows_by_monitor = by_monitor
                self._focused_window = focused_rec
                
                # Add events to ring buffer
                for ev in events:
                    self._recent_events.append(ev)
            
            # Log summary if there were events
            if events:
                logger = _get_logger()
                if logger:
                    focus_title = (focused_rec.get("title", "")[:30] if focused_rec else "None")
                    focus_pid = focused_rec.get("pid", "?") if focused_rec else "?"
                    logger.info(
                        f'[WORLD] windows={len(windows)} monitors={len(self._monitors)} '
                        f'focus="{focus_title}" (pid={focus_pid})'
                    )
                    for ev in events:
                        logger.debug(
                            f'[WORLD_EVT] type={ev["type"]} hwnd={ev["hwnd"]} '
                            f'monitor={ev.get("to_monitor")} title="{ev.get("title", "")[:40]}"'
                        )
            
            return windows, events
            
        except Exception as e:
            logger = _get_logger()
            if logger:
                logger.error(f"[WORLD] enumerate error: {e}")
            return [], []
    
    def get_latest_snapshot(self) -> List[Dict[str, Any]]:
        """Get the latest window snapshot without triggering a new poll."""
        with self._lock:
            return list(self._last_snapshot)
    
    def get_windows_by_monitor(self) -> Dict[int, List[Dict[str, Any]]]:
        """Get windows grouped by monitor index (1..N)."""
        with self._lock:
            return {k: list(v) for k, v in self._windows_by_monitor.items()}
    
    def get_focused_window(self) -> Optional[Dict[str, Any]]:
        """Get the currently focused window record."""
        with self._lock:
            return dict(self._focused_window) if self._focused_window else None
    
    def get_recent_events(self) -> List[Dict[str, Any]]:
        """Get recent change events (newest last)."""
        with self._lock:
            return list(self._recent_events)
    
    def get_last_snapshot_ts(self) -> float:
        """Get timestamp of last snapshot."""
        with self._lock:
            return self._last_snapshot_ts
    
    def get_monitor_count(self) -> int:
        """Get number of detected monitors."""
        with self._lock:
            return len(self._monitors) if self._monitors else 1
    
    # ========================================================================
    # Windows API wrappers
    # ========================================================================
    
    def _enumerate_monitors(self) -> List[Dict[str, Any]]:
        """
        Enumerate connected monitors using Windows API.
        
        Uses EnumDisplayMonitors + GetMonitorInfoW to collect monitor rects.
        Monitors are sorted left-to-right, then top-to-bottom, and assigned
        indices 1..N.
        
        DPI awareness is set on first call to ensure correct coordinates.
        
        Returns:
            List of monitor dicts with rect, work area, primary flag, and index.
        """
        # Ensure DPI awareness is set for correct monitor coordinates
        _ensure_dpi_awareness()
        
        monitors: List[Dict[str, Any]] = []
        
        if HAS_PYWIN32:
            try:
                # pywin32's EnumDisplayMonitors returns a list directly (not callback-based)
                # Each item is a tuple: (hMonitor, hdcMonitor, pyRect)
                for monitor_tuple in win32api.EnumDisplayMonitors(None, None):
                    try:
                        hMonitor = monitor_tuple[0]
                        info = win32api.GetMonitorInfo(hMonitor)
                        monitors.append({
                            "handle": hMonitor,
                            "rect": list(info.get("Monitor", (0, 0, 1920, 1080))),
                            "work": list(info.get("Work", (0, 0, 1920, 1080))),
                            "primary": info.get("Flags", 0) & 1,
                        })
                    except Exception as e:
                        logger = _get_logger()
                        if logger:
                            logger.debug(f"[WORLD] GetMonitorInfo error: {e}")
            except Exception as e:
                logger = _get_logger()
                if logger:
                    logger.warning(f"[WORLD] EnumDisplayMonitors (pywin32) failed: {e}")
        
        # Fall back to ctypes if pywin32 didn't work or isn't available
        if not monitors and HAS_CTYPES:
            try:
                user32 = ctypes.windll.user32
                
                MONITORENUMPROC = ctypes.WINFUNCTYPE(
                    ctypes.c_bool,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(wintypes.RECT),
                    ctypes.c_void_p,
                )
                
                class MONITORINFO(ctypes.Structure):
                    _fields_ = [
                        ("cbSize", wintypes.DWORD),
                        ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD),
                    ]
                
                def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
                    try:
                        mi = MONITORINFO()
                        mi.cbSize = ctypes.sizeof(MONITORINFO)
                        if user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi)):
                            monitors.append({
                                "handle": hMonitor,
                                "rect": [
                                    mi.rcMonitor.left,
                                    mi.rcMonitor.top,
                                    mi.rcMonitor.right,
                                    mi.rcMonitor.bottom,
                                ],
                                "work": [
                                    mi.rcWork.left,
                                    mi.rcWork.top,
                                    mi.rcWork.right,
                                    mi.rcWork.bottom,
                                ],
                                "primary": mi.dwFlags & 1,
                            })
                    except Exception:
                        pass
                    return True
                
                user32.EnumDisplayMonitors(
                    None, None, MONITORENUMPROC(callback), None
                )
            except Exception:
                pass
        
        if not monitors:
            # Fallback: assume one monitor
            monitors = [{
                "handle": 0,
                "rect": [0, 0, 1920, 1080],
                "work": [0, 0, 1920, 1040],
                "primary": True,
            }]
        
        # Sort monitors: primary first, then left-to-right, then top-to-bottom
        # This ensures "monitor 1" is always the primary display
        monitors.sort(key=lambda m: (0 if m.get("primary") else 1, m["rect"][0], m["rect"][1]))
        
        # Assign 1-based indices
        for i, m in enumerate(monitors):
            m["index"] = i + 1
        
        return monitors
    
    def _get_monitor_for_rect(
        self, rect: Optional[List[int]]
    ) -> int:
        """
        Determine which monitor a window rect is on.
        
        Uses center point of window to determine primary monitor.
        Returns 1-based monitor index.
        """
        if not rect or len(rect) != 4:
            return 1
        
        # Get center point
        cx = (rect[0] + rect[2]) // 2
        cy = (rect[1] + rect[3]) // 2
        
        for mon in self._monitors:
            mr = mon.get("rect", [0, 0, 1920, 1080])
            if mr[0] <= cx < mr[2] and mr[1] <= cy < mr[3]:
                return mon.get("index", 1)
        
        # Fallback: first monitor
        return 1
    
    def _get_foreground_hwnd(self) -> Optional[int]:
        """Get hwnd of currently focused window."""
        if HAS_PYWIN32:
            try:
                return win32gui.GetForegroundWindow()
            except Exception:
                return None
        elif HAS_CTYPES:
            try:
                user32 = ctypes.windll.user32
                return user32.GetForegroundWindow()
            except Exception:
                return None
        return None
    
    def _enumerate_windows(self) -> List[Dict[str, Any]]:
        """
        Enumerate all visible top-level windows.
        
        Returns list of window record dicts.
        """
        windows: List[Dict[str, Any]] = []
        
        if HAS_PYWIN32:
            def callback(hwnd, extra):
                try:
                    if not win32gui.IsWindowVisible(hwnd):
                        return True
                    
                    title = win32gui.GetWindowText(hwnd) or ""
                    
                    # Get rect
                    try:
                        rect = list(win32gui.GetWindowRect(hwnd))
                    except Exception:
                        rect = [0, 0, 0, 0]
                    
                    # Get process info
                    pid = 0
                    process_name = ""
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        if pid:
                            proc = psutil.Process(pid)
                            process_name = proc.name()
                    except Exception:
                        pass
                    
                    # Only include if has title or process
                    if title or process_name:
                        windows.append({
                            "hwnd": hwnd,
                            "title": title,
                            "process": process_name,
                            "pid": pid,
                            "rect": rect,
                            "is_visible": True,
                            "monitor": 1,  # Will be set later
                        })
                except Exception:
                    pass
                return True
            
            try:
                win32gui.EnumWindows(callback, None)
            except Exception:
                pass
                
        elif HAS_CTYPES:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            
            EnumWindowsProc = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p
            )
            
            def callback(hwnd, lParam):
                try:
                    if not user32.IsWindowVisible(hwnd):
                        return True
                    
                    # Get title
                    length = user32.GetWindowTextLengthW(hwnd)
                    title = ""
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        title = buf.value or ""
                    
                    # Get rect
                    class RECT(ctypes.Structure):
                        _fields_ = [
                            ("left", ctypes.c_long),
                            ("top", ctypes.c_long),
                            ("right", ctypes.c_long),
                            ("bottom", ctypes.c_long),
                        ]
                    
                    rect_struct = RECT()
                    rect = [0, 0, 0, 0]
                    if user32.GetWindowRect(hwnd, ctypes.byref(rect_struct)):
                        rect = [
                            rect_struct.left,
                            rect_struct.top,
                            rect_struct.right,
                            rect_struct.bottom,
                        ]
                    
                    # Get PID
                    pid = wintypes.DWORD()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    pid_val = pid.value
                    
                    # Get process name
                    process_name = ""
                    try:
                        PROCESS_QUERY_INFORMATION = 0x0400
                        PROCESS_VM_READ = 0x0010
                        hProcess = kernel32.OpenProcess(
                            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                            False, pid_val
                        )
                        if hProcess:
                            try:
                                buf = ctypes.create_unicode_buffer(260)
                                size = wintypes.DWORD(260)
                                # Try QueryFullProcessImageNameW
                                try:
                                    kernel32.QueryFullProcessImageNameW(
                                        hProcess, 0, buf, ctypes.byref(size)
                                    )
                                    import os
                                    process_name = os.path.basename(buf.value or "")
                                except Exception:
                                    pass
                            finally:
                                kernel32.CloseHandle(hProcess)
                    except Exception:
                        pass
                    
                    if title or process_name:
                        windows.append({
                            "hwnd": hwnd,
                            "title": title,
                            "process": process_name,
                            "pid": pid_val,
                            "rect": rect,
                            "is_visible": True,
                            "monitor": 1,
                        })
                except Exception:
                    pass
                return True
            
            try:
                user32.EnumWindows(EnumWindowsProc(callback), None)
            except Exception:
                pass
        
        return windows
    
    def _apply_filters(
        self, windows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply ignore filters to window list."""
        if not self._ignore_processes and not self._ignore_titles:
            return windows
        
        filtered = []
        for w in windows:
            proc = (w.get("process") or "").lower()
            title = (w.get("title") or "").lower()
            
            # Check process ignore
            if proc in self._ignore_processes:
                continue
            
            # Check title ignore (substring match)
            skip = False
            for ignore_substr in self._ignore_titles:
                if ignore_substr in title:
                    skip = True
                    break
            if skip:
                continue
            
            filtered.append(w)
        
        return filtered


# ============================================================================
# Module-level singleton (for easy access from world_state)
# ============================================================================
_watcher_instance: Optional[WindowWatcher] = None
_watcher_lock = threading.Lock()


def get_window_watcher() -> Optional[WindowWatcher]:
    """Get the singleton WindowWatcher instance (if enabled)."""
    global _watcher_instance
    with _watcher_lock:
        return _watcher_instance


def init_window_watcher() -> Optional[WindowWatcher]:
    """
    Initialize and start the window watcher (if enabled in config).
    
    Returns the WindowWatcher instance or None if disabled.
    """
    global _watcher_instance
    
    from wyzer.core.config import Config
    
    if not getattr(Config, "WINDOW_WATCHER_ENABLED", True):
        return None
    
    with _watcher_lock:
        if _watcher_instance is not None:
            return _watcher_instance
        
        _watcher_instance = WindowWatcher(
            poll_ms=getattr(Config, "WINDOW_WATCHER_POLL_MS", 500),
            ignore_processes=getattr(Config, "WINDOW_WATCHER_IGNORE_PROCESSES", []),
            ignore_titles=getattr(Config, "WINDOW_WATCHER_IGNORE_TITLES", []),
            max_events=getattr(Config, "WINDOW_WATCHER_MAX_EVENTS", 25),
        )
        
        return _watcher_instance


def stop_window_watcher() -> None:
    """Stop the window watcher if running."""
    global _watcher_instance
    with _watcher_lock:
        if _watcher_instance is not None:
            _watcher_instance.stop()
            _watcher_instance = None
