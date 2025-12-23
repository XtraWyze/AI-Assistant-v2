"""
Open target tool - opens folders, files, apps, or URLs based on user query.
"""
import os
import time
import subprocess
from typing import Dict, Any
from pathlib import Path
from wyzer.tools.tool_base import ToolBase
from wyzer.local_library import resolve_target


class OpenTargetTool(ToolBase):
    """Tool to open folders, files, apps, or URLs"""
    
    def __init__(self):
        """Initialize open_target tool"""
        super().__init__()
        self._name = "open_target"
        self._description = "Open a folder, file, app, or URL based on natural language query (e.g., 'downloads', 'chrome', 'my documents')"
        self._args_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Natural language query for what to open (e.g., 'downloads', 'notepad', 'my pictures')"
                },
                "open_mode": {
                    "type": "string",
                    "enum": ["default", "folder", "file", "app"],
                    "description": "Optional: force specific open mode"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Open a target based on query.
        
        Args:
            query: Natural language query
            open_mode: Optional mode override
            _resolved_uwp_path: (Internal) Phase 10.1 - Pre-resolved UWP app ID for stable replay
            _resolved_launch: (Internal) Phase 10.1 - Pre-resolved launch info for stable replay
            _resolved_path: (Internal) Phase 10.1 - Pre-resolved path for stable replay
            
        Returns:
            Dict with status and resolved info, or error
        """
        start_time = time.perf_counter()
        
        query = kwargs.get("query", "").strip()
        open_mode = kwargs.get("open_mode", "default")
        
        # Phase 10.1: Check for pre-resolved hints (for deterministic replay)
        resolved_uwp_path = kwargs.get("_resolved_uwp_path")
        resolved_launch = kwargs.get("_resolved_launch")
        resolved_path = kwargs.get("_resolved_path")
        
        # If we have pre-resolved info, use it directly (skip resolution)
        if resolved_uwp_path:
            return self._run_uwp_direct(resolved_uwp_path, query, start_time)
        if resolved_launch:
            return self._run_game_direct(resolved_launch, query, start_time)
        if resolved_path:
            return self._run_path_direct(resolved_path, query, open_mode, start_time)
        
        if not query:
            return {
                "error": {
                    "type": "invalid_query",
                    "message": "Query cannot be empty"
                }
            }
        
        try:
            # Resolve target
            resolved = resolve_target(query)
            
            if resolved.get("type") == "unknown":
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "error": {
                        "type": "not_found",
                        "message": f"Could not find a match for '{query}'"
                    },
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            # Check confidence threshold
            if resolved.get("confidence", 0) < 0.3:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "error": {
                        "type": "low_confidence",
                        "message": f"Low confidence match for '{query}'"
                    },
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            # Open based on type
            target_type = resolved.get("type")
            target_path = resolved.get("path", "")
            
            if target_type == "url":
                # Delegate to open_website (use webbrowser)
                import webbrowser
                url = resolved.get("url", query)
                if not url.startswith(("http://", "https://")):
                    url = f"https://{url}"
                webbrowser.open(url)
                
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "status": "opened",
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            elif target_type == "game":
                # Launch game based on launch type
                launch_info = resolved.get("launch", {})
                launch_type = launch_info.get("type", "")
                launch_target = launch_info.get("target", "")
                
                try:
                    if launch_type == "steam_uri":
                        # Open Steam URI
                        import webbrowser
                        webbrowser.open(launch_target)
                    
                    elif launch_type == "epic_uri":
                        # Open Epic Games launcher URI
                        import webbrowser
                        webbrowser.open(launch_target)
                    
                    elif launch_type == "exe":
                        # Launch executable directly
                        subprocess.Popen([launch_target], shell=False)
                    
                    elif launch_type == "shortcut":
                        # Open .lnk shortcut
                        os.startfile(launch_target)
                    
                    elif launch_type == "uwp":
                        # Launch UWP app via explorer
                        subprocess.run(["explorer", launch_target], check=False)
                    
                    else:
                        end_time = time.perf_counter()
                        latency_ms = int((end_time - start_time) * 1000)
                        
                        return {
                            "error": {
                                "type": "unsupported_launch_type",
                                "message": f"Unsupported game launch type: {launch_type}"
                            },
                            "resolved": resolved,
                            "latency_ms": latency_ms
                        }
                    
                    end_time = time.perf_counter()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    return {
                        "status": "opened",
                        "resolved": resolved,
                        "game_name": resolved.get("game_name", ""),
                        "latency_ms": latency_ms
                    }
                    
                except Exception as e:
                    end_time = time.perf_counter()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    return {
                        "error": {
                            "type": "launch_error",
                            "message": f"Failed to launch game: {str(e)}"
                        },
                        "resolved": resolved,
                        "latency_ms": latency_ms
                    }
            
            elif target_type == "uwp":
                # Launch UWP app via explorer shell:AppsFolder
                app_id = resolved.get("path", "")
                
                try:
                    subprocess.Popen(
                        ["explorer.exe", f"shell:AppsFolder\\{app_id}"],
                        shell=False
                    )
                    
                    end_time = time.perf_counter()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    return {
                        "status": "opened",
                        "resolved": resolved,
                        "app_name": resolved.get("app_name", ""),
                        "latency_ms": latency_ms
                    }
                    
                except Exception as e:
                    end_time = time.perf_counter()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    return {
                        "error": {
                            "type": "launch_error",
                            "message": f"Failed to launch UWP app: {str(e)}"
                        },
                        "resolved": resolved,
                        "latency_ms": latency_ms
                    }
            
            elif target_type == "folder":
                # Open folder in Explorer
                self._open_folder(target_path)
                
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "status": "opened",
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            elif target_type == "file":
                # Open file (select in Explorer or open with default app)
                if open_mode == "folder":
                    # Open parent folder with file selected
                    self._open_file_location(target_path)
                else:
                    # Open with default app
                    os.startfile(target_path)
                
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "status": "opened",
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            elif target_type == "app":
                # Launch app
                self._launch_app(target_path)
                
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "status": "opened",
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
            
            else:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "error": {
                        "type": "unsupported_type",
                        "message": f"Unsupported target type: {target_type}"
                    },
                    "resolved": resolved,
                    "latency_ms": latency_ms
                }
                
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                },
                "latency_ms": latency_ms
            }
    
    # =========================================================================
    # Phase 10.1: Direct launch methods for deterministic replay
    # =========================================================================
    # These methods bypass resolution and launch directly using pre-resolved info.
    # This ensures stable replays (e.g., "do that again" opens the exact same app).
    
    def _run_uwp_direct(self, uwp_path: str, query: str, start_time: float) -> Dict[str, Any]:
        """
        Launch a UWP app directly using pre-resolved app ID.
        
        Phase 10.1: Used for deterministic replay - skips resolution.
        """
        try:
            subprocess.Popen(
                ["explorer.exe", f"shell:AppsFolder\\{uwp_path}"],
                shell=False
            )
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "opened",
                "resolved": {"type": "uwp", "path": uwp_path},
                "replay": True,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "launch_error",
                    "message": f"Failed to launch UWP app: {str(e)}"
                },
                "latency_ms": latency_ms
            }
    
    def _run_game_direct(self, launch_info: dict, query: str, start_time: float) -> Dict[str, Any]:
        """
        Launch a game directly using pre-resolved launch info.
        
        Phase 10.1: Used for deterministic replay - skips resolution.
        """
        launch_type = launch_info.get("type", "")
        launch_target = launch_info.get("target", "")
        
        try:
            if launch_type == "steam_uri":
                import webbrowser
                webbrowser.open(launch_target)
            
            elif launch_type == "epic_uri":
                import webbrowser
                webbrowser.open(launch_target)
            
            elif launch_type == "exe":
                subprocess.Popen([launch_target], shell=False)
            
            elif launch_type == "shortcut":
                os.startfile(launch_target)
            
            elif launch_type == "uwp":
                subprocess.run(["explorer", launch_target], check=False)
            
            else:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)
                
                return {
                    "error": {
                        "type": "unsupported_launch_type",
                        "message": f"Unsupported game launch type: {launch_type}"
                    },
                    "latency_ms": latency_ms
                }
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "opened",
                "resolved": {"type": "game", "launch": launch_info},
                "replay": True,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "launch_error",
                    "message": f"Failed to launch game: {str(e)}"
                },
                "latency_ms": latency_ms
            }
    
    def _run_path_direct(self, path: str, query: str, open_mode: str, start_time: float) -> Dict[str, Any]:
        """
        Open a file/folder/app directly using pre-resolved path.
        
        Phase 10.1: Used for deterministic replay - skips resolution.
        """
        try:
            # Determine type from path
            if os.path.isdir(path):
                self._open_folder(path)
                target_type = "folder"
            elif path.lower().endswith((".exe", ".lnk")):
                self._launch_app(path)
                target_type = "app"
            else:
                if open_mode == "folder":
                    self._open_file_location(path)
                else:
                    os.startfile(path)
                target_type = "file"
            
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "opened",
                "resolved": {"type": target_type, "path": path},
                "replay": True,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "error": {
                    "type": "launch_error",
                    "message": f"Failed to open: {str(e)}"
                },
                "latency_ms": latency_ms
            }
    
    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def _open_folder(self, path: str) -> None:
        """Open folder in Windows Explorer"""
        os.startfile(path)
    
    def _open_file_location(self, path: str) -> None:
        """Open file location in Windows Explorer with file selected"""
        subprocess.run(["explorer", "/select,", path], check=False)
    
    def _launch_app(self, path: str) -> None:
        """Launch an application"""
        # Check if it's a .lnk shortcut
        if path.lower().endswith(".lnk"):
            os.startfile(path)
        else:
            # Launch executable
            subprocess.Popen([path], shell=False)
