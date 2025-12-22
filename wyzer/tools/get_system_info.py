"""
Get system information tool.
"""
import os
import platform
import subprocess
from typing import Dict, Any
from wyzer.tools.tool_base import ToolBase


class GetSystemInfoTool(ToolBase):
    """Tool to get system information"""
    
    def __init__(self):
        """Initialize get_system_info tool"""
        super().__init__()
        self._name = "get_system_info"
        self._description = "Get basic system information (OS, CPU, RAM, GPU)"
        self._args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    
    def _get_cpu_name(self) -> str:
        """Get CPU name/model"""
        try:
            if platform.system() == "Windows":
                # Use WMIC to get CPU name on Windows
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip() and l.strip() != "Name"]
                if lines:
                    return lines[0]
            elif platform.system() == "Linux":
                # Read from /proc/cpuinfo on Linux
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line.lower():
                            return line.split(":")[1].strip()
            elif platform.system() == "Darwin":
                # Use sysctl on macOS
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.stdout.strip()
        except Exception:
            pass
        return "Unknown"
    
    def _get_nvidia_gpu_info(self) -> dict:
        """Try to get NVIDIA GPU info using nvidia-smi"""
        nvidia_gpus = {}
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) == 2:
                            name, mem_mb = parts
                            try:
                                mem_gb = round(int(mem_mb) / 1024, 1)
                                nvidia_gpus[name.strip()] = mem_gb
                            except ValueError:
                                nvidia_gpus[name.strip()] = None
        except Exception:
            pass
        return nvidia_gpus
    
    def _get_gpu_info(self) -> list:
        """Get GPU name(s)/model(s)"""
        gpus = []
        nvidia_info = {}
        
        # First try nvidia-smi for accurate NVIDIA GPU info
        if platform.system() in ("Windows", "Linux"):
            nvidia_info = self._get_nvidia_gpu_info()
        
        try:
            if platform.system() == "Windows":
                # Use WMIC to get GPU names on Windows
                result = subprocess.run(
                    ["wmic", "path", "win32_videocontroller", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip() and l.strip() != "Name"]
                
                for gpu_name in lines:
                    # Check if we have nvidia-smi info for this GPU
                    matched = False
                    for nvidia_name, vram_gb in nvidia_info.items():
                        if nvidia_name in gpu_name or gpu_name in nvidia_name:
                            if vram_gb:
                                gpus.append(f"{gpu_name} ({vram_gb} GB VRAM)")
                            else:
                                gpus.append(gpu_name)
                            matched = True
                            break
                    
                    if not matched:
                        # For non-NVIDIA GPUs, try WMIC AdapterRAM (may be inaccurate for >4GB)
                        gpus.append(gpu_name)
                
                # If we have nvidia info but no WMIC results, use nvidia-smi data directly
                if not gpus and nvidia_info:
                    for name, vram_gb in nvidia_info.items():
                        if vram_gb:
                            gpus.append(f"{name} ({vram_gb} GB VRAM)")
                        else:
                            gpus.append(name)
                    
            elif platform.system() == "Linux":
                # Try lspci for GPU info on Linux
                result = subprocess.run(
                    ["lspci"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                for line in result.stdout.split('\n'):
                    if "VGA" in line or "3D" in line or "Display" in line:
                        # Extract GPU name from lspci output
                        parts = line.split(': ', 1)
                        if len(parts) > 1:
                            gpus.append(parts[1])
                            
            elif platform.system() == "Darwin":
                # Use system_profiler on macOS
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                for line in result.stdout.split('\n'):
                    if "Chipset Model:" in line or "Chip:" in line:
                        gpus.append(line.split(':')[1].strip())
        except Exception:
            pass
        
        return gpus if gpus else ["Unknown"]
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            Dict with os, cpu_name, cpu_cores, ram_gb, gpu (if available)
        """
        try:
            result = {
                "os": platform.system(),
                "os_version": platform.version(),
                "architecture": platform.machine()
            }
            
            # Get CPU name/model
            result["cpu_name"] = self._get_cpu_name()
            
            # Try to get CPU cores
            try:
                result["cpu_cores"] = os.cpu_count() or 0
            except:
                result["cpu_cores"] = 0
            
            # Get GPU info
            result["gpu"] = self._get_gpu_info()
            
            # Try psutil for RAM info (optional)
            try:
                import psutil
                mem = psutil.virtual_memory()
                result["ram_gb"] = round(mem.total / (1024**3), 2)
                result["ram_available_gb"] = round(mem.available / (1024**3), 2)
            except ImportError:
                # psutil not available, that's okay
                result["ram_gb"] = None
            
            return result
            
        except Exception as e:
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                }
            }
