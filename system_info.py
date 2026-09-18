"""System telemetry and synchronization helpers for the JARVIS assistant.

This module intentionally focuses on structured, JSON-serializable snapshots that can
be passed directly to an LLM as system context. The goal is to provide a compact but
useful summary of machine status without exposing unsafe internals.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger("jarvis.system")


class SystemSyncManager:
    """Collects hardware and software telemetry for the local machine."""

    def __init__(self) -> None:
        self.logger = logger

    def get_cpu_stats(self) -> Dict[str, Any]:
        """Return CPU metrics including load, per-core usage, and frequency."""
        cpu_count = psutil.cpu_count(logical=True) or 0
        per_core = psutil.cpu_percent(percpu=True, interval=None)
        freq = psutil.cpu_freq(percpu=True)

        return {
            "logical_cores": cpu_count,
            "utilization_percent": round(psutil.cpu_percent(interval=None), 2),
            "per_core_percent": [round(v, 2) for v in per_core],
            "frequency_mhz": [
                {
                    "current": round(item.current, 2) if item and item.current else None,
                    "min": round(item.min, 2) if item and item.min else None,
                    "max": round(item.max, 2) if item and item.max else None,
                }
                for item in freq or []
            ],
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return memory usage summary."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "used_gb": round(mem.used / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "percent_used": round(mem.percent, 2),
            "swap_total_gb": round(swap.total / (1024 ** 3), 2),
            "swap_used_gb": round(swap.used / (1024 ** 3), 2),
            "swap_percent": round(swap.percent, 2),
        }

    def get_disk_stats(self) -> List[Dict[str, Any]]:
        """Return per-partition disk usage information."""
        partitions: List[Dict[str, Any]] = []
        for partition in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                partitions.append(
                    {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total_gb": round(usage.total / (1024 ** 3), 2),
                        "used_gb": round(usage.used / (1024 ** 3), 2),
                        "free_gb": round(usage.free / (1024 ** 3), 2),
                        "percent_used": round((usage.used / usage.total) * 100, 2),
                    }
                )
            except (PermissionError, OSError):
                self.logger.warning("Could not read disk usage for %s", partition.mountpoint)
        return partitions

    def get_battery_status(self) -> Optional[Dict[str, Any]]:
        """Return battery information if a battery is present."""
        if not hasattr(psutil, "sensors_battery"):
            return None
        battery = psutil.sensors_battery()
        if battery is None:
            return None
        return {
            "percent": round(battery.percent, 2),
            "plugged_in": bool(battery.power_plugged),
            "seconds_remaining": int(battery.secsleft),
            "time_remaining": self._format_seconds(battery.secsleft),
        }

    def _format_seconds(self, total_seconds: float) -> str:
        """Convert a duration in seconds to a human-readable string."""
        if total_seconds <= 0:
            return "0:00:00"
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_process_summary(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the top CPU- and memory-heavy running processes."""
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                if info.get("name"):
                    processes.append(
                        {
                            "pid": info.get("pid"),
                            "name": info.get("name"),
                            "cpu_percent": round(info.get("cpu_percent") or 0.0, 2),
                            "memory_percent": round(info.get("memory_percent") or 0.0, 2),
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        processes.sort(key=lambda item: (item["cpu_percent"], item["memory_percent"]), reverse=True)
        return processes[:limit]

    def get_system_context(self) -> Dict[str, Any]:
        """Return one structured context object suitable for LLM prompts."""
        return {
            "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": sys.version.split()[0],
            },
            "user": {
                "home": os.path.expanduser("~"),
                "username": os.getenv("USERNAME") or os.getenv("USER") or "unknown",
            },
            "cpu": self.get_cpu_stats(),
            "memory": self.get_memory_stats(),
            "disks": self.get_disk_stats(),
            "battery": self.get_battery_status(),
            "top_processes": self.get_process_summary(),
        }


def get_system_telemetry() -> Dict[str, Any]:
    """Convenience function used by the chat tool registry."""
    return SystemSyncManager().get_system_context()
