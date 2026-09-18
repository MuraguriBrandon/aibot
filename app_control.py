 """Application launch and termination utilities.

This engine tries to stay safe and portable by validating names, checking common app
locations for the current host OS, and terminating only processes that match the
requested application name.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger("jarvis.app_control")


class AppControl:
    """Controls the launch and shutdown of local desktop applications."""

    def __init__(self) -> None:
        self.logger = logger
        self.windows_common_dirs = [
            Path(r"C:\Program Files"),
            Path(r"C:\Program Files (x86)"),
            Path(os.path.expanduser("~\AppData\Local\Programs")),
            Path(os.path.expanduser("~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs")),
        ]
        self.known_apps = {
            "vscode": ["code.exe", "Code.exe", "code.cmd", "vscode.exe"],
            "chrome": ["chrome.exe", "google chrome.exe"],
            "notepad": ["notepad.exe"],
            "terminal": ["wt.exe", "WindowsTerminal.exe", "powershell.exe", "cmd.exe"],
            "explorer": ["explorer.exe"],
            "file explorer": ["explorer.exe"],
        }

    def _normalize_app_name(self, app_name: str) -> str:
        cleaned = (app_name or "").strip()
        if not cleaned:
            raise ValueError("An application name is required.")
        return cleaned

    def _candidate_paths_for(self, app_name: str) -> List[str]:
        """Build a list of likely executable paths for a given app name."""
        name = self._normalize_app_name(app_name)
        candidates: List[str] = []

        if os.path.exists(name):
            candidates.append(name)

        # direct PATH lookup
        executable = shutil.which(name)
        if executable:
            candidates.append(executable)

        # handle known aliases
        alias_names = self.known_apps.get(name.lower(), [])
        for alias in alias_names:
            for match in self._search_common_locations(alias):
                candidates.append(match)

        # append common direct names
        direct_names = [name]
        if not os.path.splitext(name)[1]:
            for ext in (".exe", ".bat", ".cmd", ".msi"):
                direct_names.append(f"{name}{ext}")
        for item in direct_names:
            for match in self._search_common_locations(item):
                candidates.append(match)
        return list(dict.fromkeys(candidates))

    def _search_common_locations(self, app_name: str) -> List[str]:
        """Search common Windows install locations and PATH for a matching executable."""
        matches: List[str] = []
        searched_dirs: List[Path] = []

        if os.name == "nt":
            for base_dir in self.windows_common_dirs:
                if base_dir.exists():
                    searched_dirs.append(base_dir)
            for path_dir in os.environ.get("PATH", "").split(os.pathsep):
                if path_dir:
                    searched_dirs.append(Path(path_dir))

        for directory in searched_dirs:
            try:
                if not directory.exists():
                    continue
                for child in directory.rglob(app_name):
                    if child.is_file():
                        matches.append(str(child.resolve()))
            except (PermissionError, OSError):
                continue
        return matches

    def launch_application(self, app_name: str) -> Dict[str, str]:
        """Launch an application from PATH or common install locations."""
        name = self._normalize_app_name(app_name)
        self.logger.info("Launching application: %s", name)

        candidates = self._candidate_paths_for(name)
        if not candidates:
            raise FileNotFoundError(f"Could not find an executable for '{name}' on this machine.")

        selected = candidates[0]
        try:
            if os.name == "nt":
                subprocess.Popen([selected], shell=False)
            else:
                subprocess.Popen([selected], shell=False)
            return {"status": "launched", "app_name": name, "path": selected}
        except Exception as exc:  # pragma: no cover - platform-specific
            self.logger.exception("Failed to launch %s at %s", name, selected)
            raise RuntimeError(f"Could not launch '{name}': {exc}") from exc

    def close_application(self, app_name: str) -> Dict[str, object]:
        """Terminate a process matching the requested app name."""
        name = self._normalize_app_name(app_name)
        name_lower = name.lower()
        terminated = []

        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                proc_name = (proc.info.get("name") or "").lower()
                exe_name = (proc.info.get("exe") or "").lower()
                if not proc_name and not exe_name:
                    continue
                if proc_name == name_lower or proc_name.startswith(name_lower) or exe_name.endswith(name_lower):
                    self.logger.info("Stopping process %s (%s)", proc.info.get("pid"), proc_name)
                    proc.terminate()
                    terminated.append({"pid": proc.info.get("pid"), "name": proc_name})
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return {
            "status": "terminated" if terminated else "not_found",
            "app_name": name,
            "terminated_processes": terminated,
        }


def launch_app(app_name: str) -> Dict[str, str]:
    """Convenience wrapper for the tool registry."""
    return AppControl().launch_application(app_name)


def close_app(app_name: str) -> Dict[str, object]:
    """Convenience wrapper for the tool registry."""
    return AppControl().close_application(app_name)
