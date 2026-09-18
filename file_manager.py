"""File discovery and safe file-opening utilities.

The file manager is intentionally conservative: it only searches a set of common user
locations and it validates file paths before opening or executing them. This prevents
arbitrary or unsafe paths from being launched by the model.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger("jarvis.file_manager")


class FileManager:
    """Handles scanning common user directories and launching files safely."""

    def __init__(self, allowed_roots: Optional[List[str]] = None) -> None:
        self.user_home = Path.home().expanduser().resolve()
        default_roots = [
            str(self.user_home / "Documents"),
            str(self.user_home / "Downloads"),
            str(self.user_home / "Desktop"),
            str(self.user_home / "Projects"),
            str(self.user_home / "OneDrive"),
            str(self.user_home / "Workspaces"),
        ]
        self.allowed_roots = [Path(p).expanduser().resolve() for p in (allowed_roots or default_roots) if p]
        self.logger = logger

    def _iter_files(self, root: Path) -> Iterable[Path]:
        if not root.exists():
            return []
        for current, _, files in os.walk(root):
            current_path = Path(current)
            for file_name in files:
                yield current_path / file_name

    def _safe_path(self, path: str) -> Path:
        candidate = Path(path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not any(candidate.is_relative_to(root) for root in self.allowed_roots):
            raise PermissionError(
                "Refusing to access a file outside the allowed user directories: "
                f"{candidate}"
            )
        return candidate

    def scan_common_dirs(self) -> List[Path]:
        """Return a list of accessible directories that are safe to search."""
        found: List[Path] = []
        for root in self.allowed_roots:
            if root.exists():
                found.append(root)
        return found

    def find_file(self, query_name: str, limit: int = 10) -> List[str]:
        """Search for file names using exact or fuzzy matching across common user dirs."""
        if not query_name or not query_name.strip():
            return []

        target = query_name.strip().lower()
        matches: List[tuple[float, str]] = []

        for root in self.scan_common_dirs():
            for file_path in self._iter_files(root):
                name = file_path.name.lower()
                if name == target:
                    score = 1.0
                elif target in name or name in target:
                    score = 0.9
                else:
                    score = SequenceMatcher(None, target, name).ratio()

                if score >= 0.35:
                    matches.append((score, str(file_path)))

        unique = {}
        for score, path in matches:
            unique[path] = max(unique.get(path, 0.0), score)

        ranked = sorted(unique.items(), key=lambda item: item[1], reverse=True)
        return [path for path, _ in ranked[:limit]]

    def open_file(self, file_path: str) -> str:
        """Open a safe file using the default application for the OS."""
        safe_path = self._safe_path(file_path)
        self.logger.info("Opening file: %s", safe_path)

        try:
            if os.name == "nt":
                os.startfile(str(safe_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(safe_path)])
            else:
                subprocess.Popen(["xdg-open", str(safe_path)])
            return f"Opened file: {safe_path}"
        except Exception as exc:  # pragma: no cover - platform-specific
            self.logger.exception("Failed to open file: %s", safe_path)
            raise RuntimeError(f"Could not open file '{safe_path}': {exc}") from exc


def search_and_open_file(file_name: str, file_manager: Optional[FileManager] = None) -> str:
    """Convenience wrapper used by the assistant tool registry."""
    manager = file_manager or FileManager()
    matches = manager.find_file(file_name)
    if not matches:
        raise FileNotFoundError(f"No file matching '{file_name}' was found in known user directories.")
    selected = matches[0]
    return manager.open_file(selected)
