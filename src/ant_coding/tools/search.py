"""
Codebase search operations for finding patterns, definitions, and references.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


# Binary file extensions to skip during search
_BINARY_EXTENSIONS = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".o",
        ".a",
        ".dylib",
        ".dll",
        ".exe",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".whl",
        ".egg",
    }
)

# Definition patterns for common languages
_DEFINITION_PATTERNS: Dict[str, List[str]] = {
    ".py": [
        r"^\s*class\s+{name}\b",
        r"^\s*def\s+{name}\b",
        r"^\s*async\s+def\s+{name}\b",
        r"^{name}\s*=",
    ],
    ".js": [
        r"^\s*class\s+{name}\b",
        r"^\s*function\s+{name}\b",
        r"^\s*const\s+{name}\b",
        r"^\s*let\s+{name}\b",
        r"^\s*var\s+{name}\b",
    ],
    ".ts": [
        r"^\s*class\s+{name}\b",
        r"^\s*function\s+{name}\b",
        r"^\s*const\s+{name}\b",
        r"^\s*interface\s+{name}\b",
        r"^\s*type\s+{name}\b",
    ],
}


class CodebaseSearch:
    """
    Provides code search capabilities within a workspace directory.

    Supports grep-style pattern matching, definition lookup, and reference finding.
    All operations are scoped to the workspace directory.
    """

    def __init__(self, workspace_dir: Union[str, Path]):
        """
        Initialize CodebaseSearch.

        Args:
            workspace_dir: Root directory to search within.
        """
        self.workspace_dir = Path(workspace_dir).resolve()

    def _is_searchable(self, path: Path) -> bool:
        """Check if a file should be searched (skip binary and hidden files)."""
        if path.suffix.lower() in _BINARY_EXTENSIONS:
            return False
        # Skip hidden files/directories
        for part in path.relative_to(self.workspace_dir).parts:
            if part.startswith("."):
                return False
        return True

    def _iter_files(
        self, path: Optional[str] = None, pattern: str = "**/*"
    ) -> List[Path]:
        """
        Iterate over searchable files in the workspace.

        Args:
            path: Subdirectory to search within (relative to workspace). None means workspace root.
            pattern: Glob pattern for filtering files.

        Returns:
            List of Path objects for searchable files.
        """
        search_root = self.workspace_dir
        if path:
            search_root = self.workspace_dir / path
            if not search_root.exists():
                return []

        files = []
        for p in search_root.glob(pattern):
            if p.is_file() and self._is_searchable(p):
                files.append(p)
        return sorted(files)

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        file_pattern: str = "**/*",
    ) -> List[Dict[str, Any]]:
        """
        Search for a regex pattern in files.

        Args:
            pattern: Regex pattern to search for.
            path: Subdirectory to scope search (relative to workspace).
            file_pattern: Glob pattern to filter which files to search.

        Returns:
            List of dicts with 'file', 'line_number', and 'line_content' keys.
        """
        results: List[Dict[str, Any]] = []
        try:
            compiled = re.compile(pattern)
        except re.error:
            # Fall back to literal string search if regex is invalid
            compiled = re.compile(re.escape(pattern))

        for file_path in self._iter_files(path=path, pattern=file_pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled.search(line):
                            results.append(
                                {
                                    "file": str(
                                        file_path.relative_to(self.workspace_dir)
                                    ),
                                    "line_number": line_num,
                                    "line_content": line.rstrip("\n"),
                                }
                            )
            except (UnicodeDecodeError, PermissionError):
                continue

        return results

    def find_definition(
        self,
        name: str,
        path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find where a symbol (class, function, variable) is defined.

        Args:
            name: The symbol name to search for.
            path: Subdirectory to scope search (relative to workspace).

        Returns:
            List of dicts with 'file', 'line_number', 'line_content', and 'type' keys.
        """
        results: List[Dict[str, Any]] = []

        for file_path in self._iter_files(path=path):
            suffix = file_path.suffix.lower()
            patterns = _DEFINITION_PATTERNS.get(suffix)
            if not patterns:
                continue

            compiled_patterns = []
            for p in patterns:
                try:
                    compiled_patterns.append(re.compile(p.format(name=re.escape(name))))
                except re.error:
                    continue

            if not compiled_patterns:
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        for cp in compiled_patterns:
                            if cp.search(line):
                                results.append(
                                    {
                                        "file": str(
                                            file_path.relative_to(self.workspace_dir)
                                        ),
                                        "line_number": line_num,
                                        "line_content": line.rstrip("\n"),
                                    }
                                )
                                break  # One match per line is enough
            except (UnicodeDecodeError, PermissionError):
                continue

        return results

    def find_references(
        self,
        name: str,
        path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find all references (usages) of a symbol across the codebase.

        Searches for all occurrences of the name as a whole word, excluding
        definition lines (which are found by find_definition).

        Args:
            name: The symbol name to search for.
            path: Subdirectory to scope search (relative to workspace).

        Returns:
            List of dicts with 'file', 'line_number', and 'line_content' keys.
        """
        # Use word boundary matching to avoid partial matches
        word_pattern = re.compile(r"\b" + re.escape(name) + r"\b")

        # Get definitions to exclude them from references
        definitions = self.find_definition(name, path=path)
        def_keys = {(d["file"], d["line_number"]) for d in definitions}

        results: List[Dict[str, Any]] = []

        for file_path in self._iter_files(path=path):
            try:
                rel_path = str(file_path.relative_to(self.workspace_dir))
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if word_pattern.search(line):
                            if (rel_path, line_num) not in def_keys:
                                results.append(
                                    {
                                        "file": rel_path,
                                        "line_number": line_num,
                                        "line_content": line.rstrip("\n"),
                                    }
                                )
            except (UnicodeDecodeError, PermissionError):
                continue

        return results
