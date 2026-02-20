"""
Safe file operations scoped to a workspace directory.
"""

import shutil
from pathlib import Path
from typing import List, Dict, Any, Union

class SecurityError(Exception):
    """Exception raised for security violations like path traversal."""
    pass

class FileOperations:
    """
    Provides file system operations scoped to a specific workspace.
    Prevents path traversal and allows searching/listing files.
    """
    
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir).resolve()
        if not self.workspace_dir.exists():
             self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Resolve path and check for path traversal.
        """
        resolved = (self.workspace_dir / path).resolve()
        if not str(resolved).startswith(str(self.workspace_dir)):
            raise SecurityError(f"Path traversal detected: {path}")
        return resolved

    def read_file(self, path: str) -> str:
        """Read a file from the workspace."""
        full_path = self._resolve_path(path)
        if not full_path.is_file():
             raise FileNotFoundError(f"File not found: {path}")
        return full_path.read_text()

    def write_file(self, path: str, content: str):
        """Write a file to the workspace."""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    def edit_file(self, path: str, old_str: str, new_str: str) -> bool:
        """
        Simple string replacement in a file.
        Returns True if a replacement was made.
        """
        full_path = self._resolve_path(path)
        if not full_path.is_file():
             return False
             
        content = full_path.read_text()
        if old_str not in content:
            return False
            
        new_content = content.replace(old_str, new_str)
        full_path.write_text(new_content)
        return True

    def list_files(self, pattern: str = "**/*") -> List[str]:
        """List files matching a glob pattern relative to workspace."""
        files = []
        for p in self.workspace_dir.glob(pattern):
            if p.is_file():
                # Get path relative to workspace
                rel_path = p.relative_to(self.workspace_dir)
                files.append(str(rel_path))
        return sorted(files)

    def search_files(self, query: str, pattern: str = "**/*") -> List[Dict[str, Any]]:
        """Search for a string in files matching a pattern."""
        results = []
        for file_path in self.list_files(pattern):
            full_path = self.workspace_dir / file_path
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if query in line:
                            results.append({
                                "file": file_path,
                                "line_number": i,
                                "line_content": line.strip()
                            })
            except (UnicodeDecodeError, PermissionError):
                continue
        return results

    def delete_file(self, path: str):
        """Delete a file from the workspace."""
        full_path = self._resolve_path(path)
        if full_path.is_file():
            full_path.unlink()
        elif full_path.is_dir():
            shutil.rmtree(full_path)
