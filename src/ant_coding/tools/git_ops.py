"""
Git operations for managing code changes within a workspace.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from git import Repo

class GitOperations:
    """
    Handles git operations scoped to a workspace.
    """
    
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir).resolve()
        try:
            self.repo = Repo(self.workspace_dir)
        except:
            # If not a repo, initialize one
            self.repo = Repo.init(self.workspace_dir)

    def get_diff(self, staged: bool = True) -> str:
        """Return the unified diff of the workspace."""
        if staged:
            return self.repo.git.diff(staged=True)
        return self.repo.git.diff()

    def get_status(self) -> List[Dict[str, str]]:
        """Return a list of changed files and their status."""
        status_list = []
        
        # Helper to get status from porcelain output
        # status codes: M = modified, A = added, D = deleted, R = renamed, C = copied, U = updated but unmerged
        # XY where X is index and Y is working tree
        try:
            status_output = self.repo.git.status(porcelain=True)
            if not status_output:
                return []
                
            for line in status_output.splitlines():
                if not line: continue
                index_status = line[0]
                worktree_status = line[1]
                file_path = line[3:]
                
                if index_status != " " and index_status != "?":
                    status_list.append({"file": file_path, "status": "staged"})
                elif index_status == "?" or worktree_status == "?":
                    status_list.append({"file": file_path, "status": "untracked"})
                else:
                    status_list.append({"file": file_path, "status": "modified"})
        except:
            pass
            
        return status_list

    def commit(self, message: str, author_name: str = "Agent", author_email: str = "agent@ant-coding.ai") -> str:
        """Commit staged changes and return the commit hash."""
        # Ensure name/email are set locally if not global
        with self.repo.config_writer() as cw:
            cw.set_value("user", "name", author_name)
            cw.set_value("user", "email", author_email)
            
        commit_obj = self.repo.index.commit(message)
        return commit_obj.hexsha

    def create_branch(self, name: str):
        """Create and switch to a new branch."""
        new_branch = self.repo.create_head(name)
        self.repo.head.reference = new_branch
        self.repo.head.reset(index=True, working_tree=True)

    def checkout(self, name: str):
        """Checkout a branch or commit."""
        self.repo.git.checkout(name)

    def add(self, path: str = "."):
        """Stage changes for commit."""
        self.repo.git.add(path)
