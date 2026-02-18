"""
Isolated workspace for task execution and git-based patch generation.
"""

import os
import shutil
import subprocess
import tempfile
import logging
import asyncio
from pathlib import Path
from typing import Tuple, Optional, Union
from git import Repo
from ant_coding.tasks.types import Task

logger = logging.getLogger(__name__)

class TaskWorkspace:
    """
    Manages an isolated directory for a task.
    Handles git cloning, patch generation, and command execution.
    """
    
    def __init__(self, task: Task, base_dir: Optional[Union[str, Path]] = None):
        self.task = task
        self.base_dir = Path(base_dir) if base_dir else Path("/tmp/ant-coding")
        self.workspace_dir: Optional[Path] = None
        self.repo: Optional[Repo] = None

    async def setup(self):
        """
        Set up the workspace: create directory, clone repo or init git.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique directory name
        prefix = f"{self.task.id}_"
        self.workspace_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=self.base_dir))
        
        repo_url = self.task.metadata.get("repo_url")
        base_commit = self.task.metadata.get("base_commit")
        
        if repo_url:
            logger.info(f"Cloning {repo_url} into {self.workspace_dir}")
            self.repo = Repo.clone_from(repo_url, self.workspace_dir)
            if base_commit:
                logger.info(f"Checking out {base_commit}")
                self.repo.git.checkout(base_commit)
        else:
            # For custom tasks without a repo, initialize a fresh git repo
            logger.info(f"Initializing fresh git repo in {self.workspace_dir}")
            self.repo = Repo.init(self.workspace_dir)
            
        # Ensure workspace is writable
        os.chmod(self.workspace_dir, 0o755)

    async def get_patch(self) -> str:
        """
        Return the current git diff of the workspace.
        """
        if not self.repo:
             return ""
        
        # Track all changes including untracked files
        self.repo.git.add(A=True)
        return self.repo.git.diff(staged=True)

    async def run_command(self, command: str, timeout: Optional[int] = None) -> Tuple[bool, str]:
        """
        Execute a shell command inside the workspace.
        
        Args:
            command: The command to run.
            timeout: Timeout in seconds.
            
        Returns:
            A tuple of (success, output).
        """
        if not self.workspace_dir:
            return False, "Workspace not set up"
            
        timeout = timeout or self.task.timeout_seconds
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.workspace_dir
            )
            
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
                success = process.returncode == 0
                return success, stdout.decode()
            except asyncio.TimeoutError:
                process.kill()
                return False, f"Command timed out after {timeout}s"
                
        except Exception as e:
            return False, str(e)

    async def teardown(self):
        """
        Delete the workspace directory.
        """
        if self.workspace_dir and self.workspace_dir.exists():
            logger.info(f"Removing workspace {self.workspace_dir}")
            shutil.rmtree(self.workspace_dir)
            self.workspace_dir = None
            self.repo = None
