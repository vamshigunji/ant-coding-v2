"""
Sandboxed code execution utility.
"""

import asyncio
import subprocess
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class CodeExecutor:
    """
    Executes code or shell commands in a (basic) sandboxed environment.
    Supports timeouts and output capture.
    """
    
    def __init__(self, timeout: int = 30):
        self.default_timeout = timeout

    async def execute(
        self, 
        code: str, 
        language: str = "python", 
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a block of code.
        
        Args:
            code: The source code to run.
            language: The language (currently only 'python' is fully supported).
            timeout: Optional override for the default timeout.
            
        Returns:
            A dictionary with success, stdout, stderr, and exit_code.
        """
        if language.lower() != "python":
            return {
                "success": False, 
                "stdout": "", 
                "stderr": f"Unsupported language: {language}", 
                "exit_code": -1
            }

        timeout = timeout or self.default_timeout
        
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            # We use 'python3' for execution
            result = await self.run_command(f"python3 {tmp_path}", timeout=timeout)
            return result
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def run_command(
        self, 
        command: str, 
        cwd: Optional[Union[str, Path]] = None, 
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a shell command.
        
        Args:
            command: The command to execute.
            cwd: Optional working directory.
            timeout: Optional override for the default timeout.
            
        Returns:
            A dictionary with success, stdout, stderr, and exit_code.
        """
        timeout = timeout or self.default_timeout
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                exit_code = process.returncode
                success = exit_code == 0
                
                return {
                    "success": success,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "exit_code": exit_code
                }
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout}s",
                    "exit_code": -1
                }
                
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1
            }
