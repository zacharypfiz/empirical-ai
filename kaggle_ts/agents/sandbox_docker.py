from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

from .base import RunResult


THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class DockerSandbox:
    """Execute code inside a Docker container with resource caps and no network.

    Requirements: Docker must be installed and the image available locally.
    """

    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        cpus: float = 1.0,
        memory: str = "2g",
        workdir_in_container: str = "/work",
        mounts: Optional[Dict[str, str]] = None,
        python_exe: str = "python",
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.workdir_in_container = workdir_in_container
        self.mounts = mounts or {}
        self.python_exe = python_exe
        self.env = env or {}

    async def run(self, code: str, *, timeout_s: int) -> RunResult:
        host_dir = Path(tempfile.mkdtemp(prefix="ktsd-"))
        try:
            (host_dir / "main.py").write_text(code, encoding="utf-8")
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--cpus",
                str(self.cpus),
                "--memory",
                str(self.memory),
                "-w",
                self.workdir_in_container,
                "-v",
                f"{host_dir}:{self.workdir_in_container}:rw",
            ]
            # additional mounts
            for h, c in self.mounts.items():
                # allow mode in container path like /data:ro
                if ":" in c:
                    container_path, mode = c.rsplit(":", 1)
                    cmd.extend(["-v", f"{h}:{container_path}:{mode}"])
                else:
                    cmd.extend(["-v", f"{h}:{c}:ro"])
            # env caps
            for k, v in {**THREAD_CAPS, **self.env}.items():
                cmd.extend(["-e", f"{k}={v}"])
            cmd.append(self.image)
            cmd.extend([self.python_exe, "main.py"])

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                async with asyncio.timeout(timeout_s):
                    out, err = await proc.communicate()
            except TimeoutError:
                try:
                    proc.kill()
                finally:
                    return RunResult("", "timeout", {}, True, "timeout")

            stdout = out.decode(errors="replace") if out else ""
            stderr = err.decode(errors="replace") if err else ""
            artifacts: Dict[str, str] = {}
            sub = host_dir / "submission.csv"
            if sub.exists():
                artifacts["submission.csv"] = str(sub)
            err_str = None if proc.returncode == 0 else f"exit={proc.returncode}"
            return RunResult(stdout, stderr, artifacts, False, err_str)
        finally:
            shutil.rmtree(host_dir, ignore_errors=True)
