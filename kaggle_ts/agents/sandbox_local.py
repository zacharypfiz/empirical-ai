from __future__ import annotations

import asyncio
import sys
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

from .base import RunResult


THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class LocalSandbox:
    """Execute generated Python code in an isolated temp directory.

    - Writes code to main.py
    - Runs with thread caps and no network (best-effort: inherits current env)
    - Captures stdout/stderr
    - Returns artifacts if files like submission.csv exist in workdir
    """

    def __init__(self, env: Dict[str, str] | None = None) -> None:
        self._env_overrides = env or {}

    async def run(self, code: str, *, timeout_s: int) -> RunResult:
        workdir = Path(tempfile.mkdtemp(prefix="kts-"))
        try:
            (workdir / "main.py").write_text(code, encoding="utf-8")
            env = os.environ.copy()
            env.update(THREAD_CAPS)
            env.update(self._env_overrides)
            # Best-effort: disable pip & network inside code
            env.setdefault("NO_NETWORK", "1")
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "main.py",
                cwd=str(workdir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                async with asyncio.timeout(timeout_s):
                    out, err = await proc.communicate()
            except TimeoutError:
                proc.kill()
                return RunResult("", "timeout", {}, True, "timeout")

            stdout = out.decode(errors="replace") if out else ""
            stderr = err.decode(errors="replace") if err else ""

            artifacts: Dict[str, str] = {}
            sub = workdir / "submission.csv"
            if sub.exists():
                artifacts["submission.csv"] = str(sub)

            return RunResult(stdout, stderr, artifacts, False, None if proc.returncode == 0 else f"exit={proc.returncode}")
        finally:
            # Keep directory for debugging? For now, clean up.
            shutil.rmtree(workdir, ignore_errors=True)
