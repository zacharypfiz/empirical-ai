from __future__ import annotations

import asyncio
import hashlib
import random
from typing import Optional

from .base import LLMProvider, RunResult, Sandbox, Scorer


class StubLLMProvider(LLMProvider):
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rnd = random.Random(seed)

    async def generate(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        # Trivial mutation: append a comment with a random tweak
        tweak = self._rnd.randint(0, 10_000)
        return (prompt.split("Parent code:\n", 1)[-1]).rstrip() + f"\n# tweak: {tweak}"


class StubSandbox(Sandbox):
    def __init__(self, *, min_ms: int = 50, max_ms: int = 200) -> None:
        self._min = min_ms
        self._max = max_ms

    async def run(self, code: str, *, timeout_s: int) -> RunResult:
        # Simulate work with a short sleep, no errors/timeouts
        sleep_ms = random.randint(self._min, self._max)
        try:
            async with asyncio.timeout(timeout_s):
                await asyncio.sleep(sleep_ms / 1000.0)
        except TimeoutError:  # pragma: no cover
            return RunResult("", "", {}, True, "timeout")

        stdout = f"ran {sleep_ms}ms"
        stderr = ""
        # Simulate an artifact
        artifacts = {"submission.csv": "/tmp/fake_submission.csv"}
        return RunResult(stdout, stderr, artifacts, False, None)


class StubScorer(Scorer):
    async def run(self, artifacts: dict, *, higher_is_better: bool) -> float:
        # Deterministic score from artifact path hash; higher is better
        key = artifacts.get("submission.csv", "")
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        score = (h % 1_000_000) / 1_000_000.0
        return score if higher_is_better else -score

