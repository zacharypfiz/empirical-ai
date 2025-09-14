from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Optional


class Agent(Protocol):
    async def run(self, input: Any) -> Any:  # pragma: no cover (interface)
        ...


class LLMProvider(Protocol):
    async def generate(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:  # pragma: no cover
        ...


class LLMRewriter:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def run(self, context: Dict[str, Any]) -> str:
        # If a full prompt is provided, use it directly. Otherwise, build a simple one.
        if "prompt" in context:
            prompt = str(context["prompt"])  # type: ignore[assignment]
        else:
            parent_code: str = context.get("parent_code", "")
            instruction: str = context.get("instruction", "Improve the code.")
            feedback: str = context.get("feedback", "")
            prompt = f"""
You are an expert Python developer.
Instruction: {instruction}
Feedback: {feedback}
Parent code:\n{parent_code}
""".strip()
        return await self.provider.generate(prompt, max_tokens=2048)


@dataclass
class RunResult:
    stdout: str
    stderr: str
    artifacts: Dict[str, str]
    timed_out: bool
    error: Optional[str]


class Sandbox(Protocol):
    async def run(self, code: str, *, timeout_s: int) -> RunResult:  # pragma: no cover
        ...


class Scorer(Protocol):
    async def run(self, artifacts: Dict[str, str], *, higher_is_better: bool) -> float:  # pragma: no cover
        ...
