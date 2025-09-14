from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Optional
import re


class Agent(Protocol):
    async def run(self, input: Any) -> Any:  # pragma: no cover (interface)
        ...


class LLMProvider(Protocol):
    async def generate(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:  # pragma: no cover
        ...


class LLMRewriter:
    def __init__(self, provider: LLMProvider, *, default_max_tokens: Optional[int] = None) -> None:
        self.provider = provider
        self.default_max_tokens = default_max_tokens

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
        max_tokens = self.default_max_tokens if self.default_max_tokens is not None else 2048
        return await self.provider.generate(prompt, max_tokens=max_tokens)


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


def sanitize_code_output(text: str) -> str:
    """Extract runnable Python from LLM output.

    - If fenced blocks (``` or ```python) are present, return the largest fenced block's body.
    - Otherwise, strip common preambles and trailing backticks.
    """
    if not text:
        return text
    # Find all fenced code blocks
    blocks = []
    for m in re.finditer(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        blocks.append(m.group(1))
    if blocks:
        # pick the largest block
        code = max(blocks, key=lambda s: len(s))
        return code.strip()
    # Strip inline fences if someone left a trailing ```
    t = text.strip()
    t = re.sub(r"^```(?:python|py)?\s*\n", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\n```\s*$", "", t)
    # Remove common prose preambles
    t = re.sub(r"^\s*Here'?s\b.*?\n+", "", t)
    return t.strip()
