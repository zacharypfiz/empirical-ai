from __future__ import annotations

from typing import List, Tuple

from ..core.types import Node
from ..config import idea_llm_provider


PROMPT = """
You are an expert data scientist.
Two solutions below approach the problem differently. Analyze their core technical differences and propose a new hybrid improvement instruction to guide a rewrite.

Provide a single, concise instruction (2-4 sentences) focusing on concrete changes.

Solution A (score={score_a}):
---
{code_a}
---

Solution B (score={score_b}):
---
{code_b}
---

Instruction:
""".strip()


async def synthesize_hybrids(pairs: List[Tuple[Node, Node]], max_tokens: int = 512) -> List[str]:
    provider = idea_llm_provider()
    out: List[str] = []
    for a, b in pairs:
        prompt = PROMPT.format(
            score_a=a.score if a.score is not None else float("nan"),
            score_b=b.score if b.score is not None else float("nan"),
            code_a=a.code,
            code_b=b.code,
        )
        try:
            text = await provider.generate(prompt, max_tokens=max_tokens)
        except Exception:
            # Skip this pair on provider failure; keep search moving
            continue
        instr = (text or "").strip().splitlines()[0].strip()
        if instr:
            out.append(instr)
    return out
