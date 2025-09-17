from __future__ import annotations

import asyncio
import json
from pathlib import Path
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


def _append_log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


async def synthesize_hybrids(
    pairs: List[Tuple[Node, Node]],
    *,
    max_tokens: int = 512,
    log_path: str | Path | None = None,
) -> List[str]:
    provider = idea_llm_provider()
    out: List[str] = []
    log_file: Path | None = None
    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    for a, b in pairs:
        prompt = PROMPT.format(
            score_a=a.score if a.score is not None else float("nan"),
            score_b=b.score if b.score is not None else float("nan"),
            code_a=a.code,
            code_b=b.code,
        )
        try:
            text = await provider.generate(prompt, max_tokens=max_tokens)
            raw_text = (text or "").strip()
        except Exception as exc:
            raw_text = ""
            if log_file is not None:
                await asyncio.to_thread(
                    _append_log,
                    log_file,
                    {
                        "pair": [a.id, b.id],
                        "error": f"provider_error: {exc}",
                    },
                )
            # Skip this pair on provider failure; keep search moving
            continue
        if log_file is not None:
            await asyncio.to_thread(
                _append_log,
                log_file,
                {
                    "pair": [a.id, b.id],
                    "raw": raw_text,
                },
            )
        if not raw_text:
            continue
        first_line = raw_text.splitlines()[0].strip()
        if first_line:
            out.append(first_line)
    return out
