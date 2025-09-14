from __future__ import annotations

import re
from typing import Dict, List


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2 - 3
    return text[:half] + "..." + text[-half:]


def build_feedback(prev_score: float | None, logs: Dict[str, str] | None, max_chars: int = 800) -> str:
    parts = []
    if prev_score is not None:
        parts.append(f"prev_score={prev_score}")
    if logs:
        stderr = str(logs.get("stderr", ""))
        stdout = str(logs.get("stdout", ""))
        if stderr:
            parts.append("stderr_tail=" + truncate_middle(stderr, max_chars))
        if stdout:
            parts.append("stdout_tail=" + truncate_middle(stdout, max_chars))
    return " | ".join(parts)


def approx_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text)


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    toks = approx_tokenize(text)
    if len(toks) <= max_tokens:
        return text
    keep_head = max_tokens // 2 - 3
    keep_tail = max_tokens - keep_head - 3
    head = " ".join(toks[:keep_head])
    tail = " ".join(toks[-keep_tail:])
    return f"{head} ... {tail}"


def build_improve_prompt(context: Dict[str, str]) -> str:
    parent = context.get("parent_code", "").strip()
    instruction = context.get("instruction", "Improve the code.").strip()
    feedback = context.get("feedback", "").strip()
    challenge = context.get("challenge", "").strip()
    header = "You are an expert Python developer for Kaggle competitions.\n"
    max_prompt_tokens = int(context.get("max_prompt_tokens", 2048))
    parent_budget = max(128, int(max_prompt_tokens * 0.6))
    challenge_budget = max(64, int(max_prompt_tokens * 0.25))
    feedback_budget = max(32, int(max_prompt_tokens * 0.15))

    parent_trunc = truncate_by_tokens(parent, parent_budget)
    challenge_trunc = truncate_by_tokens(challenge, challenge_budget) if challenge else ""
    feedback_trunc = truncate_by_tokens(feedback, feedback_budget) if feedback else ""
    dataset_root = context.get("dataset_root", "").strip()

    parts: List[str] = [header]
    if challenge_trunc:
        parts.append(f"Challenge context:\n{challenge_trunc}\n\n")
    if dataset_root:
        parts.append(f"Dataset root: {dataset_root}\n\n")
    parts.append(f"Instruction: {instruction}\n")
    parts.append(f"Feedback: {feedback_trunc}\n\n")
    parts.append("Parent code:\n" + parent_trunc + "\n")
    parts.append("Rewrite the code with clear improvements. Output only the new Python code.")
    return "".join(parts)
