from __future__ import annotations

import asyncio
import itertools
import uuid
from typing import Callable, Dict, List

from ..agents.base import LLMRewriter, RunResult, Sandbox, Scorer, sanitize_code_output
from ..core.controller import select_parents
from ..core.types import Node, SolutionTree
from ..agents.prompting import build_improve_prompt, build_feedback
from ..persistence.jsonl import RunStore
from ..research.recombine import synthesize_hybrids
from ..config import idea_llm_provider, embeddings_provider


def _make_node_id() -> str:
    return uuid.uuid4().hex[:12]


async def expand_once(
    parent: Node,
    rewriter: LLMRewriter,
    sandbox: Sandbox,
    scorer: Scorer,
    *,
    timeout_seconds: int,
    higher_is_better: bool,
    instruction: str | None = None,
    challenge_text: str | None = None,
    max_prompt_tokens: int = 2048,
    dataset_root_hint: str | None = None,
    llm_timeout_seconds: int = 60,
    validation_labels_path: str | None = None,
    validation_view: bool = False,
) -> Node:
    fb = build_feedback(parent.score, parent.logs if isinstance(parent.logs, dict) else {}, max_chars=800)
    prompt = build_improve_prompt(
        {
            "parent_code": parent.code,
            "instruction": instruction or "Improve the code for better validation performance.",
            "feedback": fb,
            "challenge": challenge_text or "",
            "max_prompt_tokens": str(max_prompt_tokens),
            "dataset_root": dataset_root_hint or "",
            "validation_labels": (validation_labels_path or "") if not validation_view else "",
            "validation_view": "true" if validation_view else "false",
        }
    )
    try:
        child_code = await asyncio.wait_for(rewriter.run({"prompt": prompt}), timeout=llm_timeout_seconds)
        child_code = sanitize_code_output(child_code)
        # Guard: if LLM returned prompt text or error banner, fallback to parent
        bad_markers = [
            "Gemini API error",
            "You are an expert Python developer",
            "Challenge context:",
            "Rewrite the code",
        ]
        if not child_code.strip() or any(m in child_code for m in bad_markers):
            raise RuntimeError("non-code output from LLM")
    except Exception:
        # Treat LLM failure as a no-op mutation with a marker comment
        child_code = parent.code + "\n# llm_timeout_or_error"
    run: RunResult = await sandbox.run(child_code, timeout_s=timeout_seconds)
    if run.error or run.timed_out:
        score = float("-1e300")
    else:
        score = await scorer.run(run.artifacts, higher_is_better=higher_is_better)
    child = Node(
        id=_make_node_id(),
        parent_id=parent.id,
        code=child_code,
        score=score,
        visits=1,
        logs={
            "stdout": run.stdout,
            "stderr": run.stderr,
            "artifacts": run.artifacts,
            "timed_out": run.timed_out,
            "error": run.error,
            "instruction": instruction or "",
        },
    )
    return child


async def run_search(
    *,
    max_nodes: int,
    k_parallel: int,
    c_puct: float,
    timeout_seconds: int,
    higher_is_better: bool,
    tree: SolutionTree,
    rewriter: LLMRewriter,
    sandbox: Sandbox,
    scorer: Scorer,
    seeds: list[str] | None = None,
    store: RunStore | None = None,
    challenge_text: str | None = None,
    recombine_after: int | None = None,
    recombine_top: int = 4,
    max_prompt_tokens: int = 2048,
    dataset_root_hint: str | None = None,
    dataset_root_prompt_hint: str | None = None,
    llm_timeout_seconds: int = 60,
    emb_timeout_seconds: int = 30,
    validation_labels_path: str | None = None,
    force_validation_view: bool = False,
) -> Node | None:
    if tree.size() == 0:
        # Initialize with a simple baseline root
        root = Node(id=_make_node_id(), parent_id=None, code="print('hello world')", score=0.0, visits=1)
        tree.add(root)

    # Semaphore for sandbox concurrency
    sem = asyncio.Semaphore(max(1, k_parallel))

    # dynamic seeds buffer
    seed_buffer: List[str] = list(seeds or [])
    seed_index = 0

    def next_seed() -> str | None:
        nonlocal seed_index
        if not seed_buffer:
            return None
        s = seed_buffer[seed_index % len(seed_buffer)]
        seed_index += 1
        return s

    # Auto-detect validation view (host path) unless forced
    validation_view = bool(force_validation_view)
    if not validation_view and validation_labels_path and dataset_root_hint:
        try:
            import os, csv
            test_path = os.path.join(dataset_root_hint, "test.csv")
            if os.path.exists(test_path):
                with open(test_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                validation_view = ("Survived" not in header)
        except Exception:
            validation_view = False

    async def _task(parent: Node, seed_instruction: str | None) -> Node:
        async with sem:
            return await expand_once(
                parent,
                rewriter,
                sandbox,
                scorer,
                timeout_seconds=timeout_seconds,
                higher_is_better=higher_is_better,
                instruction=seed_instruction,
                challenge_text=challenge_text,
                max_prompt_tokens=max_prompt_tokens,
                dataset_root_hint=(dataset_root_prompt_hint or dataset_root_hint),
                llm_timeout_seconds=llm_timeout_seconds,
                validation_labels_path=validation_labels_path,
                validation_view=validation_view,
            )

    while tree.size() < max_nodes:
        to_expand = select_parents(tree, c_puct=c_puct, k=min(k_parallel, max_nodes - tree.size()))
        if not to_expand:
            # Defensive: break to avoid stalling if selection returns no candidates
            break
        tasks = []
        for p in to_expand:
            instr = next_seed()
            tasks.append(asyncio.create_task(_task(p, instr)))
        children = await asyncio.gather(*tasks)
        for parent, child in zip(to_expand, children):
            # Persist artifacts before storing node records
            if store is not None:
                try:
                    persisted = store.persist_artifacts(child.id, child.logs.get("artifacts", {}))
                    if persisted:
                        child.logs["artifacts"] = persisted
                except Exception:
                    pass
            tree.add(child)
            # Backpropagate visit increments up the ancestor chain
            current = parent
            while True:
                current.visits += 1
                if current.parent_id is None:
                    break
                try:
                    current = tree.get(current.parent_id)
                except KeyError:
                    break
            if store is not None:
                store.append_node(child)
                store.write_logs(
                    child.id,
                    stdout=str(child.logs.get("stdout", "")),
                    stderr=str(child.logs.get("stderr", "")),
                    meta={k: v for k, v in child.logs.items() if k not in ("stdout", "stderr")},
                )
        # Simple progress print (can be replaced by logging)
        best = tree.best()
        if best is not None:
            print(f"nodes={tree.size()} best_score={best.score:.6f} best_id={best.id}")

        # Recombination trigger
        if recombine_after and tree.size() >= recombine_after:
            ranked = sorted([n for n in tree.all().values() if n.score is not None], key=lambda n: n.score, reverse=True)
            top = ranked[: max(2, recombine_top)]
            pairs: List[tuple[Node, Node]] = []
            embprov = embeddings_provider()
            if embprov and len(top) >= 2:
                try:
                    vecs = await asyncio.wait_for(embprov.embed([n.code for n in top]), timeout=emb_timeout_seconds)
                    remaining = list(range(len(top)))
                    def cosine(i: int, j: int) -> float:
                        import math
                        va = vecs[i]; vb = vecs[j]
                        da = math.sqrt(sum(x*x for x in va))
                        db = math.sqrt(sum(x*x for x in vb))
                        if da == 0 or db == 0:
                            return 0.0
                        dot = sum(a*b for a, b in zip(va, vb))
                        return dot / (da * db)
                    while len(remaining) >= 2:
                        i = remaining.pop(0)
                        best_j = None; best_d = -1.0
                        for j in remaining:
                            d = 1.0 - cosine(i, j)
                            if d > best_d:
                                best_d, best_j = d, j
                        if best_j is None:
                            break
                        remaining.remove(best_j)
                        pairs.append((top[i], top[best_j]))
                except Exception:
                    for i in range(0, len(top) - 1, 2):
                        pairs.append((top[i], top[i + 1]))
            else:
                for i in range(0, len(top) - 1, 2):
                    pairs.append((top[i], top[i + 1]))
            if pairs:
                hybrids = await synthesize_hybrids(pairs)
                for h in hybrids:
                    if h not in seed_buffer:
                        seed_buffer.append(h)
                recombine_after = None

    return tree.best()
