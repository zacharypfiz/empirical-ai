from __future__ import annotations

import asyncio
import itertools
import uuid
from typing import Any, Callable, Dict, List
from pathlib import Path
import shutil

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
    llm_error_text: str | None = None
    raw_llm_output: str | None = None
    try:
        raw_llm_output = await asyncio.wait_for(
            rewriter.run({"prompt": prompt}), timeout=llm_timeout_seconds
        )
        child_code = sanitize_code_output(raw_llm_output)
        # Guard: if LLM returned prompt text or error banner, fallback to parent
        bad_markers = [
            "Gemini API error",
            "You are an expert Python developer",
            "Challenge context:",
            "Rewrite the code",
        ]
        if not child_code.strip() or any(m in child_code for m in bad_markers):
            raise RuntimeError("non-code output from LLM")
    except Exception as e:
        llm_error_text = f"llm_exception: {e}"
        # Treat LLM failure as a no-op mutation with a marker comment
        child_code = parent.code + "\n# llm_timeout_or_error"
    # Quick syntax pre-check to avoid sandbox runs on obviously broken code
    try:
        compile(child_code, "<node>", "exec")
    except SyntaxError as se:
        if "llm_timeout_or_error" not in child_code:
            llm_error_text = f"syntax_precheck: {se}"
            child_code = parent.code + "\n# llm_timeout_or_error"
    run: RunResult = await sandbox.run(child_code, timeout_s=timeout_seconds)
    if run.error or run.timed_out:
        score = float("-1e300")
    else:
        score = await scorer.run(run.artifacts, higher_is_better=higher_is_better)
    logs: dict[str, Any] = {
        "stdout": run.stdout,
        "stderr": run.stderr,
        "artifacts": run.artifacts,
        "timed_out": run.timed_out,
        "error": run.error,
        "instruction": instruction or "",
        **({"llm_error": llm_error_text} if llm_error_text else {}),
    }
    if raw_llm_output is not None:
        max_chars = 4000
        raw_trimmed = raw_llm_output if len(raw_llm_output) <= max_chars else raw_llm_output[:max_chars] + "..."
        logs["llm_raw"] = raw_trimmed

    child = Node(
        id=_make_node_id(),
        parent_id=parent.id,
        code=child_code,
        score=score,
        visits=1,
        logs=logs,
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
        # Initialize with a robust baseline that trains a quick model and writes submission.csv.
        # Keeps deps to pandas/numpy/scikit-learn only and avoids fragile column handling.
        baseline_code = (
            "import os, pandas as pd\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "\n"
            "ROOT = os.environ.get('DATASET_ROOT', '.')\n"
            "train = pd.read_csv(os.path.join(ROOT, 'train.csv'))\n"
            "test = pd.read_csv(os.path.join(ROOT, 'test.csv'))\n"
            "id_col = 'PassengerId'\n"
            "label_col = 'Survived'\n"
            "# Simple, fast features with light FE\n"
            "def add_basic_features(df):\n"
            "    if 'SibSp' in df.columns and 'Parch' in df.columns:\n"
            "        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n"
            "        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)\n"
            "    return df\n"
            "train = add_basic_features(train.copy())\n"
            "test = add_basic_features(test.copy())\n"
            "X = train.drop([label_col], axis=1)\n"
            "y = train[label_col]\n"
            "test_ids = test[id_col] if id_col in test.columns else None\n"
            "# Candidate feature sets (present columns only)\n"
            "num_cols_all = ['Age','Fare','SibSp','Parch','FamilySize','IsAlone']\n"
            "cat_cols_all = ['Pclass','Sex','Embarked']\n"
            "num_cols = [c for c in num_cols_all if c in X.columns]\n"
            "cat_cols = [c for c in cat_cols_all if c in X.columns]\n"
            "# Always drop PassengerId before preprocessing to avoid train/test schema mismatch\n"
            "cols_to_fit = [c for c in X.columns if c not in (set(num_cols) | set(cat_cols) | {id_col})]\n"
            "# Preprocess numeric and categorical; drop anything else\n"
            "pre = ColumnTransformer(\n"
            "    transformers=[\n"
            "        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),\n"
            "        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),\n"
            "    ],\n"
            "    remainder='drop'\n"
            ")\n"
            "clf = Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=500))])\n"
            "X_fit = X.drop(columns=[id_col], errors='ignore')\n"
            "X_test = test.drop(columns=[id_col], errors='ignore')\n"
            "clf.fit(X_fit, y)\n"
            "pred = clf.predict(X_test)\n"
            "sub = pd.DataFrame({id_col: test_ids if test_ids is not None else range(len(pred)), label_col: pred})\n"
            "sub.to_csv('submission.csv', index=False)\n"
            "print('baseline submission written:', len(sub))\n"
        )
        root = Node(id=_make_node_id(), parent_id=None, code=baseline_code, score=0.0, visits=1)
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
                    # Persist artifacts to runs dir
                    original_artifacts = dict(child.logs.get("artifacts", {}))
                    persisted = store.persist_artifacts(child.id, original_artifacts)
                    if persisted:
                        child.logs["artifacts"] = persisted
                    # Cleanup temp artifact dirs (kts-art-*), if any
                    for src in (original_artifacts or {}).values():
                        try:
                            p = Path(src)
                            parent_dir = p.parent
                            if parent_dir.name.startswith("kts-art-"):
                                shutil.rmtree(parent_dir, ignore_errors=True)
                        except Exception:
                            continue
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
                hybrid_log = None
                if store is not None:
                    hybrid_log = store.base / "logs" / "recombine_raw.jsonl"
                hybrids = await synthesize_hybrids(pairs, log_path=hybrid_log)
                for h in hybrids:
                    if h not in seed_buffer:
                        seed_buffer.append(h)
                recombine_after = None

    return tree.best()
