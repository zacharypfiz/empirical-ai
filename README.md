# Empirical AI: LLM + Tree Search Pipeline

A minimal, reproducible pipeline that iteratively rewrites, executes, and scores Python code for a Kaggle‑style challenge using a flat PUCT controller, sandboxed runs, and optional research seeding, recombination, and embeddings.

## Quick Start

- Prereqs: uv, Python 3.10+, optional Docker for sandboxing.
- Optional LLM: set `GEMINI_API_KEY` in `.env` (falls back to stub without it).

Project layout assumptions
- Challenge text: `challenge/challenge.md`
- Dataset root: `challenge/data` (train/test/labels)

## Setup
- Install optional Gemini SDK (for LLM + embeddings):
  - `uv add google-genai`
- Configure `.env` as needed:
  - `GEMINI_API_KEY=...`
  - Optional model buckets: `CODE_MODEL`, `IDEA_MODEL`, `EMBEDDING_MODEL`

## Pipeline Steps

1) Generate strategy ideas (from challenge.md)
- `uv run python -m kaggle_ts.cli research --challenge-path challenge/challenge.md --out runs/ideas.jsonl --max-ideas 8`

2) Run search (K-parallel, flat PUCT)
- Local (stub sandbox/scorer):
  - `uv run python -m kaggle_ts.cli search --max-nodes 200 --k 8 --c-puct 1.2 --challenge-path challenge/challenge.md --dataset-root challenge/data --ideas-path runs/ideas.jsonl --seeds 3`
- With validation + metric (e.g., Titanic accuracy):
  - `uv run python -m kaggle_ts.cli search --max-nodes 200 --k 8 --c-puct 1.2 --metric accuracy --validation-labels challenge/data/val_labels.csv --id-col PassengerId --label-col Survived --challenge-path challenge/challenge.md --dataset-root challenge/data`
- With Docker sandbox (auto-mounts dataset to `/data:ro`):
  - Build image (optional): `docker build -t my-kaggle:py311 -f docker/Dockerfile .`
  - Run:
    - `uv run python -m kaggle_ts.cli search --docker --docker-image my-kaggle:py311 --docker-cpus 2 --docker-memory 4g --challenge-path challenge/challenge.md --dataset-root challenge/data --max-nodes 200 --k 8 --c-puct 1.2`
- Recombination (one-time, after N nodes):
  - Add: `--recombine-after 200 --recombine-top 8`
- Prompt token budget (approximate):
  - Add: `--max-prompt-tokens 2048`

3) Post-hoc embeddings (optional)
- `uv run python -m kaggle_ts.cli embed --nodes runs/nodes.jsonl --out runs/embeddings.jsonl --batch-size 16`

## What Gets Produced
- `runs/nodes.jsonl`: one JSON record per generated node (code, score, visits, parent, logs).
- `runs/logs/<node_id>/`: `stdout.txt`, `stderr.txt`, and `meta.json` per node.
- `runs/ideas.jsonl`: synthesized strategy seeds.
- `runs/embeddings.jsonl`: code embeddings (when `GEMINI_API_KEY` is set).

## Key Commands (at a glance)
- Ideas: `uv run python -m kaggle_ts.cli research --challenge-path challenge/challenge.md --out runs/ideas.jsonl`
- Search (local): `uv run python -m kaggle_ts.cli search --max-nodes 200 --k 8 --c-puct 1.2 --challenge-path challenge/challenge.md --dataset-root challenge/data`
- Search (accuracy): `... --metric accuracy --validation-labels challenge/data/val_labels.csv --id-col PassengerId --label-col Survived`
- Search (Docker): `... --docker --docker-image my-kaggle:py311 --docker-cpus 2 --docker-memory 4g`
- Embeddings: `uv run python -m kaggle_ts.cli embed --nodes runs/nodes.jsonl --out runs/embeddings.jsonl`

## Notes
- Provider selection is automatic: if `GEMINI_API_KEY` is set, Gemini is used; otherwise a stub provider runs.
- The sandbox enforces timeouts and thread caps; Docker runs with `--network none`, CPU and memory limits, and dataset mounts.
- The controller follows the paper: flat selection over all nodes, rank-normalized scores, flat prior, visit backpropagation, and K-parallel expansions per round.
