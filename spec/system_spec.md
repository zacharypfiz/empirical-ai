# System Spec: Kaggle Tree Search (TS)

This spec defines the minimal, implementation‑facing contracts and runtime for a system that generates and evaluates Python solutions for a Kaggle task via a flat PUCT controller with K‑parallel expansions per round.

## 1) Scope
- Goal: Maximize validation score by iteratively rewriting, running, and scoring code.
- Success: Best node’s raw TaskScore is highest among all generated nodes; runs persist with reproducibility.
- Out of scope: UI, full Docker infra, production queueing. Provide clean interfaces so these can be added later.

## 2) Runtime Topology
- Loop: flat PUCT selection → choose K parents → for each parent: rewrite → sandbox run → score → persist node.
- Node lifecycle: select parent → construct prompt → generate child code → execute with timeout → score → store code, logs, score, parent link.
- Failure policy: exception/timeout/malformed output → assign penalty score (e.g., −inf) and persist logs; failed nodes remain selectable.

## 3) Configuration (defaults as examples)
- `max_nodes`: int, total nodes to generate (e.g., 500)
- `k_parallel`: int, parents expanded per round (e.g., 8)
- `c_puct`: float, exploration constant (e.g., 1.2)
- `timeout_seconds`: per sandbox run (e.g., 1800)
- `penalty_score`: float for failures (e.g., -1e300)
- `higher_is_better`: bool; if false, negate metric (e.g., MSE)
- Paths: dataset root, validation split, outputs/runs dir

## 4) Data Model
- Node fields:
  - `id: str`, `parent_id: str | None`
  - `code: str`, `score: float | None`
  - `visits: int` (V(u)), `created_at: iso8601`
  - `logs: dict` (e.g., `exit_code`, `timed_out`, `stderr_tail`, `artifacts`)
- Tree responsibilities:
  - `add(node)`, `get(id)`, `all()`, `best()`
  - Maintain `N_total = sum(1 + visits)` and rank cache for scores.

## 5) Controller (Flat PUCT + RankScore)
- Selection domain: all existing nodes (no recursive descent).
- RankScore: map finite scores to [0,1] via rank‑min‑max; failed/None → worst.
- Prior `P_T(u)`: start as 1.0; later allow heuristic from logs/complexity.
- Score: `S(u) = RankScore(u) + c_puct * P_T(u) * sqrt(N_total) / (1 + V(u))`
- Selection: return top‑`K` parents by `S(u)` each round.

## 6) Agents & Providers (interfaces)
```python
class Agent(Protocol):
    async def run(self, input: Any) -> Any: ...

class LLMProvider(Protocol):
    async def generate(self, prompt: str, *, max_tokens: int | None = None) -> str: ...

class LLMRewriter:
    def __init__(self, provider: LLMProvider): ...
    async def run(self, context: dict) -> str:  # returns child code
        ...

@dataclass
class RunResult:
    stdout: str
    stderr: str
    artifacts: dict  # e.g., {"submission.csv": "/path/..."}
    timed_out: bool
    error: str | None

class Sandbox:
    async def run(self, code: str, *, timeout_s: int) -> RunResult: ...

class Scorer:
    async def run(self, artifacts: dict, *, higher_is_better: bool) -> float: ...
```

## 7) Orchestration & Parallelism
- Round‑based: compute selection scores → choose `K` parents → expand concurrently.
- Concurrency: `asyncio`/`anyio` with `Semaphore(K)` or global max.
- Per‑run limits: timeout must be enforced in sandbox; optionally set CPU/mem caps.
- Expansion DAG (per parent): build prompt → `LLMRewriter.run` → `Sandbox.run` → `Scorer.run` → `Tree.add` (increment parent visits).

## 8) Persistence
- v1 (default): JSONL + files
  - `runs/nodes.jsonl`: append Node JSON per creation
  - `runs/logs/{node_id}/stdout.txt`, `stderr.txt`, artifacts copied/linked
- v2 (optional): SQLite
  - `nodes(id TEXT PK, parent_id TEXT, score REAL, visits INT, code TEXT, created_at TEXT)`
  - `logs(node_id TEXT, key TEXT, value TEXT)`

## 9) Prompt Blocks (LLM Rewriter)
- Context: competition summary, data schema, eval metric, constraints.
- Parent: full parent script (truncate tail if needed with clear notice).
- Feedback: parent score + key stdout/stderr snippets.
- Instruction: one of
  - Simple improve: "Improve code to increase [metric]; previous score=[x]."
  - Idea injection: textual method summary to implement.
  - Recombination: merge strengths of two parents (provide diff/summary).
- Truncation: budget tokens; if truncating code/logs, keep head + tail with delimiters.

## 10) Failure Handling
- Sandbox exceptions/timeouts/malformed artifacts → `penalty_score` and persist logs.
- Controller continues to consider failed nodes with worst rank; visits still count.

## 11) Observability
- Per node: `exit_code`, `timed_out`, `duration_s`, `stderr_tail`, `artifact_paths`.
- Run metrics: nodes/sec, success rate, best score over time.

## 12) CLI (uv)
- Init: `uv run python -m kaggle_ts.cli init-run --dataset ./data --metric mse`
- Search: `uv run python -m kaggle_ts.cli search --max-nodes 200 --k 8 --c-puct 1.2`
- Best: `uv run python -m kaggle_ts.cli best`

## 13) Roadmap & Extensibility
- Start with stubs: mock LLM (minor code mutation), local sandbox, real scorer.
- Swap providers: Gemini for LLM, Docker for sandbox, SQLite for storage.
- Advanced: idea seeding (multiple roots), recombination rounds using pairwise prompts, embedding‑based post‑hoc analysis.

```
Implementation note: use uv for env/package management; keep modules small, typed, and provider‑agnostic so components can be reused across different competitions.
```
