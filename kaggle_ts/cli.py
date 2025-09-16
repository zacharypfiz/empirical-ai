from __future__ import annotations

import argparse
import asyncio
import os
import subprocess

from .agents.base import LLMRewriter
from .agents.stubs import StubLLMProvider, StubSandbox, StubScorer
from .core.types import SolutionTree
from .orchestrator.loop import run_search


def _ensure_docker_ready(image: str) -> None:
    """Fail fast if Docker CLI or daemon is unavailable."""
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on host setup
        raise SystemExit("Docker executable not found; re-run without --docker") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown error").strip()
        msg = f"Unable to talk to Docker daemon: {detail}"
        raise SystemExit(msg) from exc
    # Optionally emit info when the daemon is reachable but image missing
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        print(f"Warning: Docker image '{image}' not present locally; it will be pulled on first use.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("kaggle-ts")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_search = sub.add_parser("search", help="Run flat PUCT search")
    s_search.add_argument("--max-nodes", type=int, default=50)
    s_search.add_argument("--k", dest="k_parallel", type=int, default=4)
    s_search.add_argument("--c-puct", type=float, default=1.2)
    s_search.add_argument("--timeout-seconds", type=int, default=20)
    s_search.add_argument("--metric", choices=["accuracy", "mse"], default="accuracy")
    s_search.add_argument("--ideas-path", type=str, default=None, help="JSONL ideas file for seeding")
    s_search.add_argument("--seeds", type=int, default=0, help="Use first N ideas as instruction seeds")
    s_search.add_argument("--validation-labels", type=str, default=None, help="Path to validation labels CSV")
    s_search.add_argument("--challenge-path", type=str, default="challenge/challenge.md", help="Path to challenge description for prompt context")
    s_search.add_argument("--dataset-root", type=str, default="challenge/data", help="Path to dataset root directory")
    s_search.add_argument("--docker", action="store_true", help="Use Docker sandbox instead of local")
    s_search.add_argument("--docker-image", type=str, default="python:3.11-slim")
    s_search.add_argument("--docker-cpus", type=float, default=1.0)
    s_search.add_argument("--docker-memory", type=str, default="2g")
    s_search.add_argument("--mount", action="append", default=None, help="Host:Container[:ro|rw] mount (repeatable)")
    s_search.add_argument("--recombine-after", type=int, default=0, help="Trigger recombination after this many nodes (0=disable)")
    s_search.add_argument("--recombine-top", type=int, default=4, help="Top-N nodes used to build recombination pairs")
    s_search.add_argument("--max-prompt-tokens", type=int, default=2048, help="Approximate max tokens for prompt truncation")
    s_search.add_argument("--id-col", type=str, default=None, help="Override ID column name for metrics")
    s_search.add_argument("--label-col", type=str, default=None, help="Override label column name for metrics")
    s_search.add_argument("--pred-col", type=str, default=None, help="Override prediction column name for metrics")
    s_search.add_argument("--llm-timeout-seconds", type=int, default=60, help="Timeout for LLM generation per expansion")
    s_search.add_argument("--emb-timeout-seconds", type=int, default=30, help="Timeout for embedding calls during recombination")
    s_search.add_argument("--runs-dir", type=str, default="runs", help="Directory to store nodes/logs/artifacts")
    s_search.add_argument("--fresh", action="store_true", help="Clear prior nodes/logs/artifacts in runs dir before starting")
    s_search.add_argument("--code-max-output-tokens", type=int, default=None, help="Max output tokens for code generation (env CODE_MAX_OUTPUT_TOKENS as fallback)")
    # Validation-view is the only supported validation workflow now; no separate flag needed.

    s_research = sub.add_parser("research", help="Synthesize strategy ideas from challenge.md")
    s_research.add_argument("--challenge-path", type=str, default="challenge/challenge.md")
    s_research.add_argument("--out", type=str, default="runs/ideas.jsonl")
    s_research.add_argument("--max-ideas", type=int, default=8)
    s_embed = sub.add_parser("embed", help="Compute embeddings for generated nodes' code")
    s_embed.add_argument("--nodes", type=str, default="runs/nodes.jsonl")
    s_embed.add_argument("--out", type=str, default="runs/embeddings.jsonl")
    s_embed.add_argument("--batch-size", type=int, default=16)

    s_makeval = sub.add_parser("make-val", help="Create validation labels CSV from train.csv")
    s_makeval.add_argument("--train-path", type=str, default="challenge/data/train.csv")
    s_makeval.add_argument("--out", type=str, default="challenge/data/val_labels.csv")
    s_makeval.add_argument("--id-col", type=str, default="PassengerId")
    s_makeval.add_argument("--label-col", type=str, default="Survived")
    s_makeval.add_argument("--test-size", type=float, default=0.2)
    s_makeval.add_argument("--random-state", type=int, default=42)
    s_makeval.add_argument("--no-stratify", action="store_true")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "search":
        higher_is_better = True if args.metric == "accuracy" else False
        tree = SolutionTree()
        from .config import code_llm_provider
        from .agents.stubs import StubSandbox, StubScorer

        # Code output token budget (flag > env > default)
        code_max_env = os.getenv("CODE_MAX_OUTPUT_TOKENS")
        try:
            code_max_tokens = args.code_max_output_tokens if args.code_max_output_tokens is not None else (int(code_max_env) if code_max_env else None)
        except Exception:
            code_max_tokens = None
        rewriter = LLMRewriter(code_llm_provider(), default_max_tokens=code_max_tokens)
        challenge_text = None
        if args.challenge_path and os.path.exists(args.challenge_path):
            with open(args.challenge_path, "r", encoding="utf-8") as f:
                challenge_text = f.read()
        dataset_root_hint = None
        if args.dataset_root:
            dataset_root_hint = os.path.abspath(args.dataset_root)

        # Auto-select scorer
        if args.validation_labels and os.path.exists(args.validation_labels):
            if args.metric == "accuracy":
                from .agents.scorer_local import LocalAccuracyScorer

                scorer = LocalAccuracyScorer(labels_path=args.validation_labels, id_col=args.id_col or "PassengerId", label_col=args.label_col or "Survived")
            else:
                from .agents.scorer_local import LocalMSEScorer

                scorer = LocalMSEScorer(labels_path=args.validation_labels, id_col=args.id_col, label_col=args.label_col, pred_col=args.pred_col)
        else:
            from .agents.stubs import StubScorer
            scorer = StubScorer()

        # Select sandbox
        if args.docker:
            from .agents.sandbox_docker import DockerSandbox

            mounts = {}
            if args.mount:
                for m in args.mount:
                    try:
                        host, container = m.split(":", 1)
                        mounts[host] = container
                    except ValueError:
                        print(f"Invalid mount format: {m}. Expected Host:Container[:mode]")
            # auto-mount dataset root to /data:ro
            if dataset_root_hint and os.path.isdir(dataset_root_hint):
                mounts.setdefault(dataset_root_hint, "/data:ro")
            # Validation labels handling: mount labels dir if not inside dataset root
            env_map = {"DATASET_ROOT": "/data" if dataset_root_hint else ""}
            if args.validation_labels:
                labels_abs = os.path.abspath(args.validation_labels)
                set_env_label = None
                if dataset_root_hint and labels_abs.startswith(os.path.abspath(dataset_root_hint) + os.sep):
                    # inside dataset root; refer via /data relative
                    rel = os.path.relpath(labels_abs, os.path.abspath(dataset_root_hint))
                    set_env_label = "/data/" + rel.replace(os.sep, "/")
                else:
                    labels_dir = os.path.dirname(labels_abs)
                    mounts.setdefault(labels_dir, "/labels:ro")
                    set_env_label = "/labels/" + os.path.basename(labels_abs)
                env_map["VALIDATION_LABELS"] = set_env_label
            _ensure_docker_ready(args.docker_image)
            sandbox = DockerSandbox(
                image=args.docker_image,
                cpus=args.docker_cpus,
                memory=args.docker_memory,
                mounts=mounts,
                env=env_map,
            )
        else:
            try:
                from .agents.sandbox_local import LocalSandbox

                env = {}
                if dataset_root_hint and os.path.isdir(dataset_root_hint):
                    env["DATASET_ROOT"] = dataset_root_hint
                if args.validation_labels and os.path.exists(args.validation_labels):
                    env["VALIDATION_LABELS"] = os.path.abspath(args.validation_labels)
                sandbox = LocalSandbox(env=env)
            except Exception:
                from .agents.stubs import StubSandbox

                sandbox = StubSandbox()
        seeds: list[str] | None = None
        if args.ideas_path and args.seeds > 0:
            import json
            with open(args.ideas_path, "r", encoding="utf-8") as f:
                ideas = [json.loads(line) for line in f if line.strip()]
            seeds = [i.get("summary") or i.get("title") for i in ideas][: args.seeds]
        from .persistence.jsonl import RunStore
        store = RunStore(base_dir=args.runs_dir)
        if args.fresh:
            store.clean()
        # Validation-view is the default when validation labels are provided
        force_validation_view = bool(args.validation_labels)

        best = asyncio.run(
            run_search(
                max_nodes=args.max_nodes,
                k_parallel=args.k_parallel,
                c_puct=args.c_puct,
                timeout_seconds=args.timeout_seconds,
                higher_is_better=higher_is_better,
                tree=tree,
                rewriter=rewriter,
                sandbox=sandbox,
                scorer=scorer,
                seeds=seeds,
                store=store,
                challenge_text=challenge_text,
                recombine_after=(args.recombine_after or None),
                recombine_top=args.recombine_top,
                max_prompt_tokens=args.max_prompt_tokens,
                dataset_root_hint=dataset_root_hint,  # host path for detection
                dataset_root_prompt_hint=("/data" if (args.docker and dataset_root_hint) else dataset_root_hint),
                llm_timeout_seconds=args.llm_timeout_seconds,
                emb_timeout_seconds=args.emb_timeout_seconds,
                validation_labels_path=("/data/val_labels.csv" if (args.docker and args.validation_labels) else (os.path.abspath(args.validation_labels) if args.validation_labels else None)),
                force_validation_view=force_validation_view,
            )
        )
        if best is None:
            print("No best node found.")
        else:
            print("Best node:")
            print(f"  id: {best.id}")
            print(f"  score: {best.score}")
            print(f"  parent: {best.parent_id}")
    elif args.cmd == "research":
        from pathlib import Path
        from .research.ideas import synthesize_ideas, write_ideas_jsonl

        challenge_text = Path(args.challenge_path).read_text(encoding="utf-8")
        ideas = asyncio.run(synthesize_ideas(challenge_text, max_ideas=args.max_ideas))
        write_ideas_jsonl(args.out, ideas)
        print(f"Wrote {len(ideas)} ideas to {args.out}")
    elif args.cmd == "embed":
        from .analysis.embeddings import embed_nodes
        n = asyncio.run(embed_nodes(args.nodes, args.out, args.batch_size))
        print(f"Wrote embeddings for {n} nodes to {args.out}")
    elif args.cmd == "make-val":
        from .utils.split_val import make_validation_labels
        n = make_validation_labels(
            train_path=args.train_path,
            out_path=args.out,
            id_col=args.id_col,
            label_col=args.label_col,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=(not args.no_stratify),
        )
        print(f"Wrote {n} validation rows to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
