from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from ..agents.sandbox_local import LocalSandbox
from ..agents.scorer_local import LocalAccuracyScorer, LocalMSEScorer


async def _run_once(code_path: str, dataset_root: str, timeout_s: int) -> str:
    code = Path(code_path).read_text(encoding="utf-8")
    # Ensure DATASET_ROOT is absolute so it resolves correctly from sandbox temp dirs
    dataset_root_abs = os.path.abspath(dataset_root)
    if not os.path.exists(os.path.join(dataset_root_abs, "train.csv")):
        raise SystemExit(f"Missing train.csv at {os.path.join(dataset_root_abs, 'train.csv')}")
    if not os.path.exists(os.path.join(dataset_root_abs, "test.csv")):
        raise SystemExit(f"Missing test.csv at {os.path.join(dataset_root_abs, 'test.csv')}")
    sandbox = LocalSandbox(env={"DATASET_ROOT": dataset_root_abs})
    run = await sandbox.run(code, timeout_s=timeout_s)
    if run.error or run.timed_out:
        raise SystemExit(f"Run failed: error={run.error} timed_out={run.timed_out}\nSTDERR:\n{run.stderr}")
    sub = run.artifacts.get("submission.csv")
    if not sub:
        raise SystemExit("No submission.csv produced by the code.")
    return sub


def main() -> None:
    ap = argparse.ArgumentParser("manual-eval")
    ap.add_argument("--code-path", required=True, help="Path to Python file to run (generates submission.csv)")
    ap.add_argument("--dataset-root", required=True, help="Path to dataset root containing train.csv and test.csv")
    ap.add_argument("--labels-path", required=True, help="Path to validation labels CSV")
    ap.add_argument("--metric", choices=["accuracy", "mse"], default="accuracy")
    ap.add_argument("--timeout-seconds", type=int, default=60)
    args = ap.parse_args()

    sub_path = asyncio.run(_run_once(args.code_path, args.dataset_root, args.timeout_seconds))

    if args.metric == "accuracy":
        scorer = LocalAccuracyScorer(labels_path=args.labels_path)
        higher_is_better = True
    else:
        from ..agents.scorer_local import LocalMSEScorer as _MSE
        scorer = _MSE(labels_path=args.labels_path)
        higher_is_better = False

    score = asyncio.run(scorer.run({"submission.csv": sub_path}, higher_is_better=higher_is_better))
    print(f"score={score}")
    # Basic diagnostics when scoring fails sentinel-style
    if (higher_is_better and score == float("-inf")) or ((not higher_is_better) and score == float("inf")):
        try:
            import csv
            # Load labels
            labels = {}
            with open(args.labels_path, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    labels[str(row.get("PassengerId", "")).strip()] = row.get("Survived", "")
            # Load submission
            n_sub = 0
            n_ok_cols = False
            n_match = 0
            with open(sub_path, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                cols = set(r.fieldnames or [])
                n_ok_cols = ("PassengerId" in cols and "Survived" in cols)
                for row in r:
                    n_sub += 1
                    pid = str(row.get("PassengerId", "")).strip()
                    if pid and pid in labels:
                        n_match += 1
            print(f"diag: labels={len(labels)} submission_rows={n_sub} ok_cols={n_ok_cols} matched_ids={n_match}")
        except Exception as e:
            print(f"diag: unable to analyze failure: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
