"""Fast holdout utility (CLI).

Usage (uv):
  uv run python -m kaggle_ts.utils.split_val --dataset-root challenge/data \
      --out-root runs/val_dataset --test-size 0.2

This will:
  - Create challenge/data/val_labels.csv (PassengerId,Survived) from a stratified split of train.csv
  - Write runs/val_dataset/train.csv (train minus val IDs)
  - Write runs/val_dataset/test.csv (validation rows without Survived)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Dict, List


def _stratified_split_indexes(labels: List[str], test_size: float, rnd: random.Random) -> List[int]:
    by_label: Dict[str, List[int]] = {}
    for i, y in enumerate(labels):
        by_label.setdefault(str(y), []).append(i)
    selected: List[int] = []
    for _, idxs in by_label.items():
        n = len(idxs)
        k = max(1, int(round(n * test_size)))
        rnd.shuffle(idxs)
        selected.extend(idxs[:k])
    return selected


def _random_split_indexes(n: int, test_size: float, rnd: random.Random) -> List[int]:
    idxs = list(range(n))
    rnd.shuffle(idxs)
    k = max(1, int(round(n * test_size)))
    return idxs[:k]


def main() -> None:
    ap = argparse.ArgumentParser("fast-holdout")
    ap.add_argument("--dataset-root", required=True, help="Path containing train.csv")
    ap.add_argument("--out-root", default="runs/val_dataset", help="Output dataset dir for validation view")
    ap.add_argument("--id-col", default="PassengerId")
    ap.add_argument("--label-col", default="Survived")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--labels-path", default=None, help="Optional explicit path for val_labels.csv")
    args = ap.parse_args()

    train_path = os.path.join(args.dataset_root, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)

    with open(train_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        header = reader.fieldnames or []
    if not rows:
        raise ValueError("Empty train CSV")
    if args.id_col not in rows[0] or args.label_col not in rows[0]:
        raise ValueError(f"Missing required columns: {args.id_col}, {args.label_col}")

    labels = [row[args.label_col] for row in rows]
    rnd = random.Random(args.random_state)
    if args.no_stratify:
        sel_idx = set(_random_split_indexes(len(rows), args.test_size, rnd))
    else:
        sel_idx = set(_stratified_split_indexes(labels, args.test_size, rnd))

    # Write labels CSV
    labels_out = args.labels_path or os.path.join(args.dataset_root, "val_labels.csv")
    os.makedirs(os.path.dirname(labels_out) or ".", exist_ok=True)
    with open(labels_out, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow([args.id_col, args.label_col])
        for i in sorted(sel_idx):
            row = rows[i]
            w.writerow([row[args.id_col], row[args.label_col]])

    # Build validation-view dataset
    os.makedirs(args.out_root, exist_ok=True)

    # New train.csv: rows not in validation
    with open(os.path.join(args.out_root, "train.csv"), "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=header)
        w.writeheader()
        for i, row in enumerate(rows):
            if i in sel_idx:
                continue
            w.writerow(row)

    # New test.csv: validation rows without label column
    test_header = [c for c in header if c != args.label_col]
    with open(os.path.join(args.out_root, "test.csv"), "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=test_header)
        w.writeheader()
        for i in sorted(sel_idx):
            r = dict(rows[i])
            r.pop(args.label_col, None)
            w.writerow(r)

    print(f"Wrote {labels_out}")
    print(f"Wrote {os.path.join(args.out_root, 'train.csv')} and {os.path.join(args.out_root, 'test.csv')}")


if __name__ == "__main__":
    main()
