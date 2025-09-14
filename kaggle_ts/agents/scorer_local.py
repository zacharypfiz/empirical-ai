from __future__ import annotations

import csv
from typing import Dict

from .validation import csv_has_columns


class LocalAccuracyScorer:
    """Compute accuracy given a ground truth CSV and a submission CSV.

    Expects both files to have columns PassengerId,Survived.
    """

    def __init__(self, *, labels_path: str, id_col: str = "PassengerId", label_col: str = "Survived") -> None:
        self.labels_path = labels_path
        self.id_col = id_col
        self.label_col = label_col
        self._labels = self._load_labels(labels_path)

    def _load_labels(self, path: str) -> Dict[str, int]:
        labels: Dict[str, int] = {}
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row[self.id_col]).strip()
                labels[pid] = int(row[self.label_col]) if row[self.label_col] != "" else 0
        return labels

    async def run(self, artifacts: Dict[str, str], *, higher_is_better: bool) -> float:
        sub_path = artifacts.get("submission.csv")
        if not sub_path:
            return float("-inf") if higher_is_better else float("inf")
        if not csv_has_columns(sub_path, [self.id_col, self.label_col]):
            return float("-inf") if higher_is_better else float("inf")
        correct = 0
        total = 0
        try:
            with open(sub_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = str(row.get(self.id_col, "")).strip()
                    if not pid or pid not in self._labels:
                        continue
                    pred = int(row.get(self.label_col, 0))
                    if pred == self._labels[pid]:
                        correct += 1
                    total += 1
        except Exception:
            return float("-inf") if higher_is_better else float("inf")
        if total == 0:
            return float("-inf") if higher_is_better else float("inf")
        acc = correct / total
        return acc if higher_is_better else -acc


class LocalMSEScorer:
    """Compute MSE given a ground truth CSV and a submission CSV.

    Expects both files to have columns Id/PassengerId and Target/Survived.
    Column names are auto-detected among common aliases.
    """

    def __init__(self, *, labels_path: str, id_col: str | None = None, label_col: str | None = None, pred_col: str | None = None) -> None:
        self.labels_path = labels_path
        self.id_col = id_col
        self.label_col = label_col
        self.pred_col = pred_col
        self._labels = self._load_labels(labels_path)

    @staticmethod
    def _detect_cols(header: Dict[str, int]) -> tuple[str, str]:
        id_candidates = ["Id", "PassengerId", "id", "passengerid"]
        y_candidates = ["Target", "Survived", "y", "target"]
        id_col = next((c for c in id_candidates if c in header), None)
        y_col = next((c for c in y_candidates if c in header), None)
        if not id_col or not y_col:
            raise ValueError("Unable to detect id/label columns in labels file")
        return id_col, y_col

    def _load_labels(self, path: str) -> Dict[str, float]:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            cols = {c: i for i, c in enumerate(header)}
            id_col = self.id_col
            y_col = self.label_col
            if not id_col or not y_col:
                id_col, y_col = self._detect_cols(cols)
            labels: Dict[str, float] = {}
            for row in reader:
                pid = str(row[id_col]).strip()
                y = float(row[y_col]) if row[y_col] != "" else 0.0
                labels[pid] = y
        return labels

    async def run(self, artifacts: Dict[str, str], *, higher_is_better: bool) -> float:
        sub_path = artifacts.get("submission.csv")
        if not sub_path:
            return float("inf") if not higher_is_better else float("-inf")
        try:
            with open(sub_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
                cols = {c: i for i, c in enumerate(header)}
                id_col_use = self.id_col or id_col
                y_col_use = self.pred_col or y_col
                if not self.id_col or not self.pred_col:
                    id_col_use, y_col_use = self._detect_cols(cols)
                se = 0.0
                n = 0
                for row in reader:
                    pid = str(row.get(id_col_use, "")).strip()
                    if not pid or pid not in self._labels:
                        continue
                    try:
                        pred = float(row.get(y_col_use, 0.0))
                    except Exception:
                        continue
                    err = pred - self._labels[pid]
                    se += err * err
                    n += 1
            if n == 0:
                return float("inf") if not higher_is_better else float("-inf")
            mse = se / n
            # For MSE, lower is better → return negative if higher_is_better
            return -mse if higher_is_better else mse
        except Exception:
            return float("inf") if not higher_is_better else float("-inf")
