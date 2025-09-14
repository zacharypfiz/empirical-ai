from __future__ import annotations

import csv
from typing import Iterable


def csv_has_columns(path: str, required: Iterable[str]) -> bool:
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        cols = {c.strip() for c in header}
        return all(col in cols for col in required)
    except Exception:
        return False

