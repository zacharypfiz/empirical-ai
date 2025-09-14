from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from ..core.types import Node


class RunStore:
    def __init__(self, base_dir: str | os.PathLike = "runs") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / "logs").mkdir(parents=True, exist_ok=True)
        self.nodes_path = self.base / "nodes.jsonl"

    def append_node(self, node: Node) -> None:
        rec = {
            "id": node.id,
            "parent_id": node.parent_id,
            "score": node.score,
            "visits": node.visits,
            "created_at": node.created_at,
            "code": node.code,
            "logs": node.logs,
        }
        with self.nodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def write_logs(self, node_id: str, *, stdout: str = "", stderr: str = "", meta: Dict[str, Any] | None = None) -> None:
        log_dir = self.base / "logs" / node_id
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
        (log_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")
        if meta is not None:
            (log_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

