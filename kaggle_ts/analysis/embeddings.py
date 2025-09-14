from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from ..config import embeddings_provider


def _iter_nodes_jsonl(path: str | Path):
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                nid = rec.get("id")
                code = rec.get("code")
                if nid and code:
                    yield nid, code
            except Exception:
                continue


async def embed_nodes(nodes_jsonl: str = "runs/nodes.jsonl", out_jsonl: str = "runs/embeddings.jsonl", batch_size: int = 16) -> int:
    provider = embeddings_provider()
    if provider is None:
        return 0
    items = list(_iter_nodes_jsonl(nodes_jsonl))
    if not items:
        return 0
    total = 0
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            texts = [code for _, code in batch]
            vecs: List[List[float]] = await provider.embed(texts)
            for (nid, _), v in zip(batch, vecs):
                out.write(json.dumps({"id": nid, "embedding": v}) + "\n")
                total += 1
    return total

