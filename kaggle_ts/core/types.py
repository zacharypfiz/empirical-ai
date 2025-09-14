from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time


@dataclass
class Node:
    id: str
    parent_id: Optional[str]
    code: str
    score: Optional[float]
    visits: int = 0
    created_at: float = field(default_factory=lambda: time.time())
    logs: Dict[str, Any] = field(default_factory=dict)


class SolutionTree:
    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}

    def add(self, node: Node) -> None:
        self._nodes[node.id] = node

    def get(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def all(self) -> Dict[str, Node]:
        return self._nodes

    def best(self) -> Optional[Node]:
        scored = [n for n in self._nodes.values() if n.score is not None]
        if not scored:
            return None
        return max(scored, key=lambda n: n.score)  # higher is better convention

    def size(self) -> int:
        return len(self._nodes)

