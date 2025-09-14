from __future__ import annotations

from math import sqrt
from typing import Dict, List, Optional, Tuple

from .types import Node, SolutionTree


def _rank_scores(nodes: List[Node]) -> Dict[str, float]:
    # Map finite scores to [0,1] by rank; None treated as worst.
    pairs: List[Tuple[str, Optional[float]]] = [
        (n.id, n.score if n.score is not None else float("-inf")) for n in nodes
    ]
    # Sort ascending by score for rank assignment
    pairs.sort(key=lambda x: x[1])
    # Handle all -inf case
    finite = [s for _, s in pairs if s != float("-inf")]
    if not finite:
        return {nid: 0.0 for nid, _ in pairs}
    # Assign ranks 0..(len-1)
    rank: Dict[str, int] = {nid: i for i, (nid, _) in enumerate(pairs)}
    n = len(pairs)
    if n <= 1:
        return {nid: 0.0 for nid, _ in pairs}
    return {nid: rank[nid] / (n - 1) for nid, _ in pairs}


def select_parents(tree: SolutionTree, c_puct: float, k: int) -> List[Node]:
    nodes = list(tree.all().values())
    if not nodes:
        return []
    rank = _rank_scores(nodes)
    # Total visits across all nodes per the paper's algorithm
    n_total = sum(n.visits for n in nodes)
    if n_total <= 0:
        n_total = 1
    # Flat prior P_T(u) = 1/|T| per paper (scales c_puct consistently)
    flat_prior = 1.0 / max(1, len(nodes))
    scored: List[Tuple[float, Node]] = []
    for n in nodes:
        s_rank = rank.get(n.id, 0.0)
        prior = flat_prior
        bonus = c_puct * prior * sqrt(n_total) / (1.0 + n.visits)
        s = s_rank + bonus
        scored.append((s, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[: max(k, 0)]]
