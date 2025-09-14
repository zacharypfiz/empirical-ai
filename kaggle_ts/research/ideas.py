from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
import re
from pathlib import Path
from typing import List, Optional

from ..config import idea_llm_provider


@dataclass
class Idea:
    id: str
    title: str
    summary: str
    sources: list[str]


PROMPT = """
You are an expert Kaggle competitor.
Given the challenge description below, propose 5-10 distinct, high-impact modeling strategies.
Return output as JSON Lines (one object per line) with keys: "title", "summary", and optional "sources" (list of URLs).
Do not include any prose before or after the JSONL. Do not wrap in Markdown fences.

Challenge description:
---
{challenge}
---
""".strip()


def _normalize_idea_dict(d: dict) -> Optional[Idea]:
    try:
        title = str(d.get("title", "")).strip()
        summary = str(d.get("summary", "")).strip()
        sources = d.get("sources") or []
        if not title or not summary:
            return None
        if not isinstance(sources, list):
            sources = [str(sources)]
        sources = [str(s).strip() for s in sources if str(s).strip()]
        return Idea(id=uuid.uuid4().hex[:12], title=title, summary=summary, sources=sources)
    except Exception:
        return None


def _parse_jsonl(text: str) -> List[Idea]:
    ideas: List[Idea] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        idea = _normalize_idea_dict(obj)
        if idea:
            ideas.append(idea)
    return ideas


def _parse_fenced_json(text: str) -> List[Idea]:
    ideas: List[Idea] = []
    for block in re.findall(r"```(?:json)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        try:
            obj = json.loads(block)
            if isinstance(obj, list):
                for o in obj:
                    idea = _normalize_idea_dict(o)
                    if idea:
                        ideas.append(idea)
            elif isinstance(obj, dict):
                idea = _normalize_idea_dict(obj)
                if idea:
                    ideas.append(idea)
        except Exception:
            # Try JSONL inside the block
            ideas.extend(_parse_jsonl(block))
    return ideas


def _parse_markdown_like(text: str) -> List[Idea]:
    ideas: List[Idea] = []
    title: Optional[str] = None
    summary_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-* ").strip()
        if not line:
            continue
        m = re.match(r"(?i)^\*\*?\s*title\s*:?\s*(.+?)\*?\*$", line)
        if not m:
            m = re.match(r"(?i)^title\s*:\s*(.+)$", line)
        if m:
            if title and summary_lines:
                ideas.append(Idea(id=uuid.uuid4().hex[:12], title=title.strip(), summary=" ".join(summary_lines).strip(), sources=[]))
            title = m.group(1).strip()
            summary_lines = []
            continue
        m = re.match(r"(?i)^\*\*?\s*summary\s*:?\s*(.+)$", line)
        if not m:
            m = re.match(r"(?i)^summary\s*:\s*(.+)$", line)
        if m:
            summary_lines.append(m.group(1).strip())
            continue
        # Fallback bullet: "Title — Summary" or "Title - Summary"
        if title is None:
            mm = re.match(r"^(.*?)\s+[—\-]\s+(.*)$", line)
            if mm:
                ideas.append(Idea(id=uuid.uuid4().hex[:12], title=mm.group(1).strip(), summary=mm.group(2).strip(), sources=[]))
                continue
        if summary_lines:
            summary_lines.append(line)
    if title and summary_lines:
        ideas.append(Idea(id=uuid.uuid4().hex[:12], title=title.strip(), summary=" ".join(summary_lines).strip(), sources=[]))
    return ideas


async def synthesize_ideas(challenge_text: str, *, max_ideas: int = 10) -> List[Idea]:
    provider = idea_llm_provider()
    prompt = PROMPT.format(challenge=challenge_text)
    # Allow override via env IDEA_MAX_OUTPUT_TOKENS; default to 1024 for concise ideas
    import os
    idea_max = os.getenv("IDEA_MAX_OUTPUT_TOKENS")
    try:
        max_out = int(idea_max) if idea_max else 1024
    except Exception:
        max_out = 1024
    text = await provider.generate(prompt, max_tokens=max_out)
    # Parse in order of strictness
    ideas = _parse_jsonl(text)
    if not ideas:
        ideas = _parse_fenced_json(text)
    if not ideas:
        ideas = _parse_markdown_like(text)
    if not ideas:
        # Fallback: emit a few canonical Titanic strategies for offline/demo
        defaults = [
            ("Baseline: Sex/Pclass/Embarked features + Logistic", "Engineer basic features (Sex, Pclass, Embarked, SibSp/Parch bins), impute Age with median by Pclass/Sex, train LogisticRegression with stratified CV."),
            ("Tree Models: RandomForest/XGBoost", "One-hot encode categorical vars, impute Age/Fare, tune max_depth and n_estimators with CV, use class_weight or scale_pos_weight."),
            ("Family Size & Titles", "Extract Title from Name (Mr, Mrs, Miss, etc.), create FamilySize (SibSp+Parch+1), IsAlone, and TicketGroup size; model via GradientBoosting or XGBoost."),
            ("Cabin Deck + Fare Binning", "Parse Cabin deck letter, bin Fare (quantiles), create interaction terms (Pclass*Sex, Age*Pclass), try LightGBM with early stopping."),
            ("Stacking/Blending", "Blend Logistic, RandomForest, and XGBoost via Logistic meta-model with out-of-fold predictions; use 5-fold stratified CV and threshold tuning."),
        ]
        ideas = [Idea(id=uuid.uuid4().hex[:12], title=t, summary=s, sources=[]) for t, s in defaults]
    # Deduplicate by title and clip
    uniq: dict[str, Idea] = {}
    for idea in ideas:
        if idea.title not in uniq:
            uniq[idea.title] = idea
    return list(uniq.values())[:max_ideas]


def write_ideas_jsonl(path: str | Path, ideas: List[Idea]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for idea in ideas:
            f.write(json.dumps(asdict(idea), ensure_ascii=False) + "\n")
