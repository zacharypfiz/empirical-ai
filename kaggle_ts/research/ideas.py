from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
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
Each strategy should include a short title and a 2-4 sentence summary with concrete actions.
Avoid generic advice; be specific (features, models, validation approach).

Challenge description:
---
{challenge}
---

Output as bullet points with 'Title: ...' and 'Summary: ...'.
""".strip()


async def synthesize_ideas(challenge_text: str, *, max_ideas: int = 10) -> List[Idea]:
    provider = idea_llm_provider()
    prompt = PROMPT.format(challenge=challenge_text)
    text = await provider.generate(prompt, max_tokens=2048)
    # Simple parse: look for lines starting with Title: / Summary:
    ideas: List[Idea] = []
    title: Optional[str] = None
    summary_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip("- ").strip()
        if not line:
            continue
        if line.lower().startswith("title:"):
            # flush previous
            if title and summary_lines:
                ideas.append(
                    Idea(id=uuid.uuid4().hex[:12], title=title.strip(), summary=" ".join(summary_lines).strip(), sources=[])
                )
            title = line.split(":", 1)[1].strip()
            summary_lines = []
        elif line.lower().startswith("summary:"):
            summary_lines.append(line.split(":", 1)[1].strip())
        else:
            if summary_lines:
                summary_lines.append(line)
    if title and summary_lines:
        ideas.append(Idea(id=uuid.uuid4().hex[:12], title=title.strip(), summary=" ".join(summary_lines).strip(), sources=[]))
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
    return ideas[:max_ideas]


def write_ideas_jsonl(path: str | Path, ideas: List[Idea]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for idea in ideas:
            f.write(json.dumps(asdict(idea)) + "\n")
