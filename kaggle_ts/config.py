from __future__ import annotations

import os
from typing import Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _has_gemini() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def default_llm_provider():
    """Return the default LLM provider for general use (code-oriented).

    Uses provider-agnostic envs where possible:
      - CODE_MODEL (preferred), CODE_THINKING_BUDGET
    Falls back to provider-specific envs for compatibility.
    If no remote provider is configured, returns a stub provider.
    """
    return code_llm_provider()


def code_llm_provider():
    """Provider bucket for code generation (Pro-like by default)."""
    if not _has_gemini():
        from .agents.stubs import StubLLMProvider

        return StubLLMProvider(seed=42)
    from .providers.gemini import GeminiProvider

    # Provider-agnostic first
    model = os.getenv("CODE_MODEL")
    tb_env: Optional[str] = os.getenv("CODE_THINKING_BUDGET")
    # Back-compat fallbacks
    if not model:
        model = os.getenv("GEMINI_MODEL_CODE") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    if tb_env is None:
        tb_env = os.getenv("GEMINI_THINKING_BUDGET_CODE") or os.getenv("GEMINI_THINKING_BUDGET")
    thinking_budget = int(tb_env) if tb_env else None
    return GeminiProvider(model=model, thinking_budget=thinking_budget)


def idea_llm_provider():
    """Provider bucket for idea synthesis/recombination (Flash-like by default)."""
    if not _has_gemini():
        from .agents.stubs import StubLLMProvider

        return StubLLMProvider(seed=1337)
    from .providers.gemini import GeminiProvider

    model = os.getenv("IDEA_MODEL")
    tb_env: Optional[str] = os.getenv("IDEA_THINKING_BUDGET")
    if not model:
        model = os.getenv("GEMINI_MODEL_IDEA") or "gemini-2.5-flash"
    if tb_env is None:
        tb_env = os.getenv("GEMINI_THINKING_BUDGET_IDEA")
    thinking_budget = int(tb_env) if tb_env else None
    return GeminiProvider(model=model, thinking_budget=thinking_budget)


def embeddings_provider():
    if not _has_gemini():
        return None
    from .providers.gemini import GeminiEmbeddings

    model = os.getenv("EMBEDDING_MODEL") or os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001"
    return GeminiEmbeddings(model=model)
