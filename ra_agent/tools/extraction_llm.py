from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os

from ..utils.logging_utils import timeit

try:
    # LangChain imports are optional until runtime
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain_anthropic import ChatAnthropic  # type: ignore
except Exception:  # pragma: no cover - optional dependencies
    ChatOpenAI = None
    ChatAnthropic = None


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 800


def _llm_from_env() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    return LLMConfig(provider=provider, model=model, temperature=0.0)


def _build_llm(cfg: LLMConfig):
    if cfg.provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain_openai not installed; please install to use LLM extraction")
        return ChatOpenAI(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    elif cfg.provider == "anthropic":
        if ChatAnthropic is None:
            raise RuntimeError("langchain_anthropic not installed; please install to use LLM extraction")
        return ChatAnthropic(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    else:
        # Default to OpenAI if unknown
        if ChatOpenAI is None:
            raise RuntimeError("Unsupported LLM provider and langchain_openai missing")
        return ChatOpenAI(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)


SYSTEM_PROMPT = (
    "You are a meticulous financial data extraction agent. Read the provided filing text and extract the requested metric. "
    "Do not guess. If not present, return found=false. Prefer values in the Summary Compensation Table when asked about CEO total compensation. "
    "Output strictly as a compact JSON object with keys: found (bool), value (string, raw as shown), currency (string or empty), year (string or empty), section (string or empty), evidence (short snippet)."
)


def _chunk(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    step = max_chars - 1000  # overlap
    i = 0
    while i < len(text):
        chunks.append(text[i : i + max_chars])
        i += step
    return chunks


@timeit
def llm_extract_metric(text: str, metric: str, hint: Optional[str] = None, form: Optional[str] = None) -> Dict[str, Any]:
    """Use an LLM to extract a metric from filing text without regex.

    Returns dict with keys: metric, value, context, section, currency, year, confidence (heuristic), raw_json.
    """
    cfg = _llm_from_env()
    llm = _build_llm(cfg)

    user_intro_parts = [
        f"Metric: {metric}",
        f"Filing type: {form or 'Unknown'}",
    ]
    if hint:
        user_intro_parts.append(f"Hint: {hint}")
    intro = "\n".join(user_intro_parts)

    # Process in chunks to stay within context
    best: Dict[str, Any] = {"found": False}
    for chunk in _chunk(text, max_chars=12000):
        prompt = (
            f"{intro}\n\nText (excerpt):\n" + chunk[:12000] + "\n\nReturn JSON only."
        )
        try:
            resp = llm.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            content = getattr(resp, "content", "") or str(resp)
        except Exception as e:  # pragma: no cover
            content = "{}"

        # Avoid strict JSON parsing to reduce failures; do simple extraction
        # Expect keys like: found, value, currency, year, section, evidence
        found_flag = "\"found\": true" in content.lower()
        if found_flag:
            # Heuristic: take this result as best and stop
            best = {
                "found": True,
                "raw_json": content,
            }
            break

    if not best.get("found"):
        return {
            "metric": metric,
            "value": "",
            "context": "",
            "section": "",
            "currency": "",
            "year": "",
            "confidence": 0.0,
        }

    # Lightweight field scraping from JSON-ish content
    def _grab(key: str) -> str:
        import re

        pat = rf'\"{key}\"\s*:\s*\"(.*?)\"'
        m = re.search(pat, best["raw_json"], flags=re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    value = _grab("value")
    currency = _grab("currency")
    year = _grab("year")
    section = _grab("section")
    evidence = _grab("evidence")

    return {
        "metric": metric,
        "value": value,
        "context": evidence,
        "section": section,
        "currency": currency,
        "year": year,
        "confidence": 0.8 if value else 0.0,
    }
