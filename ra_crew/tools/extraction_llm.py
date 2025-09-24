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
    "You are a meticulous financial data extraction agent. Your job is to thoroughly search the ENTIRE document for the requested metric. "
    "The hint provided is guidance to help you understand what to look for, but you must examine all relevant sections, tables, and data throughout the document. "
    "Do NOT just grab the first number you see near the hint words. Look for the most complete, accurate, and contextually appropriate value. "
    "Search compensation tables, summary tables, footnotes, and detailed breakdowns. "
    "If multiple values exist for the same metric (e.g., different years), extract the most recent or specifically requested one. "
    "Do not guess. If genuinely not present after thorough search, return found=false. "
    "Output strictly as a compact JSON object with keys: found (bool), value (string, raw as shown), currency (string or empty), year (string or empty), section (string or empty), evidence (short snippet showing context)."
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
def llm_extract_metric(text: str, metric: str, hint: Optional[str] = None, form: Optional[str] = None, xml: Optional[str] = None) -> Dict[str, Any]:
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
    all_results: List[Dict[str, Any]] = []
    xml_excerpt = (xml or "").strip()
    if xml_excerpt:
        xml_excerpt = xml_excerpt[:6000]
    
    for chunk in _chunk(text, max_chars=12000):
        if xml_excerpt:
            prompt = (
                f"{intro}\n\nIMPORTANT: Search this entire text excerpt thoroughly. The hint guides you but examine ALL tables, sections, and data.\n\n"
                f"XML (excerpt):\n{xml_excerpt}\n\nText (excerpt):\n" + chunk[:8000] + 
                "\n\nIf XML is present, prefer numeric facts from it. Search thoroughly before responding. Return JSON only."
            )
        else:
            prompt = (
                f"{intro}\n\nIMPORTANT: Search this entire text excerpt thoroughly. The hint guides you but examine ALL tables, sections, and data.\n\n"
                f"Text (excerpt):\n" + chunk[:12000] + 
                "\n\nSearch thoroughly before responding. Return JSON only."
            )
        try:
            resp = llm.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            content = getattr(resp, "content", "") or str(resp)
        except Exception as e:  # pragma: no cover
            content = "{}"

        # Check if this chunk found something
        found_flag = "\"found\": true" in content.lower()
        if found_flag:
            all_results.append({
                "found": True,
                "raw_json": content,
                "chunk_index": len(all_results)
            })
    
    # If multiple results, pick the one with the most detailed evidence or most recent
    best: Dict[str, Any] = {"found": False}
    if all_results:
        # For now, take the last result found (often more complete/recent)
        # TODO: Could implement more sophisticated selection logic
        best = all_results[-1]

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

    # Derive evidence offsets if possible
    evidence_start = -1
    evidence_end = -1
    evidence_snippet = evidence
    if evidence:
        try:
            # Case-insensitive search first, then refine for exact casing if found
            lowered_text = text.lower()
            lowered_e = evidence.lower()
            idx = lowered_text.find(lowered_e)
            if idx != -1:
                evidence_start = idx
                evidence_end = idx + len(evidence)
        except Exception:  # pragma: no cover
            pass

    return {
        "metric": metric,
        "value": value,
        "context": evidence,  # backward compatibility
        "evidence_snippet": evidence_snippet,
        "evidence_start": evidence_start,
        "evidence_end": evidence_end,
        "section": section,
        "currency": currency,
        "year": year,
        "confidence": 0.8 if value else 0.0,
    }
