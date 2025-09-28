from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
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
    "You are a meticulous financial data extraction agent for SEC filings. Your task: find the requested metric."
    " Always examine the ENTIRE provided excerpt (and XML excerpt if present)."
    " 1. Never rely only on nearby hint terms — scan summary compensation tables, footnotes, narrative discussion, and preceding prior-year rows."
    " 2. If the requested year is not present, you MUST fall back to the most recent prior year available (e.g. request 2023 but table only shows 2022, 2021, 2020 -> choose 2022)."
    " 3. Prefer structured tables / XML tagged facts over narrative text."
    " 4. Do not fabricate values. If absolutely nothing relevant exists, set found=false."
    " Output strictly COMPACT JSON with keys: found (bool), requested_year (string), extracted_year (string), value (string), currency (string), section (string), evidence (short snippet), fallback_used (bool)."
)


def _chunk(text: str, max_chars: int, overlap: int) -> List[str]:
    """Split text into overlapping character chunks.

    Args:
        text: Full filing text.
        max_chars: Maximum characters per chunk.
        overlap: Characters of backward overlap between consecutive chunks.
    """
    if max_chars <= 0:
        return [text]
    if len(text) <= max_chars:
        return [text]
    step = max_chars - overlap
    if step <= 0:
        step = max_chars
    chunks: List[str] = []
    for i in range(0, len(text), step):
        chunks.append(text[i : i + max_chars])
    return chunks


def _extract_field(raw_json: str, keys: List[str]) -> str:
    """Extract first matching string field from raw JSON-ish text given list of candidate keys (case-insensitive)."""
    import re
    for k in keys:
        pat = rf'"{k}"\s*:\s*"(.*?)"'
        m = re.search(pat, raw_json, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def _parse_bool(raw_json: str, key: str) -> bool:
    import re, json
    # Quick direct parse attempt
    pat = rf'"{key}"\s*:\s*(true|false)'
    m = re.search(pat, raw_json, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    # fallback: attempt json loads if clean
    try:
        obj = json.loads(raw_json)
        if isinstance(obj, dict) and key in obj:
            return bool(obj[key])
    except Exception:
        pass
    return False


@timeit
def llm_extract_metric(
    text: str,
    metric: str,
    hint: Optional[str] = None,
    form: Optional[str] = None,
    xml: Optional[str] = None,
    requested_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Use an LLM to extract a metric from filing text without regex.

    Returns dict with keys: metric, value, context, section, currency, year, confidence (heuristic), raw_json.
    """
    cfg = _llm_from_env()
    llm = _build_llm(cfg)

    user_intro_parts = [
        f"Metric: {metric}",
        f"Filing type: {form or 'Unknown'}",
    ]
    if requested_year is not None:
        user_intro_parts.append(f"Requested year: {requested_year}")
    if hint:
        user_intro_parts.append(f"Hint: {hint}")
    intro = "\n".join(user_intro_parts)

    # Process in chunks to stay within context
    all_results: List[Dict[str, Any]] = []
    xml_excerpt = (xml or "").strip()
    if xml_excerpt:
        xml_excerpt = xml_excerpt[:6000]
    
    # Configurable chunking via env vars
    max_chars = int(os.getenv("LLM_CHUNK_SIZE", "12000"))
    overlap = int(os.getenv("LLM_CHUNK_OVERLAP", "1000"))

    for chunk in _chunk(text, max_chars=max_chars, overlap=overlap):
        if xml_excerpt:
            prompt = (
                f"{intro}\n\nIMPORTANT: Search this entire text excerpt thoroughly. The hint guides you but examine ALL tables, sections, data, and prior-year rows.\n"
                "If requested_year is absent, pick most recent earlier year.\n\n"
                f"XML (excerpt):\n{xml_excerpt}\n\nText (excerpt):\n" + chunk[:8000] + 
                "\n\nIf XML is present, prefer numeric facts from it. Search thoroughly before responding. Return JSON only."
            )
        else:
            prompt = (
                f"{intro}\n\nIMPORTANT: Search this entire text excerpt thoroughly. The hint guides you but examine ALL tables, sections, data, and prior-year rows.\n"
                "If requested_year is absent, pick most recent earlier year.\n\n"
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

    raw = best["raw_json"]
    value = _extract_field(raw, ["value"]) or ""
    currency = _extract_field(raw, ["currency", "curr", "unit"]) or ""
    extracted_year = _extract_field(raw, ["extracted_year", "year"]) or ""
    requested_year_str = str(requested_year) if requested_year is not None else _extract_field(raw, ["requested_year"]) or ""
    section = _extract_field(raw, ["section", "table", "source_section"]) or ""
    evidence = _extract_field(raw, ["evidence", "context", "snippet"]) or ""
    fallback_used = _parse_bool(raw, "fallback_used") or (
        bool(requested_year_str) and extracted_year and requested_year_str != extracted_year
    )

    # Confidence heuristic: value present + extracted_year present
    confidence = 0.9 if value and extracted_year else (0.7 if value else 0.0)

    return {
        "metric": metric,
        "value": value,
        "context": evidence,
        "section": section,
        "currency": currency,
        "requested_year": requested_year_str,
        "extracted_year": extracted_year,
        "fallback_used": fallback_used,
        "confidence": confidence,
    }
