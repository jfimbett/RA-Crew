from __future__ import annotations

"""Concrete Python tool functions exposed to CrewAI agents.

These tools provide deterministic, side-effect aware operations that agents can
invoke instead of hallucinating work. Tools are intentionally narrow and
return structured JSON-serialisable objects.

NOTE: Tools avoid performing network I/O directly except the retriever helper
which leverages existing SEC EDGAR utility functions that already implement
rate limiting and caching semantics.
"""
from typing import List, Dict, Any, Optional
import os

from .sec_edgar import (
    get_cik_for_ticker,
    list_company_filings,
    download_filing,
    html_to_text,
)
from .cleaning import clean_text, extract_tables
from .extraction_llm import llm_extract_metric, llm_extract_metric_multi
from .calculator import compute_metric, compute_derived_metrics
from .validation import validate_values, validate_extraction_evidence
from .exporter import export_rows
from .rag import SimpleRAG
from ..config import settings
from ..utils.identifiers import is_cik, normalize_cik, ticker_to_cik
from ..utils.logging_utils import logger


def make_tool(func, name: str, description: str):
    """Create a CrewAI BaseTool from a plain Python function.

    This avoids LangChain's StructuredTool so that CrewAI Agent(tools=...) accepts it.
    It infers a pydantic args_schema from the function signature and wires _run.
    """
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field, create_model
    import inspect as _inspect

    sig = _inspect.signature(func)
    annotations = getattr(func, "__annotations__", {})
    fields = {}
    for p in sig.parameters.values():
        if p.kind in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD):
            # Skip *args/**kwargs in schema; CrewAI tools expect explicit fields
            continue
        ann = annotations.get(p.name, Any)
        default = p.default if p.default is not _inspect._empty else ...
        fields[p.name] = (ann, Field(default))
    # Build args schema model
    ArgsModel: type[BaseModel] = create_model(f"{name}_args", **fields) if fields else create_model(f"{name}_args")

    # Define attributes for dynamic class creation, include _run abstract impl now
    def _run(self, **kwargs):  # type: ignore[no-untyped-def]
        return func(**kwargs)

    from pydantic import Field as _PydField
    attrs = {
        "__module__": __name__,
        "__annotations__": {
            "name": str,
            "description": str,
            "args_schema": type[BaseModel],
        },
        "name": name,
        "description": description,
        "args_schema": ArgsModel,
        "_run": _run,
    }
    PyFuncTool = type(f"{name}_Tool", (BaseTool,), attrs)  # type: ignore[misc]
    return PyFuncTool()


def tool_resolve_cik(ticker_or_cik: str) -> Dict[str, Any]:
    """Resolve an identifier to a 10-digit CIK."""
    """Resolve a user-provided ticker or CIK into a canonical CIK and metadata.

    Returns {input, cik, ticker, name} when available.
    """
    if is_cik(ticker_or_cik):
        cik = normalize_cik(ticker_or_cik)
    else:
        cik = ticker_to_cik(ticker_or_cik) or get_cik_for_ticker(ticker_or_cik) or ""
    return {"input": ticker_or_cik, "cik": cik, "resolved": bool(cik)}


def tool_list_filings(cik: str, years: List[int], filing_types: List[str]) -> Dict[str, Any]:
    """List company filings from SEC EDGAR for a CIK filtered by years and form types.

    Returns {cik, years, forms, count, filings[]} where filings are raw metadata records.
    """
    filings = list_company_filings(cik, years, filing_types)
    return {"cik": cik, "years": years, "forms": filing_types, "count": len(filings), "filings": filings}


def tool_download_and_clean_filings(cik: str, filings: List[Dict[str, Any]], limit: int = 3) -> Dict[str, Any]:
    """Download up to `limit` filings, persist cleaned text & tables, return manifest."""
    """Download a small batch of filings and return cleaned text payloads per filing.

    Returns {cik, count, items[{accession, form, year, text}]}.
    """
    out: List[Dict[str, Any]] = []
    os.makedirs(settings.data_dir, exist_ok=True)
    for filing in filings[:limit]:
        try:
            html = download_filing(cik, filing["accessionNumber"], filing["primaryDocument"])
            text = html_to_text(html)
            cleaned = clean_text(text)
            tables = extract_tables(html)
            cik_nozero = str(int(cik))
            acc_no = filing["accessionNumber"].replace("-", "")
            base_dir = os.path.join(settings.data_dir, "filings", cik_nozero, acc_no)
            os.makedirs(base_dir, exist_ok=True)
            text_path = os.path.join(base_dir, "cleaned.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            import orjson
            tables_path = os.path.join(base_dir, "tables.json")
            with open(tables_path, "wb") as f:
                f.write(orjson.dumps(tables))
            out.append({
                "accession_number": filing["accessionNumber"],
                "form": filing["form"],
                "filing_date": filing["filingDate"],
                "cleaned_path": text_path,
                "tables_path": tables_path,
            })
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"tool_download_and_clean_filings error: {e}")
            continue
    return {"cik": cik, "processed": len(out), "records": out}


def tool_rag_select_sections(texts: List[str], query: str) -> Dict[str, Any]:
    """Select relevant sections via a simple RAG windowing against provided texts for a query.

    Returns {query, sections[], characters}.
    """
    rag = SimpleRAG(chunk_size=2000, overlap=200)
    sections: List[str] = []
    for idx, text in enumerate(texts):
        ctx = rag.extract_context_for_query(text, {"doc_index": idx}, query)
        sections.append(ctx)
    return {"query": query, "sections": sections, "characters": sum(len(s) for s in sections)}


def tool_llm_extract(text: str, metric: str, hint: Optional[str] = None, form: Optional[str] = None) -> Dict[str, Any]:
    """Extract one metric from text using an LLM, returning multi-year mapping when present.

    Returns {metric, found, years{year->{value,name,title,section,currency,evidence}}, value, year, raw[]}.
    """
    # Prefer multi-year extraction; falls back to single-year shape via back-compat fields
    return llm_extract_metric_multi(text, metric, hint=hint, form=form)


def tool_compute_metric(formula: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a single numeric formula from primitive variables.

    Returns {formula, value, variables}.
    """
    return compute_metric(formula, variables)


def tool_compute_derived(calculation_expressions: Dict[str, str], primitive: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute derived metrics year-wise given expressions and primitive values.

    Returns {metrics{metric->{year->value}}}.
    """
    return compute_derived_metrics(calculation_expressions, primitive)


def tool_validate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate rows (identifier, metric, year, value, form) for plausibility and consistency."""
    return validate_values(rows)


def tool_validate_extraction(extracted: Dict[str, Any], cleaned_texts: List[str]) -> Dict[str, Any]:
    """Validate that each executive/value in extracted structure appears verbatim in cleaned_texts corpus."""
    return validate_extraction_evidence(extracted, cleaned_texts)


def tool_export(rows: List[Dict[str, Any]], fmt: str, name: str) -> Dict[str, Any]:
    """Export rows to JSON or CSV with a base filename, returning path and count."""
    path = export_rows(rows, fmt, name=name)
    return {"path": path, "format": fmt, "rows": len(rows)}


def _build_wrapped_map():
    return {
        "retriever": [
            make_tool(tool_resolve_cik, "resolve_cik", "Resolve ticker or raw CIK to normalized 10-digit CIK"),
            make_tool(tool_list_filings, "list_filings", "List SEC filings for CIK filtered by years & forms"),
            make_tool(tool_download_and_clean_filings, "download_and_clean", "Download and clean a limited set of filings"),
        ],
        "cleaner": [
            make_tool(tool_download_and_clean_filings, "download_and_clean", "Idempotent download/clean of filings"),
        ],
        "researcher": [
            make_tool(tool_rag_select_sections, "rag_select", "RAG select relevant sections for a query from texts"),
            make_tool(tool_llm_extract, "llm_extract", "LLM-based structured metric extraction from text"),
        ],
        "calculator": [
            make_tool(tool_compute_metric, "compute_metric", "Compute a single numeric formula from primitives"),
            make_tool(tool_compute_derived, "compute_derived", "Compute multiple derived metrics year-wise"),
        ],
        "validator": [
            make_tool(tool_validate, "validate_metrics", "Validate extracted/derived metrics for consistency"),
            make_tool(tool_validate_extraction, "validate_extraction_evidence", "Check that each extracted executive/value has verbatim evidence in corpus"),
        ],
        "exporter": [
            make_tool(tool_export, "export_results", "Export rows/metrics to JSON or CSV"),
        ],
        "coordinator": [],
        "analyst": [],
    }


AGENT_TOOL_MAP = _build_wrapped_map()
