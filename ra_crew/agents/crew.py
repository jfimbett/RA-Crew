from __future__ import annotations

"""CrewAI orchestration with dynamic metric-restricted prompts.

Rebuilt to remove hard‑coded CFO examples that previously caused unrequested
metrics to appear in results. Prompts now strictly enforce inclusion of only
the metrics a user specifies.
"""

from typing import Any, Dict, List, Optional
import os

from crewai import Agent, Task, Crew, Process  # type: ignore

from ..config import settings
from ..utils.logging_utils import setup_logging, logger
from ..tools.sec_edgar import (
    get_cik_for_ticker,
    list_company_filings,
    download_filing,
    html_to_text,
)
from ..tools.cleaning import clean_text, extract_tables
from ..tools.rag import SimpleRAG


# ---------------------------------------------------------------------------
# Hints loader
# ---------------------------------------------------------------------------
def _load_metric_hint(metric: str) -> Optional[str]:
    """Load a short hint string for a metric from hints/default.json if available."""
    try:
        hints_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "hints", "default.json")
        hints_path = os.path.normpath(hints_path)
        import json
        with open(hints_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        entry = payload.get(metric)
        if not entry:
            return None
        # Turn priority sections and keywords into a compact guidance string
        sections = entry.get("priority_sections") or []
        keywords = entry.get("keywords") or []
        notes = entry.get("notes") or ""
        parts: List[str] = []
        if sections:
            parts.append("priority sections: " + ", ".join(sections))
        if keywords:
            parts.append("keywords: " + ", ".join(keywords))
        if notes:
            parts.append(notes)
        return "; ".join(parts)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------
def _build_filing_url(cik: str, accession_no: str, primary_document: str) -> str:
    """Construct the public EDGAR URL for a filing document."""
    cik_nozero = str(int(cik))
    acc_no_nodash = accession_no.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{acc_no_nodash}/{primary_document}"
def retrieve_sec_data(ticker: str, years: List[int], filing_types: List[str]) -> Dict[str, Any]:
    ticker = ticker.upper()
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return {"error": f"No CIK found for {ticker}"}

    filings = list_company_filings(cik, years, filing_types)
    if not filings:
        return {"error": f"No filings of types {filing_types} for {ticker} in years {years}"}

    documents: List[Dict[str, Any]] = []
    for filing in filings[:3]:  # limit for performance
        try:
            html = download_filing(cik, filing["accessionNumber"], filing["primaryDocument"])
            text = clean_text(html_to_text(html))
            tables = extract_tables(html)
            url = _build_filing_url(cik, filing["accessionNumber"], filing["primaryDocument"])
            documents.append(
                {
                    "form": filing["form"],
                    "filing_date": filing["filingDate"],
                    "accession": filing["accessionNumber"],
                    "primary_document": filing["primaryDocument"],
                    "url": url,
                    "tables": tables,
                    "text": text,
                }
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to process filing {filing.get('accessionNumber')}: {e}")
            continue

    return {"cik": cik, "ticker": ticker, "documents": documents}


# ---------------------------------------------------------------------------
# Crew construction
# ---------------------------------------------------------------------------
def build_crew() -> Crew:
    """Construct Crew with dynamic prompts allowing only requested metrics."""
    setup_logging()

    researcher = Agent(
        role="Financial Data Researcher",
        goal=(
            "Extract ONLY requested financial metrics from SEC filings without adding extras; never invent values. "
            "Every number you output must be seen verbatim in the provided content or be an exact sum of visible components."
        ),
        backstory=(
            "You scan SEC filings (10-K, DEF 14A, etc.) to locate precise tabular or narrative data for "
            "user-specified metrics. If verification is not possible from the provided content, you set the value to null."
        ),
        allow_delegation=False,
        verbose=False,
    )

    analyst = Agent(
        role="Financial Data Validation Analyst",
        goal="Validate JSON strictly contains only requested metrics with accurate year/value pairs",
        backstory=(
            "You ensure structural correctness, strip formatting from numbers, remove unrequested metrics, "
            "and set absent metrics to null without invention."
        ),
        allow_delegation=False,
        verbose=False,
    )

    example_block = (
        "Example JSON (ONLY metrics in {metrics}):\n"
        "{\n"
        "  \"<REQUESTED METRIC NAME>\": {\n"
        "    \"<YEAR_A>\": {\n"
        "      \"value\": 1234567,\n"
        "      \"name\": \"Executive Name\",\n"
        "      \"title\": \"Full Title\",\n"
        "      \"source\": \"Table or section name\",\n"
        "      \"source_url\": \"https://www.sec.gov/Archives/...\",\n"
        "      \"page\": 42\n"
        "    },\n"
        "    \"<YEAR_B>\": {\n"
        "      \"value\": 1200000\n"
        "    }\n"
        "  }\n"
        "}"
    )

    research_task = Task(
        description=(
            "You are given relevant sections from ONE SEC filing (10-K, DEF 14A, etc.). Extract ONLY metrics in {metrics}.\n"
            "Context: ticker={ticker} years={years} types={filing_types} hint={hint}.\n"
            "Important year rule: The provided 'years' are ONLY used to select which filing(s) to open. Do NOT assume the metric must exist for that/those year(s).\n"
            "Instructions:\n"
            "1. Treat all provided snippets as one filing. Parse the ENTIRE filing content (tables + narrative) and reconstruct multi-line titles when needed.\n"
            "2. For each requested metric, include EXACTLY the fiscal years that are actually reported in THIS filing (e.g., current year and prior comparative years). Do not invent years.\n"
            "3. Prefer realized compensation totals from authoritative tables such as the 'Summary Compensation Table' or equivalent executive compensation tables.\n"
            "   - If a 'Total' column is explicitly shown for the CEO/PEO row, use it.\n"
            "   - If no explicit 'Total' is shown, compute it as an exact sum of clearly listed components in THIS filing (salary, bonus, stock/option awards, non-equity incentive, pension/change in pension value, all other). Use only numbers found in the same context.\n"
            "   - Do NOT use forward-looking, target, proposed, or 'future equity award decision' amounts as realized compensation. Only report values tied to past fiscal years disclosed in the filing.\n"
            "4. Numbers must be exact (no $, commas, or text). Every numeric value MUST appear verbatim in the content or be an exact sum of components from the same table/section. If you cannot verify a year's value, set it to null.\n"
            "5. If a requested metric is absent for ALL years in the filing, set that metric to null (do not list empty year objects).\n"
            "6. OUTPUT VALID JSON ONLY. Top-level keys must be EXACTLY the requested metrics.\n"
            "7. NEVER add extra metrics or years that are not present.\n\n"
            "FILING CONTENT START:\n{sec_filing_content}\nFILING CONTENT END.\n\n"
            "For each year entry you output, include 'source_url' (required) and 'page' if discernible from the context header; 'source' should be a brief table/section label when evident.\n"
            f"{example_block}\n"
            "Return ONLY the JSON."
        ),
        agent=researcher,
        expected_output="JSON mapping requested metrics -> {year: {value,...}} or null",
    )

    analysis_task = Task(
        description=(
            "Validate and clean the researcher JSON for EXACT keys in {metrics}.\n"
            "- The provided 'years' were ONLY used to select the filing; do not require those years in output.\n"
            "- Keep ONLY years that are evidenced in THIS filing's content.\n"
            "- Prefer totals from realized compensation tables (e.g., 'Summary Compensation Table').\n"
            "- Reject values that appear to come from forward-looking/target/proposed/'future equity award decision' sections unless they clearly state actual totals for past fiscal years.\n"
            "- Every numeric value must appear verbatim in the content or be an exact sum of listed components in the same context. If unverifiable, set to null.\n"
            "- Strip $ and commas. No extra text. Return VALID JSON only.\n\n"
            "FILING CONTENT START:\n{sec_filing_content}\nFILING CONTENT END."
        ),
        agent=analyst,
        expected_output="Filtered JSON limited to requested metrics",
        context=[research_task],
    )

    verbose_mode = os.getenv("CREW_VERBOSE", "false").lower() == "true"
    log_file = None
    if verbose_mode:
        os.makedirs(settings.outputs_dir, exist_ok=True)
        log_file = os.path.join(settings.outputs_dir, "crew_execution.log")
        logger.info(f"CrewAI execution will be logged to {log_file}")

    return Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=verbose_mode,
        output_log_file=log_file,
    )


# ---------------------------------------------------------------------------
# Wrapper with RAG + retrieval
# ---------------------------------------------------------------------------
class SECDataCrew:
    """Wrapper: retrieve filings, apply simple RAG, then run crew."""

    def __init__(self) -> None:
        self.crew = build_crew()

    def kickoff(self, inputs: Dict[str, Any]) -> str:
        ticker: str = inputs.get("ticker", "")
        years: List[int] = inputs.get("years", [])
        filing_types: List[str] = inputs.get("filing_types", [])
        metrics: List[str] = inputs.get("metrics", [])
        raw_output: bool = bool(inputs.get("raw_output", False))
        hint: str = inputs.get("hint", "")
        output_format: str = inputs.get("output_format", "json")
        retrieval: str = inputs.get("retrieval", "llm")

        logger.info(f"Retrieving filings for {ticker} types={filing_types} years={years} metrics={metrics}")
        sec_data = retrieve_sec_data(ticker, years, filing_types)
        if "error" in sec_data:
            return f"ERROR: {sec_data['error']}"

        relevant_sections: List[str] = []
        if retrieval == "llm":
            # Provide the full (cleaned) document content for each filing, plus compact table previews
            logger.info("Using direct LLM parsing (no RAG)")
            for idx, doc in enumerate(sec_data.get("documents", []), start=1):
                header = (
                    f"\n=== {doc['form']} Filing {doc['filing_date']} (Doc {idx}) ===\n"
                    f"URL: {doc.get('url','')}\n"
                    f"If you cite values from this section, include 'source_url' as this URL and the 'page' if visible.\n"
                )
                # Build table previews
                table_snippets: List[str] = []
                for t in (doc.get("tables") or [])[:8]:
                    cols = [c.lower() for c in t.get("columns", [])]
                    if any(
                        (
                            "compensation" in c or "total" in c or "pay" in c or "remuneration" in c or "ceo" in c or "chief executive" in c or "named executive" in c
                        )
                        for c in cols
                    ):
                        rows = t.get("rows", [])[:5]
                        import json as _json
                        table_snippets.append(_json.dumps({"columns": t.get("columns"), "rows": rows}))
                tables_section = ("\n\n[Tables JSON Preview]\n" + "\n".join(table_snippets)) if table_snippets else ""
                # Full cleaned text
                ctx = doc["text"]
                relevant_sections.append(header + ctx + tables_section)
        else:
            # Default: RAG-based focused context
            rag = SimpleRAG(chunk_size=2000, overlap=200)
            base_query = " ".join(metrics) if metrics else "compensation"
            aug_terms: List[str] = []
            mq = base_query.lower()
            if "compensation" in mq or "pay" in mq or "remuneration" in mq:
                aug_terms += [
                    "total",
                    "compensation",
                    "pay",
                    "remuneration",
                    "executive",
                    "named executive officers",
                    "proxy statement",
                ]
            if "ceo" in mq or "chief executive" in mq:
                aug_terms += [
                    "chief executive officer",
                    "principal executive officer",
                ]
            query_terms = base_query + (" " + " ".join(aug_terms) if aug_terms else "")
            if hint:
                query_terms += f" {hint}"
            for idx, doc in enumerate(sec_data.get("documents", []), start=1):
                logger.info(f"RAG scanning {doc['form']} {doc['filing_date']}")
                ctx = rag.extract_context_for_query(
                    text=doc["text"],
                    source_info={"form": doc["form"], "filing_date": doc["filing_date"], "ticker": ticker},
                    query=query_terms,
                )
                header = (
                    f"\n=== {doc['form']} Filing {doc['filing_date']} (Doc {idx}) ===\n"
                    f"URL: {doc.get('url','')}\n"
                    f"If you cite values from this section, include 'source_url' as this URL and the 'page' if visible.\n"
                )
                table_snippets: List[str] = []
                for t in (doc.get("tables") or [])[:8]:
                    cols = [c.lower() for c in t.get("columns", [])]
                    if any(
                        (
                            "compensation" in c or "total" in c or "pay" in c or "remuneration" in c or "ceo" in c or "chief executive" in c or "named executive" in c
                        )
                        for c in cols
                    ):
                        rows = t.get("rows", [])[:5]
                        import json as _json
                        table_snippets.append(_json.dumps({"columns": t.get("columns"), "rows": rows}))
                tables_section = ("\n\n[Tables JSON Preview]\n" + "\n".join(table_snippets)) if table_snippets else ""
                relevant_sections.append(header + ctx + tables_section)

        sec_filing_content = "\n".join(relevant_sections)

        # Helper: run crew once and parse JSON if possible
        def _run_and_parse(_inputs: Dict[str, Any]):
            logger.info(f"Executing crew using retrieval='{retrieval}' with hint={'yes' if _inputs.get('hint') else 'no'}")
            res = self.crew.kickoff(inputs=_inputs)
            try:
                import json as _json
                parsed = _json.loads(str(res))
            except Exception:
                parsed = None
            return res, parsed

        # Quality heuristic: any requested metric has at least one non-null value
        def _has_meaningful_values(parsed_json: Any) -> bool:
            try:
                if isinstance(parsed_json, list) and parsed_json and isinstance(parsed_json[0], dict):
                    obj = parsed_json[0]
                elif isinstance(parsed_json, dict):
                    obj = parsed_json
                else:
                    return False
                for m in metrics:
                    v = obj.get(m)
                    if isinstance(v, dict):
                        for y, vv in v.items():
                            if isinstance(vv, dict):
                                if vv.get("value") not in (None, "", []):
                                    return True
                            elif vv not in (None, "", []):
                                return True
                    elif v not in (None, "", []):
                        return True
                return False
            except Exception:
                return False

        # First pass: use user-provided hint if present; otherwise no hint
        user_provided_hint = bool(hint)
        crew_inputs = {**inputs, "sec_filing_content": sec_filing_content, "hint": hint}
        result, parsed = _run_and_parse(crew_inputs)

        # If poor quality and no user hint, try metric-based hint on second pass
        if not user_provided_hint and not _has_meaningful_values(parsed):
            metric_hint_parts: List[str] = []
            for m in metrics or []:
                h = _load_metric_hint(m)
                if h:
                    metric_hint_parts.append(h)
            combined_hint = "; ".join(metric_hint_parts)
            if combined_hint:
                logger.info("First pass had no meaningful values; retrying with built-in metric hints")
                crew_inputs2 = {**crew_inputs, "hint": combined_hint}
                result2, parsed2 = _run_and_parse(crew_inputs2)
                # Prefer second result if it looks better
                if _has_meaningful_values(parsed2):
                    result, parsed = result2, parsed2
                else:
                    logger.info("Hinted second pass did not improve results; keeping first pass output")
        logger.info("Crew execution complete")

        # Persist result
        try:
            import json, orjson  # type: ignore
            os.makedirs(settings.outputs_dir, exist_ok=True)
            base_name = f"crew_results_{ticker}_{'_'.join(map(str, years))}"
            output_file = os.path.join(settings.outputs_dir, f"{base_name}.{output_format}")
            try:
                parsed = json.loads(str(result))
            except json.JSONDecodeError:
                parsed = None
            # Filter to only requested metrics and flatten rows
            def _flatten(filtered: Dict[str, Any]) -> List[Dict[str, Any]]:
                rows: List[Dict[str, Any]] = []
                doc_urls = [d.get("url") for d in sec_data.get("documents", []) if d.get("url")]
                single_doc_url = doc_urls[0] if len(doc_urls) == 1 else None
                for metric in metrics:
                    data = filtered.get(metric)
                    if data is None:
                        rows.append({
                            "ticker": ticker,
                            "cik": sec_data.get("cik"),
                            "metric": metric,
                            "year": None,
                            "value": None,
                            "name": None,
                            "title": None,
                            "source": single_doc_url or None,
                            "source_url": single_doc_url or None,
                            "page": None,
                        })
                        continue
                    if isinstance(data, dict):
                        for year_str, obj in data.items():
                            try:
                                year_i = int(year_str)
                            except Exception:
                                year_i = None
                            if isinstance(obj, dict):
                                src_url = obj.get("source_url") or single_doc_url
                                src = obj.get("source") or src_url
                                rows.append({
                                    "ticker": ticker,
                                    "cik": sec_data.get("cik"),
                                    "metric": metric,
                                    "year": year_i,
                                    "value": obj.get("value"),
                                    "name": obj.get("name"),
                                    "title": obj.get("title"),
                                    "source": src,
                                    "source_url": src_url,
                                    "page": obj.get("page"),
                                })
                            else:
                                # Year present and mapped to a scalar (e.g., number) or null
                                src_url = single_doc_url
                                val = None
                                if isinstance(obj, (int, float)):
                                    val = obj
                                elif isinstance(obj, str):
                                    # Best-effort numeric coercion
                                    s = obj.replace(",", "").replace("$", "").strip()
                                    try:
                                        val = float(s) if "." in s else int(s)
                                    except Exception:
                                        val = obj  # keep raw string
                                rows.append({
                                    "ticker": ticker,
                                    "cik": sec_data.get("cik"),
                                    "metric": metric,
                                    "year": year_i,
                                    "value": val,
                                    "name": None,
                                    "title": None,
                                    "source": src_url,
                                    "source_url": src_url,
                                    "page": None,
                                })
                    # else ignore unexpected shapes
                return rows

            flattened_rows: List[Dict[str, Any]] = []
            if isinstance(parsed, dict):
                # keep only requested metrics
                filtered = {k: v for k, v in parsed.items() if k in set(metrics or [])}
                flattened_rows = _flatten(filtered)
            elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                # some models return a list with one dict
                filtered = {k: v for k, v in parsed[0].items() if k in set(metrics or [])}
                flattened_rows = _flatten(filtered)
            else:
                logger.warning("Crew result not JSON or not in expected shape; saving raw text only")

            # Write minimal output
            if output_format == "json":
                with open(output_file, "wb") as f:
                    f.write(orjson.dumps(flattened_rows, option=orjson.OPT_INDENT_2))
            elif output_format == "csv":
                import pandas as pd  # type: ignore
                pd.DataFrame(flattened_rows).to_csv(output_file, index=False)
            logger.info(f"Saved minimal flattened output to {output_file}")

            # Optional raw sidecar
            if raw_output:
                raw_path = os.path.join(settings.outputs_dir, f"{base_name}.raw.txt")
                with open(raw_path, "w", encoding="utf-8") as fh:
                    fh.write(str(result))
                logger.info(f"Saved raw crew output to {raw_path}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to persist crew output: {e}")

        return result


__all__ = ["build_crew", "SECDataCrew"]
