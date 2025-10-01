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
    """Construct Crew with seven agents in a sequential pipeline.

    Agents:
    - Graduate Assistant Supervisor (orchestrates and plans)
    - Data Retriever (retrieve SEC filings)
    - Data Cleaner (normalize text and parse tables)
    - Financial Data Extractor (extract requested metrics from content)
    - Data Calculator (compute exact totals or derived values when needed)
    - Financial Data Validation Analyst (validate/clean JSON strictly)
    - Data Exporter (prepare/save final output)
    """
    setup_logging()

    # Orchestrator
    supervisor = Agent(
        role="Graduate Assistant Supervisor",
        goal="Plan and oversee the data pipeline from retrieval to export; coordinate agents to achieve the goal.",
        backstory=(
            "You are a diligent graduate assistant orchestrating a research pipeline. "
            "You design the steps, ensure hand-offs between agents, and keep scope focused on requested metrics."
        ),
        allow_delegation=True,
        verbose=False,
    )

    # Retrieval & cleaning stages (tool-backed by Python functions but represented as agents)
    retriever = Agent(
        role="Data Retriever",
        goal="Retrieve relevant SEC filings (HTML) for the given ticker, years, and filing types.",
        backstory="You locate and fetch filings content and metadata from EDGAR.",
        allow_delegation=False,
        verbose=False,
    )

    cleaner = Agent(
        role="Data Cleaner",
        goal="Normalize HTML to text and extract relevant tables for later parsing.",
        backstory="You convert noisy HTML to clean text and compact table previews.",
        allow_delegation=False,
        verbose=False,
    )

    # Extraction, calculation, validation, export
    extractor = Agent(
        role="Financial Data Extractor",
        goal=(
            "Extract ONLY requested metrics from filings content; never invent values; "
            "use exact numbers seen or precise sums of visible components."
        ),
        backstory="You carefully read SEC filings and isolate just the requested metrics.",
        allow_delegation=False,
        verbose=False,
    )

    calculator = Agent(
        role="Data Calculator",
        goal="Where totals are not explicit, compute exact sums from listed components from the same context.",
        backstory="You aggregate components into exact totals when appropriate (no estimates).",
        allow_delegation=False,
        verbose=False,
    )

    validator = Agent(
        role="Financial Data Validation Analyst",
        goal="Validate JSON strictly contains only requested metrics with accurate year/value pairs",
        backstory=(
            "You ensure structural correctness, strip formatting, remove unrequested metrics, and set absent metrics to null."
        ),
        allow_delegation=False,
        verbose=False,
    )

    exporter = Agent(
        role="Data Exporter",
        goal="Prepare final flattened rows and save to the requested format (JSON/CSV).",
        backstory="You finalize outputs for downstream analysis and storage.",
        allow_delegation=False,
        verbose=False,
    )

    # Shared example block for JSON shape
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

    # Tasks
    plan_task = Task(
        description=(
            "Create a concise plan for the pipeline given: ticker={ticker} years={years} types={filing_types} metrics={metrics}.\n"
            "Steps: retrieve -> clean -> extract -> calculate -> validate -> export. Focus only on requested metrics. Return a short checklist."
        ),
        agent=supervisor,
        expected_output="Short plan with ordered steps and notes",
    )

    retrieve_task = Task(
        description=(
            "Retrieve relevant filings for ticker={ticker}, years={years}, types={filing_types}.\n"
            "If available, use provided seed context: {sec_retrieval_hint}.\n"
            "Return a compact JSON with a 'documents' array (form, filing_date, url, primary_document)."
        ),
        agent=retriever,
        expected_output="JSON with documents metadata",
        context=[plan_task],
    )

    clean_task = Task(
        description=(
            "Convert HTML to clean text and extract compact table previews for each document.\n"
            "Use provided raw content if present. Return a JSON mirroring 'documents' but with 'text' and 'tables' fields populated."
        ),
        agent=cleaner,
        expected_output="JSON with documents including text and tables",
        context=[retrieve_task],
    )

    extract_task = Task(
        description=(
            "You are given relevant sections from ONE SEC filing (10-K, DEF 14A, etc.). Extract ONLY metrics in {metrics}.\n"
            "Context: ticker={ticker} years={years} types={filing_types} hint={hint}.\n"
            "Important year rule: The provided 'years' are ONLY used to select which filing(s) to open. Do NOT assume the metric must exist for that/those year(s).\n"
            "Instructions:\n"
            "1. Treat all provided snippets as one filing. Parse the ENTIRE filing content (tables + narrative) and reconstruct multi-line titles when needed.\n"
            "2. For each requested metric, include EXACTLY the fiscal years that are actually reported in THIS filing. Do not invent years.\n"
            "3. Prefer realized compensation totals from authoritative tables such as the 'Summary Compensation Table' or equivalent tables.\n"
            "4. Numbers must be exact (no $, commas, or text). Values must appear verbatim or be exact sums of components from the same context.\n"
            "5. If a requested metric is absent for ALL years, set that metric to null.\n"
            "6. OUTPUT VALID JSON ONLY. Top-level keys must be EXACTLY the requested metrics.\n\n"
            "FILING CONTENT START:\n{sec_filing_content}\nFILING CONTENT END.\n\n"
            "For each year entry include 'source_url' and 'page' if discernible; 'source' can be a brief table/section label.\n"
            f"{example_block}\n"
            "Return ONLY the JSON."
        ),
        agent=extractor,
        expected_output="JSON mapping requested metrics -> {year: {value,...}} or null",
        context=[clean_task],
    )

    calculate_task = Task(
        description=(
            "Given the extracted JSON, where totals are missing but components exist, compute exact sums strictly from the same context.\n"
            "Do NOT introduce new years or metrics. Return JSON in the same shape, updated only where sums are certain."
        ),
        agent=calculator,
        expected_output="JSON with exact totals filled where applicable",
        context=[extract_task],
    )

    validate_task = Task(
        description=(
            "Validate and clean the extracted JSON for EXACT keys in {metrics}.\n"
            "- The 'years' were used only to select the filing; do not force them in output.\n"
            "- Keep ONLY years evidenced in THIS filing.\n"
            "- Prefer realized compensation totals; reject forward-looking/target amounts.\n"
            "- Every number must appear verbatim or be an exact sum of listed components in the same context.\n"
            "- Strip $ and commas. Return VALID JSON only.\n\n"
            "FILING CONTENT START:\n{sec_filing_content}\nFILING CONTENT END."
        ),
        agent=validator,
        expected_output="Filtered JSON limited to requested metrics",
        context=[calculate_task],
    )

    export_task = Task(
        description=(
            "Prepare for export without altering content. Do not add metrics beyond {metrics}.\n"
            "Return EXACTLY the validated JSON from the previous step with no extra narration or text."
        ),
        agent=exporter,
        expected_output="VALID JSON identical to validated output",
        context=[validate_task],
    )

    verbose_mode = os.getenv("CREW_VERBOSE", "false").lower() == "true"
    log_file = None
    if verbose_mode:
        os.makedirs(settings.outputs_dir, exist_ok=True)
        log_file = os.path.join(settings.outputs_dir, "crew_execution.log")
        logger.info(f"CrewAI execution will be logged to {log_file}")

    return Crew(
        agents=[supervisor, retriever, cleaner, extractor, calculator, validator, exporter],
        tasks=[plan_task, retrieve_task, clean_task, extract_task, calculate_task, validate_task, export_task],
        process=Process.sequential,
        verbose=verbose_mode,
        output_log_file=log_file,
    )


# ---------------------------------------------------------------------------
# Wrapper with RAG + retrieval
# ---------------------------------------------------------------------------
class SECDataCrew:
    """Wrapper: retrieve filings, build full-document context, then run crew (no RAG)."""

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
        # RAG removed: always build full-document context

        logger.info(f"Retrieving filings for {ticker} types={filing_types} years={years} metrics={metrics}")
        sec_data = retrieve_sec_data(ticker, years, filing_types)
        if "error" in sec_data:
            return f"ERROR: {sec_data['error']}"

        relevant_sections: List[str] = []
        logger.info("Building full-document context (RAG disabled)")
        for idx, doc in enumerate(sec_data.get("documents", []), start=1):
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
            ctx = doc["text"]
            relevant_sections.append(header + ctx + tables_section)

        sec_filing_content = "\n".join(relevant_sections)

        # Helper: run crew once and parse JSON if possible
        def _run_and_parse(_inputs: Dict[str, Any]):
            logger.info(f"Executing crew (RAG disabled) with hint={'yes' if _inputs.get('hint') else 'no'}")
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
