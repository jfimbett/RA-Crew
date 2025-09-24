from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from crewai import Agent, Task, Crew, Process

from ..config import settings
from ..utils.logging_utils import setup_logging
from ..tools.sec_edgar import get_cik_for_ticker, list_company_filings, download_filing, html_to_text
from ..tools.cleaning import clean_text
from ..tools.extraction import extract_metric
from ..tools.calculator import compute_metric
from ..tools.validation import validate_values
from ..tools.exporter import export_rows


def _llm_config() -> Dict[str, Any]:
    # Keep flexible provider choice; defer concrete instantiation to CrewAI defaults via env
    # Users can configure provider via environment variables
    return {
        "temperature": 0.1,
        "model": os.getenv("MODEL_NAME", "gpt-4o-mini"),  # default if using OpenAI
    }


import os


def _agent(name: str, role: str, goal: str, backstory: str, tools: Optional[List[Any]] = None, allow_delegation: bool = False) -> Agent:
    return Agent(
        name=name,
        role=role,
        goal=goal,
        backstory=backstory,
        llm=_llm_config(),
        verbose=True,
        tools=tools or [],
        allow_delegation=allow_delegation,
    )


def build_crew() -> Crew:
    setup_logging()

    data_retriever = _agent(
        name="DataRetriever",
        role="Retrieve SEC EDGAR filings",
        goal=(
            "Connect to SEC EDGAR, respect rate limits (10 req/s), use headers with identity, "
            "fetch filings for tickers/years and convert HTML to text."
        ),
        backstory=(
            "Expert at SEC EDGAR endpoints, pagination, and robust retrieval, returning raw and text versions."
        ),
    )

    data_cleaner = _agent(
        name="DataCleaner",
        role="Clean and normalize text",
        goal="Strip HTML residue, normalize whitespace, remove boilerplate, segment sections, save JSON/CSV.",
        backstory="Experienced data wrangler for financial filings.",
    )

    data_extractor = _agent(
        name="DataExtractor",
        role="Extract variables and facts",
        goal=(
            "Given cleaned text and a target variable/metric, locate values using hybrid search (BM25/RAG) and regex."
        ),
        backstory="Finds needles in haystacks across 10-K/Q and proxies.",
    )

    data_calculator = _agent(
        name="DataCalculator",
        role="Compute metrics",
        goal="Use Python to compute requested metrics strictly via code, not mental math.",
        backstory="Careful with units and numeric precision.",
    )

    data_validator = _agent(
        name="DataValidator",
        role="Validate extracted data",
        goal="Check reasonableness, units, missingness, duplicates across companies/periods.",
        backstory="Auditor mindset.",
    )

    data_exporter = _agent(
        name="DataExporter",
        role="Export final data",
        goal="Write outputs as JSON or CSV as requested.",
        backstory="Organized and consistent file outputs.",
    )

    graduate_assistant = _agent(
        name="GraduateAssistant",
        role="Orchestrate crew",
        goal=(
            "Delegate tasks to retrieve, clean, extract, compute, validate, and export data across companies and years."
        ),
        backstory="Manages multi-company, multi-period workflows with error handling and retries.",
        allow_delegation=True,
    )

    # Define high-level placeholder tasks; the CLI will feed inputs
    def _run_retrieve(inputs: Dict[str, Any]) -> Dict[str, Any]:
        ticker: str = inputs["ticker"]
        years: list[int] = inputs["years"]
        filing_types: list[str] = inputs["filing_types"]
        out: list[dict] = []
        cik = get_cik_for_ticker(ticker)
        if not cik:
            return {"filings": out, "error": f"No CIK for {ticker}"}
        filings = list_company_filings(cik, years, filing_types)
        for f in filings:
            html = download_filing(cik, f["accessionNumber"], f["primaryDocument"])
            text = html_to_text(html)
            # Persist to data folder
            cik_nozero = str(int(cik))
            acc_no = f["accessionNumber"].replace("-", "")
            base_dir = os.path.join(settings.data_dir, "filings", cik_nozero, acc_no)
            os.makedirs(base_dir, exist_ok=True)
            html_path = os.path.join(base_dir, f["primaryDocument"])
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(html)
            out.append({"meta": {**f, "cik": cik, "paths": {"html": html_path}}, "html": html, "text": text})
        return {"filings": out}

    t_retrieve = Task(
        description=(
            "For each company and year, find target filings and return text. Ensure SEC headers include user identity: "
            f"{settings.edgar_identity}. Respect 10 req/s rate limit."
        ),
        agent=data_retriever,
        expected_output="Raw HTML and plain text per filing.",
        context={"runner": _run_retrieve},
    )

    def _run_clean(prev: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: list[dict] = []
        for doc in prev.get("filings", []):
            cleaned.append({"meta": doc["meta"], "text": clean_text(doc["text"])})
        return {"cleaned": cleaned}

    t_clean = Task(
        description="Clean and normalize text, remove HTML noise, segment sections, store intermediate artifacts.",
        agent=data_cleaner,
        expected_output="Cleaned text with section markers in JSON.",
        context={"runner": _run_clean},
    )

    def _run_extract(prev: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        metrics: list[str] = inputs.get("metrics", [])
        rows: list[dict] = []
        for doc in prev.get("cleaned", []):
            text = doc["text"]
            meta = doc["meta"]
            for m in metrics:
                res = extract_metric(text, m)
                rows.append(
                    {
                        "ticker": inputs["ticker"],
                        "year": int(meta["filingDate"][:4]),
                        "metric": res["metric"],
                        "value": res["value"],
                        "context": res["context"],
                        "form": meta["form"],
                    }
                )
        return {"rows": rows}

    t_extract = Task(
        description="Extract requested variables or needed components for a metric using hybrid search and regex.",
        agent=data_extractor,
        expected_output="Key-value pairs with provenance (section, page, snippet).",
        context={"runner": _run_extract},
    )

    def _run_calc(prev: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder: no bespoke formulas provided yet; pass-through
        return prev

    t_calc = Task(
        description="Compute metrics via Python code execution based on extracted variables, handling units.",
        agent=data_calculator,
        expected_output="Computed metric values with formula and inputs.",
        context={"runner": _run_calc},
    )

    def _run_validate(prev: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        report = validate_values(prev.get("rows", []))
        return {**prev, "validation": report}

    t_validate = Task(
        description="Validate values, check units, detect outliers and duplicates across firm-periods.",
        agent=data_validator,
        expected_output="Validation report with flags and clean dataset.",
        context={"runner": _run_validate},
    )

    def _run_export(prev: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        fmt = inputs.get("output_format", "json")
        path = export_rows(prev.get("rows", []), fmt, name=f"results_{inputs['ticker']}_{inputs['years'][0]}")
        return {**prev, "output_path": path}

    t_export = Task(
        description="Export outputs to JSON/CSV as requested with consistent schema.",
        agent=data_exporter,
        expected_output="Saved files paths for final dataset and logs.",
        context={"runner": _run_export},
    )

    crew = Crew(
        agents=[
            graduate_assistant,
            data_retriever,
            data_cleaner,
            data_extractor,
            data_calculator,
            data_validator,
            data_exporter,
        ],
        tasks=[t_retrieve, t_clean, t_extract, t_calc, t_validate, t_export],
        process=Process.sequential,
        verbose=True,
    )
    return crew
