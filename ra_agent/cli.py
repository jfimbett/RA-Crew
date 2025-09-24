from __future__ import annotations

import os
from typing import Optional, List
import typer
from tqdm import tqdm
from rich import print
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from .config import settings
from .utils.logging_utils import setup_logging
from .tools.sec_edgar import get_cik_for_ticker, list_company_filings, download_filing, html_to_text, is_xml_content
from .tools.cleaning import clean_text, extract_tables
from .tools.extraction_llm import llm_extract_metric
from .tools.validation import validate_values
from .tools.exporter import export_rows
from .utils.identifiers import is_cik, normalize_cik, ticker_to_cik
from loguru import logger


app = typer.Typer(add_completion=False)


def _parse_companies(companies: Optional[str]) -> List[tuple[str, int]]:
    pairs: List[tuple[str, int]] = []
    if not companies:
        return pairs
    for item in companies.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            ticker, year = item.split(":", 1)
            pairs.append((ticker.strip(), int(year)))
        else:
            raise typer.BadParameter("Expected format TICKER:YEAR, e.g., AAPL:2023")
    return pairs


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


@app.command()
def main(
    companies: Optional[str] = typer.Option(
        None, help="Comma-separated list like 'AAPL:2023,MSFT:2022'"
    ),
    companies_file: Optional[str] = typer.Option(
        None, help="Path to file with 'TICKER:YEAR' per line"
    ),
    filings: str = typer.Option("10-K", help="Comma-separated filing types, e.g., 10-K,DEF 14A"),
    metrics: Optional[str] = typer.Option(None, help="Comma-separated metrics to extract"),
    metrics_file: Optional[str] = typer.Option(None, help="Path to a file with metrics, one per line"),
    output_format: str = typer.Option("json", help="json or csv"),
    hint: Optional[str] = typer.Option(None, help="Free-text hint to guide extraction (optional)"),
    interactive: bool = typer.Option(False, help="Launch interactive wizard"),
    verbose: bool = typer.Option(False, help="Increase verbosity"),
    use_crew: bool = typer.Option(False, help="Use CrewAI agents (shows agent activity)"),
):
    """Run the SEC filings crew over the specified companies and years."""
    # Initialize logging with appropriate level
    if verbose:
        setup_logging(level="DEBUG")
        logger.debug("Verbose mode enabled")
    else:
        setup_logging()

    # Configure CrewAI verbosity if using crew
    if use_crew:
        os.environ["CREW_VERBOSE"] = "true"
        # Also set OPENAI/LLM debugging if available
        os.environ["LANGCHAIN_VERBOSE"] = "true"
        # Enable verbose automatically for crew mode
        verbose = True

    if interactive:
        print(Panel.fit("[bold cyan]RA-Agent Interactive Wizard[/bold cyan]", border_style="cyan"))
        id_input = Prompt.ask("Enter company identifier ([green]ticker[/green] or [yellow]CIK[/yellow])", default="AAPL")
        year = int(Prompt.ask("Enter year", default="2024"))
        filings = Prompt.ask("Filing types (comma-separated)", default="DEF 14A")
        metric = Prompt.ask("Metric to extract", default="Total CEO compensation")
        # Ask for hint text (recommended)
        hint = Prompt.ask("Optional free-text hint (press Enter to skip)", default="") or None
        companies = f"{id_input}:{year}"
        metrics = metric

    pairs: List[tuple[str, int]] = []
    pairs.extend(_parse_companies(companies))
    if companies_file:
        for line in _read_lines(companies_file):
            t, y = line.split(":", 1)
            pairs.append((t.strip(), int(y)))

    metric_list: List[str] = []
    if metrics:
        metric_list.extend([m.strip() for m in metrics.split(",") if m.strip()])
    if metrics_file:
        metric_list.extend(_read_lines(metrics_file))

    if not pairs:
        raise typer.BadParameter("Provide companies via --companies or --companies-file")

    filing_types = [f.strip() for f in filings.split(",") if f.strip()]

    # Print a clear delegation graph and pipeline overview before processing
    console = Console()
    try:
        from .agents.crew import build_crew
        crew = build_crew()
        # Delegation graph (who delegates to whom)
        console.print(Panel.fit("[bold magenta]Crew Agents & Delegation[/bold magenta]", border_style="magenta"))
        tree = Tree("Delegation")
        # Find the orchestrator (allow_delegation True)
        orchestrators = [ag for ag in crew.agents if getattr(ag, "allow_delegation", False)]
        if orchestrators:
            ga = orchestrators[0]
            ga_node = tree.add(f"[yellow]{ga.name}[/yellow]: {ga.role}")
            for ag in crew.agents:
                if ag is ga:
                    continue
                ga_node.add(f"[cyan]{ag.name}[/cyan]: {ag.role}")
        else:
            agents_node = tree.add("Agents")
            for ag in crew.agents:
                agents_node.add(f"[cyan]{ag.name}[/cyan]: {ag.role}")
        console.print(tree)
        console.print(Rule("Pipeline"))
        # Sequential pipeline based on tasks order
        pipeline = " -> ".join(f"[green]{t.agent.name}[/green]" for t in crew.tasks)
        console.print(pipeline)
    except Exception:
        # Fallback static graph if CrewAI isn't available
        console.print(Panel.fit("[bold magenta]Crew Agents & Delegation[/bold magenta]", border_style="magenta"))
        tree = Tree("Delegation")
        ga_node = tree.add("[yellow]GraduateAssistant[/yellow]: Orchestrate crew")
        for name, role in [
            ("DataRetriever", "Retrieve SEC EDGAR filings"),
            ("DataCleaner", "Clean and normalize text"),
            ("DataExtractor", "Extract variables and facts"),
            ("DataCalculator", "Compute metrics"),
            ("DataValidator", "Validate extracted data"),
            ("DataExporter", "Export final data"),
        ]:
            ga_node.add(f"[cyan]{name}[/cyan]: {role}")
        console.print(tree)
        console.print(Rule("Pipeline"))
        console.print(
            "[green]DataRetriever[/green] -> [green]DataCleaner[/green] -> [green]DataExtractor[/green] -> "
            "[green]DataCalculator[/green] -> [green]DataValidator[/green] -> [green]DataExporter[/green]"
        )

    # No JSON section hints supported

    if use_crew:
        # Use CrewAI agents
        if verbose:
            typer.echo("[INFO] Using CrewAI agents workflow...")
        
        try:
            from .agents.crew import build_crew
            crew = build_crew()
            
            results = []
            for identifier, year in pairs:
                try:
                    if is_cik(identifier):
                        cik = normalize_cik(identifier)
                        ticker = identifier
                    else:
                        ticker = identifier
                        cik = ticker_to_cik(ticker) or get_cik_for_ticker(ticker) or ""
                    
                    if not cik:
                        results.append({"id": identifier, "year": year, "error": "CIK not found"})
                        continue
                    
                    if verbose:
                        typer.echo(f"[INFO] Running crew for {identifier} (CIK {cik}) year {year}")
                    
                    # Set up crew inputs
                    inputs = {
                        "ticker": ticker,
                        "years": [year],
                        "filing_types": filing_types,
                        "metrics": metric_list,
                        "hint": hint,
                        "output_format": output_format
                    }
                    
                    # Execute the crew - this will show agent activity
                    crew_result = crew.kickoff(inputs=inputs)
                    
                    # Process crew results
                    result_data = {
                        "identifier": identifier,
                        "ticker": ticker,
                        "cik": cik,
                        "year": year,
                        "crew_output": str(crew_result),
                        "hint": hint
                    }
                    results.append(result_data)
                    
                except Exception as e:
                    results.append({"identifier": identifier, "year": year, "error": str(e)})
            
        except ImportError:
            typer.echo("[ERROR] CrewAI not available. Falling back to direct processing...")
            use_crew = False
    
    if not use_crew:
        # Direct processing (existing code)
        results = []
        pbar = tqdm(pairs, desc="Processing companies")
        for identifier, year in pbar:
            try:
                pbar.set_description(f"Processing {identifier}:{year}")
                pbar.refresh()
                
                # Detect ticker vs CIK
                if is_cik(identifier):
                    cik = normalize_cik(identifier)
                    ticker = identifier  # ticker unknown; keep id for output label
                else:
                    ticker = identifier
                    cik = ticker_to_cik(ticker) or get_cik_for_ticker(ticker) or ""
                if not cik:
                    results.append({"id": identifier, "year": year, "error": "CIK not found"})
                    continue
                
                if verbose:
                    tqdm.write(f"[INFO] Processing {identifier} (CIK {cik}) for year {year} and filings {filing_types}")
                
                pbar.set_description(f"Retrieving filings for {identifier}")
                pbar.refresh()
                filings = list_company_filings(cik, [year], filing_types)
                
                if verbose:
                    tqdm.write(f"[DEBUG] Found {len(filings)} filings for CIK {cik} in {year}")
                
                docs = []
                for f in filings:
                    pbar.set_description(f"Downloading {f['form']} for {identifier}")
                    pbar.refresh()
                    
                    html = download_filing(cik, f["accessionNumber"], f["primaryDocument"])
                    text = html_to_text(html)
                    # Persist cleaned text and tables under data/filings/CIK/ACCESSION/
                    cik_nozero = str(int(cik))
                    acc_no = f["accessionNumber"].replace("-", "")
                    base_dir = os.path.join(settings.data_dir, "filings", cik_nozero, acc_no)
                    os.makedirs(base_dir, exist_ok=True)
                    # Optionally persist XML if detected
                    xml_path = None
                    if is_xml_content(html):
                        xml_path = os.path.join(base_dir, "source.xml")
                        with open(xml_path, "w", encoding="utf-8") as xf:
                            xf.write(html)
                        if verbose:
                            tqdm.write(f"[DEBUG] Saved XML source to {xml_path}")
                    # Save cleaned text
                    cleaned_text = clean_text(text)
                    text_path = os.path.join(base_dir, "cleaned.txt")
                    with open(text_path, "w", encoding="utf-8") as fh:
                        fh.write(cleaned_text)
                    # Save tables (JSON)
                    tables = extract_tables(html)
                    import orjson
                    tables_path = os.path.join(base_dir, "tables.json")
                    with open(tables_path, "wb") as fh:
                        fh.write(orjson.dumps(tables, option=orjson.OPT_INDENT_2))
                    if verbose:
                        tqdm.write(f"[DEBUG] Saved cleaned text to {text_path} and {len(tables)} tables to {tables_path}")
                    docs.append({
                        "meta": {**f, "cik": cik, "paths": {"text": text_path, "tables": tables_path, "xml": xml_path}},
                        "text": cleaned_text,
                        "html": html,
                    })
                cleaned = docs  # already cleaned and saved
                rows = []
                for doc in cleaned:
                    for m in metric_list:
                        pbar.set_description(f"Extracting '{m}' from {doc['meta']['form']}")
                        pbar.refresh()
                        
                        # Use LLM-based extraction (no regex), with optional free-text hint
                        xml_content = None
                        xml_p = doc["meta"].get("paths", {}).get("xml")
                        if xml_p and os.path.exists(xml_p):
                            try:
                                with open(xml_p, "r", encoding="utf-8") as xf:
                                    xml_content = xf.read()
                            except Exception:
                                xml_content = None
                        res = llm_extract_metric(doc["text"], m, hint=hint, form=doc["meta"].get("form"), xml=xml_content)
                        if verbose:
                            tqdm.write(f"[DEBUG] Extracted {m}: value='{res.get('value','')[:120]}' form={doc['meta'].get('form')} year={doc['meta']['filingDate'][:4]}")
                        rows.append(
                            {
                                "identifier": identifier,
                                "ticker": ticker,
                                "cik": cik,
                                "year": int(doc["meta"]["filingDate"][:4]),
                                "metric": res["metric"],
                                "value": res["value"],
                                "context": res["context"],
                                "form": doc["meta"]["form"],
                                "file_path_text": doc["meta"].get("paths", {}).get("text"),
                                "file_path_tables": doc["meta"].get("paths", {}).get("tables"),
                                "file_path_xml": doc["meta"].get("paths", {}).get("xml"),
                            }
                        )
                pbar.set_description(f"Validating and exporting for {identifier}")
                pbar.refresh()
                
                report = validate_values(rows)
                out_path = export_rows(rows, output_format, name=f"results_{identifier}_{year}")
                
                # Show validation results
                if verbose and report:
                    tqdm.write(f"[VALIDATION] {report.get('summary', 'No summary')}")
                    if report.get('issues'):
                        tqdm.write(f"[VALIDATION ISSUES] {'; '.join(report['issues'][:3])}{'...' if len(report['issues']) > 3 else ''}")
                    if report.get('warnings'):
                        tqdm.write(f"[VALIDATION WARNINGS] {'; '.join(report['warnings'][:2])}{'...' if len(report['warnings']) > 2 else ''}")
                    tqdm.write(f"[INFO] Exported results to {out_path}")
                elif not verbose:
                    if report and not report.get('valid', True):
                        tqdm.write(f"[WARNING] Validation issues found for {identifier}")
                
                results.append({"identifier": identifier, "ticker": ticker, "cik": cik, "year": year, "output": out_path, "validation": report, "hint": hint})
            except Exception as e:
                results.append({"identifier": identifier, "year": year, "error": str(e)})
        
        pbar.close()

    # Save a session summary
    os.makedirs(settings.outputs_dir, exist_ok=True)
    out_path = os.path.join(settings.outputs_dir, "session_summary.json")
    import orjson

    with open(out_path, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    typer.echo(f"Saved summary to {out_path}")


if __name__ == "__main__":
    app()
