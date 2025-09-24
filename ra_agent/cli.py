from __future__ import annotations

import os
from typing import Optional, List
import typer
from tqdm import tqdm
from rich import print
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from .config import settings
from .utils.logging_utils import setup_logging
from .tools.sec_edgar import get_cik_for_ticker, list_company_filings, download_filing, html_to_text
from .tools.cleaning import clean_text, extract_tables
from .tools.extraction_llm import llm_extract_metric
from .tools.validation import validate_values
from .tools.exporter import export_rows
from .utils.identifiers import is_cik, normalize_cik, ticker_to_cik


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
):
    """Run the SEC filings crew over the specified companies and years."""
    setup_logging()
    if verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"

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

    # Print a simple agents/tasks overview before processing
    try:
        from .agents.crew import build_crew

        crew = build_crew()
        console = Console()
        print(Panel.fit("[bold magenta]Crew Overview[/bold magenta]", border_style="magenta"))
        tree = Tree("Workflow")
        agents_node = tree.add("Agents")
        for ag in crew.agents:
            agents_node.add(f"[cyan]{ag.name}[/cyan]: {ag.role}")
        tasks_node = tree.add("Sequential Tasks")
        for t in crew.tasks:
            tasks_node.add(f"[green]{t.agent.name}[/green] → {t.description[:60]}...")
        console.print(tree)
    except Exception:
        # Non-fatal if CrewAI isn't available in this run context
        pass

    # No JSON section hints supported

    results = []
    for identifier, year in tqdm(pairs, desc="Companies"):
        try:
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
            filings = list_company_filings(cik, [year], filing_types)
            docs = []
            for f in filings:
                html = download_filing(cik, f["accessionNumber"], f["primaryDocument"])
                text = html_to_text(html)
                # Persist cleaned text and tables under data/filings/CIK/ACCESSION/
                cik_nozero = str(int(cik))
                acc_no = f["accessionNumber"].replace("-", "")
                base_dir = os.path.join(settings.data_dir, "filings", cik_nozero, acc_no)
                os.makedirs(base_dir, exist_ok=True)
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
                docs.append({
                    "meta": {**f, "cik": cik, "paths": {"text": text_path, "tables": tables_path}},
                    "text": cleaned_text,
                    "html": html,
                })
            cleaned = docs  # already cleaned and saved
            rows = []
            for doc in cleaned:
                for m in metric_list:
                    # Use LLM-based extraction (no regex), with optional free-text hint
                    res = llm_extract_metric(doc["text"], m, hint=hint, form=doc["meta"].get("form"))
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
                        }
                    )
            report = validate_values(rows)
            out_path = export_rows(rows, output_format, name=f"results_{identifier}_{year}")
            results.append({"identifier": identifier, "ticker": ticker, "cik": cik, "year": year, "output": out_path, "validation": report, "hint": hint})
        except Exception as e:
            results.append({"identifier": identifier, "year": year, "error": str(e)})

    # Save a session summary
    os.makedirs(settings.outputs_dir, exist_ok=True)
    out_path = os.path.join(settings.outputs_dir, "session_summary.json")
    import orjson

    with open(out_path, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    typer.echo(f"Saved summary to {out_path}")


if __name__ == "__main__":
    app()
