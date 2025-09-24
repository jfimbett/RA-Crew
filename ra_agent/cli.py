from __future__ import annotations

import os
from typing import Optional, List
import typer
from tqdm import tqdm

from .config import settings
from .utils.logging_utils import setup_logging
from .tools.sec_edgar import get_cik_for_ticker, list_company_filings, download_filing, html_to_text
from .tools.cleaning import clean_text
from .tools.extraction import extract_metric
from .tools.validation import validate_values
from .tools.exporter import export_rows


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
    verbose: bool = typer.Option(False, help="Increase verbosity"),
):
    """Run the SEC filings crew over the specified companies and years."""
    setup_logging()
    if verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"

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

    results = []
    for ticker, year in tqdm(pairs, desc="Companies"):
        try:
            cik = get_cik_for_ticker(ticker)
            if not cik:
                results.append({"ticker": ticker, "year": year, "error": "CIK not found"})
                continue
            filings = list_company_filings(cik, [year], filing_types)
            docs = []
            for f in filings:
                html = download_filing(cik, f["accessionNumber"], f["primaryDocument"])
                text = html_to_text(html)
                docs.append({"meta": f, "text": text})
            cleaned = [
                {"meta": d["meta"], "text": clean_text(d["text"])}
                for d in docs
            ]
            rows = []
            for doc in cleaned:
                for m in metric_list:
                    res = extract_metric(doc["text"], m)
                    rows.append(
                        {
                            "ticker": ticker,
                            "year": int(doc["meta"]["filingDate"][:4]),
                            "metric": res["metric"],
                            "value": res["value"],
                            "context": res["context"],
                            "form": doc["meta"]["form"],
                        }
                    )
            report = validate_values(rows)
            out_path = export_rows(rows, output_format, name=f"results_{ticker}_{year}")
            results.append({"ticker": ticker, "year": year, "output": out_path, "validation": report})
        except Exception as e:
            results.append({"ticker": ticker, "year": year, "error": str(e)})

    # Save a session summary
    os.makedirs(settings.outputs_dir, exist_ok=True)
    out_path = os.path.join(settings.outputs_dir, "session_summary.json")
    import orjson

    with open(out_path, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    typer.echo(f"Saved summary to {out_path}")


if __name__ == "__main__":
    app()
