 # An AI Agent for Financial Research  
RA-Agent: SEC Filings Retrieval and Extraction Crew
===================================================

This project builds a collaborative crew of AI agents (CrewAI + LangChain) to retrieve, clean, extract, validate, and export non-structured financial data from SEC EDGAR filings (e.g., 10-K, 10-Q, 8-K, DEF 14A).

Key features
- Multi-agent workflow: DataRetriever, DataCleaner, DataExtractor, DataCalculator, DataValidator, DataExporter, GraduateAssistant.
- SEC EDGAR retrieval with respectful rate limiting and realistic headers.
- Text cleaning, HTML parsing, and structured outputs (JSON/CSV).
- Extraction via hybrid search (BM25 + regex) or RAG.
- Unit-aware validation and cross-company/period sanity checks.
- CLI to run over companies and years or from a file with progress bars.
- Logging to console and rotating files with verbosity control.

Quickstart (Windows, cmd)
1) Create and activate a virtual environment
```
python -m venv .venv
.venv\\Scripts\\activate
```

2) Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
If you've previously installed `langchain` 0.2.x, uninstall it before installing to avoid resolver conflicts, because this project uses the newer modular packages:

```bash
pip uninstall -y langchain langchain-text-splitters langchain-community
pip install -r requirements.txt
```

3) Configure environment
- Copy `.env.example` to `.env` and fill values.
- Provide your SEC header identity (name/email) as shown; replace with your own.

4) Run examples
```
python -m ra_agent.cli --companies "AAPL:2023" --filings 10-K --metrics "Total CEO compensation"

# Or from file
python -m ra_agent.cli --companies-file examples\\companies.example --filings 10-K,DEF 14A --metrics-file examples\\metrics.example
```

Environment variables
- `LLM_PROVIDER`: one of `openai`, `azure-openai`, `anthropic`, `groq` (flexible; only install keys you use).
- `OPENAI_API_KEY` or respective provider keys.
- `EDGAR_IDENTITY`: e.g., `First Last contact@email.com` (mandatory for SEC).
- `LOG_LEVEL`: `INFO` (default), `DEBUG`, `WARNING`.

Repository commands
```
# Format/lint (optional if you add tools)
pip install ruff black mypy
ruff check ra_agent
black ra_agent
mypy ra_agent
```

Notes
- The code respects SEC guidance on headers and pacing. You are responsible for lawful, compliant use.
- Do not push secrets. `.env` is gitignored.

License
MIT (add your preferred license).
