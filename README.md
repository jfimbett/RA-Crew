# RA-Crew: SEC Filings AI Extraction Crew
===================================================

A collaborative crew of AI agents built with CrewAI and LangChain to retrieve, clean, extract, validate, and export non-structured financial data from SEC EDGAR filings (10-K, 10-Q, 8-K, DEF 14A, etc.).

## Key Features

- **Multi-agent workflow**: DataRetriever, DataCleaner, DataExtractor, DataCalculator, DataValidator, DataExporter, GraduateAssistant
- **SEC EDGAR integration**: Respectful rate limiting with realistic headers and automatic ticker-to-CIK mapping
- **RAG-enhanced extraction**: Intelligent document chunking and context search for large filings
- **Text cleaning**: HTML parsing with table structure preservation for financial data
- **Structured outputs**: JSON/CSV exports with multi-year compensation data
- **LLM validation**: Unit-aware validation and cross-company/period sanity checks
- **CLI interface**: Interactive wizard and batch processing with progress bars
- **Comprehensive logging**: Console and rotating file logs with verbosity control

## Installation & Setup

### Prerequisites

1. **Install Anaconda** (recommended Python distribution):
   - Download from [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)
   - Follow the installation instructions for your operating system

### Environment Setup

1. **Create a Python 3.12 environment**:
   ```bash
   conda create -n ra-crew python=3.12
   conda activate ra-crew
   ```

2. **Clone and navigate to the repository**:
   ```bash
   git clone <your-repo-url>
   cd RA-Crew
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **Note**: If you previously installed `langchain` 0.2.x, uninstall it first:
   ```bash
   pip uninstall -y langchain langchain-text-splitters langchain-community
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   ```
   
   Edit `.env` and provide your values:
   ```bash
   # Required: Choose your LLM provider (openai, azure-openai, anthropic, groq)
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_key_here
   
   # Required: SEC identity (replace with your real name and email)
   EDGAR_IDENTITY=Your Name your.email@domain.com
   
   # Optional: Logging level
   LOG_LEVEL=INFO
   ```

## Usage

### Interactive Mode (Recommended for Beginners)

Launch the interactive wizard for guided usage:

```bash
python -m ra_crew.cli --interactive
```

The wizard will prompt you for:
- Company identifier (ticker or CIK)
- Year 
- Filing types
- Metrics to extract
- Optional hint text

### Command Line Usage

#### Basic Examples

Extract CEO compensation from Apple's 2023 proxy statement:
```bash
python -m ra_crew.cli --companies "AAPL:2023" --filings "DEF 14A" --metrics "Total CEO compensation"
```

Multiple companies and years:
```bash
python -m ra_crew.cli --companies "AAPL:2023,MSFT:2022,TSLA:2023" --filings "10-K,DEF 14A" --metrics "Total CEO compensation,Total CFO compensation"
```

#### Using File Inputs

Create input files for batch processing:

**examples/companies.example**:
```
AAPL:2023
MSFT:2022
TSLA:2023
```

**examples/metrics.example**:
```
Total CEO compensation
Total CEO salary
Total CEO bonus
Total CEO stock awards
Total CFO compensation
```

Run with files:
```bash
python -m ra_crew.cli --companies-file examples/companies.example --metrics-file examples/metrics.example --filings "DEF 14A"
```

#### Advanced Options

**Enable CrewAI agent verbosity** (see what agents are doing):
```bash
python -m ra_crew.cli --companies "AAPL:2023" --filings "DEF 14A" --metrics "Total CEO compensation" --use-crew --verbose
```

**Provide extraction hints** (improve accuracy):
```bash
python -m ra_crew.cli --companies "AAPL:2023" --filings "DEF 14A" --metrics "Total CEO compensation" --hint "Look for executive compensation tables and summary compensation table"
```

**Different output formats**:
```bash
# JSON output (default)
python -m ra_crew.cli --companies "AAPL:2023" --filings "10-K" --metrics "Total CEO compensation" --output-format json

# CSV output
python -m ra_crew.cli --companies "AAPL:2023" --filings "10-K" --metrics "Total CEO compensation" --output-format csv
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--companies` | Comma-separated TICKER:YEAR pairs | `"AAPL:2023,MSFT:2022"` |
| `--companies-file` | File with TICKER:YEAR per line | `examples/companies.example` |
| `--filings` | Comma-separated filing types | `"10-K,DEF 14A,10-Q"` |
| `--metrics` | Comma-separated metrics to extract | `"Total CEO compensation"` |
| `--metrics-file` | File with metrics per line | `examples/metrics.example` |
| `--output-format` | Output format | `json` or `csv` |
| `--hint` | Free-text hint to guide extraction | `"Look for compensation tables"` |
| `--interactive` | Launch interactive wizard | Flag |
| `--verbose` | Increase verbosity | Flag |
| `--use-crew` | Enable CrewAI agents (shows activity) | Flag |

## Ticker and CIK Support

The system automatically handles both stock tickers and CIK codes:

- **Stock tickers**: Use familiar symbols like `AAPL`, `MSFT`, `TSLA`
- **CIK codes**: Use SEC Central Index Key numbers like `0000320193` (Apple)

The system uses a comprehensive ticker-to-CIK mapping and automatically converts tickers to CIK codes when communicating with the SEC EDGAR database, ensuring compliance with SEC requirements.

## Output

Results are saved to the `outputs/` directory:

- **Individual results**: `results_{ticker}_{year}.json` or `.csv`
- **Session summary**: `session_summary.json` with validation reports
- **Execution logs**: `logs/ra_crew.log` with detailed processing information

### JSON Output Format

```json
{
  "identifier": "AAPL",
  "ticker": "AAPL", 
  "cik": "0000320193",
  "year": 2023,
  "metric": "Total CEO compensation",
  "value": "$99,420,097",
  "context": "Tim Cook received total compensation of...",
  "form": "DEF 14A",
  "validation": {
    "valid": true,
    "summary": "All values validated successfully"
  }
}
```

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `LLM_PROVIDER` | Yes | LLM service to use | `openai`, `azure-openai`, `anthropic`, `groq` |
| `OPENAI_API_KEY` | Conditional | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Conditional | Anthropic API key | Required if using Anthropic |
| `GROQ_API_KEY` | Conditional | Groq API key | Required if using Groq |
| `EDGAR_IDENTITY` | Yes | SEC required identity | `John Doe john@example.com` |
| `LOG_LEVEL` | No | Logging verbosity | `INFO`, `DEBUG`, `WARNING` |

## Development

Format and lint code (optional):
```bash
pip install ruff black mypy
ruff check ra_crew
black ra_crew  
mypy ra_crew

## Customizing Prompts (YAML Externalization)

Static agent roles, goals, backstories, and task instructions now live in YAML under `prompts/`:

- `prompts/agents.yaml`: Defines each agent (`id`, `name`, `role`, `goal`, `backstory`).
- `prompts/tasks.yaml`: Defines each task (`id`, `agent`, `description`, `expected_output`, optional `context`).

### Placeholders

Task descriptions support runtime placeholders interpolated before execution:

| Placeholder | Injected Value |
|-------------|----------------|
| `{metrics}` | List of metrics requested |
| `{ticker}` | Company ticker or CIK |
| `{years}` | List of target years |
| `{filing_types}` | List of SEC form types (e.g., 10-K, DEF 14A) |
| `{hint}` | User-provided hint text (may be empty) |
| `{sec_filing_content}` | RAG-extracted relevant filing sections |
| `{derived_metrics}` | List of derived metrics to compute |
| `{calculation_expressions}` | Mapping of derived metric -> formula expression |
| `{output_format}` | Desired export format (json or csv) |

If you add new placeholders, ensure they are passed in the crew kickoff inputs and wired through the interpolation dictionary in `ra_crew/agents/crew.py`.

### Editing Prompts

1. Open the YAML file you want to change.
2. Modify only the free text; keep YAML keys intact.
3. Avoid accidental `{placeholder}` removals unless you remove their logic.
4. No restart needed—subsequent runs load the modified YAML (cache resets on process start).

### Adding a New Task

1. Add a new task block to `prompts/tasks.yaml` with a unique `id`.
2. Reference an existing agent by its `agent` field.
3. (Optional) Add `context: [other_task_id]` if it depends on prior task output.
4. Update orchestration in `crew.py` only if task ordering changes; otherwise order follows YAML list order.
5. For derived computations, ensure you also pass `derived_metrics` and `calculation_expressions` via kickoff inputs.

### Common Pitfalls

- Placeholder typo => interpolation error (see logs for missing key).
- Invalid YAML (indentation) => loader exception on startup.
- Large `{sec_filing_content}` may exceed model context; tune RAG chunking in `rag.py` if needed.

## Batch Input Example

An example company list file is provided for multi-company processing:

`examples/companies.example` (format: `TICKER:YEAR` per line):
```
AAPL:2023
MSFT:2022
TSLA:2023
```

Metrics example (`examples/metrics.example`):
```
Total CEO compensation
Total CFO compensation
Total CEO salary
```

Use with CLI flags `--companies-file` and `--metrics-file`.
```

## Important Notes

- **SEC Compliance**: The system respects SEC rate limits and header requirements. You are responsible for lawful, compliant use.
- **API Keys**: Never commit API keys. The `.env` file is gitignored for security.
- **Data Storage**: Downloaded SEC filings are cached in the `data/` directory (also gitignored).
- **Rate Limiting**: Built-in rate limiting (10 requests/second) to respect SEC guidelines.

## License

MIT License (see LICENSE file for details).

## Troubleshooting

### Common Issues

1. **"CIK not found for ticker"**: 
   - Check ticker spelling
   - Try using the CIK code directly
   - Ensure company is publicly traded and files with SEC

2. **"No filings found"**:
   - Verify the year and filing type combination
   - Some companies may not file certain forms in specific years

3. **Rate limiting errors**:
   - The system automatically handles rate limits
   - For persistent issues, check your EDGAR_IDENTITY format

4. **Missing API key errors**:
   - Ensure your `.env` file is properly configured
   - Verify the API key is valid and has sufficient credits

### Getting Help

1. Check the logs in `logs/ra_crew.log` for detailed error information
2. Run with `--verbose` flag for additional debugging output
3. Use `--use-crew` to see detailed agent activity and decision-making process

---

## Development Credits

This project was enhanced using **GitHub Copilot** running on **GPT-5** and **Claude Sonnet 4**, leveraging advanced AI assistance for code development, documentation, and architecture design.
