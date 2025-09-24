@echo off
echo Testing RA-Agent improvements...
echo.

echo === Testing verbose direct processing ===
python -m ra_agent.cli --companies "AAPL:2024" --filings "DEF 14A" --metrics "Total CEO compensation" --hint "Look for the summary compensation table and total compensation column, not just salary" --verbose

echo.
echo === Testing CrewAI agents with verbosity ===
python -m ra_agent.cli --companies "AAPL:2024" --filings "DEF 14A" --metrics "Total CEO compensation" --hint "Look for the summary compensation table and total compensation column, not just salary" --use-crew --verbose

echo.
echo Test completed. Check the outputs and validation reports.