@echo off
echo Testing CrewAI verbosity with --use-crew flag...

REM Test with --use-crew flag (should automatically enable verbosity)
echo ===== Testing --use-crew (verbose should be automatic) =====
python -m ra_agent --use-crew --companies AAPL:2023 --metrics "CEO Total Compensation"

echo.
echo Test completed! Check above output for CrewAI agent activity.
pause