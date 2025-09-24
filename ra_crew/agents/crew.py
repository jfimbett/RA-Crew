from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from crewai import Agent, Task, Crew, Process

from ..config import settings
from ..utils.logging_utils import setup_logging, logger
from ..tools.sec_edgar import get_cik_for_ticker, list_company_filings, download_filing, html_to_text, is_xml_content
from ..tools.cleaning import clean_text, extract_tables
from ..tools.extraction_llm import llm_extract_metric
from ..tools.calculator import compute_metric
from ..tools.validation import validate_values
from ..tools.exporter import export_rows
from ..tools.rag import SimpleRAG
from ..utils.identifiers import is_cik, normalize_cik, ticker_to_cik


def _agent(name: str, role: str, goal: str, backstory: str, tools: Optional[List[Any]] = None, allow_delegation: bool = False) -> Agent:
    # Get verbosity from environment or default to True for agents
    verbose_mode = os.getenv("CREW_VERBOSE", "false").lower() == "true"
    
    # Use string model name instead of dict - CrewAI will handle the LLM creation
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    return Agent(
        name=name,
        role=role,
        goal=goal,
        backstory=backstory,
        llm=model_name,  # Pass string model name, not dict
        verbose=verbose_mode,
        tools=tools or [],
        allow_delegation=allow_delegation,
    )


def retrieve_sec_data(ticker: str, years: List[int], filing_types: List[str]) -> Dict[str, Any]:
    """Retrieve actual SEC filing data for the ticker and years specified."""
    try:
        # Get CIK for ticker
        if is_cik(ticker):
            cik = normalize_cik(ticker)
        else:
            cik = ticker_to_cik(ticker) or get_cik_for_ticker(ticker) or ""
            
        if not cik:
            return {"error": f"CIK not found for ticker {ticker}"}
            
        logger.info(f"Retrieving SEC data for {ticker} (CIK {cik}) years {years} filing types {filing_types}")
        
        # Get filings for all years
        filings = list_company_filings(cik, years, filing_types)
        
        if not filings:
            return {"error": f"No {filing_types} filings found for {ticker} in {years}"}
            
        logger.info(f"Found {len(filings)} filings")
        
        # Download and process each filing
        documents = []
        for filing in filings[:3]:  # Limit to 3 most recent filings
            try:
                html = download_filing(cik, filing["accessionNumber"], filing["primaryDocument"])
                text = html_to_text(html)
                cleaned_text = clean_text(text)
                
                # Save the filing data
                cik_nozero = str(int(cik))
                acc_no = filing["accessionNumber"].replace("-", "")
                base_dir = os.path.join(settings.data_dir, "filings", cik_nozero, acc_no)
                os.makedirs(base_dir, exist_ok=True)
                
                text_path = os.path.join(base_dir, "cleaned.txt")
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                
                documents.append({
                    "filing_meta": filing,
                    "text": cleaned_text,
                    "text_path": text_path,
                    "form": filing["form"],
                    "filing_date": filing["filingDate"]
                })
                logger.info(f"Retrieved and processed {filing['form']} filing from {filing['filingDate']}")
                
            except Exception as e:
                logger.error(f"Error processing filing {filing['accessionNumber']}: {e}")
                continue
        
        return {
            "ticker": ticker,
            "cik": cik,
            "documents": documents,
            "years": years,
            "filing_types": filing_types
        }
        
    except Exception as e:
        logger.error(f"Error retrieving SEC data: {e}")
        return {"error": f"Failed to retrieve SEC data: {str(e)}"}


def build_crew() -> Crew:
    """Build a CrewAI crew for financial data extraction that actually works."""
    setup_logging()
    
    # Enable LangChain debugging if crew verbosity is on
    if os.getenv("CREW_VERBOSE", "false").lower() == "true":
        os.environ["LANGCHAIN_VERBOSE"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"  # Avoid external tracing

    # Create agents with proper CrewAI patterns and tool access
    researcher = _agent(
        name="FinancialResearcher",
        role="SEC Filing Data Extractor",
        goal="Extract specific financial metrics as JSON from SEC filings for all available years",
        backstory=(
            "You are an expert at reading SEC filings and extracting financial data from complex table structures. "
            "You understand that executive titles often span multiple rows in tables (e.g., 'Luca Maestri' on row 1, "
            "'Senior Vice President,' on row 2, 'Chief Financial Officer' on row 3). "
            "You always search for functional titles like 'Chief Financial Officer', 'Chief Executive Officer' "
            "regardless of their position in the table structure. You extract EXACT numbers for ALL years present "
            "and work with any type of SEC filing: 10-K, 10-Q, DEF 14A, 8-K, etc. "
            "CRITICAL: You NEVER use placeholder values like 99999, 0, or estimates. If data doesn't exist, you use null."
        ),
        # For now, no tools - agents will work with provided data
        tools=[]
    )

    analyst = _agent(
        name="DataAnalyst", 
        role="JSON Validator and Formatter",
        goal="Validate extracted multi-year data and output clean JSON with numerical values",
        backstory=(
            "You are a JSON formatting specialist for multi-year financial data. You ensure that "
            "extracted data contains ALL available years from the filing, properly formatted as valid JSON. "
            "You remove formatting (commas, dollar signs) from numbers and validate data consistency "
            "across years. You work with any SEC filing type and any financial metric. "
            "CRITICAL: You NEVER allow placeholder values like 99999, 0, or estimates in the final output. "
            "Missing data must be null or omitted entirely. Your output is always valid JSON with complete year coverage."
        )
    )

    # Create tasks with dynamic descriptions that use input variables
    research_task = Task(
        description=(
            "IMPORTANT: SEC filings often contain compensation or financial data from MULTIPLE YEARS. "
            "For example, proxy statements contain 3+ years of data, 10-K filings contain historical data. "
            "\nYou will be provided with actual SEC filing content. "
            "Extract the requested metrics: {metrics} from the SEC filing text provided below. "
            "Filing details: Ticker {ticker}, Filing Year {years}, Filing types {filing_types} "
            "Hint: {hint} "
            "\n\nSEC FILING CONTENT:\n{sec_filing_content}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Look for compensation tables, financial statements, or relevant sections\n"
            "2. IMPORTANT: Executive titles may span multiple rows in tables. For example:\n"
            "   Row 1: 'Luca Maestri' 'Senior Vice President,'\n"
            "   Row 2: [blank] 'Chief Financial Officer'\n"
            "   This person is the CFO - look for 'Chief Financial Officer' anywhere near the name\n"
            "3. Search for executives by their functional titles: CEO, CFO, COO, CTO, etc.\n"
            "4. Extract data for ALL YEARS available in the document (typically 3+ years)\n"
            "5. Extract EXACT numbers from tables - DO NOT estimate or round\n"
            "6. For each requested metric, provide data for ALL available years in JSON format:\n"
            "{{\n"
            "  \"Total CEO compensation\": {{\n"
            "    \"2022\": {{\n"
            "      \"value\": [exact number],\n"
            "      \"name\": \"[executive name]\",\n"
            "      \"title\": \"[full title from table]\",\n"
            "      \"source\": \"[table/section name]\"\n"
            "    }},\n"
            "    \"2021\": {{\n"
            "      \"value\": [exact number],\n"
            "      \"name\": \"[executive name]\",\n"
            "      \"title\": \"[full title from table]\",\n"
            "      \"source\": \"[table/section name]\"\n"
            "    }}\n"
            "  }},\n"
            "  \"Total CFO compensation\": {{\n"
            "    \"2022\": {{\n"
            "      \"value\": [exact number],\n"
            "      \"name\": \"[executive name]\",\n"
            "      \"title\": \"[full title from table]\",\n"
            "      \"source\": \"[table/section name]\"\n"
            "    }}\n"
            "  }}\n"
            "}}\n"
            "CRITICAL: If data is NOT found for a specific year or metric, use null (not 99999, not 0, not any made-up number). "
            "Only include years where you actually found the data in the filing. "
            "REMEMBER: Look for 'Chief Financial Officer', 'Chief Executive Officer', etc. even if they appear on separate rows from the name."
        ),
        agent=researcher,
        expected_output=(
            "JSON object with requested metrics, each containing data for ALL years found in the filing, including executives whose titles span multiple table rows"
        )
    )

    analysis_task = Task(
        description=(
            "Validate the extracted JSON data and ensure it contains ALL available years and executives. "
            "Your job is to:\n"
            "1. Verify the numbers match what's in the SEC filing content for ALL years\n"
            "2. Ensure the JSON is valid and well-structured\n"
            "3. Check that years are complete (don't miss any years from the original data)\n"
            "4. IMPORTANT: Verify that executives with multi-row titles were found correctly:\n"
            "   - CFO might be listed as 'Senior Vice President, Chief Financial Officer'\n"
            "   - Look for anyone with 'Chief Financial Officer' in their title\n"
            "   - Look for anyone with 'Chief Executive Officer' in their title\n"
            "5. Output the final JSON with clean numerical values (no commas, dollar signs)\n"
            "\nYour output must be valid JSON only, nothing else.\n"
            "CRITICAL: NEVER use placeholder values like 99999, 0, or any made-up numbers. "
            "If data is not found, use null or omit the year entirely. "
            "Only include years where actual data exists in the filing.\n"
            "Format with ALL available years and proper executive identification:\n"
            "{{\n"
            "  \"Total CEO compensation\": {{\n"
            "    \"2022\": {{\n"
            "      \"value\": 99420097,\n"
            "      \"name\": \"Tim Cook\",\n"
            "      \"title\": \"Chief Executive Officer\",\n"
            "      \"source\": \"Summary Compensation Table\"\n"
            "    }},\n"
            "    \"2021\": {{\n"
            "      \"value\": 98734394,\n"
            "      \"name\": \"Tim Cook\",\n"
            "      \"title\": \"Chief Executive Officer\",\n"
            "      \"source\": \"Summary Compensation Table\"\n"
            "    }}\n"
            "  }},\n"
            "  \"Total CFO compensation\": {{\n"
            "    \"2022\": {{\n"
            "      \"value\": 27151798,\n"
            "      \"name\": \"Luca Maestri\",\n"
            "      \"title\": \"Senior Vice President, Chief Financial Officer\",\n"
            "      \"source\": \"Summary Compensation Table\"\n"
            "    }},\n"
            "    \"2021\": {{\n"
            "      \"value\": 26978503,\n"
            "      \"name\": \"Luca Maestri\",\n"
            "      \"title\": \"Senior Vice President, Chief Financial Officer\",\n"
            "      \"source\": \"Summary Compensation Table\"\n"
            "    }}\n"
            "  }}\n"
            "}}\n"
            "If data for 2023 doesn't exist, DO NOT include a 2023 entry. If an entire metric is not found, use null for the whole metric."
        ),
        agent=analyst,
        expected_output=(
            "Valid JSON object with clean numerical values for each requested metric across ALL available years, correctly identifying executives with multi-row titles"
        ),
        context=[research_task]
    )

    # Get verbosity setting from environment
    verbose_mode = os.getenv("CREW_VERBOSE", "false").lower() == "true"
    
    # Create log file path
    log_file = None
    if verbose_mode:
        os.makedirs(settings.outputs_dir, exist_ok=True)
        log_file = os.path.join(settings.outputs_dir, "crew_execution.log")
        logger.info(f"CrewAI execution will be logged to {log_file}")
    
    # Create crew with proper configuration
    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=verbose_mode,
        output_log_file=log_file,
    )
    return crew


class SECDataCrew:
    """Custom crew wrapper that retrieves SEC data before processing."""
    
    def __init__(self):
        self.crew = build_crew()
    
    def kickoff(self, inputs: Dict[str, Any]) -> str:
        """Custom kickoff that retrieves SEC data first, then processes it with RAG."""
        ticker = inputs.get("ticker", "")
        years = inputs.get("years", [])
        filing_types = inputs.get("filing_types", [])
        metrics = inputs.get("metrics", [])
        hint = inputs.get("hint", "")
        output_format = inputs.get("output_format", "json")
        
        logger.info(f"Starting SEC data retrieval for {ticker}")
        
        # Retrieve actual SEC data
        sec_data = retrieve_sec_data(ticker, years, filing_types)
        
        if "error" in sec_data:
            return f"ERROR: {sec_data['error']}"
        
        # Initialize RAG system
        rag = SimpleRAG(chunk_size=2000, overlap=200)
        
        # Create query from requested metrics
        search_query = " ".join(metrics) if metrics else "compensation"
        if hint:
            search_query += f" {hint}"
        
        logger.info(f"Using RAG to search for: {search_query}")
        
        # Extract relevant sections using RAG for each document
        relevant_sections = []
        for doc in sec_data["documents"]:
            logger.info(f"Processing {doc['form']} filing from {doc['filing_date']} with RAG")
            
            # Use RAG to get most relevant sections
            relevant_content = rag.extract_context_for_query(
                text=doc['text'],
                source_info={
                    "form": doc['form'],
                    "filing_date": doc['filing_date'],
                    "ticker": ticker
                },
                query=search_query
            )
            
            relevant_sections.append(
                f"\n=== {doc['form']} Filing from {doc['filing_date']} ===\n"
                f"{relevant_content}"
            )
        
        # Combine all relevant sections
        sec_filing_content = "\n".join(relevant_sections)
        
        logger.info(f"RAG extracted {len(sec_filing_content)} characters of relevant content from {len(sec_data['documents'])} documents")
        
        # Update inputs with RAG-extracted relevant content
        crew_inputs = {
            **inputs,
            "sec_filing_content": sec_filing_content
        }
        
        logger.info(f"Processing relevant sections with CrewAI")
        
        # Execute the crew with RAG-extracted relevant content
        result = self.crew.kickoff(inputs=crew_inputs)
        
        logger.info("Crew processing completed")
        
        # Export results to JSON/CSV like the direct processing does
        try:
            import orjson
            import json
            os.makedirs(settings.outputs_dir, exist_ok=True)
            
            # Try to parse the crew result as JSON
            try:
                crew_result_json = json.loads(str(result))
                logger.info("Successfully parsed crew result as JSON")
            except json.JSONDecodeError:
                logger.warning("Crew result is not valid JSON, storing as text")
                crew_result_json = None
            
            # Create structured result data
            result_data = {
                "identifier": ticker,
                "ticker": ticker,
                "cik": sec_data.get("cik", ""),
                "years": years,
                "metrics": metrics,
                "filing_types": filing_types,
                "hint": hint,
                "crew_result": crew_result_json if crew_result_json else str(result),
                "crew_result_raw": str(result),  # Keep raw for debugging
                "processing_method": "CrewAI with RAG",
                "documents_processed": len(sec_data.get("documents", [])),
                "rag_query": search_query
            }
            
            # Save individual result
            output_file = os.path.join(settings.outputs_dir, f"crew_results_{ticker}_{'_'.join(map(str, years))}.{output_format}")
            
            if output_format == "json":
                with open(output_file, "wb") as f:
                    f.write(orjson.dumps([result_data], option=orjson.OPT_INDENT_2))
            elif output_format == "csv":
                import pandas as pd
                df = pd.DataFrame([result_data])
                df.to_csv(output_file, index=False)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        return result
