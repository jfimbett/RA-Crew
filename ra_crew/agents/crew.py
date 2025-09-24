from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from crewai import Agent, Task, Crew, Process

from ..config import settings
from ..utils.logging_utils import setup_logging, logger
from ..utils.prompt_loader import load_prompts, build_agent_objects, build_task_objects
from ..tools.sec_edgar import get_cik_for_ticker, list_company_filings, download_filing, html_to_text, is_xml_content
from ..tools.cleaning import clean_text, extract_tables
from ..tools.extraction_llm import llm_extract_metric
from ..tools.calculator import compute_metric
from ..tools.validation import validate_values
from ..tools.exporter import export_rows
from ..tools.rag import SimpleRAG
from ..utils.identifiers import is_cik, normalize_cik, ticker_to_cik


def _agent(*args, **kwargs):  # Backward compatibility placeholder (no longer used directly)
    raise RuntimeError("_agent factory is deprecated. Agents now loaded from prompts/agents.yaml")


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
    """Build a CrewAI crew using externalized YAML prompts."""
    setup_logging()

    # Enable LangChain debugging if crew verbosity is on
    verbose_mode = os.getenv("CREW_VERBOSE", "false").lower() == "true"
    if verbose_mode:
        os.environ["LANGCHAIN_VERBOSE"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"  # Avoid external tracing

    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # Load prompts and build agents
    prompts = load_prompts()
    agent_objs = build_agent_objects(model_name=model_name, verbose=verbose_mode, prompts=prompts)

    # Placeholder interpolation values will be provided later (empty defaults here)
    interpolation_defaults = {
        "metrics": "[]",
        "ticker": "",
        "years": "[]",
        "filing_types": "[]",
        "hint": "",
        "sec_filing_content": "",
    }
    task_objs = build_task_objects(agent_objs, verbose=verbose_mode, interpolation_kwargs=interpolation_defaults, prompts=prompts)

    # Create log file path
    log_file = None
    if verbose_mode:
        os.makedirs(settings.outputs_dir, exist_ok=True)
        log_file = os.path.join(settings.outputs_dir, "crew_execution.log")
        logger.info(f"CrewAI execution will be logged to {log_file}")

    crew = Crew(
        agents=list(agent_objs.values()),
        tasks=list(task_objs.values()),
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
        
        # Rebuild tasks with actual interpolation values now that we have content
        try:
            prompts = load_prompts()
            verbose_mode = os.getenv("CREW_VERBOSE", "false").lower() == "true"
            model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
            # Rebuild agents (reuse existing model/verbosity) and tasks with real placeholders
            agent_objs = build_agent_objects(model_name=model_name, verbose=verbose_mode, prompts=prompts)
            interpolation_values = {
                "metrics": metrics,
                "ticker": ticker,
                "years": years,
                "filing_types": filing_types,
                "hint": hint,
                "sec_filing_content": sec_filing_content,
            }
            task_objs = build_task_objects(agent_objs, verbose=verbose_mode, interpolation_kwargs=interpolation_values, prompts=prompts)
            # Replace crew tasks with re-interpolated ones preserving order
            self.crew.tasks = list(task_objs.values())
        except Exception as e:
            logger.error(f"Failed to rebuild tasks with interpolation: {e}")
        
        logger.info("Processing relevant sections with CrewAI (interpolated tasks)")
        result = self.crew.kickoff(inputs=inputs)
        
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
