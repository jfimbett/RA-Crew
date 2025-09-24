from __future__ import annotations

import re
from bs4 import BeautifulSoup
import pandas as pd
from pandas.errors import EmptyDataError
from io import StringIO
from ..tools.sec_edgar import is_xml_content
from typing import Dict
from ..utils.logging_utils import timeit


@timeit
def clean_text(raw_html_or_text: str) -> str:
    """Clean HTML while preserving table structure and important formatting."""
    # If contains tags, strip them but preserve structure
    if "<" in raw_html_or_text and ">" in raw_html_or_text:
        soup = BeautifulSoup(raw_html_or_text, "lxml")
        
        # Remove unnecessary elements
        for el in soup(["script", "style", "noscript"]):
            el.decompose()
        
        # Preserve table structure by adding extra spacing
        for table in soup.find_all("table"):
            # Add spacing around table
            table.insert_before("\n\n=== TABLE START ===\n")
            table.insert_after("\n=== TABLE END ===\n\n")
            
            # Add spacing between table rows
            for tr in table.find_all("tr"):
                tr.insert_after("\n")
                # Add spacing between cells in same row
                for td in tr.find_all(["td", "th"]):
                    if td.next_sibling:
                        td.insert_after(" | ")
        
        # Preserve paragraph breaks
        for p in soup.find_all("p"):
            p.insert_after("\n\n")
        
        # Preserve div breaks  
        for div in soup.find_all("div"):
            div.insert_after("\n")
            
        text = soup.get_text()
    else:
        text = raw_html_or_text
    
    # Clean up while preserving important structure
    text = re.sub(r"\r", "\n", text)
    # Don't collapse all whitespace - preserve table formatting
    text = re.sub(r"\n{4,}", "\n\n", text)  # Limit excessive newlines but keep some
    text = re.sub(r" {3,}", "  ", text)     # Limit excessive spaces but keep some alignment
    text = re.sub(r"\t", " ", text)         # Convert tabs to spaces
    
    return text.strip()


@timeit
def extract_tables(html: str, max_tables: int = 50) -> list[dict]:
    """Extract HTML tables into JSON-serializable dicts.

    Returns a list of {columns: [...], rows: [{col: val, ...}, ...]} objects.
    Limits to max_tables to avoid huge memory use.
    """
    tables: list[dict] = []
    if not html or "<table" not in html.lower():
        return tables
    # If content is XML/XBRL, do not attempt HTML table extraction
    if is_xml_content(html):
        return tables
    try:
        dfs = pd.read_html(StringIO(html), flavor="lxml")
    except (ValueError, EmptyDataError):  # no tables or parse issues
        return tables
    except Exception:
        # As a fallback, try the html5lib parser
        try:
            dfs = pd.read_html(StringIO(html), flavor="bs4")
        except Exception:
            return tables
    for i, df in enumerate(dfs[:max_tables]):
        # Reset index and convert to records
        try:
            df = df.reset_index(drop=True)
            records = df.to_dict(orient="records")
            tables.append({
                "columns": list(df.columns.map(str)),
                "rows": records,
            })
        except Exception:
            continue
    return tables
