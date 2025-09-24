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
    # If contains tags, strip them
    if "<" in raw_html_or_text and ">" in raw_html_or_text:
        soup = BeautifulSoup(raw_html_or_text, "lxml")
        for el in soup(["script", "style", "noscript"]):
            el.decompose()
        text = soup.get_text("\n")
    else:
        text = raw_html_or_text
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n ?", "\n", text)
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
