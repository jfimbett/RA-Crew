from __future__ import annotations

import os
import re
import time
import json
from typing import Dict, List, Optional, Tuple
import requests
import random
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup

from ..config import settings
from ..utils.logging_utils import timeit


SEC_RATE_LIMIT_CALLS = int(os.getenv("SEC_RATE_CALLS", "10"))
SEC_RATE_LIMIT_PERIOD = int(os.getenv("SEC_RATE_PERIOD", "1"))  # default SEC published guidance


def _extract_email(identity: str) -> Optional[str]:
    m = re.search(r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}", identity)
    return m.group(0) if m else None


def build_sec_headers() -> Dict[str, str]:
    """Construct SEC-compliant headers strictly from environment variables.

    Required env vars:
      - SEC_APP_NAME: short identifier of your application (e.g., RA-Crew)
      - SEC_CONTACT_EMAIL: valid email address for contact per SEC guidelines
    Optional env vars:
      - SEC_IDENTITY_EXTRA: free-form note (e.g., research purpose)

    We purposely DO NOT fabricate or guess values. If mandatory variables are
    missing, we raise an exception to force explicit user configuration.
    """
    app = os.getenv("SEC_APP_NAME")
    email = os.getenv("SEC_CONTACT_EMAIL")
    extra = os.getenv("SEC_IDENTITY_EXTRA", "")

    if not app or not email:
        raise RuntimeError(
            "Missing SEC_APP_NAME or SEC_CONTACT_EMAIL env vars. Populate these in .env to proceed."
        )
    ua = f"{app} (Contact: {email}{'; ' + extra if extra else ''})"

    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.sec.gov/edgar/searchedgar/companysearch.html",
        "Origin": "https://www.sec.gov",
        "Cache-Control": "max-age=0",
        "Pragma": "no-cache",
    }
    headers["From"] = email
    return headers


# Create a session to reuse connections and apply defaults consistently
SESSION = requests.Session()
SESSION.headers.update(build_sec_headers())


EDGAR_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession_no_nodashes}/{filename}"


@sleep_and_retry
@limits(calls=SEC_RATE_LIMIT_CALLS, period=SEC_RATE_LIMIT_PERIOD)
def _get(url: str, *, accept: Optional[str] = None, referer: Optional[str] = None) -> requests.Response:
    headers: Dict[str, str] = {}
    if accept:
        headers["Accept"] = accept
    if referer:
        headers["Referer"] = referer
    # Ensure headers still contain current identity (in case env rotated mid-run)
    if "User-Agent" not in SESSION.headers:
        SESSION.headers.update(build_sec_headers())
    # small jitter to avoid bursty patterns
    time.sleep(random.uniform(0.05, 0.2))
    return SESSION.get(url, headers=headers, timeout=30)


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def _get_json(url: str) -> dict:
    r = _get(
        url,
        accept="application/json, text/javascript, */*; q=0.1",
        referer="https://www.sec.gov/edgar/searchedgar/companysearch.html",
    )
    r.raise_for_status()
    return r.json()


@timeit
def get_cik_for_ticker(ticker: str) -> Optional[str]:
    data = _get_json(EDGAR_TICKER_URL)
    ticker_upper = ticker.upper()
    for _, rec in data.items():
        if rec.get("ticker", "").upper() == ticker_upper:
            cik = str(rec.get("cik_str", "")).zfill(10)
            return cik
    return None


def _match_filing_type(form: str, target_types: List[str]) -> bool:
    form_upper = form.upper()
    return any(form_upper.startswith(t.strip().upper()) for t in target_types)


@timeit
def list_company_filings(cik: str, years: List[int], filing_types: List[str]) -> List[dict]:
    url = EDGAR_SUBMISSIONS.format(cik=cik)
    submissions = _get_json(url)
    recent = submissions.get("filings", {}).get("recent", {})
    out: List[dict] = []
    for i, form in enumerate(recent.get("form", [])):
        if not _match_filing_type(form, filing_types):
            continue
        year = int(recent.get("filingDate", [""])[i][:4] or 0)
        if years and year not in years:
            continue
        out.append(
            {
                "accessionNumber": recent.get("accessionNumber", [""])[i],
                "primaryDocument": recent.get("primaryDocument", [""])[i],
                "filingDate": recent.get("filingDate", [""])[i],
                "form": form,
            }
        )
    return out


def _acc_no_nodash(accession_no: str) -> str:
    return accession_no.replace("-", "")


@timeit
@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def download_filing(cik: str, accession_no: str, primary_document: str) -> str:
    cik_nozero = str(int(cik))  # remove leading zeros
    acc_no_nodashes = _acc_no_nodash(accession_no)
    url = EDGAR_ARCHIVES.format(
        cik_nozero=cik_nozero, accession_no_nodashes=acc_no_nodashes, filename=primary_document
    )
    r = _get(
        url,
        accept="text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        referer="https://www.sec.gov/edgar/browse/",
    )
    r.raise_for_status()
    return r.text


def is_xml_content(html: str) -> bool:
    content_head = (html or "").lstrip()[:512].lower()
    return (
        content_head.startswith("<?xml")
        or content_head.startswith("<xml")
        or ("xmlns" in content_head and "<html" not in content_head)
    )


@timeit
def html_to_text(html: str) -> str:
    """Convert HTML or XML content to plain text using an appropriate parser."""
    parser = "xml" if is_xml_content(html) else "lxml"
    try:
        soup = BeautifulSoup(html, parser)
    except Exception:
        # Fallback to lxml HTML parser if XML parser not available
        soup = BeautifulSoup(html, "lxml")
    for el in soup(["script", "style", "noscript"]):
        el.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text
