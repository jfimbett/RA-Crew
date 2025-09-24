from __future__ import annotations

import os
import re
import time
import json
from typing import Dict, List, Optional, Tuple
import requests
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup

from ..config import settings
from ..utils.logging_utils import timeit


SEC_RATE_LIMIT_CALLS = 10
SEC_RATE_LIMIT_PERIOD = 1  # 10 requests per second max per SEC guidance


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": f"RA-Agent/0.1 ({settings.edgar_identity})",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }


EDGAR_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession_no_nodashes}/{filename}"


@sleep_and_retry
@limits(calls=SEC_RATE_LIMIT_CALLS, period=SEC_RATE_LIMIT_PERIOD)
def _get(url: str) -> requests.Response:
    return requests.get(url, headers=_headers(), timeout=30)


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def _get_json(url: str) -> dict:
    r = _get(url)
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
def download_filing(cik: str, accession_no: str, primary_document: str) -> str:
    cik_nozero = str(int(cik))  # remove leading zeros
    acc_no_nodashes = _acc_no_nodash(accession_no)
    url = EDGAR_ARCHIVES.format(
        cik_nozero=cik_nozero, accession_no_nodashes=acc_no_nodashes, filename=primary_document
    )
    r = _get(url)
    r.raise_for_status()
    return r.text


@timeit
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for el in soup(["script", "style", "noscript"]):
        el.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text
