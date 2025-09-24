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


SEC_RATE_LIMIT_CALLS = 10
SEC_RATE_LIMIT_PERIOD = 1  # 10 requests per second max per SEC guidance


def _extract_email(identity: str) -> Optional[str]:
    m = re.search(r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}", identity)
    return m.group(0) if m else None


def _base_headers() -> Dict[str, str]:
    # Realistic browser-like headers while preserving SEC-required identity in UA
    # Example UA embeds app name and contact per SEC guidelines
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36 "
        f"RA-Agent/0.1 (Contact: {settings.edgar_identity})"
    )
    headers = {
        "User-Agent": ua,
        # Note: Accept is set per-request (JSON vs HTML). This is a safe default.
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        # Avoid setting Host manually; requests will manage it.
        # Provide a reasonable referer to look more like a browser navigation.
        "Referer": "https://www.sec.gov/edgar/searchedgar/companysearch.html",
        "Origin": "https://www.sec.gov",
        "Cache-Control": "max-age=0",
        "Pragma": "no-cache",
        # Client hints used by Chromium-based browsers
        "sec-ch-ua": '"Chromium";v="125", "Not.A/Brand";v="24", "Google Chrome";v="125"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        # Fetch metadata headers
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
    }
    email = _extract_email(settings.edgar_identity)
    if email:
        headers["From"] = email
    return headers


# Create a session to reuse connections and apply defaults consistently
SESSION = requests.Session()
SESSION.headers.update(_base_headers())


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


@timeit
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for el in soup(["script", "style", "noscript"]):
        el.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text
