from __future__ import annotations

import os
from typing import Optional

MAPPING_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "utils", "ticker.txt")


def _load_local_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not os.path.exists(MAPPING_PATH):
        return mapping
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                ticker = parts[0].strip().lower()
                cik = parts[1].strip()
                mapping[ticker] = cik.zfill(10)
    return mapping


_CACHE = _load_local_mapping()


def is_cik(s: str) -> bool:
    return s.isdigit() and 7 <= len(s) <= 10


def normalize_cik(s: str) -> str:
    return s.zfill(10)


def ticker_to_cik(ticker: str) -> Optional[str]:
    return _CACHE.get(ticker.lower())
