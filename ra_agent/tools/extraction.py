from __future__ import annotations

import re
from typing import Dict, List, Tuple
from rapidfuzz import process, fuzz
from ..utils.logging_utils import timeit


@timeit
def extract_metric(text: str, metric: str) -> Dict[str, str]:
    # Heuristics: find best-matching lines and regex for numbers
    lines = text.split("\n")
    choices = [ln[:200] for ln in lines]
    best = process.extract(metric, choices, scorer=fuzz.token_sort_ratio, limit=5)
    numeric_pattern = re.compile(r"\$?\(?\b[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?\)?\b")
    for score_item in best:
        idx = score_item[2]
        window = "\n".join(lines[max(0, idx - 2) : min(len(lines), idx + 3)])
        nums = numeric_pattern.findall(window)
        if nums:
            value = nums[0]
            return {"metric": metric, "value": value, "context": window}
    return {"metric": metric, "value": "", "context": ""}
