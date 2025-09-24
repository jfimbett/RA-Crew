from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any
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


def _windows_by_sections(text: str, sections: List[str], span: int = 4000) -> List[str]:
    """Internal helper retained for potential future use.

    Deprecated: JSON-based section hints are no longer supported at the CLI layer.
    """
    lowered = text.lower()
    windows: List[str] = []
    for sec in sections:
        if not sec:
            continue
        s = sec.lower()
        idx = lowered.find(s)
        if idx != -1:
            start = max(0, idx - span // 2)
            end = min(len(text), idx + span // 2)
            windows.append(text[start:end])
    return windows


"""JSON-based section hints path has been removed; use extract_metric_with_hint_text instead."""


@timeit
def extract_metric_with_hint_text(text: str, metric: str, hint_text: str | None, span: int = 5000) -> Dict[str, str]:
    """Use a free-text hint to locate a nearby window and extract the metric.

    Example hint: "In DEF 14A, the Summary Compensation Table contains the CEO total compensation."
    """
    if not hint_text:
        return extract_metric(text, metric)
    # Find the best matching location in the text using fuzzy line matching
    lines = text.split("\n")
    choices = [ln[:200] for ln in lines]
    best = process.extractOne(hint_text, choices, scorer=fuzz.token_sort_ratio)
    if best is not None:
        idx = best[2]
        # Build a window around that line by character count
        # Approximate char index by summing lengths of preceding lines
        char_pos = sum(len(l) + 1 for l in lines[:idx])
        start = max(0, char_pos - span // 2)
        end = min(len(text), char_pos + span // 2)
        window = text[start:end]
        res = extract_metric(window, metric)
        if res.get("value"):
            return res
    # Fallback
    return extract_metric(text, metric)
