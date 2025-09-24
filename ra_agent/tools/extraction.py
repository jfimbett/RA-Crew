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


@timeit
def extract_metric_with_hints(text: str, metric: str, hints: Dict[str, Any] | None) -> Dict[str, str]:
    """Use section hints to narrow the search window; fallback to full-text heuristic."""
    if not hints:
        return extract_metric(text, metric)
    try:
        # Normalize hints structure
        hint_items = hints.get("hints", []) if isinstance(hints, dict) else []
        sections: List[str] = []
        for h in hint_items:
            # If metrics filter present, ensure it matches loosely
            mlist = [m.lower() for m in h.get("metrics", [])]
            if mlist and metric.lower() not in mlist:
                continue
            sections.extend(h.get("sections", []))
        sections = [s for s in sections if s]
        if not sections:
            return extract_metric(text, metric)
        windows = _windows_by_sections(text, sections)
        for w in windows:
            res = extract_metric(w, metric)
            if res.get("value"):
                return res
        # Fallback to full text
        return extract_metric(text, metric)
    except Exception:
        # Defensive fallback
        return extract_metric(text, metric)


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
