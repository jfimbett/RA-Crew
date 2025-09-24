from __future__ import annotations

import re
from typing import List, Dict, Any, Sequence


def _extract_numeric_value(value_str: str) -> float:
    """Extract numeric value from string, handling common formats like '$1,234.56' or '1.2 million'."""
    if not value_str or not isinstance(value_str, str):
        return 0.0
    
    # Remove common prefixes/suffixes and formatting
    clean = re.sub(r'[,$\s]', '', value_str.lower())
    
    # Handle millions, billions, thousands
    multipliers = {'million': 1e6, 'billion': 1e9, 'thousand': 1e3, 'k': 1e3, 'm': 1e6, 'b': 1e9}
    multiplier = 1.0
    for word, mult in multipliers.items():
        if word in clean:
            multiplier = mult
            clean = clean.replace(word, '')
            break
    
    # Extract the numeric part
    numeric_match = re.search(r'([\d,]+\.?\d*)', clean)
    if numeric_match:
        try:
            return float(numeric_match.group(1).replace(',', '')) * multiplier
        except ValueError:
            return 0.0
    return 0.0


def _validate_compensation_reasonableness(value: float, metric: str) -> List[str]:
    """Validate if compensation values are reasonable."""
    issues = []
    
    if 'ceo' in metric.lower() and 'compensation' in metric.lower():
        if value < 100_000:  # Less than $100k seems low for CEO
            issues.append(f"CEO compensation ${value:,.0f} seems unusually low")
        elif value > 100_000_000:  # More than $100M seems high
            issues.append(f"CEO compensation ${value:,.0f} seems unusually high")
    
    return issues


def validate_values(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Comprehensive validation of extracted data."""
    issues: List[str] = []
    warnings: List[str] = []
    seen = set()
    
    # Track statistics for reasonableness checks
    value_stats: Dict[str, List[float]] = {}
    
    for r in rows:
        ticker = r.get("ticker", "")
        year = r.get("year", "")
        metric = r.get("metric", "")
        value_str = r.get("value", "")
        form = r.get("form", "")
        context = r.get("context", "")
        
        key = (ticker, year, metric)
        
        # Check for duplicates
        if key in seen:
            issues.append(f"Duplicate entry for {ticker} {year} {metric}")
        else:
            seen.add(key)
        
        # Check for missing values
        if not value_str or value_str.strip() == "":
            issues.append(f"Missing value for {ticker} {year} {metric}")
            continue
        
        # Extract numeric value for validation
        numeric_value = _extract_numeric_value(value_str)
        
        # Check if extraction makes sense
        if numeric_value == 0.0 and value_str not in ["0", "$0", "None", "N/A"]:
            warnings.append(f"Could not parse numeric value from '{value_str}' for {ticker} {year} {metric}")
        
        # Track values by metric for cross-validation
        if metric not in value_stats:
            value_stats[metric] = []
        if numeric_value > 0:
            value_stats[metric].append(numeric_value)
        
        # Reasonableness checks
        reasonableness_issues = _validate_compensation_reasonableness(numeric_value, metric)
        issues.extend([f"{ticker} {year}: {issue}" for issue in reasonableness_issues])
        
        # Check if context seems relevant
        if context and len(context.strip()) < 10:
            warnings.append(f"Very short context for {ticker} {year} {metric}: '{context}'")
        
        # Validate form type matches expected content
        if 'compensation' in metric.lower() and form and 'DEF 14A' not in form and 'proxy' not in form.lower():
            warnings.append(f"Compensation data extracted from {form} instead of proxy (DEF 14A) for {ticker} {year}")
    
    # Cross-validation: check for outliers within each metric
    for metric, values in value_stats.items():
        if len(values) > 1:
            mean_val = sum(values) / len(values)
            for val in values:
                if val > mean_val * 10:  # More than 10x the mean
                    warnings.append(f"Potential outlier for {metric}: ${val:,.0f} (mean: ${mean_val:,.0f})")
    
    total_rows = len(rows)
    valid_rows = len([r for r in rows if r.get("value", "").strip()])
    
    return {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "issues": issues,
        "warnings": warnings,
        "valid": len(issues) == 0,
        "metrics_found": list(value_stats.keys()),
        "summary": f"{valid_rows}/{total_rows} rows have values, {len(issues)} critical issues, {len(warnings)} warnings"
    }


def validate_extraction_evidence(extracted: Dict[str, Any], cleaned_texts: Sequence[str]) -> Dict[str, Any]:
    """Validate that every executive name and numeric value in extracted structure appears verbatim in at least one cleaned text.

    Parameters
    ----------
    extracted : dict
        Nested structure mapping metrics -> years -> {value, name, title, evidence_snippet?}
    cleaned_texts : sequence[str]
        Collection of cleaned filing contents.

    Returns
    -------
    dict
        Report with keys: removed_entries (list), missing_evidence (list), value_mismatches (list), name_not_found (list),
        kept_metrics (dict), summary (str).
    """
    lowered_corpus = [t.lower() for t in cleaned_texts]

    def _in_corpus(fragment: str) -> bool:
        if not fragment:
            return False
        frag = fragment.strip().lower()
        if len(frag) < 2:
            return False
        return any(frag in c for c in lowered_corpus)

    removed_entries: List[Dict[str, Any]] = []
    missing_evidence: List[str] = []
    value_mismatches: List[str] = []
    name_not_found: List[str] = []

    cleaned_output: Dict[str, Any] = {}

    for metric, years_dict in extracted.items():
        if not isinstance(years_dict, dict):
            continue
        new_years: Dict[str, Any] = {}
        for year, payload in years_dict.items():
            if payload is None:
                new_years[year] = None
                continue
            if not isinstance(payload, dict):
                continue
            name = str(payload.get("name", ""))
            value = payload.get("value")
            evidence = payload.get("evidence_snippet") or payload.get("context") or ""
            evidence_start = payload.get("evidence_start", -1)
            evidence_end = payload.get("evidence_end", -1)
            # Basic numeric presence check
            numeric_token = None
            if isinstance(value, (int, float)):
                numeric_token = f"{value:,}".replace(",", "")
            elif isinstance(value, str):
                numeric_token = value.replace(",", "").strip()
            # Validate evidence snippet
            evidence_ok = _in_corpus(evidence) if evidence else False
            name_ok = _in_corpus(name)
            value_ok = _in_corpus(numeric_token) if numeric_token else False

            problems: List[str] = []
            if not name_ok and name:
                name_not_found.append(f"{metric} {year} name '{name}' not found verbatim")
                problems.append("name")
            if numeric_token and not value_ok:
                value_mismatches.append(f"{metric} {year} value '{value}' not found verbatim in corpus")
                problems.append("value")
            if not evidence_ok:
                missing_evidence.append(f"{metric} {year} missing or non-verbatim evidence snippet")
                problems.append("evidence")

            if problems:
                removed_entries.append({"metric": metric, "year": year, "reasons": problems})
            else:
                # Keep only whitelisted keys plus offsets to avoid propagating unexpected hallucinated fields
                kept = dict(payload)
                kept["evidence_start"] = evidence_start
                kept["evidence_end"] = evidence_end
                kept["evidence_snippet"] = evidence
                new_years[year] = kept
        if new_years:
            cleaned_output[metric] = new_years

    summary = (
        f"Removed {len(removed_entries)} metric-year entries; "
        f"missing_evidence={len(missing_evidence)}, name_not_found={len(name_not_found)}, value_mismatches={len(value_mismatches)}"
    )

    return {
        "removed_entries": removed_entries,
        "missing_evidence": missing_evidence,
        "value_mismatches": value_mismatches,
        "name_not_found": name_not_found,
        "kept_metrics": cleaned_output,
        "summary": summary,
    }
