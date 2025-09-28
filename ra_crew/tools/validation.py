from __future__ import annotations

import re
from typing import List, Dict, Any


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
        requested_year = r.get("requested_year") or r.get("year") or r.get("filing_year")
        extracted_year = r.get("extracted_year") or r.get("year")
        metric = r.get("metric", "")
        value_str = r.get("value", "")
        form = r.get("form", "")
        context = r.get("context", "")
        fallback_used = r.get("fallback_used")

        year_key = extracted_year or requested_year or ""
        key = (ticker, year_key, metric)
        
        # Check for duplicates
        if key in seen:
            issues.append(f"Duplicate entry for {ticker} {year_key} {metric}")
        else:
            seen.add(key)
        
        # Check for missing values
        if not value_str or value_str.strip() == "":
            issues.append(f"Missing value for {ticker} {year_key} {metric}")
            continue
        
        # Extract numeric value for validation
        numeric_value = _extract_numeric_value(value_str)

        # Check if extraction makes sense
        if numeric_value == 0.0 and value_str not in ["0", "$0", "None", "N/A"]:
            warnings.append(f"Could not parse numeric value from '{value_str}' for {ticker} {year_key} {metric}")

        # Track values by metric for cross-validation
        if metric not in value_stats:
            value_stats[metric] = []
        if numeric_value > 0:
            value_stats[metric].append(numeric_value)

        # Reasonableness checks
        reasonableness_issues = _validate_compensation_reasonableness(numeric_value, metric)
        issues.extend([f"{ticker} {year_key}: {issue}" for issue in reasonableness_issues])

        # Check if context seems relevant
        if context and len(context.strip()) < 10:
            warnings.append(f"Very short context for {ticker} {year_key} {metric}: '{context}'")

        if fallback_used and requested_year and extracted_year and str(requested_year) != str(extracted_year):
            warnings.append(f"Fallback year used for {ticker} {metric}: requested {requested_year} -> extracted {extracted_year}")
        
        # Validate form type matches expected content
        if 'compensation' in metric.lower() and form and 'DEF 14A' not in form and 'proxy' not in form.lower():
            warnings.append(f"Compensation data extracted from {form} instead of proxy (DEF 14A) for {ticker} {year_key}")
    
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
