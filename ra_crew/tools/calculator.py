from __future__ import annotations

from typing import Dict, Any, List
import re


def compute_metric(formula: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    # Evaluate strictly with provided variables
    allowed_names = {k: v for k, v in variables.items()}
    allowed_names["__builtins__"] = {}
    try:
        value = eval(formula, allowed_names)  # noqa: S307 - controlled context
        return {"formula": formula, "variables": variables, "value": value}
    except Exception as e:
        return {"formula": formula, "variables": variables, "error": str(e)}


_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_PY_KEYWORDS = {"for", "if", "else", "and", "or", "not", "in", "is", "True", "False", "None"}


def _extract_vars(formula: str) -> List[str]:
    names = set(_IDENT_RE.findall(formula))
    return [n for n in names if n not in _PY_KEYWORDS and not n.isupper() and not n.startswith("__")]


def compute_derived_metrics(calculation_expressions: Dict[str, str], primitive: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute derived metrics year-wise.

    Parameters
    ----------
    calculation_expressions: mapping of derived metric name -> python expression using primitive metric identifiers
    primitive: mapping primitive_metric -> {year: value or {value: v}}
    """
    results: Dict[str, Any] = {}
    # Build per-year primitive numeric lookup
    # Normalize primitive values structure
    norm: Dict[str, Dict[str, float]] = {}
    for metric, year_map in primitive.items():
        norm[metric] = {}
        for year, val in year_map.items():
            if isinstance(val, dict):
                val_raw = val.get("value")
            else:
                val_raw = val
            try:
                norm[metric][str(year)] = float(val_raw) if val_raw not in (None, "", "null") else None
            except (TypeError, ValueError):
                norm[metric][str(year)] = None

    # Collect all years across primitives
    all_years = set()
    for ymap in norm.values():
        all_years.update(ymap.keys())

    for dname, formula in calculation_expressions.items():
        vars_needed = _extract_vars(formula)
        d_result: Dict[str, Any] = {}
        for year in all_years:
            inputs = {}
            missing = []
            for v in vars_needed:
                val = norm.get(v, {}).get(year)
                if val is None:
                    missing.append(v)
                else:
                    inputs[v] = val
            if missing:
                d_result[year] = {"value": None, "missing": missing}
                continue
            eval_out = compute_metric(formula, inputs)
            d_result[year] = {"value": eval_out.get("value"), "inputs": inputs, "formula": formula, "error": eval_out.get("error")}
        results[dname] = d_result
    return results
