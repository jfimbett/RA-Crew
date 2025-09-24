from __future__ import annotations

from typing import List, Dict, Any


def validate_values(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[str] = []
    seen = set()
    for r in rows:
        key = (r.get("ticker"), r.get("year"), r.get("metric"))
        if key in seen:
            issues.append(f"Duplicate entry for {key}")
        else:
            seen.add(key)
        val = r.get("value")
        if val in (None, ""):
            issues.append(f"Missing value for {key}")
    return {"issues": issues, "valid": len(issues) == 0}
