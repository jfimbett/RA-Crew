from __future__ import annotations

from typing import Dict, Any


def compute_metric(formula: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    # Evaluate strictly with provided variables
    allowed_names = {k: v for k, v in variables.items()}
    allowed_names["__builtins__"] = {}
    try:
        value = eval(formula, allowed_names)  # noqa: S307 - controlled context
        return {"formula": formula, "variables": variables, "value": value}
    except Exception as e:
        return {"formula": formula, "variables": variables, "error": str(e)}
