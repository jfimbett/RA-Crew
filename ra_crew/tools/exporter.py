from __future__ import annotations

import os
import pandas as pd
import orjson
from typing import List, Dict, Any
from ..config import settings


def export_rows(rows: List[Dict[str, Any]], fmt: str, name: str = "results") -> str:
    os.makedirs(settings.outputs_dir, exist_ok=True)
    if fmt.lower() == "csv":
        df = pd.DataFrame(rows)
        path = os.path.join(settings.outputs_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        return path
    else:
        path = os.path.join(settings.outputs_dir, f"{name}.json")
        with open(path, "wb") as f:
            f.write(orjson.dumps(rows, option=orjson.OPT_INDENT_2))
        return path
