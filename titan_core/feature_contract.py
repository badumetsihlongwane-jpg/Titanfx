from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Dict, List


def validate_schema(schema: Dict) -> None:
    required = {"pairs", "feats_per_node", "shared_cols", "node_cols"}
    missing = required - set(schema.keys())
    if missing:
        raise ValueError(f"Schema missing required keys: {sorted(missing)}")

    feats = int(schema["feats_per_node"])
    for pair in schema["pairs"]:
        cols = schema["node_cols"].get(pair)
        if cols is None:
            raise ValueError(f"Schema missing node columns for pair '{pair}'")
        if len(cols) != feats:
            raise ValueError(
                f"Schema pair '{pair}' has {len(cols)} features, expected {feats}"
            )


def save_feature_schema(path: str, pairs: List[str], feats_per_node: int,
                        shared_cols: List[str], node_cols: Dict[str, List[str]]) -> Dict:
    schema = {
        "schema_version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": pairs,
        "feats_per_node": int(feats_per_node),
        "shared_cols": shared_cols,
        "node_cols": node_cols,
    }
    validate_schema(schema)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    return schema


def load_feature_schema(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validate_schema(schema)
    return schema
