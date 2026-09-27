from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .data import data_root
from .state import ROOT, content_hash


def append_result(record: dict[str, object]) -> Path:
    warehouse = ROOT / "results" / "warehouse" / "results.parquet"
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "run_id": content_hash(record)[:16],
        **record,
    }
    incoming = pd.DataFrame([row])
    if warehouse.exists():
        existing = pd.read_parquet(warehouse)
        if row["run_id"] in set(existing["run_id"]):
            return warehouse
        incoming = pd.concat([existing, incoming], ignore_index=True)
    temporary = warehouse.with_suffix(".parquet.tmp")
    incoming.to_parquet(temporary, index=False)
    os.replace(temporary, warehouse)
    return warehouse


def dataset_fingerprint() -> str:
    manifest = data_root() / "manifests" / "splits_databento_mbo_v3.json"
    return content_hash(json.loads(manifest.read_text(encoding="utf-8")))
