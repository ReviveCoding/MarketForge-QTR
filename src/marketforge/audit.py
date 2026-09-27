from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from .data import data_root
from .probe import atomic_json


def audit_mbo(path: Path) -> Path:
    con = duckdb.connect()
    escaped = str(path).replace("'", "''")
    relation = f"read_csv_auto('{escaped}', header=true, timestampformat='%Y-%m-%dT%H:%M:%S.%nZ')"
    summary = con.execute(
        f"""SELECT count(*) AS row_count,
        min(ts_event) AS min_ts_event, max(ts_event) AS max_ts_event,
        min(ts_recv) AS min_ts_recv, max(ts_recv) AS max_ts_recv,
        count(DISTINCT symbol) AS instruments,
        count(*) FILTER (WHERE size < 0) AS negative_sizes,
        count(*) FILTER (WHERE price <= 0 AND action NOT IN ('R')) AS invalid_prices,
        count(*) FILTER (WHERE ts_recv < ts_event) AS recv_before_event,
        count(*) FILTER (WHERE action NOT IN ('A','C','M','R','T','F','N')) AS unknown_actions,
        count(*) FILTER (WHERE side NOT IN ('A','B','N')) AS unknown_sides
        FROM {relation}"""
    ).fetchone()
    columns = [column[0] for column in con.description]
    metrics = dict(zip(columns, summary, strict=True))
    for key, value in list(metrics.items()):
        if hasattr(value, "isoformat"):
            metrics[key] = value.isoformat()
    severities: list[dict[str, object]] = []
    for name in ("negative_sizes", "invalid_prices", "unknown_actions", "unknown_sides"):
        if metrics[name]:
            severities.append({"severity": "ERROR", "check": name, "count": metrics[name]})
    if metrics["recv_before_event"]:
        severities.append(
            {
                "severity": "WARN",
                "check": "recv_before_event",
                "count": metrics["recv_before_event"],
            }
        )
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_file": str(path),
        "parser": "duckdb-read_csv_auto-v1",
        "metrics": metrics,
        "findings": severities,
        "quality_status": "ERROR"
        if any(x["severity"] == "ERROR" for x in severities)
        else "PASS_WITH_WARNINGS"
        if severities
        else "PASS",
    }
    target = data_root() / "manifests" / "qa_databento_GLBX.MDP3_mbo.json"
    atomic_json(target, payload)
    con.close()
    return target


def read_quality(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
