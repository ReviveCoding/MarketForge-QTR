from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from .data import data_root, sha256_file
from .probe import atomic_json


def _epoch_ns(column: str) -> str:
    return (
        f"epoch_ns(strptime(substr({column}, 1, 19), '%Y-%m-%dT%H:%M:%S')) "
        f"+ CAST(substr({column}, 21, 9) AS BIGINT)"
    )


def canonicalize_mbo(raw: Path) -> Path:
    root = data_root()
    target = root / "canonical" / "databento_mbo_v2"
    marker = root / "manifests" / "canonical_databento_mbo_v2.json"
    raw_hash = sha256_file(raw)
    if marker.exists() and target.exists():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current.get("raw_sha256") == raw_hash:
            return marker
    if target.exists():
        raise RuntimeError(
            "canonical target exists without matching completion marker; manual review required"
        )
    staging = target.parent / f".stage-{uuid.uuid4().hex}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=true")
    raw_sql = str(raw).replace("'", "''")
    stage_sql = str(staging).replace("'", "''")
    null_lob = ",\n".join(
        f"NULL::DOUBLE AS {side}_{kind}_{level:02d}"
        for level in range(1, 11)
        for side in ("bid", "ask")
        for kind in ("px", "sz", "ct")
    )
    query = f"""SELECT
        row_number() OVER ()::UBIGINT AS ingest_index,
        'databento'::VARCHAR AS source, 'CME_GLOBEX'::VARCHAR AS venue,
        'futures'::VARCHAR AS asset_class, symbol::VARCHAR AS instrument,
        symbol::VARCHAR AS raw_contract, instrument_id::UBIGINT AS instrument_id,
        {_epoch_ns("ts_event")}::BIGINT AS ts_event_ns,
        {_epoch_ns("ts_recv")}::BIGINT AS ts_recv_ns,
        CAST(substr(ts_event, 1, 10) AS DATE) AS trading_date,
        substr(ts_event, 1, 10)::VARCHAR AS session_id, sequence::UBIGINT AS sequence,
        'MBO'::VARCHAR AS event_type, action::VARCHAR AS action, side::VARCHAR AS side,
        order_id::UBIGINT AS order_id, price::DOUBLE AS price, size::UBIGINT AS size,
        {null_lob},
        NULL::DOUBLE AS tick_size, NULL::DATE AS expiration,
        NULL::INTEGER AS days_to_expiry, NULL::INTEGER AS roll_rank,
        false AS is_roll_window, 0::UBIGINT AS missing_mask,
        flags::UBIGINT AS source_flags, 0::UBIGINT AS qa_flags
        FROM read_csv('{raw_sql}', header=true,
          types={{'ts_event':'VARCHAR','ts_recv':'VARCHAR'}}, auto_detect=true)"""
    con.execute(
        f"COPY ({query}) TO '{stage_sql}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (source, instrument, trading_date), ROW_GROUP_SIZE 100000)"
    )
    metrics = con.execute(
        f"SELECT count(*), min(ts_event_ns), max(ts_event_ns) FROM ({query})"
    ).fetchone()
    con.close()
    os.replace(staging, target)
    parquet_files = list(target.rglob("*.parquet"))
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "raw_path": str(raw),
        "raw_sha256": raw_hash,
        "canonical_root": str(target),
        "parser_version": "databento-mbo-canonical-v2-ordered",
        "row_count": metrics[0],
        "timestamp_bounds_ns": [metrics[1], metrics[2]],
        "parquet_files": len(parquet_files),
        "parquet_bytes": sum(path.stat().st_size for path in parquet_files),
        "dataset_fingerprint": sha256_file(raw),
    }
    atomic_json(marker, payload)
    return marker
