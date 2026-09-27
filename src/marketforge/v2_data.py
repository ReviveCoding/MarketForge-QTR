from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from .data import data_root, sha256_file
from .probe import atomic_json
from .state import ROOT, content_hash

NEW_PUBLIC_FILES = {
    "xnas_mbo": ROOT / "data/raw/databento/xnas-itch-equities-mbo.csv",
    "xnas_mbp10": ROOT / "data/raw/databento/xnas-itch-equities-mbp10.csv",
    "ifeu_mbo": ROOT / "data/raw/databento/ifeu-impact-futures-mbo.csv",
    "ifeu_mbp10": ROOT / "data/raw/databento/ifeu-impact-futures-mbp10.csv",
}


def _coverage(path: Path) -> dict[str, object]:
    con = duckdb.connect()
    row = con.execute(
        """SELECT count(*), count(DISTINCT symbol), count(DISTINCT instrument_id),
        min(ts_event)::VARCHAR, max(ts_event)::VARCHAR
        FROM read_csv_auto(?)""",
        [str(path)],
    ).fetchone()
    symbols = con.execute(
        "SELECT symbol, instrument_id, count(*) n FROM read_csv_auto(?) GROUP BY ALL ORDER BY n DESC",
        [str(path)],
    ).fetchall()
    con.close()
    return {
        "rows": row[0],
        "symbol_count": row[1],
        "instrument_count": row[2],
        "timestamp_bounds": [row[3], row[4]],
        "symbols": [
            {"symbol": symbol, "instrument_id": instrument_id, "rows": count}
            for symbol, instrument_id, count in symbols
        ],
    }


def register_new_public_data() -> Path:
    output = data_root() / "manifests" / "v2_public_data_acquisition.json"
    if output.exists():
        return output
    missing = [str(path) for path in NEW_PUBLIC_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    free = shutil.disk_usage(ROOT).free
    reserve = 40 * 1024**3
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provider": "Databento official public samples",
        "official_page": "https://databento.com/tick-data",
        "terms_page": "https://databento.com/legal",
        "local_research_only_no_redistribution": True,
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "coverage": _coverage(path),
            }
            for name, path in NEW_PUBLIC_FILES.items()
        },
        "free_bytes_after_acquisition": free,
        "reserve_bytes": reserve,
        "reserve_satisfied": free >= reserve,
    }
    if not payload["reserve_satisfied"]:
        raise RuntimeError("40 GiB storage reserve violated")
    atomic_json(output, payload)
    return output


def isolate_v2_lockbox() -> Path:
    acquisition = json.loads(register_new_public_data().read_text(encoding="utf-8"))
    output = data_root() / "final_lockbox_v2" / "LOCKBOX_POLICY.json"
    if output.exists():
        return output
    xnas = {name: acquisition["files"][name] for name in ("xnas_mbo", "xnas_mbp10")}
    payload: dict[str, object] = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "SEALED_UNTIL_FINAL_FREEZE_V2",
        "role": "STRICT_SOURCE_ASSET_INSTRUMENT_OOD_FINAL_LOCKBOX_V2",
        "source": "databento",
        "dataset": "XNAS.ITCH",
        "instrument": "NVDA",
        "instrument_id": 11667,
        "trading_date": "2025-09-16",
        "files": xnas,
        "metadata_inspection_only": ["row counts", "symbols", "timestamps", "action counts"],
        "outcomes_predictions_labels_not_inspected": True,
        "development_access": "DENY",
        "v1_lockbox_reused": False,
        "v1_artifacts_excluded_from_v2_selection": True,
    }
    payload["policy_hash"] = content_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return output


def canonicalize_ifeu_v2() -> Path:
    raw = NEW_PUBLIC_FILES["ifeu_mbo"]
    target = data_root() / "canonical_v2" / "ifeu_impact_mbo_v2"
    marker = data_root() / "manifests" / "canonical_ifeu_impact_mbo_v3.json"
    raw_hash = sha256_file(raw)
    if marker.exists() and target.exists():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current["raw_sha256"] == raw_hash:
            return marker
    if target.exists():
        raise RuntimeError("unmarked v2 canonical target exists")
    staging = target.parent / f".stage-{uuid.uuid4().hex}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    raw_sql = str(raw).replace("'", "''")
    stage_sql = str(staging).replace("'", "''")
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=true")
    query = f"""WITH ordered AS (
      SELECT row_number() OVER ()::UBIGINT AS ingest_index, *,
        epoch_ns(CAST(ts_event AS TIMESTAMPTZ))::BIGINT AS ts_event_exact_ns,
        epoch_ns(CAST(ts_recv AS TIMESTAMPTZ))::BIGINT AS ts_recv_exact_ns
      FROM read_csv_auto('{raw_sql}')
    ), reset_marked AS (
      SELECT *, sum(CASE WHEN action='R' THEN 1 ELSE 0 END) OVER (
        PARTITION BY instrument_id ORDER BY ingest_index ROWS UNBOUNDED PRECEDING
      )::INTEGER AS reset_generation
      FROM ordered
    ) SELECT
      ingest_index, 'databento'::VARCHAR AS source, 'IFEU_IMPACT'::VARCHAR AS venue,
      'futures'::VARCHAR AS asset_class, symbol::VARCHAR AS instrument,
      symbol::VARCHAR AS contract, instrument_id::UBIGINT AS instrument_id,
      ts_event_exact_ns AS ts_event_ns, ts_recv_exact_ns AS ts_recv_ns,
      CAST(timezone('Europe/London', CAST(ts_recv AS TIMESTAMPTZ)) AS DATE) AS trading_date,
      concat(CAST(CAST(timezone('Europe/London', CAST(ts_recv AS TIMESTAMPTZ)) AS DATE) AS VARCHAR), '-', reset_generation)::VARCHAR AS session_id,
      reset_generation, sequence::UBIGINT AS sequence, action::VARCHAR AS action,
      side::VARCHAR AS side, order_id::UBIGINT AS order_id, price::DOUBLE AS price,
      size::UBIGINT AS size, flags::UBIGINT AS source_flags,
      0.01::DOUBLE AS tick_size, 1000.0::DOUBLE AS contract_multiplier,
      10.0::DOUBLE AS tick_value, 'USD'::VARCHAR AS currency
    FROM reset_marked"""
    con.execute(
        f"COPY ({query}) TO '{stage_sql}' (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (source, instrument, trading_date), ROW_GROUP_SIZE 100000)"
    )
    metrics = con.execute(
        f"SELECT count(*), min(ts_event_ns), max(ts_event_ns), count(DISTINCT trading_date), max(reset_generation) FROM ({query})"
    ).fetchone()
    con.close()
    os.replace(staging, target)
    payload = {
        "schema_version": 3,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "raw_path": str(raw),
        "raw_sha256": raw_hash,
        "canonical_root": str(target),
        "parser": "v3-exact-timestamps-reception-session-snapshot-aware",
        "session_clock": "ts_recv",
        "event_clock": "ts_event",
        "snapshot_semantics": "historical ts_event retained; grouping uses ordered reception stream",
        "row_count": metrics[0],
        "timestamp_bounds_ns": [metrics[1], metrics[2]],
        "calendar_dates_present": metrics[3],
        "reset_generations": metrics[4],
        "instrument_spec_key": "databento:IFEU.IMPACT:BRN",
        "final_lockbox_v2_used": False,
    }
    atomic_json(marker, payload)
    return marker


def canonicalize_es_v2() -> Path:
    source_marker_path = data_root() / "manifests" / "canonical_databento_mbo_v2.json"
    source_marker = json.loads(source_marker_path.read_text(encoding="utf-8"))
    source_root = Path(source_marker["canonical_root"])
    target = data_root() / "canonical_v2" / "glbx_es_mbo_v1"
    marker = data_root() / "manifests" / "canonical_glbx_es_mbo_v2.json"
    source_hash = content_hash(source_marker)
    if target.exists() and marker.exists():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current["source_manifest_hash"] == source_hash:
            return marker
    if target.exists():
        raise RuntimeError("unmarked ES v2 canonical target exists")
    staging = target.parent / f".stage-{uuid.uuid4().hex}"
    parquet_glob = str(source_root / "**" / "*.parquet").replace("\\", "/").replace("'", "''")
    stage_sql = str(staging).replace("'", "''")
    con = duckdb.connect()
    con.execute("SET threads=1")
    query = f"""WITH ordered AS (
      SELECT *, sum(CASE WHEN action='R' THEN 1 ELSE 0 END) OVER (
        PARTITION BY instrument_id ORDER BY ingest_index ROWS UNBOUNDED PRECEDING
      )::INTEGER AS reset_generation
      FROM read_parquet('{parquet_glob}', hive_partitioning=true)
    ) SELECT ingest_index, source, venue, asset_class, instrument,
      raw_contract AS contract, instrument_id, ts_event_ns, ts_recv_ns,
      CAST(timezone('America/Chicago', make_timestamp_ns(ts_event_ns)) + INTERVAL '7 hours' AS DATE) AS trading_date,
      concat(CAST(CAST(timezone('America/Chicago', make_timestamp_ns(ts_event_ns)) + INTERVAL '7 hours' AS DATE) AS VARCHAR), '-', reset_generation)::VARCHAR AS session_id,
      reset_generation, sequence, action, side, order_id, price, size, source_flags,
      0.25::DOUBLE AS tick_size, 50.0::DOUBLE AS contract_multiplier,
      12.5::DOUBLE AS tick_value, 'USD'::VARCHAR AS currency
    FROM ordered ORDER BY ingest_index"""
    con.execute(
        f"COPY ({query}) TO '{stage_sql}' (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (source, instrument, trading_date), ROW_GROUP_SIZE 100000)"
    )
    metrics = con.execute(
        f"SELECT count(*), min(ts_event_ns), max(ts_event_ns), count(DISTINCT trading_date), max(reset_generation) FROM ({query})"
    ).fetchone()
    con.close()
    os.replace(staging, target)
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest": str(source_marker_path),
        "source_manifest_hash": source_hash,
        "canonical_root": str(target),
        "parser": "v2-audited-repartition-grouped",
        "row_count": metrics[0],
        "timestamp_bounds_ns": [metrics[1], metrics[2]],
        "trading_days": metrics[3],
        "reset_generations": metrics[4],
        "instrument_spec_key": "databento:GLBX.MDP3:ES",
        "final_lockbox_v2_used": False,
    }
    atomic_json(marker, payload)
    return marker


def canonicalize_xnas_final_v2() -> Path:
    """Open and canonicalize the sealed XNAS lockbox only after the v2 freeze."""
    freeze = ROOT / "artifacts/final_freeze_manifest_v2.json"
    opened = data_root() / "final_lockbox_v2/OPENED_ONCE.json"
    if not freeze.is_file() or not opened.is_file():
        raise PermissionError("XNAS final lockbox requires the frozen v2 protocol and open marker")
    raw = NEW_PUBLIC_FILES["xnas_mbo"]
    target = data_root() / "final_lockbox_v2/canonical_xnas_mbo_v2"
    marker = data_root() / "final_lockbox_v2/manifests/canonical_xnas_mbo_v2.json"
    raw_hash = sha256_file(raw)
    if marker.exists() and target.exists():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current["raw_sha256"] == raw_hash:
            return marker
    if target.exists():
        raise RuntimeError("unmarked XNAS final canonical target exists")
    staging = target.parent / f".stage-{uuid.uuid4().hex}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    raw_sql = str(raw).replace("'", "''")
    stage_sql = str(staging).replace("'", "''")
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=true")
    query = f"""WITH ordered AS (
      SELECT row_number() OVER ()::UBIGINT AS ingest_index, *,
        epoch_ns(CAST(ts_event AS TIMESTAMPTZ))::BIGINT AS ts_event_exact_ns,
        epoch_ns(CAST(ts_recv AS TIMESTAMPTZ))::BIGINT AS ts_recv_exact_ns
      FROM read_csv_auto('{raw_sql}')
    ), reset_marked AS (
      SELECT *, sum(CASE WHEN action='R' THEN 1 ELSE 0 END) OVER (
        PARTITION BY instrument_id ORDER BY ingest_index ROWS UNBOUNDED PRECEDING
      )::INTEGER AS reset_generation
      FROM ordered
    ) SELECT ingest_index, 'databento'::VARCHAR AS source,
      'XNAS_ITCH'::VARCHAR AS venue, 'equity'::VARCHAR AS asset_class,
      symbol::VARCHAR AS instrument, symbol::VARCHAR AS contract,
      instrument_id::UBIGINT AS instrument_id, ts_event_exact_ns AS ts_event_ns,
      ts_recv_exact_ns AS ts_recv_ns,
      CAST(timezone('America/New_York', CAST(ts_recv AS TIMESTAMPTZ)) AS DATE) AS trading_date,
      concat(CAST(CAST(timezone('America/New_York', CAST(ts_recv AS TIMESTAMPTZ)) AS DATE) AS VARCHAR), '-', reset_generation)::VARCHAR AS session_id,
      reset_generation, sequence::UBIGINT AS sequence, action::VARCHAR AS action,
      side::VARCHAR AS side, order_id::UBIGINT AS order_id, price::DOUBLE AS price,
      size::UBIGINT AS size, flags::UBIGINT AS source_flags,
      0.01::DOUBLE AS tick_size, 1.0::DOUBLE AS contract_multiplier,
      0.01::DOUBLE AS tick_value, 'USD'::VARCHAR AS currency
    FROM reset_marked"""
    con.execute(
        f"COPY ({query}) TO '{stage_sql}' (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (source, instrument, trading_date), ROW_GROUP_SIZE 100000)"
    )
    metrics = con.execute(
        f"SELECT count(*), min(ts_event_ns), max(ts_event_ns), count(DISTINCT trading_date), max(reset_generation) FROM ({query})"
    ).fetchone()
    con.close()
    os.replace(staging, target)
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "raw_path": str(raw),
        "raw_sha256": raw_hash,
        "canonical_root": str(target),
        "parser": "v2-exact-timestamps-reception-session-snapshot-aware",
        "session_clock": "ts_recv",
        "event_clock": "ts_event",
        "row_count": metrics[0],
        "timestamp_bounds_ns": [metrics[1], metrics[2]],
        "calendar_dates_present": metrics[3],
        "reset_generations": metrics[4],
        "instrument_spec_key": "databento:XNAS.ITCH:NVDA",
        "final_lockbox_v2_used": True,
    }
    atomic_json(marker, payload)
    return marker
