from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_DB = ROOT / "artifacts" / "pipeline_state.sqlite3"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def content_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def database() -> Iterator[sqlite3.Connection]:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(STATE_DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """CREATE TABLE IF NOT EXISTS stage_runs (
        stage TEXT PRIMARY KEY, status TEXT NOT NULL, input_hash TEXT NOT NULL,
        started_at TEXT, finished_at TEXT, attempts INTEGER NOT NULL DEFAULT 0,
        evidence_json TEXT NOT NULL DEFAULT '[]', error TEXT)"""
    )
    try:
        yield con
        con.commit()
    finally:
        con.close()


def record(
    stage: str, status: str, input_hash: str, evidence: list[str], error: str | None = None
) -> None:
    allowed = {
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "BLOCKED_EXTERNAL_DATA",
        "SKIPPED",
        "SKIPPED_RESOURCE_LIMIT",
        "SKIPPED_INSUFFICIENT_SAMPLE",
        "SUPERSEDED",
    }
    if status not in allowed:
        raise ValueError(status)
    now = utc_now()
    with database() as con:
        prior = con.execute(
            "SELECT attempts, started_at FROM stage_runs WHERE stage=?", (stage,)
        ).fetchone()
        attempts = (prior[0] if prior else 0) + (status == "RUNNING")
        started = now if status == "RUNNING" else (prior[1] if prior else now)
        finished = now if status in allowed - {"PENDING", "RUNNING"} else None
        con.execute(
            """INSERT OR REPLACE INTO stage_runs
            (stage,status,input_hash,started_at,finished_at,attempts,evidence_json,error)
            VALUES (?,?,?,?,?,?,?,?)""",
            (stage, status, input_hash, started, finished, attempts, json.dumps(evidence), error),
        )


def rows() -> list[dict[str, object]]:
    with database() as con:
        cols = [x[1] for x in con.execute("PRAGMA table_info(stage_runs)")]
        return [
            dict(zip(cols, row, strict=True))
            for row in con.execute("SELECT * FROM stage_runs ORDER BY stage")
        ]
