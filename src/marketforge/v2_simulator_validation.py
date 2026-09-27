from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np

from .data import data_root
from .features import F_LAST, Book
from .probe import atomic_json
from .state import ROOT


def _mbp_targets(
    path: Path, stride: int = 5000
) -> tuple[dict[tuple[int, int, int], tuple[float, ...]], dict[str, int]]:
    escaped = str(path).replace("'", "''")
    fields = [
        f"{side}_{kind}_{level:02d}"
        for level in range(10)
        for side in ("bid", "ask")
        for kind in ("px", "sz", "ct")
    ]
    con = duckdb.connect()
    rows = con.execute(
        f"""SELECT instrument_id, sequence,
        epoch_ns(CAST(ts_event AS TIMESTAMPTZ))::BIGINT AS ts_event_ns,
        {",".join(fields)} FROM (
          SELECT row_number() OVER () rn,* FROM read_csv_auto('{escaped}', header=true)
        ) WHERE rn % {stride} = 1"""
    ).fetchall()
    total, distinct = con.execute(
        f"SELECT count(*), count(DISTINCT (instrument_id,sequence,ts_event)) FROM read_csv_auto('{escaped}', header=true)"
    ).fetchone()
    con.close()
    return (
        {
            (int(row[0]), int(row[1]), int(row[2])): tuple(float(value) for value in row[3:])
            for row in rows
        },
        {
            "total_rows": int(total),
            "distinct_instrument_sequence_timestamp": int(distinct),
            "duplicate_composite_rows": int(total - distinct),
        },
    )


def _reconstruct(
    canonical_root: Path, targets: dict[tuple[int, int, int], tuple[float, ...]]
) -> dict[str, object]:
    parquet_glob = str(canonical_root / "**" / "*.parquet").replace("\\", "/").replace("'", "''")
    con = duckdb.connect()
    reader = con.execute(
        f"SELECT instrument_id,action,side,order_id,price,size,source_flags,sequence,ts_event_ns "
        f"FROM read_parquet('{parquet_glob}', hive_partitioning=true) ORDER BY ingest_index"
    ).fetch_record_batch(rows_per_batch=65_536)
    books: dict[int, Book] = {}
    matched = price_matches = size_matches = count_matches = comparisons = 0
    reconstruction_errors = 0
    examples: list[dict[str, object]] = []
    for batch in reader:
        values = batch.to_pydict()
        columns = list(values)
        for row in zip(*(values[column] for column in columns), strict=True):
            (
                instrument_id,
                action,
                side,
                order_id,
                price,
                size,
                flags,
                sequence,
                ts_event_ns,
            ) = row
            book = books.setdefault(int(instrument_id), Book())
            before = book.errors
            book.apply(action, side, order_id, price, size)
            reconstruction_errors += book.errors - before
            key = (int(instrument_id), int(sequence), int(ts_event_ns))
            if not flags & F_LAST or key not in targets:
                continue
            observed = targets.pop(key)
            reconstructed: list[float] = []
            bids, asks = book.top("B"), book.top("A")
            for level in range(10):
                for levels in (bids, asks):
                    reconstructed.extend(levels[level] if level < len(levels) else (np.nan, 0, 0))
            for index, (actual, expected) in enumerate(zip(reconstructed, observed, strict=True)):
                equal = bool(np.isclose(actual, expected, equal_nan=True))
                comparisons += 1
                matched += equal
                metric = index % 3
                price_matches += equal and metric == 0
                size_matches += equal and metric == 1
                count_matches += equal and metric == 2
                if not equal and len(examples) < 20:
                    examples.append(
                        {
                            "instrument_id": int(instrument_id),
                            "sequence": int(sequence),
                            "ts_event_ns": int(ts_event_ns),
                            "level": index // 6 + 1,
                            "side": "bid" if index % 6 < 3 else "ask",
                            "field": ("price", "size", "count")[metric],
                            "reconstructed": actual,
                            "observed": expected,
                        }
                    )
    con.close()
    denominator = max(comparisons, 1)
    per_metric = max(comparisons // 3, 1)
    return {
        "compared_fields": comparisons,
        "overall_match_rate": matched / denominator,
        "price_match_rate": price_matches / per_metric,
        "size_match_rate": size_matches / per_metric,
        "count_match_rate": count_matches / per_metric,
        "reconstruction_errors": reconstruction_errors,
        "unmatched_sample_events": len(targets),
        "mismatch_examples": examples,
    }


def validate_simulator_v2() -> Path:
    v1 = json.loads(
        (data_root() / "manifests/simulator_validation_v1.json").read_text(encoding="utf-8")
    )
    failed_ice_report = json.loads(
        (
            ROOT
            / "artifacts/superseded_v2_drafts/ice_snapshot_ts_event_grouping_20260927/manifests/simulator_validation_v2.json"
        ).read_text(encoding="utf-8")
    )
    es_queue = json.loads(
        (data_root() / "manifests/development_replay_glbx_es_v2.json").read_text(encoding="utf-8")
    )["historical_order_validation"]
    ice_l2 = json.loads(
        (data_root() / "manifests/development_replay_ifeu_brn_v2.json").read_text(encoding="utf-8")
    )
    available_gate = (
        v1["actual_mbo_vs_mbp10"]["overall_match_rate"] >= 0.999
        and es_queue["eligibility_rate"] >= 0.9
        and not ice_l2["l3_targets_available"]
    )
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "es_mbo_vs_mbp10": {
            "status": "REUSED_AUDITED_V1_ENGINEERING_EVIDENCE",
            "source": "data/manifests/simulator_validation_v1.json",
            "metrics": v1["actual_mbo_vs_mbp10"],
        },
        "ice_mbo_vs_mbp10": {
            "status": "FAILED_UNRECOVERED_CLEAR_EXCLUDED_FROM_L3_TARGETS",
            "metrics": failed_ice_report["ice_mbo_vs_mbp10"],
            "development_replacement": "official paired MBP-10 L2 with l3_targets_available=0",
        },
        "historical_real_order_queue_validation": {
            "glbx_es": es_queue,
            "ifeu_brn": "NOT_APPLICABLE_TO_TRAINING_AFTER_MBO_VALIDATION_FAILURE",
        },
        "counterfactual_label_type": "SIMULATED_L3_FOR_ES_ONLY",
        "fill_conditional_adverse_selection": True,
        "small_order_no_endogenous_impact": True,
        "validation_claim": "calibration/error evidence, not perfect observed fills",
        "passed_available_data_gate": available_gate,
        "all_mbo_sources_passed": False,
        "execution_scope": "ES only; ICE is L2 forecasting/representation only",
        "headline_live_pnl_permitted": False,
        "final_lockbox_v2_used": False,
    }
    target = data_root() / "manifests/simulator_validation_v2.json"
    atomic_json(target, report)
    return target
