from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np

from .data import CME_MBO_SAMPLE, CME_MBP10_SAMPLE, acquire, data_root
from .features import F_LAST, Book
from .probe import atomic_json
from .simulator import QueueSimulator


def _mbp_targets(
    path: Path, stride: int = 5000
) -> tuple[dict[tuple[int, int], tuple[float, ...]], dict[str, int]]:
    escaped = str(path).replace("'", "''")
    fields = [
        f"{side}_{kind}_{level:02d}"
        for level in range(10)
        for side in ("bid", "ask")
        for kind in ("px", "sz", "ct")
    ]
    con = duckdb.connect()
    query = f"""SELECT sequence,
    epoch_ns(strptime(substr(ts_event,1,19), '%Y-%m-%dT%H:%M:%S')) + CAST(substr(ts_event,21,9) AS BIGINT) AS ts_event_ns,
    {",".join(fields)} FROM (SELECT row_number() OVER () rn,* FROM read_csv('{escaped}', header=true,
    types={{'ts_event':'VARCHAR'}})) WHERE rn % {stride} = 1"""
    rows = con.execute(query).fetchall()
    total, distinct = con.execute(
        f"SELECT count(*), count(DISTINCT sequence) FROM read_csv_auto('{escaped}', header=true)"
    ).fetchone()
    con.close()
    return (
        {(int(row[0]), int(row[1])): tuple(float(value) for value in row[2:]) for row in rows},
        {
            "total_rows": int(total),
            "distinct_sequences": int(distinct),
            "duplicate_sequence_rows": int(total - distinct),
        },
    )


def _actual_reconstruction(
    mbo_root: Path, targets: dict[tuple[int, int], tuple[float, ...]]
) -> dict[str, object]:
    parquet_glob = str(mbo_root / "**" / "*.parquet").replace("\\", "/").replace("'", "''")
    con = duckdb.connect()
    reader = con.execute(
        f"SELECT action,side,order_id,price,size,source_flags,sequence,ts_event_ns FROM read_parquet('{parquet_glob}') ORDER BY ingest_index"
    ).fetch_record_batch(rows_per_batch=65536)
    book = Book()
    matched = price_matches = size_matches = count_matches = 0
    level_matches = [0] * 10
    level_comparisons = [0] * 10
    mismatch_examples: list[dict[str, object]] = []
    comparisons = 0
    for batch in reader:
        columns = batch.to_pydict()
        for action, side, order_id, price, size, flags, sequence, ts_event_ns in zip(
            *(columns[name] for name in columns), strict=True
        ):
            book.apply(action, side, order_id, price, size)
            key = (int(sequence), int(ts_event_ns))
            if not flags & F_LAST or key not in targets:
                continue
            observed = targets[key]
            bids, asks = book.top("B"), book.top("A")
            reconstructed: list[float] = []
            for level in range(10):
                for levels in (bids, asks):
                    reconstructed.extend(levels[level] if level < len(levels) else (np.nan, 0, 0))
            for index, (actual, expected) in enumerate(zip(reconstructed, observed, strict=True)):
                comparisons += 1
                equal = bool(np.isclose(actual, expected, equal_nan=True))
                matched += equal
                metric = index % 3
                level = index // 6
                level_comparisons[level] += 1
                level_matches[level] += equal
                price_matches += equal and metric == 0
                size_matches += equal and metric == 1
                count_matches += equal and metric == 2
                if not equal and len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "sequence": int(sequence),
                            "ts_event_ns": int(ts_event_ns),
                            "level": level + 1,
                            "side": "bid" if (index % 6) < 3 else "ask",
                            "field": ("price", "size", "count")[metric],
                            "reconstructed": actual,
                            "observed": expected,
                        }
                    )
            targets.pop(key)
    con.close()
    denominator = max(comparisons, 1)
    per_metric = max(comparisons // 3, 1)
    return {
        "compared_fields": comparisons,
        "overall_match_rate": matched / denominator,
        "price_match_rate": price_matches / per_metric,
        "size_match_rate": size_matches / per_metric,
        "count_match_rate": count_matches / per_metric,
        "reconstruction_errors": book.errors,
        "level_match_rates": [level_matches[i] / max(level_comparisons[i], 1) for i in range(10)],
        "mismatch_examples": mismatch_examples,
        "unmatched_sample_sequences": len(targets),
    }


def _synthetic_invariants() -> dict[str, bool]:
    sim = QueueSimulator()
    order = sim.submit("B", 100.0, 2, send_ns=100, latency_ns=50)
    sim.market_add("B", 100.0, 3, now_ns=120)
    sim.market_trade("B", 100.0, 5, now_ns=140)
    no_fill_before_arrival = order.filled == 0
    sim.market_cancel("B", 100.0, 3, now_ns=160, ahead_of_us=True)
    sim.market_trade("B", 100.0, 2, now_ns=170)
    position_conservation = sim.position == order.filled == 2
    cash_conservation = sim.cash == -200.0 and sim.nav(101.0) == 2.0
    latency_ordering = no_fill_before_arrival and all(
        ts >= order.arrival_ns for ts in sim.fill_times
    )
    valid_queue = order.queue_ahead >= 0 and order.filled <= order.size
    return {
        "no_fill_before_arrival": no_fill_before_arrival,
        "position_conservation": position_conservation,
        "cash_conservation": cash_conservation,
        "latency_ordering": latency_ordering,
        "valid_queue_state": valid_queue,
    }


def validate_simulator(canonical_root: Path) -> Path:
    mbo = acquire(CME_MBO_SAMPLE)
    mbp = acquire(CME_MBP10_SAMPLE)
    targets, mbp_metadata = _mbp_targets(mbp)
    target_count = len(targets)
    actual = _actual_reconstruction(canonical_root, targets)
    invariants = _synthetic_invariants()
    passed = (
        actual["overall_match_rate"] >= 0.999
        and actual["reconstruction_errors"] == 0
        and all(invariants.values())
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "mbo_file": str(mbo),
        "mbp10_file": str(mbp),
        "sampled_mbp_events": target_count,
        "mbp_metadata": mbp_metadata,
        "actual_mbo_vs_mbp10": actual,
        "synthetic_invariants": invariants,
        "SV1_mbo_reconstruction": passed,
        "SV2_top_levels_vs_mbp": actual["overall_match_rate"] >= 0.999,
        "SV3_l3_queue": invariants["valid_queue_state"],
        "SV4_l2_approximation": "NOT_ESTABLISHED",
        "SV5_fill_calibration": "SYNTHETIC_ONLY",
        "SV6_latency_ordering": invariants["latency_ordering"],
        "headline_pnl_permitted": False,
        "passed_available_data_gate": passed,
    }
    target = data_root() / "manifests" / "simulator_validation_v1.json"
    atomic_json(target, report)
    return target
