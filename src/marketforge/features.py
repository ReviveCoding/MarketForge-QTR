from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .data import data_root
from .probe import atomic_json

F_LAST = 128


def _totals(
    bids: list[tuple[float, int, int]], asks: list[tuple[float, int, int]], n: int
) -> tuple[float, float]:
    return float(sum(x[1] for x in bids[:n])), float(sum(x[1] for x in asks[:n]))


def _imbalance(pair: tuple[float, float]) -> float:
    total = pair[0] + pair[1]
    return (pair[0] - pair[1]) / total if total else 0.0


@dataclass(slots=True)
class Order:
    side: str
    price: float
    size: int


class Book:
    def __init__(self) -> None:
        self.orders: dict[int, Order] = {}
        self.levels: dict[str, dict[float, list[int]]] = {"B": {}, "A": {}}
        self.heaps: dict[str, list[float]] = {"B": [], "A": []}
        self.errors = 0

    def clear(self) -> None:
        self.orders.clear()
        self.levels = {"B": {}, "A": {}}
        self.heaps = {"B": [], "A": []}

    def _change_level(self, side: str, price: float, size_delta: int, count_delta: int) -> None:
        levels = self.levels[side]
        if price not in levels:
            levels[price] = [0, 0]
            heapq.heappush(self.heaps[side], -price if side == "B" else price)
        level = levels[price]
        level[0] += size_delta
        level[1] += count_delta
        if level[0] <= 0 or level[1] <= 0:
            levels.pop(price, None)

    def apply(self, action: str, side: str, order_id: int, price: float | None, size: int) -> None:
        if action in {"T", "F", "N"}:
            return
        if action == "R":
            self.clear()
            return
        if action == "A" and price is not None:
            if order_id in self.orders:
                old = self.orders.pop(order_id)
                self._change_level(old.side, old.price, -old.size, -1)
                self.errors += 1
            self.orders[order_id] = Order(side, price, size)
            self._change_level(side, price, size, 1)
            return
        existing = self.orders.get(order_id)
        if existing is None:
            self.errors += 1
            return
        if action == "C":
            cancelled = min(size, existing.size)
            self._change_level(existing.side, existing.price, -cancelled, 0)
            existing.size -= cancelled
            if existing.size == 0:
                self._change_level(existing.side, existing.price, 0, -1)
                self.orders.pop(order_id)
        elif action == "M" and price is not None:
            self._change_level(existing.side, existing.price, -existing.size, -1)
            existing.side, existing.price, existing.size = side, price, size
            self._change_level(side, price, size, 1)

    def top(self, side: str, depth: int = 10) -> list[tuple[float, int, int]]:
        levels = self.levels[side]
        prices = sorted(levels, reverse=side == "B")[:depth]
        return [(price, levels[price][0], levels[price][1]) for price in prices]


def output_schema() -> pa.Schema:
    fields = [
        ("ts_event_ns", pa.int64()),
        ("sequence", pa.uint64()),
        ("instrument", pa.string()),
        ("event_action_id", pa.int8()),
        ("event_side_id", pa.int8()),
        ("event_relative_price", pa.float64()),
        ("event_size_log", pa.float64()),
        ("delta_time_log", pa.float64()),
        ("modality_mask", pa.int8()),
        ("mid", pa.float64()),
        ("spread", pa.float64()),
        ("relative_spread", pa.float64()),
        ("microprice", pa.float64()),
        ("microprice_minus_mid", pa.float64()),
        ("imbalance_1", pa.float64()),
        ("imbalance_5", pa.float64()),
        ("imbalance_10", pa.float64()),
        ("depth_bid_1", pa.float64()),
        ("depth_ask_1", pa.float64()),
        ("depth_bid_5", pa.float64()),
        ("depth_ask_5", pa.float64()),
        ("depth_bid_10", pa.float64()),
        ("depth_ask_10", pa.float64()),
        ("depth_slope", pa.float64()),
        ("depth_convexity", pa.float64()),
        ("return_1", pa.float64()),
        ("return_10", pa.float64()),
        ("realized_vol_50", pa.float64()),
        ("ewma_vol", pa.float64()),
        ("robust_vol_50", pa.float64()),
        ("event_rate", pa.float64()),
        ("trade_rate", pa.float64()),
        ("cancel_rate", pa.float64()),
        ("modify_rate", pa.float64()),
        ("signed_volume", pa.float64()),
        ("interarrival_ns", pa.int64()),
        ("session_position", pa.float64()),
        ("information_start", pa.int64()),
        ("prediction_time", pa.int64()),
    ]
    for level in range(1, 11):
        for side in ("bid", "ask"):
            fields.extend(
                [
                    (f"{side}_px_{level:02d}", pa.float64()),
                    (f"{side}_sz_{level:02d}", pa.float64()),
                    (f"{side}_ct_{level:02d}", pa.float64()),
                ]
            )
    return pa.schema(fields)


def reconstruct_features(canonical_root: Path, sample_every: int = 100) -> Path:
    output = data_root() / "features" / "databento_mbo_features_v3.parquet"
    manifest = data_root() / "manifests" / "features_databento_mbo_v3.json"
    if output.exists() and manifest.exists():
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(temporary, output_schema(), compression="zstd")
    book = Book()
    mids: deque[float] = deque(maxlen=101)
    times: deque[int] = deque(maxlen=101)
    buffer: list[dict[str, object]] = []
    event_counter = last_capture_events = 0
    action_counts = {"T": 0, "C": 0, "M": 0}
    signed_volume = 0.0
    captures = 0
    min_ts = max_ts = last_capture_ts = None
    columns = [
        "ts_event_ns",
        "sequence",
        "instrument",
        "action",
        "side",
        "order_id",
        "price",
        "size",
        "source_flags",
    ]
    parquet_glob = str(canonical_root / "**" / "*.parquet").replace("\\", "/").replace("'", "''")
    con = duckdb.connect()
    selected = ",".join(columns)
    reader = con.execute(
        f"SELECT {selected} FROM read_parquet('{parquet_glob}', hive_partitioning=true) ORDER BY ingest_index"
    ).fetch_record_batch(rows_per_batch=65536)
    for batch in reader:
        arrays = batch.to_pydict()
        for values in zip(*(arrays[name] for name in columns), strict=True):
            ts, sequence, instrument, action, side, order_id, price, size, flags = values
            event_counter += 1
            action_counts[action] = action_counts.get(action, 0) + 1
            if action == "A" and side in {"A", "B"}:
                signed_volume += size if side == "B" else -size
            book.apply(action, side, order_id, price, size)
            if not flags & F_LAST or event_counter - last_capture_events < sample_every:
                continue
            bids, asks = book.top("B"), book.top("A")
            if not bids or not asks or bids[0][0] >= asks[0][0]:
                continue
            mid = (bids[0][0] + asks[0][0]) / 2
            spread = asks[0][0] - bids[0][0]
            b1, a1 = bids[0][1], asks[0][1]
            micro = (asks[0][0] * b1 + bids[0][0] * a1) / (b1 + a1)
            prior_mid = mids[-1] if mids else mid
            ret1 = math.log(mid / prior_mid) if prior_mid > 0 else 0.0
            mids.append(mid)
            times.append(ts)
            returns = np.diff(np.log(np.asarray(mids, dtype=float)))
            ewma = 0.0
            for value in returns[-50:]:
                ewma = 0.94 * ewma + 0.06 * float(value * value)
            d1, d5, d10 = _totals(bids, asks, 1), _totals(bids, asks, 5), _totals(bids, asks, 10)
            dt = ts - last_capture_ts if last_capture_ts is not None else 0
            elapsed = max(dt / 1e9, 1e-9)
            event_delta = event_counter - last_capture_events
            row: dict[str, object] = {
                "ts_event_ns": ts,
                "sequence": sequence,
                "instrument": instrument,
                "event_action_id": {"A": 1, "C": 2, "M": 3, "R": 4, "T": 5, "F": 6, "N": 0}[action],
                "event_side_id": {"N": 0, "B": 1, "A": 2}[side],
                "event_relative_price": ((price - mid) / 0.25) if price is not None else 0.0,
                "event_size_log": math.log1p(size),
                "delta_time_log": math.log1p(max(dt, 0)),
                "modality_mask": 3,
                "mid": mid,
                "spread": spread,
                "relative_spread": spread / mid,
                "microprice": micro,
                "microprice_minus_mid": micro - mid,
                "imbalance_1": _imbalance(d1),
                "imbalance_5": _imbalance(d5),
                "imbalance_10": _imbalance(d10),
                "depth_bid_1": d1[0],
                "depth_ask_1": d1[1],
                "depth_bid_5": d5[0],
                "depth_ask_5": d5[1],
                "depth_bid_10": d10[0],
                "depth_ask_10": d10[1],
                "depth_slope": (d10[0] + d10[1] - d1[0] - d1[1]) / 9,
                "depth_convexity": (d10[0] + d10[1]) / max(d5[0] + d5[1], 1.0),
                "return_1": ret1,
                "return_10": math.log(mid / mids[-11]) if len(mids) > 10 else 0.0,
                "realized_vol_50": float(np.std(returns[-50:])) if len(returns) > 1 else 0.0,
                "ewma_vol": math.sqrt(ewma),
                "robust_vol_50": float(np.median(np.abs(returns[-50:] - np.median(returns[-50:]))))
                if len(returns)
                else 0.0,
                "event_rate": event_delta / elapsed,
                "trade_rate": action_counts.get("T", 0) / elapsed,
                "cancel_rate": action_counts.get("C", 0) / elapsed,
                "modify_rate": action_counts.get("M", 0) / elapsed,
                "signed_volume": signed_volume,
                "interarrival_ns": dt,
                "session_position": 0.0,
                "information_start": times[0],
                "prediction_time": ts,
            }
            for level in range(10):
                for label, levels in (("bid", bids), ("ask", asks)):
                    values3 = (
                        levels[level] if level < len(levels) else (math.nan, math.nan, math.nan)
                    )
                    (
                        row[f"{label}_px_{level + 1:02d}"],
                        row[f"{label}_sz_{level + 1:02d}"],
                        row[f"{label}_ct_{level + 1:02d}"],
                    ) = values3
            buffer.append(row)
            captures += 1
            min_ts = ts if min_ts is None else min(min_ts, ts)
            max_ts = ts if max_ts is None else max(max_ts, ts)
            last_capture_ts, last_capture_events = ts, event_counter
            action_counts = {"T": 0, "C": 0, "M": 0}
            signed_volume = 0.0
            if len(buffer) >= 5000:
                writer.write_table(pa.Table.from_pylist(buffer, schema=output_schema()))
                buffer.clear()
    con.close()
    if buffer:
        writer.write_table(pa.Table.from_pylist(buffer, schema=output_schema()))
    writer.close()
    temporary.replace(output)
    atomic_json(
        manifest,
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "feature_file": str(output),
            "rows": captures,
            "timestamp_bounds_ns": [min_ts, max_ts],
            "sample_every_events": sample_every,
            "reconstruction_errors": book.errors,
            "feature_version": "mbo-past-only-v3-dual-stream",
        },
    )
    return manifest
