from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .data import data_root, sha256_file
from .features import F_LAST, _imbalance, _totals
from .probe import atomic_json
from .state import ROOT, content_hash


@dataclass(slots=True)
class PendingL2:
    feature: dict[str, object]
    mid: float
    bid: float
    ask: float
    last_mid: float
    variance: float = 0.0
    mid_100ms: float | None = None
    mid_1000ms: float | None = None

    def update(self, ts_ns: int, mid: float) -> None:
        if self.mid_1000ms is None and self.last_mid > 0 and mid > 0:
            move = math.log(mid / self.last_mid)
            self.variance += move * move
        self.last_mid = mid
        elapsed = ts_ns - int(self.feature["prediction_time"])
        if self.mid_100ms is None and elapsed >= 100_000_000:
            self.mid_100ms = mid
        if self.mid_1000ms is None and elapsed >= 1_000_000_000:
            self.mid_1000ms = mid


def _session_position(ts_ns: int) -> float:
    local = datetime.fromtimestamp(ts_ns / 1e9, UTC).astimezone(ZoneInfo("Europe/London"))
    minute = local.hour * 60 + local.minute + local.second / 60
    elapsed = (minute - 60) % (24 * 60)
    return elapsed / (22 * 60) if elapsed <= 22 * 60 else math.nan


def _levels(row: dict[str, object], side: str) -> list[tuple[float, int, int]]:
    return [
        (
            float(row[f"{side}_px_{level:02d}"]),
            int(row[f"{side}_sz_{level:02d}"]),
            int(row[f"{side}_ct_{level:02d}"]),
        )
        for level in range(10)
        if row[f"{side}_px_{level:02d}"] is not None
    ]


def _feature(
    row: dict[str, object],
    mids: deque[float],
    times: deque[int],
    last_capture_ts: int | None,
    events: int,
    actions: dict[str, int],
    signed_volume: float,
) -> dict[str, object] | None:
    bids, asks = _levels(row, "bid"), _levels(row, "ask")
    if not bids or not asks or bids[0][0] >= asks[0][0]:
        return None
    ts_ns = int(row["ts_event_ns"])
    mid = (bids[0][0] + asks[0][0]) / 2
    spread = asks[0][0] - bids[0][0]
    b1, a1 = bids[0][1], asks[0][1]
    micro = (asks[0][0] * b1 + bids[0][0] * a1) / (b1 + a1)
    prior = mids[-1] if mids else mid
    mids.append(mid)
    times.append(ts_ns)
    returns = np.diff(np.log(np.asarray(mids, dtype=np.float64)))
    ewma = 0.0
    for value in returns[-50:]:
        ewma = 0.94 * ewma + 0.06 * float(value * value)
    d1, d5, d10 = _totals(bids, asks, 1), _totals(bids, asks, 5), _totals(bids, asks, 10)
    delta = ts_ns - last_capture_ts if last_capture_ts is not None else 0
    seconds = max(delta / 1e9, 1e-9)
    local_date = (
        datetime.fromtimestamp(ts_ns / 1e9, UTC)
        .astimezone(ZoneInfo("Europe/London"))
        .date()
        .isoformat()
    )
    output: dict[str, object] = {
        "source": "databento",
        "venue": "ICE_FUTURES_EUROPE",
        "asset_class": "futures",
        "instrument": "BRN",
        "instrument_id": int(row["instrument_id"]),
        "contract": str(row["symbol"]),
        "trading_date": local_date,
        "session_id": f"{local_date}-mbp10",
        "reset_generation": 0,
        "ts_event_ns": ts_ns,
        "sequence": int(row["sequence"]),
        "event_action_id": {
            "A": 1,
            "C": 2,
            "M": 3,
            "R": 4,
            "T": 5,
            "F": 6,
            "N": 0,
        }.get(str(row["action"]), 0),
        "event_side_id": {"N": 0, "B": 1, "A": 2}.get(str(row["side"]), 0),
        "event_relative_price": ((float(row["price"]) - mid) / 0.01)
        if row["price"] is not None
        else 0.0,
        "event_size_log": math.log1p(int(row["size"])),
        "delta_time_log": math.log1p(max(delta, 0)),
        "modality_mask": 3,
        "modality_l1": 1,
        "modality_l2": 1,
        "modality_l3": 0,
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
        "return_1": math.log(mid / prior) if prior > 0 else 0.0,
        "return_10": math.log(mid / mids[-11]) if len(mids) > 10 else 0.0,
        "realized_vol_50": float(np.std(returns[-50:])) if len(returns) > 1 else 0.0,
        "ewma_vol": math.sqrt(ewma),
        "robust_vol_50": float(np.median(np.abs(returns[-50:] - np.median(returns[-50:]))))
        if len(returns)
        else 0.0,
        "event_rate": events / seconds,
        "trade_rate": actions.get("T", 0) / seconds,
        "cancel_rate": actions.get("C", 0) / seconds,
        "modify_rate": actions.get("M", 0) / seconds,
        "signed_volume": signed_volume,
        "interarrival_ns": delta,
        "session_position": _session_position(ts_ns),
        "information_start": times[0],
        "prediction_time": ts_ns,
        "tick_size": 0.01,
        "contract_multiplier": 1000.0,
        "tick_value": 10.0,
        "currency": "USD",
    }
    for level in range(10):
        for label, levels in (("bid", bids), ("ask", asks)):
            values = levels[level] if level < len(levels) else (math.nan, math.nan, math.nan)
            output[f"{label}_px_{level + 1:02d}"] = values[0]
            output[f"{label}_sz_{level + 1:02d}"] = values[1]
            output[f"{label}_ct_{level + 1:02d}"] = values[2]
    return output


def _label(pending: PendingL2) -> dict[str, object]:
    assert pending.mid_100ms is not None and pending.mid_1000ms is not None
    return {
        "prediction_time": int(pending.feature["prediction_time"]),
        "label_type": "UNAVAILABLE",
        "return_100ms": math.log(pending.mid_100ms / pending.mid),
        "return_1000ms": math.log(pending.mid_1000ms / pending.mid),
        "direction_1000ms": int(pending.mid_1000ms > pending.mid)
        - int(pending.mid_1000ms < pending.mid),
        "bid_quote_markout_100ms": pending.mid_100ms - pending.bid,
        "ask_quote_markout_100ms": pending.ask - pending.mid_100ms,
        "bid_quote_markout_1000ms": pending.mid_1000ms - pending.bid,
        "ask_quote_markout_1000ms": pending.ask - pending.mid_1000ms,
        "future_realized_vol_1000ms": math.sqrt(pending.variance),
        "bid_fill": 0,
        "ask_fill": 0,
        "bid_partial_fill": 0,
        "ask_partial_fill": 0,
        "bid_fill_fraction": 0.0,
        "ask_fill_fraction": 0.0,
        "bid_first_fill_time_ns": None,
        "ask_first_fill_time_ns": None,
        "bid_completion_time_ns": None,
        "ask_completion_time_ns": None,
        "bid_time_to_first_fill_ns": None,
        "ask_time_to_first_fill_ns": None,
        "bid_queue_ahead_at_entry": None,
        "ask_queue_ahead_at_entry": None,
        "bid_queue_trajectory": None,
        "ask_queue_trajectory": None,
        "bid_fill_conditional_markout_100ms": None,
        "ask_fill_conditional_markout_100ms": None,
        "bid_fill_conditional_adverse": None,
        "ask_fill_conditional_adverse": None,
        "bid_adverse_available": 0,
        "ask_adverse_available": 0,
        "l3_targets_available": 0,
        "order_entry_latency_ns": None,
        "fill_horizon_ns": None,
        "hypothetical_quantity": None,
        "small_order_no_endogenous_impact": None,
    }


def reconstruct_ifeu_mbp10_development(sample_every: int = 2000) -> Path:
    raw = ROOT / "data/raw/databento/ifeu-impact-futures-mbp10.csv"
    feature_path = data_root() / "features_v4_multi_market/ifeu_brn.parquet"
    label_path = data_root() / "labels_v2_l3/ifeu_brn.parquet"
    marker = data_root() / "manifests/development_replay_ifeu_brn_v2.json"
    raw_hash = sha256_file(raw)
    if marker.exists() and feature_path.exists() and label_path.exists():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior.get("input_raw_sha256") == raw_hash and prior["sample_every"] == sample_every:
            return marker
    level_columns = [
        f"{side}_{kind}_{level:02d}"
        for level in range(10)
        for side in ("bid", "ask")
        for kind in ("px", "sz", "ct")
    ]
    raw_sql = str(raw).replace("'", "''")
    con = duckdb.connect()
    selected = [
        "epoch_ns(CAST(ts_event AS TIMESTAMPTZ))::BIGINT ts_event_ns",
        "instrument_id",
        "sequence",
        "action",
        "side",
        "price",
        "size",
        "flags",
        "symbol",
        *level_columns,
    ]
    reader = con.execute(
        f"SELECT {','.join(selected)} FROM read_csv_auto('{raw_sql}', header=true)"
    ).fetch_record_batch(rows_per_batch=65_536)
    mids: deque[float] = deque(maxlen=101)
    times: deque[int] = deque(maxlen=101)
    pending: list[PendingL2] = []
    features: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    actions: dict[str, int] = {"T": 0, "C": 0, "M": 0}
    signed_volume = 0.0
    event_count = last_capture_event = 0
    last_capture_ts: int | None = None
    for batch in reader:
        arrays = batch.to_pydict()
        names = list(arrays)
        for values in zip(*(arrays[name] for name in names), strict=True):
            row = dict(zip(names, values, strict=True))
            event_count += 1
            action, side = str(row["action"]), str(row["side"])
            actions[action] = actions.get(action, 0) + 1
            if action == "A" and side in {"A", "B"}:
                signed_volume += int(row["size"]) if side == "B" else -int(row["size"])
            bid, ask = row["bid_px_00"], row["ask_px_00"]
            if bid is None or ask is None or float(bid) >= float(ask):
                continue
            ts_ns = int(row["ts_event_ns"])
            mid = (float(bid) + float(ask)) / 2
            for item in pending:
                item.update(ts_ns, mid)
            ready = [item for item in pending if item.mid_1000ms is not None]
            pending = [item for item in pending if item.mid_1000ms is None]
            for item in ready:
                features.append(item.feature)
                labels.append(_label(item))
            if event_count - last_capture_event < sample_every or not int(row["flags"]) & F_LAST:
                continue
            feature = _feature(
                row,
                mids,
                times,
                last_capture_ts,
                event_count - last_capture_event,
                actions,
                signed_volume,
            )
            if feature is None:
                continue
            pending.append(PendingL2(feature, mid, float(bid), float(ask), mid))
            last_capture_event, last_capture_ts = event_count, ts_ns
            actions = {"T": 0, "C": 0, "M": 0}
            signed_volume = 0.0
    con.close()
    if not features or len(features) != len(labels):
        raise RuntimeError("invalid ICE MBP-10 development output")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    feature_tmp, label_tmp = feature_path.with_suffix(".tmp"), label_path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(features), feature_tmp, compression="zstd")
    pq.write_table(pa.Table.from_pylist(labels), label_tmp, compression="zstd")
    feature_tmp.replace(feature_path)
    label_tmp.replace(label_path)
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": "ifeu_brn",
        "input_schema": "MBP-10",
        "input_raw_sha256": raw_hash,
        "sample_every": sample_every,
        "features": {
            "path": str(feature_path),
            "sha256": sha256_file(feature_path),
            "rows": len(features),
        },
        "labels": {
            "path": str(label_path),
            "sha256": sha256_file(label_path),
            "rows": len(labels),
            "type": "UNAVAILABLE",
        },
        "l3_targets_available": False,
        "reason": "ICE MBO has an unrecovered midstream clear; paired MBP-10 used for L2 forecasting only",
        "final_lockbox_v2_used": False,
    }
    payload["config_hash"] = content_hash({"sample_every": sample_every, "schema": "MBP-10"})
    atomic_json(marker, payload)
    return marker
