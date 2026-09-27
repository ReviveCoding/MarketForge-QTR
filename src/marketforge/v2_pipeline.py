from __future__ import annotations

import json
import math
from collections import OrderedDict, deque
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
from .v2_book import FIFOBook, MBOEvent, SimulatedPassiveOrder, finalize_simulated_l3


@dataclass(frozen=True, slots=True)
class DevelopmentSource:
    name: str
    canonical_root: Path
    manifest: Path
    timezone: str
    session_start_minute: int
    session_end_minute: int
    venue: str
    asset_class: str
    instrument: str


DEVELOPMENT_SOURCES = (
    DevelopmentSource(
        "glbx_es",
        ROOT / "data/canonical_v2/glbx_es_mbo_v1",
        ROOT / "data/manifests/canonical_glbx_es_mbo_v2.json",
        "America/Chicago",
        17 * 60,
        16 * 60,
        "CME_GLOBEX",
        "futures",
        "ES",
    ),
)


def _assert_development_path(path: Path) -> None:
    resolved = path.resolve()
    denied = {
        (ROOT / "data/raw/databento/xnas-itch-equities-mbo.csv").resolve(),
        (ROOT / "data/raw/databento/xnas-itch-equities-mbp10.csv").resolve(),
    }
    lockbox_root = (ROOT / "data/final_lockbox_v2").resolve()
    if resolved in denied or resolved == lockbox_root or lockbox_root in resolved.parents:
        raise PermissionError(f"development access denied for v2 lockbox path: {resolved}")


def _session_position(ts_ns: int, source: DevelopmentSource) -> float:
    local = datetime.fromtimestamp(ts_ns / 1e9, UTC).astimezone(ZoneInfo(source.timezone))
    minute = local.hour * 60 + local.minute + local.second / 60 + local.microsecond / 60e6
    duration = (source.session_end_minute - source.session_start_minute) % (24 * 60)
    elapsed = (minute - source.session_start_minute) % (24 * 60)
    return elapsed / duration if elapsed <= duration else math.nan


@dataclass(slots=True)
class QueueValidationCandidate:
    order_id: int
    side: str
    price: float
    entered_ns: int
    expires_ns: int
    ahead: OrderedDict[int, int]

    def reduce(self, order_id: int, quantity: int) -> None:
        current = self.ahead.get(order_id, 0)
        removed = min(current, quantity)
        if removed == 0:
            return
        remaining = current - removed
        if remaining:
            self.ahead[order_id] = remaining
        else:
            self.ahead.pop(order_id)


@dataclass(slots=True)
class PendingCapture:
    feature: dict[str, object]
    bid_order: SimulatedPassiveOrder
    ask_order: SimulatedPassiveOrder
    decision_mid: float
    decision_bid: float
    decision_ask: float
    last_mid: float
    future_variance: float = 0.0
    mid_100ms: float | None = None
    mid_1000ms: float | None = None
    bid_post_fill_markout_100ms: float | None = None
    ask_post_fill_markout_100ms: float | None = None

    @property
    def decision_time_ns(self) -> int:
        return int(self.feature["prediction_time"])

    def update_mid(self, ts_ns: int, mid: float) -> None:
        # Include the first endpoint at/after 1 s, then stop. Captures with a late
        # fill may remain pending for their +100 ms post-fill markout, but that
        # extra wait must not leak into the explicitly 1000 ms volatility target.
        if self.mid_1000ms is None and self.last_mid > 0 and mid > 0:
            move = math.log(mid / self.last_mid)
            self.future_variance += move * move
        self.last_mid = mid
        elapsed = ts_ns - self.decision_time_ns
        if self.mid_100ms is None and elapsed >= 100_000_000:
            self.mid_100ms = mid
        if self.mid_1000ms is None and elapsed >= 1_000_000_000:
            self.mid_1000ms = mid
        if (
            self.bid_order.first_fill_time_ns is not None
            and self.bid_post_fill_markout_100ms is None
            and ts_ns - self.bid_order.first_fill_time_ns >= 100_000_000
        ):
            self.bid_post_fill_markout_100ms = mid - self.bid_order.price
        if (
            self.ask_order.first_fill_time_ns is not None
            and self.ask_post_fill_markout_100ms is None
            and ts_ns - self.ask_order.first_fill_time_ns >= 100_000_000
        ):
            self.ask_post_fill_markout_100ms = self.ask_order.price - mid

    def ready(self) -> bool:
        if self.mid_1000ms is None:
            return False
        bid_ready = (
            self.bid_order.first_fill_time_ns is None
            or self.bid_post_fill_markout_100ms is not None
        )
        ask_ready = (
            self.ask_order.first_fill_time_ns is None
            or self.ask_post_fill_markout_100ms is not None
        )
        return bid_ready and ask_ready


def _book_feature_row(
    *,
    event: MBOEvent,
    book: FIFOBook,
    source: DevelopmentSource,
    reset_generation: int,
    tick_size: float,
    contract_multiplier: float,
    tick_value: float,
    currency: str,
    mids: deque[float],
    times: deque[int],
    last_capture_ts: int | None,
    events_since_capture: int,
    interval_actions: dict[str, int],
    signed_volume: float,
) -> dict[str, object] | None:
    bids, asks = book.top("B"), book.top("A")
    if not bids or not asks or bids[0][0] >= asks[0][0]:
        return None
    mid = (bids[0][0] + asks[0][0]) / 2
    spread = asks[0][0] - bids[0][0]
    bid_size, ask_size = bids[0][1], asks[0][1]
    microprice = (asks[0][0] * bid_size + bids[0][0] * ask_size) / (bid_size + ask_size)
    prior = mids[-1] if mids else mid
    mids.append(mid)
    times.append(event.ts_event_ns)
    returns = np.diff(np.log(np.asarray(mids, dtype=np.float64)))
    ewma = 0.0
    for value in returns[-50:]:
        ewma = 0.94 * ewma + 0.06 * float(value * value)
    d1, d5, d10 = _totals(bids, asks, 1), _totals(bids, asks, 5), _totals(bids, asks, 10)
    delta_ns = event.ts_event_ns - last_capture_ts if last_capture_ts is not None else 0
    elapsed_seconds = max(delta_ns / 1e9, 1e-9)
    row: dict[str, object] = {
        "source": event.source,
        "venue": source.venue,
        "asset_class": source.asset_class,
        "instrument": source.instrument,
        "instrument_id": event.instrument_id,
        "contract": event.contract,
        "trading_date": event.trading_date,
        "session_id": event.session_id,
        "reset_generation": reset_generation,
        "ts_event_ns": event.ts_event_ns,
        "sequence": event.sequence,
        "event_action_id": {"A": 1, "C": 2, "M": 3, "R": 4, "T": 5, "F": 6, "N": 0}[event.action],
        "event_side_id": {"N": 0, "B": 1, "A": 2}[event.side],
        "event_relative_price": ((event.price - mid) / tick_size) if event.price else 0.0,
        "event_size_log": math.log1p(event.size),
        "delta_time_log": math.log1p(max(delta_ns, 0)),
        "modality_mask": 7,
        "modality_l1": 1,
        "modality_l2": 1,
        "modality_l3": 1,
        "mid": mid,
        "spread": spread,
        "relative_spread": spread / mid,
        "microprice": microprice,
        "microprice_minus_mid": microprice - mid,
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
        "event_rate": events_since_capture / elapsed_seconds,
        "trade_rate": interval_actions.get("T", 0) / elapsed_seconds,
        "cancel_rate": interval_actions.get("C", 0) / elapsed_seconds,
        "modify_rate": interval_actions.get("M", 0) / elapsed_seconds,
        "signed_volume": signed_volume,
        "interarrival_ns": delta_ns,
        "session_position": _session_position(event.ts_event_ns, source),
        "information_start": times[0],
        "prediction_time": event.ts_event_ns,
        "tick_size": tick_size,
        "contract_multiplier": contract_multiplier,
        "tick_value": tick_value,
        "currency": currency,
    }
    for level in range(10):
        for label, levels in (("bid", bids), ("ask", asks)):
            values = levels[level] if level < len(levels) else (math.nan, math.nan, math.nan)
            row[f"{label}_px_{level + 1:02d}"] = values[0]
            row[f"{label}_sz_{level + 1:02d}"] = values[1]
            row[f"{label}_ct_{level + 1:02d}"] = values[2]
    return row


def _label_row(capture: PendingCapture) -> dict[str, object]:
    bid = finalize_simulated_l3(capture.bid_order)
    ask = finalize_simulated_l3(capture.ask_order)
    assert capture.mid_100ms is not None and capture.mid_1000ms is not None
    row = {
        "prediction_time": capture.decision_time_ns,
        "label_type": "SIMULATED_L3",
        "return_100ms": math.log(capture.mid_100ms / capture.decision_mid),
        "return_1000ms": math.log(capture.mid_1000ms / capture.decision_mid),
        "direction_1000ms": int(capture.mid_1000ms > capture.decision_mid)
        - int(capture.mid_1000ms < capture.decision_mid),
        "bid_quote_markout_100ms": capture.mid_100ms - capture.decision_bid,
        "ask_quote_markout_100ms": capture.decision_ask - capture.mid_100ms,
        "bid_quote_markout_1000ms": capture.mid_1000ms - capture.decision_bid,
        "ask_quote_markout_1000ms": capture.decision_ask - capture.mid_1000ms,
        "future_realized_vol_1000ms": math.sqrt(capture.future_variance),
        "bid_fill": int(bid.filled),
        "ask_fill": int(ask.filled),
        "bid_partial_fill": int(bid.partial_fill),
        "ask_partial_fill": int(ask.partial_fill),
        "bid_fill_fraction": bid.fill_fraction,
        "ask_fill_fraction": ask.fill_fraction,
        "bid_first_fill_time_ns": bid.first_fill_time_ns,
        "ask_first_fill_time_ns": ask.first_fill_time_ns,
        "bid_completion_time_ns": bid.completion_time_ns,
        "ask_completion_time_ns": ask.completion_time_ns,
        "bid_time_to_first_fill_ns": bid.time_to_first_fill_ns,
        "ask_time_to_first_fill_ns": ask.time_to_first_fill_ns,
        "bid_queue_ahead_at_entry": bid.queue_ahead_at_entry,
        "ask_queue_ahead_at_entry": ask.queue_ahead_at_entry,
        "bid_queue_trajectory": json.dumps(bid.queue_trajectory, separators=(",", ":")),
        "ask_queue_trajectory": json.dumps(ask.queue_trajectory, separators=(",", ":")),
        "bid_fill_conditional_markout_100ms": capture.bid_post_fill_markout_100ms,
        "ask_fill_conditional_markout_100ms": capture.ask_post_fill_markout_100ms,
        "bid_fill_conditional_adverse": None
        if capture.bid_post_fill_markout_100ms is None
        else int(capture.bid_post_fill_markout_100ms < 0),
        "ask_fill_conditional_adverse": None
        if capture.ask_post_fill_markout_100ms is None
        else int(capture.ask_post_fill_markout_100ms < 0),
        "bid_adverse_available": int(capture.bid_post_fill_markout_100ms is not None),
        "ask_adverse_available": int(capture.ask_post_fill_markout_100ms is not None),
        "l3_targets_available": 1,
        "order_entry_latency_ns": 1_000_000,
        "fill_horizon_ns": 1_000_000_000,
        "hypothetical_quantity": 1,
        "small_order_no_endogenous_impact": True,
    }
    return row


def reconstruct_development_source(
    source: DevelopmentSource,
    sample_every: int = 2000,
    output_root: Path | None = None,
    final_lockbox_used: bool = False,
) -> Path:
    if not final_lockbox_used:
        _assert_development_path(source.canonical_root)
    root = data_root() if output_root is None else output_root
    feature_dir = root / "features_v4_multi_market"
    label_dir = root / "labels_v2_l3"
    manifest_dir = root / "manifests"
    feature_path = feature_dir / f"{source.name}.parquet"
    label_path = label_dir / f"{source.name}.parquet"
    marker = manifest_dir / f"development_replay_{source.name}_v2.json"
    input_manifest = json.loads(source.manifest.read_text(encoding="utf-8"))
    input_hash = content_hash(input_manifest)
    if marker.exists() and feature_path.exists() and label_path.exists():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior["input_manifest_hash"] == input_hash and prior["sample_every"] == sample_every:
            return marker
    feature_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    parquet_glob = (
        str(source.canonical_root / "**" / "*.parquet").replace("\\", "/").replace("'", "''")
    )
    columns = [
        "source",
        "instrument_id",
        "contract",
        "trading_date",
        "session_id",
        "reset_generation",
        "ts_event_ns",
        "sequence",
        "action",
        "side",
        "order_id",
        "price",
        "size",
        "source_flags",
        "tick_size",
        "contract_multiplier",
        "tick_value",
        "currency",
    ]
    con = duckdb.connect()
    reader = con.execute(
        f"SELECT {','.join(columns)} FROM read_parquet('{parquet_glob}', hive_partitioning=true) ORDER BY ingest_index"
    ).fetch_record_batch(rows_per_batch=65_536)
    book = FIFOBook()
    current_key = None
    mids: deque[float] = deque(maxlen=101)
    times: deque[int] = deque(maxlen=101)
    pending: list[PendingCapture] = []
    features: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    interval_actions: dict[str, int] = {"T": 0, "C": 0, "M": 0}
    signed_volume = 0.0
    event_count = last_capture_event = 0
    last_capture_ts: int | None = None
    reconstruction_errors = 0
    dropped_boundary = 0
    validation_candidates: dict[int, QueueValidationCandidate] = {}
    validation_add_counter = validation_sampled = 0
    validation_fills = validation_eligible = validation_queue_error_sum = 0

    for batch in reader:
        arrays = batch.to_pydict()
        for values in zip(*(arrays[name] for name in columns), strict=True):
            data = dict(zip(columns, values, strict=True))
            event = MBOEvent(
                source=str(data["source"]),
                instrument_id=int(data["instrument_id"]),
                contract=str(data["contract"]),
                trading_date=str(data["trading_date"]),
                session_id=str(data["session_id"]),
                ts_event_ns=int(data["ts_event_ns"]),
                sequence=int(data["sequence"]),
                action=str(data["action"]),
                side=str(data["side"]),
                order_id=int(data["order_id"]),
                price=None if data["price"] is None else float(data["price"]),
                size=int(data["size"]),
                flags=int(data["source_flags"]),
            )
            if current_key is not None and event.key != current_key:
                dropped_boundary += len(pending)
                pending.clear()
                validation_candidates.clear()
                book = FIFOBook()
                mids.clear()
                times.clear()
                last_capture_ts = None
                last_capture_event = event_count
            current_key = event.key
            event_count += 1
            interval_actions[event.action] = interval_actions.get(event.action, 0) + 1
            if event.action == "A" and event.side in {"B", "A"}:
                signed_volume += event.size if event.side == "B" else -event.size

            for capture in pending:
                capture.bid_order.observe(event, book)
                capture.ask_order.observe(event, book)

            expired = [
                order_id
                for order_id, candidate in validation_candidates.items()
                if event.ts_event_ns > candidate.expires_ns
            ]
            for order_id in expired:
                validation_candidates.pop(order_id)
            for candidate in validation_candidates.values():
                if event.action in {"C", "M"}:
                    candidate.reduce(
                        event.order_id, event.size if event.action == "C" else 2**63 - 1
                    )
                elif event.action == "F":
                    candidate.reduce(event.order_id, event.size)
            candidate = validation_candidates.get(event.order_id)
            if candidate is not None and event.action == "F":
                queue_remaining = sum(candidate.ahead.values())
                validation_fills += 1
                validation_eligible += int(queue_remaining == 0)
                validation_queue_error_sum += queue_remaining
                validation_candidates.pop(event.order_id, None)
            elif candidate is not None and event.action in {"C", "M"}:
                validation_candidates.pop(event.order_id, None)

            if event.action == "A" and event.price is not None and event.side in {"B", "A"}:
                validation_add_counter += 1
                top = book.top(event.side, 1)
                if validation_add_counter % 5000 == 0 and top and top[0][0] == event.price:
                    validation_sampled += 1
                    validation_candidates[event.order_id] = QueueValidationCandidate(
                        event.order_id,
                        event.side,
                        event.price,
                        event.ts_event_ns,
                        event.ts_event_ns + 10_000_000_000,
                        OrderedDict(book.queue_at(event.side, event.price)),
                    )

            try:
                book.apply(event)
            except ValueError:
                reconstruction_errors += 1
                continue
            if event.action == "R":
                dropped_boundary += len(pending)
                pending.clear()
                validation_candidates.clear()
                mids.clear()
                times.clear()
                last_capture_ts = None
                last_capture_event = event_count
                continue
            if not event.flags & F_LAST:
                continue
            bids, asks = book.top("B", 1), book.top("A", 1)
            if not bids or not asks or bids[0][0] >= asks[0][0]:
                continue
            current_mid = (bids[0][0] + asks[0][0]) / 2
            for capture in pending:
                capture.update_mid(event.ts_event_ns, current_mid)
            still_pending: list[PendingCapture] = []
            for capture in pending:
                if capture.ready():
                    features.append(capture.feature)
                    labels.append(_label_row(capture))
                else:
                    still_pending.append(capture)
            pending = still_pending

            if event_count - last_capture_event < sample_every:
                continue
            feature = _book_feature_row(
                event=event,
                book=book,
                source=source,
                reset_generation=int(data["reset_generation"]),
                tick_size=float(data["tick_size"]),
                contract_multiplier=float(data["contract_multiplier"]),
                tick_value=float(data["tick_value"]),
                currency=str(data["currency"]),
                mids=mids,
                times=times,
                last_capture_ts=last_capture_ts,
                events_since_capture=event_count - last_capture_event,
                interval_actions=interval_actions,
                signed_volume=signed_volume,
            )
            if feature is None:
                continue
            decision_time = event.ts_event_ns
            pending.append(
                PendingCapture(
                    feature=feature,
                    bid_order=SimulatedPassiveOrder(
                        event.key,
                        "B",
                        bids[0][0],
                        1,
                        decision_time,
                        decision_time + 1_000_000,
                        decision_time + 1_000_000_000,
                    ),
                    ask_order=SimulatedPassiveOrder(
                        event.key,
                        "A",
                        asks[0][0],
                        1,
                        decision_time,
                        decision_time + 1_000_000,
                        decision_time + 1_000_000_000,
                    ),
                    decision_mid=current_mid,
                    decision_bid=bids[0][0],
                    decision_ask=asks[0][0],
                    last_mid=current_mid,
                )
            )
            last_capture_ts = decision_time
            last_capture_event = event_count
            interval_actions = {"T": 0, "C": 0, "M": 0}
            signed_volume = 0.0
    con.close()
    dropped_boundary += len(pending)
    if not features or len(features) != len(labels):
        raise RuntimeError(
            f"invalid v2 replay output for {source.name}: {len(features)} features, {len(labels)} labels"
        )
    feature_tmp = feature_path.with_suffix(".parquet.tmp")
    label_tmp = label_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(features), feature_tmp, compression="zstd")
    pq.write_table(pa.Table.from_pylist(labels), label_tmp, compression="zstd")
    feature_tmp.replace(feature_path)
    label_tmp.replace(label_path)
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": source.name,
        "input_manifest": str(source.manifest),
        "input_manifest_hash": input_hash,
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
            "type": "SIMULATED_L3",
        },
        "reconstruction_errors": reconstruction_errors,
        "dropped_incomplete_at_group_boundary": dropped_boundary,
        "order_entry_latency_ns": 1_000_000,
        "fill_horizon_ns": 1_000_000_000,
        "markout_horizons_ns": [100_000_000, 1_000_000_000],
        "historical_order_validation": {
            "sampling_opportunities": validation_add_counter // 5000,
            "sampled_top_level_adds": validation_sampled,
            "actual_fills_observed": validation_fills,
            "queue_eligible_at_actual_fill": validation_eligible,
            "eligibility_rate": validation_eligible / validation_fills
            if validation_fills
            else None,
            "mean_queue_ahead_error_at_actual_fill": validation_queue_error_sum / validation_fills
            if validation_fills
            else None,
            "claim": "calibration/error diagnostic, not perfection",
        },
        "limitations": ["small hypothetical order", "no endogenous impact", "public sample scope"],
        "final_lockbox_v2_used": final_lockbox_used,
    }
    atomic_json(marker, payload)
    return marker


def build_development_v2(sample_every: int = 2000) -> Path:
    from .v2_l2 import reconstruct_ifeu_mbp10_development

    markers = [
        reconstruct_development_source(source, sample_every) for source in DEVELOPMENT_SOURCES
    ]
    markers.append(reconstruct_ifeu_mbp10_development(sample_every))
    output = data_root() / "manifests/features_v4_multi_market.json"
    payload = {
        "schema_version": 4,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "sources": [json.loads(path.read_text(encoding="utf-8")) for path in markers],
        "source_balancing_supported": True,
        "instrument_balancing_supported": True,
        "modality_masks": {"L1": 1, "L2": 2, "L3": 4},
        "final_lockbox_v2_used": False,
    }
    payload["dataset_fingerprint"] = content_hash(payload)
    atomic_json(output, payload)
    labels_manifest = data_root() / "manifests/labels_v2_l3.json"
    atomic_json(
        labels_manifest,
        {
            "schema_version": 2,
            "generated_at_utc": payload["generated_at_utc"],
            "type": "MIXED_BY_AVAILABILITY",
            "sources": [item["labels"] for item in payload["sources"]],
            "side_specific": True,
            "endpoint_crossing_proxy_used_as_fill": False,
            "fill_conditional_adverse_selection": True,
            "availability_policy": {
                "ES": "SIMULATED_L3",
                "BRN": "UNAVAILABLE_AFTER_UNRECOVERED_CLEAR_USE_MBP10_L2_ONLY",
            },
            "final_lockbox_v2_used": False,
        },
    )
    return output
