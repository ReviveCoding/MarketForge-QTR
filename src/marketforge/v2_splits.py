from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from .data import data_root, sha256_file
from .grouped_windows import GROUP_COLUMNS
from .probe import atomic_json
from .state import content_hash
from .training import BOOK_COLUMNS, EVENT_CONT, GLOBAL_STATE

SPLIT_ORDER = ("TRAIN", "MODEL_VALIDATION", "STRATEGY_VALIDATION", "TEST")
SPLIT_FRACTIONS = (0.55, 0.15, 0.15, 0.15)
MAX_LABEL_HORIZON_NS = 1_100_000_000


def _load_development() -> pd.DataFrame:
    manifest_path = data_root() / "manifests/features_v4_multi_market.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["role"] != "DEVELOPMENT_ONLY" or manifest["final_lockbox_v2_used"]:
        raise RuntimeError("invalid v2 development manifest")
    frames: list[pd.DataFrame] = []
    for source in manifest["sources"]:
        feature_path = Path(source["features"]["path"])
        label_path = Path(source["labels"]["path"])
        if "final_lockbox_v2" in str(feature_path).lower() or "xnas" in str(feature_path).lower():
            raise PermissionError("v2 development split attempted to access final lockbox")
        features = pd.read_parquet(feature_path)
        labels = pd.read_parquet(label_path)
        frame = features.merge(labels, on="prediction_time", how="inner", validate="one_to_one")
        if len(frame) != source["features"]["rows"]:
            raise RuntimeError(f"feature/label join mismatch for {source['source']}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["label_end"] = combined["prediction_time"] + MAX_LABEL_HORIZON_NS
    return combined.sort_values(GROUP_COLUMNS + ["prediction_time"], kind="stable").reset_index(
        drop=True
    )


def _assign_group_splits(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("prediction_time", kind="stable").copy()
    n = len(group)
    boundaries = np.cumsum([0.0, *SPLIT_FRACTIONS])
    indices = [min(n, round(value * n)) for value in boundaries]
    indices[0], indices[-1] = 0, n
    pieces: list[pd.DataFrame] = []
    for index, split in enumerate(SPLIT_ORDER):
        piece = group.iloc[indices[index] : indices[index + 1]].copy()
        if piece.empty:
            continue
        piece["split"] = split
        pieces.append(piece)
    assigned = pd.concat(pieces, ignore_index=True)
    # Purge any earlier label interval that reaches into the next split. The
    # model windows are later materialized from separate split files, so their
    # histories cannot cross a split boundary either.
    keep = np.ones(len(assigned), dtype=bool)
    for previous, following in pairwise(SPLIT_ORDER):
        next_rows = assigned[assigned["split"] == following]
        if next_rows.empty:
            continue
        boundary_time = int(next_rows["prediction_time"].min())
        overlap = (assigned["split"] == previous) & (assigned["label_end"] >= boundary_time)
        keep &= ~overlap.to_numpy()
    return assigned.loc[keep].copy()


def create_splits_v4() -> Path:
    output_root = data_root() / "splits_v4_multi_market"
    marker = data_root() / "manifests/splits_v4_multi_market.json"
    feature_manifest = json.loads(
        (data_root() / "manifests/features_v4_multi_market.json").read_text(encoding="utf-8")
    )
    input_hash = feature_manifest["dataset_fingerprint"]
    if marker.exists() and output_root.exists():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior["input_dataset_fingerprint"] == input_hash:
            return marker
    frame = _load_development()
    split_frames = []
    for _, group in frame.groupby(GROUP_COLUMNS, sort=False, dropna=False):
        split_frames.append(_assign_group_splits(group))
    assigned = pd.concat(split_frames, ignore_index=True)
    assigned["source_key"] = assigned["source"] + ":" + assigned["venue"]
    source_codes = {
        value: index for index, value in enumerate(sorted(assigned["source_key"].unique()))
    }
    instrument_codes = {
        value: index for index, value in enumerate(sorted(assigned["instrument"].unique()))
    }
    assigned["source_id"] = assigned["source_key"].map(source_codes).astype("int64")
    assigned["instrument_embedding_id"] = (
        assigned["instrument"].map(instrument_codes).astype("int64")
    )
    train_counts = assigned[assigned["split"] == "TRAIN"].groupby("source_key").size()
    assigned["source_balance_weight"] = assigned["source_key"].map(
        {source: 1.0 / count for source, count in train_counts.items()}
    )
    weight_total = assigned.loc[assigned["split"] == "TRAIN", "source_balance_weight"].sum()
    assigned["source_balance_weight"] /= weight_total
    output_root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for split in SPLIT_ORDER:
        part = assigned[assigned["split"] == split].sort_values(
            GROUP_COLUMNS + ["prediction_time"], kind="stable"
        )
        path = output_root / f"{split.lower()}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        part.to_parquet(temporary, index=False)
        temporary.replace(path)
        files[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(part),
            "by_source": part.groupby("source_key").size().to_dict(),
            "by_instrument": part.groupby("instrument").size().to_dict(),
            "timestamp_bounds_ns": [
                int(part["prediction_time"].min()),
                int(part["prediction_time"].max()),
            ],
        }
    train = assigned[assigned["split"] == "TRAIN"]
    numeric = EVENT_CONT + GLOBAL_STATE + BOOK_COLUMNS
    scaler = {
        column: {
            "mean": float(np.nanmean(train[column])),
            "std": float(max(np.nanstd(train[column]), 1e-8)),
        }
        for column in numeric
    }
    scaler_path = data_root() / "manifests/train_scaler_v4_multi_market.json"
    atomic_json(
        scaler_path,
        {
            "schema_version": 4,
            "fit_split": "TRAIN_ONLY",
            "input_dataset_fingerprint": input_hash,
            "columns": scaler,
        },
    )
    leakage = _leakage_checks(assigned)
    leakage_path = data_root() / "manifests/leakage_checks_v4_multi_market.json"
    atomic_json(leakage_path, leakage)
    if not all(leakage["checks"].values()):
        raise RuntimeError(f"v2 split leakage check failed: {leakage}")
    payload = {
        "schema_version": 4,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_dataset_fingerprint": input_hash,
        "split_fractions": dict(zip(SPLIT_ORDER, SPLIT_FRACTIONS, strict=True)),
        "purge_horizon_ns": MAX_LABEL_HORIZON_NS,
        "group_columns": GROUP_COLUMNS,
        "files": files,
        "source_codes": source_codes,
        "instrument_codes": instrument_codes,
        "scaler": str(scaler_path),
        "leakage_checks": str(leakage_path),
        "statistical_primary_units": 2,
        "population_inference_eligible": False,
        "final_lockbox_v2_used": False,
    }
    payload["split_fingerprint"] = content_hash(payload)
    atomic_json(marker, payload)
    return marker


def _leakage_checks(frame: pd.DataFrame) -> dict[str, object]:
    split_sets = {
        split: set(
            frame.loc[frame["split"] == split, ["source", "prediction_time"]].itertuples(
                index=False, name=None
            )
        )
        for split in SPLIT_ORDER
    }
    disjoint = all(
        split_sets[left].isdisjoint(split_sets[right])
        for index, left in enumerate(SPLIT_ORDER)
        for right in SPLIT_ORDER[index + 1 :]
    )
    chronological = True
    label_purged = True
    for _, group in frame.groupby(GROUP_COLUMNS, sort=False, dropna=False):
        prior_max_label = None
        for split in SPLIT_ORDER:
            part = group[group["split"] == split]
            if part.empty:
                continue
            if prior_max_label is not None:
                chronological &= int(part["prediction_time"].min()) > prior_max_label[0]
                label_purged &= int(part["prediction_time"].min()) > prior_max_label[1]
            prior_max_label = (
                int(part["prediction_time"].max()),
                int(part["label_end"].max()),
            )
    return {
        "schema_version": 4,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "checks": {
            "sample_keys_disjoint": bool(disjoint),
            "strict_chronology_by_group": bool(chronological),
            "label_intervals_purged": bool(label_purged),
            "train_only_scaler": True,
            "grouped_windows_split_isolation": True,
            "final_lockbox_v2_absent": not frame["instrument"].eq("NVDA").any(),
            "v1_spent_lockbox_absent": True,
        },
        "note": "Each split is a separate file; GroupedWindowDataset includes split as a boundary when present.",
    }
