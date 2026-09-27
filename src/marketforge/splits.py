from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .data import data_root, sha256_file
from .probe import atomic_json

HORIZON = 10
PARTITIONS = ("TRAIN", "MODEL_VALIDATION", "STRATEGY_VALIDATION", "TEST", "FINAL_LOCKBOX")


def _safe_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(matrix.shape[1], dtype=float)
    scale = np.ones(matrix.shape[1], dtype=float)
    for index in range(matrix.shape[1]):
        finite = matrix[np.isfinite(matrix[:, index]), index]
        if finite.size:
            mean[index] = float(finite.mean())
            candidate = float(finite.std())
            scale[index] = candidate if candidate > 0 else 1.0
    return mean, scale


def build_splits(feature_file: Path) -> Path:
    root = data_root()
    manifest = root / "manifests" / "splits_databento_mbo_v3.json"
    if manifest.exists():
        return manifest
    table = pq.read_table(feature_file)
    frame = table.to_pandas()
    frame = frame.sort_values("prediction_time", kind="stable").reset_index(drop=True)
    n = len(frame)
    if n < 1000:
        raise RuntimeError(f"insufficient feature rows: {n}")
    mids = frame["mid"].to_numpy(dtype=float)
    times = frame["prediction_time"].to_numpy(dtype=np.int64)
    future_mid = np.roll(mids, -HORIZON)
    label_end = np.roll(times, -HORIZON)
    future_return = np.log(future_mid / mids)
    future_vol = np.full(n, np.nan)
    one_step = np.diff(np.log(mids), prepend=np.nan)
    for index in range(n - HORIZON):
        future_vol[index] = float(np.nanstd(one_step[index + 1 : index + HORIZON + 1]))
    usable = n - HORIZON
    boundaries = [
        0,
        int(usable * 0.60),
        int(usable * 0.70),
        int(usable * 0.80),
        int(usable * 0.90),
        usable,
    ]
    train_returns = np.abs(future_return[: boundaries[1]])
    direction_threshold = float(np.nanquantile(train_returns, 0.67))
    direction = np.where(
        future_return > direction_threshold,
        1,
        np.where(future_return < -direction_threshold, -1, 0),
    )
    frame["future_mid_10"] = future_mid
    frame["future_return_10"] = future_return
    frame["future_volatility_10"] = future_vol
    frame["direction_10"] = direction.astype(np.int8)
    frame["label_end"] = label_end
    frame["fill_metadata"] = "SIMULATED_L3"
    frame["counterfactual_bid_markout_10"] = future_mid - frame["bid_px_01"].to_numpy()
    frame["counterfactual_ask_markout_10"] = frame["ask_px_01"].to_numpy() - future_mid
    feature_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "instrument",
            "fill_metadata",
            "direction_10",
            "future_mid_10",
            "future_return_10",
            "future_volatility_10",
            "label_end",
            "counterfactual_bid_markout_10",
            "counterfactual_ask_markout_10",
        }
        and frame[column].dtype.kind in "fiu"
        and column
        not in {
            "ts_event_ns",
            "sequence",
            "information_start",
            "prediction_time",
            "event_action_id",
            "event_side_id",
            "modality_mask",
        }
    ]
    train_matrix = frame.loc[: boundaries[1] - HORIZON - 1, feature_columns].to_numpy(dtype=float)
    mean, scale = _safe_stats(train_matrix)
    scaler = {
        "fit_partition": "TRAIN",
        "columns": feature_columns,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "fit_rows": len(train_matrix),
    }
    scaler_path = root / "manifests" / "train_scaler_v3.json"
    atomic_json(scaler_path, scaler)
    split_counts: dict[str, int] = {}
    split_bounds: dict[str, list[int]] = {}
    dev_dir = root / "splits" / "development"
    lockbox_dir = root / "final_lockbox"
    dev_dir.mkdir(parents=True, exist_ok=True)
    lockbox_dir.mkdir(parents=True, exist_ok=True)
    for split_index, name in enumerate(PARTITIONS):
        start, stop = boundaries[split_index], boundaries[split_index + 1]
        if split_index > 0:
            start += HORIZON  # embargo
        stop -= HORIZON  # purge labels crossing the next boundary
        subset = frame.iloc[start:stop].copy()
        subset["partition"] = name
        matrix = subset[feature_columns].to_numpy(dtype=float)
        normalized = (matrix - mean) / scale
        for column_index, column in enumerate(feature_columns):
            subset[column] = normalized[:, column_index]
        destination = (
            lockbox_dir if name == "FINAL_LOCKBOX" else dev_dir
        ) / f"{name.lower()}.parquet"
        subset.to_parquet(destination, index=False)
        split_counts[name] = len(subset)
        split_bounds[name] = [int(subset["prediction_time"].min()), int(subset["label_end"].max())]
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "feature_sha256": sha256_file(feature_file),
        "horizon_rows": HORIZON,
        "purge_rows": HORIZON,
        "embargo_rows": HORIZON,
        "direction_threshold_train_only": direction_threshold,
        "split_counts": split_counts,
        "split_bounds": split_bounds,
        "scaler": str(scaler_path),
        "final_lockbox_root": str(lockbox_dir),
        "final_lockbox_access_policy": "DENY_UNTIL_FROZEN_FINAL_EVALUATION",
    }
    atomic_json(manifest, payload)
    return manifest


def verify_leakage(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bounds = manifest["split_bounds"]
    ordered = [bounds[name] for name in PARTITIONS]
    nonoverlap = all(ordered[index][1] < ordered[index + 1][0] for index in range(len(ordered) - 1))
    scaler = json.loads(Path(manifest["scaler"]).read_text(encoding="utf-8"))
    finite_scaler = all(math.isfinite(value) for value in scaler["mean"] + scaler["scale"])
    checks = {
        "strict_chronological_nonoverlap": nonoverlap,
        "purge_positive": manifest["purge_rows"] >= HORIZON,
        "embargo_positive": manifest["embargo_rows"] >= HORIZON,
        "scaler_train_only": scaler["fit_partition"] == "TRAIN",
        "scaler_finite": finite_scaler,
        "final_lockbox_separate_root": "final_lockbox" in manifest["final_lockbox_root"],
    }
    checks["passed"] = all(checks.values())
    return checks
