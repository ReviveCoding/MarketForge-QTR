from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, mean_squared_error

from .data import data_root
from .results import append_result, dataset_fingerprint

FEATURES = [
    "spread",
    "relative_spread",
    "microprice_minus_mid",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "depth_bid_1",
    "depth_ask_1",
    "depth_bid_5",
    "depth_ask_5",
    "depth_bid_10",
    "depth_ask_10",
    "depth_slope",
    "depth_convexity",
    "return_1",
    "return_10",
    "realized_vol_50",
    "ewma_vol",
    "robust_vol_50",
    "event_rate",
    "trade_rate",
    "cancel_rate",
    "modify_rate",
    "signed_volume",
    "interarrival_ns",
]


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
    }


def run_baselines(max_train_rows: int = 20000) -> Path:
    split_root = data_root() / "splits" / "development"
    train = pd.read_parquet(split_root / "train.parquet").tail(max_train_rows)
    valid = pd.read_parquet(split_root / "model_validation.parquet")
    x_train = np.nan_to_num(train[FEATURES].to_numpy(dtype=np.float32))
    x_valid = np.nan_to_num(valid[FEATURES].to_numpy(dtype=np.float32))
    y_train = train["direction_10"].to_numpy()
    y_valid = valid["direction_10"].to_numpy()
    fingerprint = dataset_fingerprint()
    outputs: list[dict[str, object]] = []
    majority = np.full_like(y_valid, pd.Series(y_train).mode().iloc[0])
    outputs.append(
        {"model": "majority", "metrics": _metrics(y_valid, majority), "wall_seconds": 0.0}
    )
    micro = np.sign(valid["microprice_minus_mid"].to_numpy()).astype(np.int8)
    outputs.append(
        {"model": "microprice_sign", "metrics": _metrics(y_valid, micro), "wall_seconds": 0.0}
    )
    classifiers = {
        "logistic": LogisticRegression(max_iter=500, class_weight="balanced", random_state=1701),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, random_state=1701
        ),
    }
    for name, model in classifiers.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        prediction = model.predict(x_valid)
        outputs.append(
            {
                "model": name,
                "metrics": _metrics(y_valid, prediction),
                "wall_seconds": time.perf_counter() - started,
            }
        )
    ridge = Ridge(alpha=1.0).fit(x_train, train["future_return_10"].to_numpy())
    return_prediction = ridge.predict(x_valid)
    outputs.append(
        {
            "model": "ridge_return",
            "metrics": {
                "rmse": float(
                    mean_squared_error(valid["future_return_10"], return_prediction) ** 0.5
                )
            },
            "wall_seconds": 0.0,
        }
    )
    for output in outputs:
        append_result(
            {
                "stage": "baselines",
                "status": "SUCCEEDED",
                "dataset_fingerprint": fingerprint,
                "seed": 1701,
                "hardware": "CPU",
                "precision": "float32",
                "model": output["model"],
                "metrics_json": json.dumps(output["metrics"], sort_keys=True),
                "wall_seconds": output["wall_seconds"],
                "artifact_paths_json": "[]",
            }
        )
    report = data_root() / "manifests" / "baseline_results_v1.json"
    report.write_text(
        json.dumps({"dataset_fingerprint": fingerprint, "results": outputs}, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
