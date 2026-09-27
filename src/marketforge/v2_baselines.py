from __future__ import annotations

import json
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import brier_score_loss, f1_score, mean_absolute_error

from .baselines import FEATURES
from .data import data_root, sha256_file
from .probe import atomic_json
from .state import ROOT, content_hash
from .training import gpu_lease

XGB_FEATURES = FEATURES + [
    "session_position",
    "source_id",
    "instrument_embedding_id",
    "modality_l1",
    "modality_l2",
    "modality_l3",
    "tick_size",
    "contract_multiplier",
]


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(frame[XGB_FEATURES].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _target(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = frame[name].to_numpy(np.float32)
    if "markout" in name:
        values = values / frame["tick_size"].to_numpy(np.float32)
    return values


def _device(booster: xgb.Booster) -> str:
    config = json.loads(booster.save_config())
    return str(config["learner"]["generic_param"]["device"])


def _predict_booster(booster: xgb.Booster, matrix: np.ndarray) -> tuple[np.ndarray, list[str]]:
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        prediction = booster.predict(xgb.DMatrix(matrix))
    caught.extend(str(record.message) for record in records)
    return prediction, caught


def run_xgboost_v2() -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for primary v2 XGBoost training")
    split_manifest = json.loads(
        (data_root() / "manifests/splits_v4_multi_market.json").read_text(encoding="utf-8")
    )
    train = pd.read_parquet(data_root() / "splits_v4_multi_market/train.parquet")
    valid = pd.read_parquet(data_root() / "splits_v4_multi_market/model_validation.parquet")
    strategy = pd.read_parquet(data_root() / "splits_v4_multi_market/strategy_validation.parquet")
    x_train, x_valid, x_strategy = _matrix(train), _matrix(valid), _matrix(strategy)
    weights = train["source_balance_weight"].to_numpy(np.float64)
    weights = weights / weights.mean()
    model_root = ROOT / "artifacts/xgboost_v2"
    model_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, object] = {}
    strategy_predictions: dict[str, np.ndarray] = {
        "source": strategy["source"].to_numpy(),
        "source_key": strategy["source_key"].to_numpy(),
        "instrument": strategy["instrument"].to_numpy(),
        "prediction_time": strategy["prediction_time"].to_numpy(),
    }
    all_warnings: list[str] = []
    devices: set[str] = set()
    started = time.perf_counter()
    with gpu_lease():
        direction = xgb.XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            device="cuda",
            random_state=1701,
            n_jobs=4,
        )
        direction.fit(
            x_train, train["direction_1000ms"].to_numpy(np.int64) + 1, sample_weight=weights
        )
        booster = direction.get_booster()
        devices.add(_device(booster))
        valid_probability, caught = _predict_booster(booster, x_valid)
        strategy_probability, caught_strategy = _predict_booster(booster, x_strategy)
        all_warnings.extend(caught + caught_strategy)
        valid_prediction = valid_probability.argmax(axis=1) - 1
        outputs["direction"] = {
            "validation_macro_f1": float(
                f1_score(valid["direction_1000ms"], valid_prediction, average="macro")
            )
        }
        strategy_predictions["direction_probability_down"] = strategy_probability[:, 0]
        strategy_predictions["direction_probability_flat"] = strategy_probability[:, 1]
        strategy_predictions["direction_probability_up"] = strategy_probability[:, 2]
        direction_path = model_root / "direction.json"
        direction.save_model(direction_path)

        regression_targets = {
            "return": "return_1000ms",
            "bid_quote_markout": "bid_quote_markout_1000ms",
            "ask_quote_markout": "ask_quote_markout_1000ms",
            "volatility": "future_realized_vol_1000ms",
        }
        for name, target_name in regression_targets.items():
            model = xgb.XGBRegressor(
                n_estimators=160,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                tree_method="hist",
                device="cuda",
                random_state=1701,
                n_jobs=4,
            )
            model.fit(x_train, _target(train, target_name), sample_weight=weights)
            booster = model.get_booster()
            devices.add(_device(booster))
            valid_prediction, caught = _predict_booster(booster, x_valid)
            strategy_prediction, caught_strategy = _predict_booster(booster, x_strategy)
            all_warnings.extend(caught + caught_strategy)
            outputs[name] = {
                "validation_mae": float(
                    mean_absolute_error(_target(valid, target_name), valid_prediction)
                ),
                "unit": "ticks" if "markout" in target_name else "native",
            }
            strategy_predictions[name] = strategy_prediction
            model.save_model(model_root / f"{name}.json")

        for side in ("bid", "ask"):
            target_name = f"{side}_fill"
            train_l3 = train["l3_targets_available"].to_numpy(bool)
            valid_l3 = valid["l3_targets_available"].to_numpy(bool)
            model = xgb.XGBClassifier(
                n_estimators=160,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=1701,
                n_jobs=4,
            )
            model.fit(
                x_train[train_l3],
                train.loc[train_l3, target_name].to_numpy(np.int64),
                sample_weight=weights[train_l3],
            )
            booster = model.get_booster()
            devices.add(_device(booster))
            valid_probability, caught = _predict_booster(booster, x_valid)
            strategy_probability, caught_strategy = _predict_booster(booster, x_strategy)
            all_warnings.extend(caught + caught_strategy)
            outputs[f"{side}_fill"] = {
                "validation_brier": float(
                    brier_score_loss(valid.loc[valid_l3, target_name], valid_probability[valid_l3])
                )
            }
            strategy_predictions[f"{side}_fill_probability"] = strategy_probability
            model.save_model(model_root / f"{side}_fill.json")
        torch.cuda.synchronize()
    if not devices or not all(device.startswith("cuda") for device in devices):
        raise RuntimeError(f"XGBoost did not train entirely on CUDA: {sorted(devices)}")
    prediction_path = ROOT / "artifacts/predictions_v2/xgboost_strategy_validation.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(strategy_predictions).to_parquet(prediction_path, index=False)
    report = data_root() / "manifests/xgboost_gpu_v2.json"
    model_files = {
        path.name: {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(model_root.glob("*.json"))
    }
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "xgboost_version": xgb.__version__,
        "split_fingerprint": split_manifest["split_fingerprint"],
        "features": XGB_FEATURES,
        "source_balanced_train_weights": True,
        "devices_from_booster_config": sorted(devices),
        "tree_method": "hist",
        "prediction_path": "Booster.predict(DMatrix)",
        "prediction_warnings": all_warnings,
        "mismatched_device_warning_absent": not any(
            "mismatched devices" in warning.lower() for warning in all_warnings
        ),
        "validation": outputs,
        "strategy_predictions": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "rows": len(strategy),
        },
        "models": model_files,
        "wall_seconds": time.perf_counter() - started,
        "test_used": False,
        "final_lockbox_v2_used": False,
    }
    payload["config_hash"] = content_hash(
        {"features": XGB_FEATURES, "n_estimators": 160, "max_depth": 5, "seed": 1701}
    )
    atomic_json(report, payload)
    return report
