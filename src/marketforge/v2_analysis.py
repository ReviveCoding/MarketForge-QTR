from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import data_root, sha256_file
from .model_v2 import MarketForgeM3TV2
from .probe import atomic_json
from .state import ROOT
from .training import gpu_lease
from .v2_baselines import _matrix, _predict_booster
from .v2_training import (
    V2WindowDataset,
    _load_scalers,
    _metrics,
    _model,
    _predict,
)


def _frame(split: str) -> pd.DataFrame:
    return pd.read_parquet(data_root() / f"splits_v4_multi_market/{split}.parquet")


def _dataset(split: str) -> V2WindowDataset:
    feature_scaler, target_scaler = _load_scalers()
    return V2WindowDataset(_frame(split), feature_scaler, target_scaler)


def _selected_m3t(device: torch.device) -> MarketForgeM3TV2:
    ladder = json.loads(
        (data_root() / "manifests/model_ladder_v2.json").read_text(encoding="utf-8")
    )
    checkpoint = ladder["branches"]["M3T-SSL-EPT"]["checkpoint"]
    model = _model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model.eval()


def run_simple_baselines_v2() -> None:
    train, valid = _frame("train"), _frame("model_validation")
    target_train = train["direction_1000ms"].to_numpy()
    target_valid = valid["direction_1000ms"].to_numpy()
    majority = np.full(len(valid), pd.Series(target_train).mode().iloc[0])
    microprice = np.sign(valid["microprice_minus_mid"].to_numpy()).astype(np.int64)
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=1701).fit(
        _matrix(train), target_train
    )
    logistic_prediction = logistic.predict(_matrix(valid))
    results = {
        "majority": float(f1_score(target_valid, majority, average="macro")),
        "microprice_sign": float(f1_score(target_valid, microprice, average="macro")),
        "logistic": float(f1_score(target_valid, logistic_prediction, average="macro")),
    }
    atomic_json(
        data_root() / "manifests/baselines_v2.json",
        {
            "schema_version": 2,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "partition": "MODEL_VALIDATION",
            "direction_macro_f1": results,
            "xgboost_primary": "data/manifests/xgboost_gpu_v2.json",
            "hist_gradient_boosting_primary_substitute": False,
            "test_used": False,
            "final_lockbox_v2_used": False,
        },
    )


def run_transfer_v2() -> None:
    train, valid = _frame("train"), _frame("model_validation")
    results: dict[str, object] = {}
    with gpu_lease():
        for train_instrument, test_instrument in (("ES", "BRN"), ("BRN", "ES")):
            train_part = train[train["instrument"] == train_instrument]
            test_part = valid[valid["instrument"] == test_instrument]
            model = xgb.XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.05,
                objective="multi:softprob",
                tree_method="hist",
                device="cuda",
                random_state=1701,
            )
            model.fit(
                _matrix(train_part),
                train_part["direction_1000ms"].to_numpy(np.int64) + 1,
            )
            probability, warnings_found = _predict_booster(model.get_booster(), _matrix(test_part))
            prediction = probability.argmax(axis=1) - 1
            results[f"{train_instrument}_to_{test_instrument}"] = {
                "train_rows": len(train_part),
                "test_rows": len(test_part),
                "macro_f1": float(
                    f1_score(test_part["direction_1000ms"], prediction, average="macro")
                ),
                "prediction_warnings": warnings_found,
            }
        torch.cuda.synchronize()
    atomic_json(
        data_root() / "manifests/transfer_v2.json",
        {
            "schema_version": 2,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "strict_zero_shot_xgboost": results,
            "m3t_current_protocol": "UNSUPERVISED_SEEN: both ES and BRN are present in SSL pretraining",
            "strict_zero_shot_m3t": "NOT_ESTABLISHED",
            "source_ood_final": "reserved XNAS/NVDA v2 lockbox",
            "population_transfer_inference": "SKIPPED_INSUFFICIENT_SAMPLE",
            "primary_units": 2,
            "test_used": False,
            "final_lockbox_v2_used": False,
        },
    )


def run_ablation_and_robustness_v2() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for M3T robustness inference")
    device = torch.device("cuda:0")
    dataset = _dataset("model_validation")
    model = _selected_m3t(device)
    with gpu_lease():
        base = _metrics(_predict(model, dataset, device))
        original_book = dataset.book.copy()
        dataset.book.fill(0)
        depth_dropout = _metrics(_predict(model, dataset, device))
        dataset.book[:] = original_book
        original_modality = dataset.modality.copy()
        dataset.modality[:, 2] = 0
        l3_mask_off = _metrics(_predict(model, dataset, device))
        dataset.modality[:] = original_modality
        original_source = dataset.source.copy()
        dataset.source[:] = 1 - dataset.source
        source_swap = _metrics(_predict(model, dataset, device))
        dataset.source[:] = original_source
    economics = json.loads(
        (data_root() / "manifests/development_economics_v2.json").read_text(encoding="utf-8")
    )
    active = {name: result for name, result in economics["results"].items() if name != "NO_TRADE"}
    cost_stress = {}
    for ticks in (0.0, 0.25, 0.5, 1.0):
        cost_stress[str(ticks)] = {
            name: result["price_pnl_usd"] - result["fees_usd"] - result["fills"] * 12.5 * ticks
            for name, result in active.items()
        }
    atomic_json(
        data_root() / "manifests/ablation_v2.json",
        {
            "schema_version": 2,
            "partition": "MODEL_VALIDATION",
            "base": base,
            "depth_dropout": depth_dropout,
            "l3_modality_mask_off": l3_mask_off,
            "source_embedding_swap": source_swap,
            "architecture_retraining_ablation": "STOPPED_AFTER_XGBOOST_WIN_AND_NEGATIVE_ECONOMICS",
            "test_used": False,
            "final_lockbox_v2_used": False,
        },
    )
    atomic_json(
        data_root() / "manifests/robustness_v2.json",
        {
            "schema_version": 2,
            "prediction_corruptions": {
                "depth_dropout": depth_dropout,
                "source_embedding_swap": source_swap,
            },
            "liquidation_cost_stress_realized_pnl_usd": cost_stress,
            "latency_resimulation": "NOT_ESTABLISHED: v2 labels are registered at 1 ms only",
            "queue_uncertainty": "ICE excluded after failed reconstruction; ES real-order eligibility 97.7%",
            "robust_edge": False,
            "reason": "no active strategy is profitable at the registered base cost",
            "test_used": False,
            "final_lockbox_v2_used": False,
        },
    )


def run_systems_v2() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for v2 systems benchmark")
    device = torch.device("cuda:0")
    dataset = _dataset("model_validation")
    model = _selected_m3t(device)
    first = dataset[0]
    timings = []
    with gpu_lease(), torch.inference_mode():
        for batch_size in (1, 64):
            batch = {
                name: value.unsqueeze(0).repeat((batch_size,) + (1,) * value.ndim).to(device)
                for name, value in first.items()
            }
            samples = []
            for _ in range(10):
                model(
                    batch["action"],
                    batch["side"],
                    batch["event"],
                    batch["state"],
                    batch["book"],
                    batch["source"],
                    batch["instrument"],
                    batch["modality"],
                )
            torch.cuda.synchronize()
            for _ in range(50):
                started = time.perf_counter_ns()
                model(
                    batch["action"],
                    batch["side"],
                    batch["event"],
                    batch["state"],
                    batch["book"],
                    batch["source"],
                    batch["instrument"],
                    batch["modality"],
                )
                torch.cuda.synchronize()
                samples.append((time.perf_counter_ns() - started) / 1e6)
            timings.append(
                {
                    "batch_size": batch_size,
                    "p50_ms": float(np.percentile(samples, 50)),
                    "p95_ms": float(np.percentile(samples, 95)),
                    "examples_per_second": batch_size / (np.mean(samples) / 1000),
                }
            )
    loader = DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=True)
    started = time.perf_counter()
    rows = sum(len(batch["action"]) for batch in loader)
    atomic_json(
        data_root() / "manifests/systems_benchmark_v2.json",
        {
            "schema_version": 2,
            "device": torch.cuda.get_device_name(0),
            "m3t_inference": timings,
            "dataloader": {
                "workers": 0,
                "rows": rows,
                "wall_seconds": time.perf_counter() - started,
                "windows_spawn_safe": True,
            },
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "medium_large": "STOPPED: XGBoost wins validation and all active economics are negative",
            "final_lockbox_v2_used": False,
        },
    )


def run_test_once_and_analysis_v2() -> None:
    marker = data_root() / "manifests/development_test_v2.json"
    if marker.exists():
        raise RuntimeError("v2 development TEST has already been evaluated")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for v2 M3T TEST inference")
    device = torch.device("cuda:0")
    dataset = _dataset("test")
    model = _selected_m3t(device)
    with gpu_lease():
        neural = _metrics(_predict(model, dataset, device))
    frame = _frame("test")
    booster = xgb.Booster()
    booster.load_model(ROOT / "artifacts/xgboost_v2/direction.json")
    probability, prediction_warnings = _predict_booster(booster, _matrix(frame))
    xgb_f1 = float(
        f1_score(frame["direction_1000ms"], probability.argmax(axis=1) - 1, average="macro")
    )
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_count": 1,
        "protocol_selected_before_test": True,
        "m3t_ssl_ept": neural,
        "xgboost_direction_macro_f1": xgb_f1,
        "xgboost_prediction_warnings": prediction_warnings,
        "controller_selection_changed": False,
        "economic_selection": "NO_TRADE retained; no TEST tuning",
        "final_lockbox_v2_used": False,
    }
    atomic_json(marker, payload)
    monitoring_rows = frame[["source_key", "instrument", "prediction_time"]].copy()
    monitoring_rows["prediction"] = probability.argmax(axis=1) - 1
    monitoring_rows["target"] = frame["direction_1000ms"].to_numpy()
    monitoring_rows["error"] = (monitoring_rows["prediction"] != monitoring_rows["target"]).astype(
        int
    )
    monitoring_rows["rolling_error"] = monitoring_rows.groupby("instrument")["error"].transform(
        lambda series: series.rolling(64, min_periods=16).mean()
    )
    monitoring_rows["state"] = np.where(
        monitoring_rows["rolling_error"].fillna(0) < 0.6, "GREEN", "WATCH"
    )
    log_path = ROOT / "artifacts/monitoring_v2/shadow_log.parquet"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    monitoring_rows.to_parquet(log_path, index=False)
    atomic_json(
        data_root() / "manifests/monitoring_v2.json",
        {
            "schema_version": 2,
            "partition": "TEST",
            "rows": len(monitoring_rows),
            "states": monitoring_rows["state"].value_counts().to_dict(),
            "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
            "execution": "shadow prediction; NO_TRADE selected",
            "final_lockbox_v2_used": False,
        },
    )
    atomic_json(
        data_root() / "manifests/statistical_inference_v2.json",
        {
            "schema_version": 2,
            "status": "SKIPPED_INSUFFICIENT_SAMPLE",
            "primary_unit": "instrument_x_trading_day",
            "available_development_units": 2,
            "confidence_intervals": "NOT ESTABLISHED",
            "pseudo_replication_avoided": True,
        },
    )
    atomic_json(
        data_root() / "manifests/selection_bias_v2.json",
        {
            "schema_version": 2,
            "status": "SKIPPED_INSUFFICIENT_SAMPLE",
            "recorded_model_branches": 6,
            "controller_rules": 1,
            "white_reality_check": "NOT ESTABLISHED",
            "spa": "NOT ESTABLISHED",
            "cscv_pbo": "NOT ESTABLISHED",
            "deflated_sharpe_ratio": "NOT ESTABLISHED",
        },
    )
    atomic_json(
        data_root() / "manifests/failure_analysis_v2.json",
        {
            "schema_version": 2,
            "where_marketforge_wins": "SSL-EPT improves on SUP-EPT direction and fill calibration on MODEL_VALIDATION",
            "where_it_ties": "return and volatility errors are broadly similar across neural branches",
            "where_it_fails": [
                "XGBoost has higher validation macro-F1 and lower fill Brier",
                "all active ES SIMULATED_L3 strategies lose after registered costs",
                "ICE MBO reconstruction fails after an unrecovered clear and is excluded from L3",
                "two development primary units cannot support population inference",
            ],
            "complexity_justified": False,
            "medium_large_scaling": "STOPPED",
            "final_lockbox_v2_used": False,
        },
    )


def run_remaining_development_v2() -> None:
    run_simple_baselines_v2()
    run_transfer_v2()
    run_ablation_and_robustness_v2()
    run_systems_v2()
    run_test_once_and_analysis_v2()
