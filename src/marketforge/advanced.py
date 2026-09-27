from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from .baselines import FEATURES
from .data import data_root
from .economics import _simulate
from .model import MarketForgeM3T
from .probe import atomic_json
from .strategy import Prediction, quote_controller, risk_gate
from .training import GLOBAL_STATE, WindowDataset, gpu_lease

ROOT = Path(data_root()).parent


def _split(name: str) -> pd.DataFrame:
    return pd.read_parquet(data_root() / "splits" / "development" / f"{name}.parquet").reset_index(
        drop=True
    )


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(frame[FEATURES].to_numpy(np.float32))


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
    }


def train_gpu_xgboost() -> tuple[xgb.XGBClassifier, Path]:
    train = _split("train").tail(20000)
    valid = _split("model_validation")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
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
    started = time.perf_counter()
    with gpu_lease():
        model.fit(_matrix(train), train["direction_10"].to_numpy(np.int64) + 1)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prediction = model.predict(_matrix(valid)).astype(np.int64) - 1
    config = json.loads(model.get_booster().save_config())
    device = config["learner"]["generic_param"]["device"]
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"XGBoost silently failed to use CUDA: {device}")
    checkpoint = ROOT / "artifacts" / "checkpoints" / "xgboost_gpu_v1.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(checkpoint)
    report = data_root() / "manifests" / "xgboost_gpu_v1.json"
    atomic_json(
        report,
        {
            "xgboost_version": xgb.__version__,
            "device_from_booster_config": device,
            "tree_method": "hist",
            "train_rows": len(train),
            "validation_rows": len(valid),
            "validation": _metrics(valid["direction_10"].to_numpy(), prediction),
            "wall_seconds": elapsed,
            "checkpoint": str(checkpoint),
            "final_lockbox_used": False,
        },
    )
    return model, report


def run_transfer_study(model: xgb.XGBClassifier) -> Path:
    frame = _split("strategy_validation")
    prediction = model.predict(_matrix(frame)).astype(np.int64) - 1
    spread_bucket = pd.qcut(frame["relative_spread"].rank(method="first"), 3, labels=False)
    volatility_bucket = pd.qcut(frame["ewma_vol"].rank(method="first"), 3, labels=False)
    regimes: dict[str, dict[str, float | int]] = {}
    for family, bucket in (("liquidity", spread_bucket), ("volatility", volatility_bucket)):
        for level in range(3):
            selected = bucket.to_numpy() == level
            regimes[f"{family}_{level}"] = {
                "rows": int(selected.sum()),
                **_metrics(frame.loc[selected, "direction_10"].to_numpy(), prediction[selected]),
            }
    report = data_root() / "manifests" / "transfer_v1.json"
    atomic_json(
        report,
        {
            "temporal_oos_regime_results": regimes,
            "strict_unseen_instrument": "BLOCKED_EXTERNAL_DATA",
            "cross_source": "BLOCKED_EXTERNAL_DATA",
            "fx_to_futures": "BLOCKED_EXTERNAL_DATA",
            "futures_to_fx": "BLOCKED_EXTERNAL_DATA",
            "reason": "public core contains one ESZ5 instrument on one trading date",
            "claim": "NOT ESTABLISHED",
            "final_lockbox_used": False,
        },
    )
    return report


def run_robustness(model: xgb.XGBClassifier) -> Path:
    frame = _split("strategy_validation")
    train = _split("train")
    target = frame["direction_10"].to_numpy()
    base = _matrix(frame)
    corruptions: dict[str, np.ndarray] = {"clean": base.copy()}
    rng = np.random.default_rng(1701)
    missing = base.copy()
    missing[rng.random(missing.shape) < 0.05] = 0
    corruptions["missing_events_5pct"] = missing
    jitter = base.copy()
    jitter[:, FEATURES.index("interarrival_ns")] *= rng.lognormal(0, 0.25, len(frame))
    corruptions["timestamp_jitter"] = jitter
    delayed = np.vstack([base[:1], base[:-1]])
    corruptions["delayed_features_one_event"] = delayed
    size_corrupt = base.copy()
    for name in ("signed_volume", "depth_bid_1", "depth_ask_1", "depth_bid_5", "depth_ask_5"):
        size_corrupt[:, FEATURES.index(name)] *= 1.5
    corruptions["size_corruption_50pct"] = size_corrupt
    depth_drop = base.copy()
    for name in FEATURES:
        if "depth" in name:
            depth_drop[:, FEATURES.index(name)] = 0
    corruptions["depth_dropout"] = depth_drop
    corruptions["feed_gap_10pct"] = np.where(rng.random(base.shape) < 0.10, 0, base)
    stale = np.vstack([np.repeat(base[:1], 5, axis=0), base[:-5]])
    corruptions["stale_quote_five_events"] = stale
    spread_shock = base.copy()
    spread_shock[:, FEATURES.index("spread")] *= 2
    spread_shock[:, FEATURES.index("relative_spread")] *= 2
    corruptions["spread_shock_2x"] = spread_shock
    vol_shock = base.copy()
    for name in ("realized_vol_50", "ewma_vol", "robust_vol_50"):
        vol_shock[:, FEATURES.index(name)] *= 2
    corruptions["volatility_shock_2x"] = vol_shock
    predictive = {}
    for name, matrix in corruptions.items():
        prediction = model.predict(matrix).astype(np.int64) - 1
        predictive[name] = _metrics(target, prediction)

    probabilities = model.predict_proba(base)
    return_scale = float(np.median(np.abs(train["future_return_10"].to_numpy())))
    predictions = [
        Prediction(
            float((probabilities[i, 2] - probabilities[i, 0]) * return_scale),
            0.0,
            float(max(frame.iloc[i]["ewma_vol"], 0)),
            0.5,
            float(1 - probabilities[i].max()),
            float(1 - probabilities[i].max()),
            True,
        )
        for i in range(len(frame))
    ]
    economic = {}
    for spread in (0.25, 0.75, 1.5):
        economic[f"quote_width_{spread}"] = _simulate(frame, predictions, spread, 0.1, "learned")
    report = data_root() / "manifests" / "robustness_v1.json"
    atomic_json(
        report,
        {
            "partition": "STRATEGY_VALIDATION",
            "predictive_stress": predictive,
            "economic_quote_width_stress": economic,
            "economic_prediction_scale": {
                "value": return_scale,
                "source": "TRAIN median absolute future_return_10",
            },
            "latency_fee_slippage_queue_claim": "NOT ESTABLISHED: no empirical L2 queue/fill calibration",
            "robust_edge": False,
            "final_lockbox_used": False,
        },
    )
    return report


def run_systems_benchmark() -> Path:
    checkpoint = ROOT / "artifacts" / "checkpoints" / "m3t-ept_seed1701.pt"
    device = torch.device("cuda:0")
    model = MarketForgeM3T(len(GLOBAL_STATE), d_model=64, layers=2, heads=4).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    dataset = WindowDataset(_split("model_validation").head(512))
    first = dataset[0]
    timings = []
    with gpu_lease(), torch.inference_mode():
        model.eval()
        for precision in ("float32", "bfloat16"):
            for batch_size in (1, 32, 256):
                inputs = [
                    item.unsqueeze(0).repeat((batch_size,) + (1,) * item.ndim).to(device)
                    for item in first[:5]
                ]
                for _ in range(10):
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=precision == "bfloat16"
                    ):
                        model(*inputs)
                torch.cuda.synchronize()
                samples = []
                for _ in range(50):
                    started = time.perf_counter_ns()
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=precision == "bfloat16"
                    ):
                        model(*inputs)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter_ns() - started) / 1e6)
                timings.append(
                    {
                        "precision": precision,
                        "batch_size": batch_size,
                        "p50_ms": float(np.percentile(samples, 50)),
                        "p95_ms": float(np.percentile(samples, 95)),
                        "p99_ms": float(np.percentile(samples, 99)),
                        "examples_per_second": float(batch_size / (np.mean(samples) / 1000)),
                    }
                )
        inputs = [item.unsqueeze(0).to(device) for item in first[:5]]
        one = model(*inputs)["direction_logits"].detach().cpu()
        model2 = MarketForgeM3T(len(GLOBAL_STATE), d_model=64, layers=2, heads=4).to(device)
        model2.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=False)["model"]
        )
        model2.eval()
        two = model2(*inputs)["direction_logits"].detach().cpu()
    loader_results = []
    for workers in (0, 2):
        started = time.perf_counter()
        count = 0
        loader = DataLoader(dataset, batch_size=64, num_workers=workers, pin_memory=True)
        for batch in loader:
            count += len(batch[0])
        loader_results.append(
            {"workers": workers, "rows": count, "wall_seconds": time.perf_counter() - started}
        )
    report = data_root() / "manifests" / "systems_benchmark_v1.json"
    atomic_json(
        report,
        {
            "device": torch.cuda.get_device_name(0),
            "inference": timings,
            "dataloader": loader_results,
            "checkpoint_reload_exact": bool(torch.equal(one, two)),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "compile": "SKIPPED_RESOURCE_LIMIT: no development edge and Windows compile adds cold-start cost",
            "multi_gpu": "NOT APPLICABLE: one NVIDIA GPU",
            "final_lockbox_used": False,
        },
    )
    return report


def run_inference_and_selection_bias() -> tuple[Path, Path]:
    economics = json.loads(
        (data_root() / "manifests" / "development_economics_v1.json").read_text(encoding="utf-8")
    )
    effects = {
        "P3_calibrated_m3t_minus_xgb_proxy": economics["results"]["M3T_EPT_CAL"]["net_pnl"]
        - economics["results"]["XGBOOST_PROXY_HGB"]["net_pnl"],
        "P4_calibrated_m3t_minus_glft": economics["results"]["M3T_EPT_CAL"]["net_pnl"]
        - economics["results"]["GLFT"]["net_pnl"],
    }
    inference = data_root() / "manifests" / "statistical_inference_v1.json"
    atomic_json(
        inference,
        {
            "primary_unit": "instrument_x_trading_day",
            "available_units": 1,
            "point_effects_noninferential": effects,
            "confidence_intervals": "NOT ESTABLISHED",
            "day_instrument_regime_win_rates": "NOT ESTABLISHED",
            "reason": "one instrument-day cannot support dependence-preserving inference",
            "pseudo_replication_avoided": True,
        },
    )
    bias = data_root() / "manifests" / "selection_bias_v1.json"
    atomic_json(
        bias,
        {
            "controller_trials": economics["controller_grid_trials"],
            "recorded_model_trials": 6,
            "white_reality_check": "NOT ESTABLISHED",
            "spa": "NOT ESTABLISHED",
            "cscv_pbo": "NOT ESTABLISHED",
            "deflated_sharpe_ratio": "NOT ESTABLISHED",
            "reason": "one instrument-day and nonvalidated fill simulator make these diagnostics uninterpretable",
            "search_counts_preserved": True,
        },
    )
    return inference, bias


def run_monitoring(model: xgb.XGBClassifier) -> Path:
    frame = _split("strategy_validation")
    probabilities = model.predict_proba(_matrix(frame))
    predicted = probabilities.argmax(axis=1) - 1
    target = frame["direction_10"].to_numpy()
    rolling_error = pd.Series(predicted != target).rolling(256, min_periods=32).mean().to_numpy()
    confidence = probabilities.max(axis=1)
    calibration_gap = (
        pd.Series(np.abs(confidence - (predicted == target)))
        .rolling(256, min_periods=32)
        .mean()
        .to_numpy()
    )
    rows = []
    for index, row in frame.iterrows():
        prediction = Prediction(
            float(probabilities[index, 2] - probabilities[index, 0]),
            0.0,
            float(max(row.ewma_vol, 0)),
            0.5,
            float(1 - confidence[index]),
            float(1 - confidence[index]),
            True,
        )
        quote = quote_controller(float(row.mid), 0.0, prediction, 0.75)
        approved, reason = risk_gate(quote, prediction, 0.0, 1.0, 0.0, float(row.mid))
        error = rolling_error[index]
        gap = calibration_gap[index]
        state = (
            "GREEN"
            if (np.isnan(error) or error < 0.65) and (np.isnan(gap) or gap < 0.35)
            else "WATCH"
        )
        rows.append(
            {
                "sequence": int(row.sequence),
                "model_version": "xgboost_gpu_v1",
                "prediction": int(predicted[index]),
                "confidence": float(confidence[index]),
                "proposed_bid": quote.bid,
                "proposed_ask": quote.ask,
                "risk_approved": approved,
                "risk_reason": reason,
                "realized_direction": int(target[index]),
                "rolling_error": None if np.isnan(error) else float(error),
                "rolling_calibration_gap": None if np.isnan(gap) else float(gap),
                "state": state,
            }
        )
    log_path = ROOT / "artifacts" / "monitoring" / "shadow_log_v1.parquet"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(log_path, index=False)
    report = data_root() / "manifests" / "monitoring_v1.json"
    states = pd.Series([row["state"] for row in rows]).value_counts().to_dict()
    atomic_json(
        report,
        {
            "partition": "STRATEGY_VALIDATION",
            "rows": len(rows),
            "states": states,
            "log": str(log_path),
            "execution": "counterfactual only",
            "final_lockbox_used": False,
        },
    )
    return report
