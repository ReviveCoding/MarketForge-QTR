from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import data_root, sha256_file
from .model_v2 import MarketForgeM3TV2
from .probe import atomic_json
from .specs import load_registry
from .state import ROOT, content_hash
from .v2_training import _datasets, _model, _predict


def export_m3t_strategy_predictions() -> Path:
    ladder = json.loads(
        (data_root() / "manifests/model_ladder_v2.json").read_text(encoding="utf-8")
    )
    checkpoint = Path(ladder["branches"]["M3T-SSL-EPT"]["checkpoint"])
    calibration = ladder["branches"]["M3T-SSL-EPT-CAL"]["calibration"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for M3T prediction; CPU fallback forbidden")
    device = torch.device("cuda:0")
    model: MarketForgeM3TV2 = _model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    datasets, target_scaler = _datasets()
    dataset = datasets["STRATEGY_VALIDATION"]
    prediction = _predict(model, dataset, device)
    rows = dataset.frame.iloc[dataset.valid_ends].reset_index(drop=True)
    output = rows[
        [
            "source_key",
            "instrument",
            "prediction_time",
            "tick_size",
            "contract_multiplier",
            "tick_value",
        ]
    ].copy()
    for side in ("bid", "ask"):
        scaled = prediction[f"{side}_quote_markout"]
        affine = calibration[f"{side}_markout_affine"]
        calibrated = affine["slope"] * scaled + affine["intercept"]
        scale = target_scaler[f"{side}_quote_markout_1000ms"]
        output[f"{side}_markout_ticks"] = calibrated * scale.std + scale.mean
        platt = calibration[f"{side}_fill_platt"]
        output[f"{side}_fill_probability"] = 1 / (
            1 + np.exp(-(platt["slope"] * prediction[f"{side}_fill_logit"] + platt["intercept"]))
        )
        output[f"{side}_adverse_probability"] = 1 / (
            1 + np.exp(-prediction[f"{side}_adverse_logit"])
        )
    return_scale = target_scaler["return_1000ms"]
    output["expected_return"] = prediction["return_mean"] * return_scale.std + return_scale.mean
    vol_scale = target_scaler["future_realized_vol_1000ms"]
    output["volatility"] = prediction["volatility"] * vol_scale.std
    path = ROOT / "artifacts/predictions_v2/m3t_ssl_ept_cal_strategy_validation.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False)
    return path


def _decision(prediction: pd.Series, side: str, cost_ticks: float) -> bool:
    expected_ticks = float(prediction[f"{side}_markout_ticks"])
    fill_probability = float(prediction[f"{side}_fill_probability"])
    adverse = float(prediction[f"{side}_adverse_probability"])
    return fill_probability * (expected_ticks - cost_ticks) > 0 and adverse < 0.7


def _simulate(
    frame: pd.DataFrame,
    prediction: pd.DataFrame | None,
    policy: str,
) -> dict[str, object]:
    es = load_registry()["databento:GLBX.MDP3:ES"]
    totals = {
        "price_pnl_usd": 0.0,
        "spread_capture_usd": 0.0,
        "fees_usd": 0.0,
        "slippage_usd": 0.0,
        "liquidation_cost_usd": 0.0,
        "hedge_cost_usd": 0.0,
        "realized_pnl_usd": 0.0,
        "unrealized_pnl_usd": 0.0,
        "nav_usd": 0.0,
    }
    fills = quotes = 0
    nav_path: list[float] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        selected = {"bid": False, "ask": False}
        if policy == "FIXED_PASSIVE_BOTH":
            selected = {"bid": True, "ask": True}
        elif policy == "AS_LIKE_HEURISTIC":
            selected = {
                "bid": float(row["imbalance_1"]) >= -0.35,
                "ask": float(row["imbalance_1"]) <= 0.35,
            }
        elif policy == "GLFT_LIKE_HEURISTIC":
            selected = {
                "bid": float(row["imbalance_5"]) > 0,
                "ask": float(row["imbalance_5"]) < 0,
            }
        elif policy in {"M3T_COMMON_CONTROLLER", "XGBOOST_COMMON_CONTROLLER"}:
            assert prediction is not None
            pred = prediction.iloc[index]
            # Two entry/exit commissions plus a quarter-tick liquidation
            # assumption are converted to ticks before a common decision rule.
            cost_ticks = (2 * es.fee_value) / es.tick_value + 0.25
            selected = {side: _decision(pred, side, cost_ticks) for side in ("bid", "ask")}
        for side in ("bid", "ask"):
            if not selected[side]:
                continue
            quotes += 1
            fraction = float(row[f"{side}_fill_fraction"])
            markout = row[f"{side}_fill_conditional_markout_100ms"]
            if fraction <= 0 or pd.isna(markout):
                continue
            fills += 1
            quantity = fraction
            total_price_pnl = float(markout) * es.contract_multiplier * quantity
            spread_capture = (
                (
                    (float(row["mid"]) - float(row["bid_px_01"]))
                    if side == "bid"
                    else (float(row["ask_px_01"]) - float(row["mid"]))
                )
                * es.contract_multiplier
                * quantity
            )
            fees = 2 * es.fee_value * quantity
            liquidation = es.ticks_to_usd(0.25, quantity)
            realized = total_price_pnl - fees - liquidation
            totals["price_pnl_usd"] += total_price_pnl
            totals["spread_capture_usd"] += spread_capture
            totals["fees_usd"] += fees
            totals["liquidation_cost_usd"] += liquidation
            totals["realized_pnl_usd"] += realized
            totals["nav_usd"] += realized
        nav_path.append(totals["nav_usd"])
    nav = np.asarray(nav_path or [0.0])
    running_max = np.maximum.accumulate(nav)
    return {
        **totals,
        "quotes": quotes,
        "fills": fills,
        "fill_per_quote": fills / quotes if quotes else 0.0,
        "max_drawdown_usd": float(np.max(running_max - nav)),
        "ending_inventory_contracts": 0.0,
        "accounting_identity_error_usd": abs(
            totals["realized_pnl_usd"]
            - (
                totals["price_pnl_usd"]
                - totals["fees_usd"]
                - totals["slippage_usd"]
                - totals["liquidation_cost_usd"]
                - totals["hedge_cost_usd"]
            )
        ),
    }


def run_development_economics_v2() -> Path:
    m3t_path = export_m3t_strategy_predictions()
    xgb_path = ROOT / "artifacts/predictions_v2/xgboost_strategy_validation.parquet"
    frame = pd.read_parquet(data_root() / "splits_v4_multi_market/strategy_validation.parquet")
    # Only MBO-validated ES has genuine counterfactual queue labels.
    frame = frame[(frame["instrument"] == "ES") & (frame["l3_targets_available"] == 1)].copy()
    m3t = pd.read_parquet(m3t_path).merge(
        frame[["prediction_time"]], on="prediction_time", how="inner"
    )
    xgb = pd.read_parquet(xgb_path).merge(
        frame[["prediction_time"]], on="prediction_time", how="inner"
    )
    frame = frame[frame["prediction_time"].isin(m3t["prediction_time"])].sort_values(
        "prediction_time"
    )
    m3t = m3t.sort_values("prediction_time")
    xgb = xgb[xgb["prediction_time"].isin(frame["prediction_time"])].sort_values("prediction_time")
    for side in ("bid", "ask"):
        xgb[f"{side}_adverse_probability"] = 1 / (1 + np.exp(xgb[f"{side}_quote_markout"]))
        xgb[f"{side}_markout_ticks"] = xgb[f"{side}_quote_markout"]
    results = {
        "NO_TRADE": _simulate(frame, None, "NO_TRADE"),
        "FIXED_PASSIVE_BOTH": _simulate(frame, None, "FIXED_PASSIVE_BOTH"),
        "AS_LIKE_HEURISTIC": _simulate(frame, None, "AS_LIKE_HEURISTIC"),
        "GLFT_LIKE_HEURISTIC": _simulate(frame, None, "GLFT_LIKE_HEURISTIC"),
        "M3T_COMMON_CONTROLLER": _simulate(frame, m3t, "M3T_COMMON_CONTROLLER"),
        "XGBOOST_COMMON_CONTROLLER": _simulate(frame, xgb, "XGBOOST_COMMON_CONTROLLER"),
    }
    output = data_root() / "manifests/development_economics_v2.json"
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "partition": "STRATEGY_VALIDATION",
        "execution_scope": "ES SIMULATED_L3 only",
        "rows": len(frame),
        "results": results,
        "controller": {
            "common_rule": "quote side iff fill_probability * (predicted_markout_ticks - cost_ticks) > 0 and adverse_probability < 0.7",
            "round_trip_fee_usd": 2 * load_registry()["databento:GLBX.MDP3:ES"].fee_value,
            "liquidation_cost_ticks": 0.25,
            "slippage_usd": 0.0,
            "hedge_cost_usd": 0.0,
        },
        "baseline_names": {
            "AS": "AS_LIKE_HEURISTIC",
            "GLFT": "GLFT_LIKE_HEURISTIC",
            "reason": "available best-quote counterfactual labels do not identify faithful multi-level intensity parameters",
            "literature": [
                "Avellaneda and Stoikov (2008), High-frequency trading in a limit order book",
                "Guéant, Lehalle, and Fernandez-Tapia (2013), Dealing with the inventory risk",
            ],
        },
        "dimensional_accounting": {
            "price_pnl_usd": "side-specific post-fill markout * contract_multiplier * filled_contracts",
            "spread_capture_usd": "diagnostic component; not added twice to realized PnL",
            "fees_usd": "two sides per filled/then-liquidated contract",
            "liquidation_cost_usd": "0.25 tick per filled contract",
        },
        "small_order_no_endogenous_impact": True,
        "headline_live_pnl_permitted": False,
        "test_used": False,
        "final_lockbox_v2_used": False,
        "prediction_artifacts": {
            "m3t": {"path": str(m3t_path), "sha256": sha256_file(m3t_path)},
            "xgboost": {"path": str(xgb_path), "sha256": sha256_file(xgb_path)},
        },
    }
    payload["config_hash"] = content_hash(payload["controller"])
    if any(value["accounting_identity_error_usd"] > 1e-9 for value in results.values()):
        raise RuntimeError("v2 economic accounting identity failed")
    atomic_json(output, payload)
    return output
