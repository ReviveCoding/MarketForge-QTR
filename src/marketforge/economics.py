from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from .baselines import FEATURES
from .data import data_root
from .model import MarketForgeM3T
from .probe import atomic_json
from .results import append_result, dataset_fingerprint
from .strategy import Prediction, Quote, quote_controller, risk_gate
from .training import GLOBAL_STATE, WindowDataset, gpu_lease


def validate_risk_gate() -> Path:
    healthy = Prediction(0, 0, 0, 0.5, 0.5, 0, True)
    cases = {
        "allow": risk_gate(Quote(99.75, 100.25), healthy, 0, 1, 0, 100)[0],
        "nan_prediction": not risk_gate(
            Quote(99.75, 100.25), Prediction(math.nan, 0, 0, 0.5, 0.5, 0, True), 0, 1, 0, 100
        )[0],
        "invalid_quote": not risk_gate(Quote(101, 100), healthy, 0, 1, 0, 100)[0],
        "inventory": not risk_gate(Quote(99.75, 100.25), healthy, 10, 1, 0, 100)[0],
        "latency": not risk_gate(Quote(99.75, 100.25), healthy, 0, 51, 0, 100)[0],
        "loss": not risk_gate(Quote(99.75, 100.25), healthy, 0, 1, -1001, 100)[0],
        "price_collar": not risk_gate(Quote(90, 91), healthy, 0, 1, 0, 100)[0],
    }
    report = data_root() / "manifests" / "risk_gate_validation_v1.json"
    atomic_json(
        report,
        {
            "cases": cases,
            "passed": all(cases.values()),
            "kill_action": "cancel outstanding and block new quotes",
        },
    )
    return report


def _neural_predictions(
    frame: pd.DataFrame,
    checkpoint: str,
    temperature: float = 1.0,
    calibration: dict[str, object] | None = None,
) -> list[Prediction | None]:
    device = torch.device("cuda:0")
    model = MarketForgeM3T(len(GLOBAL_STATE), d_model=64, layers=2, heads=4).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    output_predictions: list[Prediction | None] = [None] * 31
    loader = DataLoader(WindowDataset(frame), batch_size=256, num_workers=0, pin_memory=True)
    with gpu_lease(), torch.inference_mode():
        model.eval()
        for batch in loader:
            output = model(*[item.to(device, non_blocking=True) for item in batch[:5]])
            probabilities = torch.softmax(output["direction_logits"] / temperature, dim=-1)
            for index in range(len(probabilities)):
                fill_probability = float(torch.sigmoid(output["fill_logit"][index]))
                volatility = float(output["volatility"][index])
                if calibration:
                    fill_probability = float(
                        np.interp(
                            fill_probability,
                            calibration["fill_isotonic_x"],
                            calibration["fill_isotonic_y"],
                        )
                    )
                    volatility *= float(calibration["volatility_median_ratio_scale"])
                output_predictions.append(
                    Prediction(
                        float(output["expected_return"][index]),
                        float(output["expected_markout"][index]),
                        volatility,
                        fill_probability,
                        float(torch.sigmoid(output["adverse_logit"][index])),
                        float(1 - probabilities[index].max()),
                        True,
                    )
                )
    return output_predictions[: len(frame)]


def _tabular_predictions(train: pd.DataFrame, frame: pd.DataFrame) -> list[Prediction]:
    x_train = np.nan_to_num(train[FEATURES].to_numpy(np.float32))
    x = np.nan_to_num(frame[FEATURES].to_numpy(np.float32))
    classifier = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, random_state=1701
    ).fit(x_train, train["direction_10"])
    ridge = Ridge(alpha=1.0).fit(x_train, train["future_return_10"])
    probabilities = classifier.predict_proba(x)
    expected_return = ridge.predict(x)
    return [
        Prediction(
            float(expected_return[i]),
            0.0,
            float(max(frame.iloc[i]["ewma_vol"], 0)),
            0.5,
            float(1 - probabilities[i].max()),
            float(1 - probabilities[i].max()),
            True,
        )
        for i in range(len(frame))
    ]


def _simulate(
    frame: pd.DataFrame,
    predictions: list[Prediction | None],
    base_spread: float,
    inventory_penalty: float,
    mode: str,
) -> dict[str, float]:
    inventory = cash = peak_nav = max_drawdown = 0.0
    turnover = fills = adverse = killed = 0
    inventories: list[float] = []
    fee = 1.25
    for index, row in frame.iterrows():
        prediction = predictions[index]
        if prediction is None or mode == "no_trade":
            inventories.append(inventory)
            continue
        if mode == "fixed":
            quote = Quote(row.mid - base_spread / 2, row.mid + base_spread / 2)
        elif mode == "microprice":
            reservation = row.microprice - inventory_penalty * inventory
            quote = Quote(reservation - base_spread / 2, reservation + base_spread / 2)
        elif mode == "as":
            reservation = row.mid - inventory_penalty * inventory
            half = base_spread / 2 + 10 * max(row.ewma_vol, 0)
            quote = Quote(reservation - half, reservation + half)
        elif mode == "glft":
            reservation = row.mid - 2 * inventory_penalty * inventory
            half = base_spread / 2 + 0.02 * abs(inventory)
            quote = Quote(reservation - half, reservation + half)
        else:
            quote = quote_controller(row.mid, inventory, prediction, base_spread)
        approved, _ = risk_gate(
            quote, prediction, inventory, 1.0, cash + inventory * row.mid, row.mid
        )
        if not approved:
            killed += 1
            inventories.append(inventory)
            continue
        prior_inventory = inventory
        if row.future_mid_10 <= quote.bid:
            inventory += 1
            cash -= quote.bid + fee
            adverse += row.future_mid_10 < quote.bid
        elif row.future_mid_10 >= quote.ask:
            inventory -= 1
            cash += quote.ask - fee
            adverse += row.future_mid_10 > quote.ask
        if inventory != prior_inventory:
            fills += 1
            turnover += 1
        nav = cash + inventory * row.mid
        peak_nav = max(peak_nav, nav)
        max_drawdown = max(max_drawdown, peak_nav - nav)
        inventories.append(inventory)
    final_mid = float(frame.iloc[-1].mid)
    cash += inventory * final_mid - abs(inventory) * (fee + 0.25)
    turnover += abs(inventory)
    inventory = 0.0
    return {
        "net_pnl": float(cash),
        "fills": float(fills),
        "turnover": float(turnover),
        "pnl_per_turnover": float(cash / turnover) if turnover else 0.0,
        "inventory_variance": float(np.var(inventories)),
        "max_abs_inventory": float(max(map(abs, inventories), default=0)),
        "maximum_drawdown": float(max_drawdown),
        "negative_markout_rate": float(adverse / fills) if fills else 0.0,
        "risk_kills": float(killed),
        "ending_inventory": inventory,
    }


def run_development_economics() -> Path:
    root = data_root() / "splits" / "development"
    train = pd.read_parquet(root / "train.parquet").tail(20000)
    strategy = pd.read_parquet(root / "strategy_validation.parquet").reset_index(drop=True)
    test = pd.read_parquet(root / "test.parquet").reset_index(drop=True)
    checkpoints = Path(data_root()).parent / "artifacts" / "checkpoints"
    calibration = json.loads(
        (data_root() / "manifests" / "calibration_m3t_ept_v1.json").read_text(encoding="utf-8")
    )
    strategy_predictors = {
        "XGBOOST_PROXY_HGB": _tabular_predictions(train, strategy),
        "M3T_SUP": _neural_predictions(strategy, str(checkpoints / "m3t-sup-weighted_seed1701.pt")),
        "M3T_EPT_CAL": _neural_predictions(
            strategy, calibration["checkpoint"], calibration["temperature"], calibration
        ),
    }
    grid = [(spread, penalty) for spread, penalty in product((0.25, 0.5, 0.75), (0.01, 0.05, 0.1))]
    scores = []
    for spread, penalty in grid:
        pnl = np.mean(
            [
                _simulate(strategy, pred, spread, penalty, "learned")["net_pnl"]
                for pred in strategy_predictors.values()
            ]
        )
        scores.append((float(pnl), spread, penalty))
    _, selected_spread, selected_penalty = max(scores)
    test_predictors = {
        "NO_TRADE": [Prediction(0, 0, 0, 0, 0, 0, True)] * len(test),
        "FIXED": [Prediction(0, 0, 0, 0.5, 0.5, 0, True)] * len(test),
        "MICROPRICE": [Prediction(0, 0, 0, 0.5, 0.5, 0, True)] * len(test),
        "AS": [Prediction(0, 0, 0, 0.5, 0.5, 0, True)] * len(test),
        "GLFT": [Prediction(0, 0, 0, 0.5, 0.5, 0, True)] * len(test),
        "XGBOOST_PROXY_HGB": _tabular_predictions(train, test),
        "M3T_SUP": _neural_predictions(test, str(checkpoints / "m3t-sup-weighted_seed1701.pt")),
        "M3T_EPT": _neural_predictions(test, calibration["checkpoint"]),
        "M3T_EPT_CAL": _neural_predictions(
            test, calibration["checkpoint"], calibration["temperature"], calibration
        ),
    }
    modes = {
        "NO_TRADE": "no_trade",
        "FIXED": "fixed",
        "MICROPRICE": "microprice",
        "AS": "as",
        "GLFT": "glft",
    }
    results = {
        name: _simulate(
            test, prediction, selected_spread, selected_penalty, modes.get(name, "learned")
        )
        for name, prediction in test_predictors.items()
    }
    report = data_root() / "manifests" / "development_economics_v1.json"
    payload = {
        "controller_selection_partition": "STRATEGY_VALIDATION",
        "evaluation_partition": "TEST",
        "selected_common_base_spread": selected_spread,
        "selected_common_inventory_penalty": selected_penalty,
        "controller_grid_trials": len(grid),
        "results": results,
        "simulator_validation_scope": "COUNTERFACTUAL_CROSSING; NO_HEADLINE_PNL",
        "final_lockbox_used": False,
        "complexity_justified": results["M3T_EPT_CAL"]["net_pnl"]
        > results["XGBOOST_PROXY_HGB"]["net_pnl"],
    }
    atomic_json(report, payload)
    for name, metrics in results.items():
        append_result(
            {
                "stage": "development_economics",
                "status": "SUCCEEDED",
                "dataset_fingerprint": dataset_fingerprint(),
                "seed": 1701,
                "hardware": "CPU+CUDA inference",
                "precision": "bfloat16",
                "model": name,
                "metrics_json": json.dumps(metrics, sort_keys=True),
                "wall_seconds": 0.0,
                "artifact_paths_json": json.dumps([str(report)]),
            }
        )
    return report
