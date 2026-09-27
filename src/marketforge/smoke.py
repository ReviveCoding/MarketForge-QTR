from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import data_root
from .probe import atomic_json
from .results import append_result, dataset_fingerprint
from .strategy import Prediction, quote_controller, risk_gate, smoke_accounting
from .training import WindowDataset, train_tiny


def run_smoke() -> object:
    model, training = train_tiny(max_rows=4096, epochs=2)
    validation_frame = pd.read_parquet(
        data_root() / "splits" / "development" / "model_validation.parquet"
    ).iloc[:512]
    loader = DataLoader(WindowDataset(validation_frame), batch_size=64, num_workers=0)
    device = torch.device("cuda:0")
    inventory = cash = pnl = 0.0
    allowed = killed = fills = 0
    latencies: list[float] = []
    row_offset = 31
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            started = time.perf_counter_ns()
            action, side, event, state, book = [item.to(device) for item in batch[:5]]
            output = model(action, side, event, state, book)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            probabilities = output["direction_logits"].softmax(-1)
            for index in range(len(action)):
                source = validation_frame.iloc[row_offset]
                prediction = Prediction(
                    float(output["expected_return"][index]),
                    float(output["expected_markout"][index]),
                    float(output["volatility"][index]),
                    float(torch.sigmoid(output["fill_logit"][index])),
                    float(torch.sigmoid(output["adverse_logit"][index])),
                    float(1 - probabilities[index].max()),
                    True,
                )
                quote = quote_controller(float(source["mid"]), inventory, prediction)
                approved, _reason = risk_gate(
                    quote,
                    prediction,
                    inventory,
                    elapsed_ms / len(action),
                    pnl,
                    float(source["mid"]),
                )
                latencies.append(elapsed_ms / len(action))
                if approved:
                    allowed += 1
                    old_inventory = inventory
                    inventory, cash, nav = smoke_accounting(
                        float(source["mid"]), float(source["future_mid_10"]), quote, inventory, cash
                    )
                    fills += int(inventory != old_inventory)
                    pnl = nav
                else:
                    killed += 1
                row_offset += 1
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "SUCCEEDED",
        "training": training,
        "predictions": allowed + killed,
        "risk_allowed": allowed,
        "risk_killed": killed,
        "simulated_fill_events": fills,
        "ending_inventory": inventory,
        "ending_cash": cash,
        "ending_nav": pnl,
        "latency_ms_p50": float(pd.Series(latencies).quantile(0.5)),
        "latency_ms_p95": float(pd.Series(latencies).quantile(0.95)),
        "execution_scope": "LOGICAL_SMOKE_ONLY_NOT_HEADLINE_ECONOMICS",
    }
    target = data_root() / "manifests" / "smoke_v1.json"
    atomic_json(target, report)
    append_result(
        {
            "stage": "smoke",
            "status": "SUCCEEDED",
            "dataset_fingerprint": dataset_fingerprint(),
            "seed": 1701,
            "hardware": "RTX 4090 Laptop GPU",
            "precision": "bfloat16",
            "model": "M3T-Tiny",
            "metrics_json": json.dumps(report, sort_keys=True),
            "wall_seconds": training["wall_seconds"],
            "artifact_paths_json": json.dumps([str(target), training["checkpoint"]]),
        }
    )
    return target
