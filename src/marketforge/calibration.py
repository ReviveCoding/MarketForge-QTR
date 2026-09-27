from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

from .data import data_root
from .model import MarketForgeM3T
from .probe import atomic_json
from .training import GLOBAL_STATE, WindowDataset, gpu_lease


def _ece(probabilities: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in pairwise(edges):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            result += selected.mean() * abs(
                (prediction[selected] == target[selected]).mean() - confidence[selected].mean()
            )
    return float(result)


def calibrate_checkpoint(checkpoint_path: str) -> object:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA model calibration requires CUDA inference")
    device = torch.device("cuda:0")
    model = MarketForgeM3T(len(GLOBAL_STATE), d_model=64, layers=2, heads=4).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)["model"]
    )
    validation = pd.read_parquet(
        data_root() / "splits" / "development" / "model_validation.parquet"
    )
    loader = DataLoader(WindowDataset(validation), batch_size=256, num_workers=0, pin_memory=True)
    logits_all, target_all, fill_logits_all, fill_all, vol_pred_all, vol_all = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with gpu_lease(), torch.inference_mode():
        model.eval()
        for batch in loader:
            output = model(*[item.to(device, non_blocking=True) for item in batch[:5]])
            logits_all.append(output["direction_logits"].float().cpu())
            target_all.append(batch[5])
            fill_logits_all.append(output["fill_logit"].float().cpu())
            fill_all.append(batch[9])
            vol_pred_all.append(output["volatility"].float().cpu())
            vol_all.append(batch[8])
    logits = torch.cat(logits_all).to(device)
    target = torch.cat(target_all).to(device)
    log_temperature = torch.nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp(), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.exp().detach().cpu())
    uncalibrated = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    calibrated = torch.softmax(logits / temperature, dim=-1).detach().cpu().numpy()
    targets = target.cpu().numpy()
    fill_probability = torch.sigmoid(torch.cat(fill_logits_all)).numpy()
    fill_target = torch.cat(fill_all).numpy()
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(fill_probability, fill_target)
    fill_calibrated = isotonic.predict(fill_probability)
    volatility_prediction = torch.cat(vol_pred_all).numpy()
    volatility_target = torch.cat(vol_all).numpy()
    positive = volatility_prediction > 1e-12
    volatility_scale = (
        float(np.median(volatility_target[positive] / volatility_prediction[positive]))
        if positive.any()
        else 1.0
    )
    report = {
        "checkpoint": checkpoint_path,
        "partition": "MODEL_VALIDATION",
        "temperature": temperature,
        "direction_nll_before": float(log_loss(targets, uncalibrated, labels=[0, 1, 2])),
        "direction_nll_after": float(log_loss(targets, calibrated, labels=[0, 1, 2])),
        "direction_ece_before": _ece(uncalibrated, targets),
        "direction_ece_after": _ece(calibrated, targets),
        "fill_brier_before": float(np.mean((fill_probability - fill_target) ** 2)),
        "fill_brier_after": float(np.mean((fill_calibrated - fill_target) ** 2)),
        "fill_isotonic_x": isotonic.X_thresholds_.tolist(),
        "fill_isotonic_y": isotonic.y_thresholds_.tolist(),
        "volatility_median_ratio_scale": volatility_scale,
        "test_used": False,
        "final_lockbox_used": False,
    }
    target_path = data_root() / "manifests" / "calibration_m3t_ept_v1.json"
    atomic_json(target_path, report)
    return target_path
