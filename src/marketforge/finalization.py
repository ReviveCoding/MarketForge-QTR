from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader

from .advanced import ROOT, _matrix
from .data import data_root, sha256_file
from .economics import _neural_predictions, _simulate
from .model import MarketForgeM3T
from .probe import atomic_json
from .results import dataset_fingerprint
from .state import content_hash
from .strategy import Prediction
from .training import GLOBAL_STATE, WindowDataset, gpu_lease

FREEZE = ROOT / "artifacts" / "final_freeze_manifest.json"
FINAL = ROOT / "artifacts" / "final_evaluation_v1.json"


def _working_tree_fingerprint() -> str:
    files = [ROOT / "pyproject.toml", ROOT / "MASTER_PROMPT.md"]
    for directory in (ROOT / "src", ROOT / "tests", ROOT / "scripts"):
        files.extend(path for path in directory.rglob("*") if path.is_file())
    return content_hash({str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(files)})


def freeze_final_protocol() -> Path:
    if FINAL.exists():
        raise RuntimeError("final evaluation already exists; protocol cannot be refrozen")
    if FREEZE.exists():
        existing = json.loads(FREEZE.read_text(encoding="utf-8"))
        seal = existing.pop("manifest_content_hash")
        if seal != content_hash(existing):
            raise RuntimeError("existing final freeze manifest seal is invalid")
        return FREEZE
    checkpoints = {
        "xgboost_gpu": ROOT / "artifacts" / "checkpoints" / "xgboost_gpu_v1.json",
        "m3t_sup": ROOT / "artifacts" / "checkpoints" / "m3t-sup-weighted_seed1701.pt",
        "m3t_ssl": ROOT / "artifacts" / "checkpoints" / "m3t-ssl_seed1701.pt",
        "m3t_ept": ROOT / "artifacts" / "checkpoints" / "m3t-ept_seed1701.pt",
    }
    for path in checkpoints.values():
        if not path.exists():
            raise FileNotFoundError(path)
    lockbox = data_root() / "final_lockbox" / "final_lockbox.parquet"
    split_manifest = json.loads(
        (data_root() / "manifests" / "splits_databento_mbo_v3.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        (data_root() / "manifests" / "calibration_m3t_ept_v1.json").read_text(encoding="utf-8")
    )
    payload: dict[str, object] = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "working_tree_fingerprint": _working_tree_fingerprint(),
        "git_sha": "UNAVAILABLE: repository has no commit",
        "dataset_fingerprint": dataset_fingerprint(),
        "feature_sha256": split_manifest["feature_sha256"],
        "final_lockbox": {
            "path": str(lockbox),
            "rows_expected": split_manifest["split_counts"]["FINAL_LOCKBOX"],
            "bounds": split_manifest["split_bounds"]["FINAL_LOCKBOX"],
            "bytes_metadata_only": lockbox.stat().st_size,
            "content_not_opened_at_freeze": True,
        },
        "model_config": {
            "m3t": {"d_model": 64, "layers": 2, "heads": 4, "fusion": "gated"},
            "xgboost": {
                "version": xgb.__version__,
                "trees": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "device": "cuda",
            },
        },
        "training_config": {
            "seed": 1701,
            "m3t_precision": "bfloat16",
            "windows_dataloader_workers": 0,
            "xgboost_train_rows": 20000,
        },
        "checkpoint_sha256": {name: sha256_file(path) for name, path in checkpoints.items()},
        "calibration": calibration,
        "controller_parameters": {"common_base_spread": 0.75, "common_inventory_penalty": 0.1},
        "strategy_parameters": {
            "fee_per_fill": 1.25,
            "liquidation_slippage": 0.25,
            "inventory_limit": 10,
            "latency_ms": 1.0,
        },
        "primary_experiments": [
            "predictive majority/XGBoost/M3T-SUP/M3T-SSL/M3T-EPT-CAL",
            "common-controller no-trade/GLFT/XGBoost/M3T-EPT-CAL",
        ],
        "claim_scope": "COUNTERFACTUAL_CROSSING; NO_HEADLINE_PNL; one instrument-day",
        "post_lockbox_tuning_prohibited": True,
    }
    payload["manifest_content_hash"] = content_hash(payload)
    atomic_json(FREEZE, payload)
    return FREEZE


def _m3t_classes(frame: pd.DataFrame, checkpoint: Path, temperature: float = 1.0) -> np.ndarray:
    device = torch.device("cuda:0")
    model = MarketForgeM3T(len(GLOBAL_STATE), d_model=64, layers=2, heads=4).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    loader = DataLoader(WindowDataset(frame), batch_size=256, num_workers=0, pin_memory=True)
    predictions = []
    with gpu_lease(), torch.inference_mode():
        model.eval()
        for batch in loader:
            output = model(*[item.to(device, non_blocking=True) for item in batch[:5]])
            predictions.extend(
                (output["direction_logits"] / temperature).argmax(-1).cpu().numpy() - 1
            )
    return np.asarray(predictions)


def _classification(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "mcc": float(matthews_corrcoef(target, prediction)),
    }


def run_final_evaluation() -> Path:
    if FINAL.exists():
        return FINAL
    if not FREEZE.exists():
        raise RuntimeError("missing final freeze manifest")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    seal = frozen.pop("manifest_content_hash")
    if seal != content_hash(frozen):
        raise RuntimeError("final freeze manifest seal mismatch")
    if frozen["working_tree_fingerprint"] != _working_tree_fingerprint():
        raise RuntimeError("working tree changed after freeze; final evaluation refused")
    lockbox = pd.read_parquet(frozen["final_lockbox"]["path"]).reset_index(drop=True)
    if len(lockbox) != frozen["final_lockbox"]["rows_expected"]:
        raise RuntimeError("final lockbox row count mismatch")
    target = lockbox["direction_10"].to_numpy()[31:]
    train = pd.read_parquet(data_root() / "splits" / "development" / "train.parquet")
    majority = np.full_like(target, int(train["direction_10"].mode().iloc[0]))
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(ROOT / "artifacts" / "checkpoints" / "xgboost_gpu_v1.json")
    xgb_prediction_full = xgb_model.predict(_matrix(lockbox)).astype(np.int64) - 1
    checkpoints = ROOT / "artifacts" / "checkpoints"
    calibration = frozen["calibration"]
    supervised = _m3t_classes(lockbox, checkpoints / "m3t-sup-weighted_seed1701.pt")
    ssl = _m3t_classes(lockbox, checkpoints / "m3t-ssl_seed1701.pt")
    ept_cal = _m3t_classes(lockbox, checkpoints / "m3t-ept_seed1701.pt", calibration["temperature"])
    predictive = {
        "MAJORITY": _classification(target, majority),
        "XGBOOST_GPU": _classification(target, xgb_prediction_full[31:]),
        "M3T_SUP": _classification(target, supervised),
        "M3T_SSL": _classification(target, ssl),
        "M3T_EPT_CAL": _classification(target, ept_cal),
    }
    probabilities = xgb_model.predict_proba(_matrix(lockbox))
    return_scale = float(np.median(np.abs(train["future_return_10"].to_numpy())))
    xgb_economic = [
        Prediction(
            float((row[2] - row[0]) * return_scale),
            0.0,
            float(max(lockbox.iloc[i]["ewma_vol"], 0)),
            0.5,
            float(1 - row.max()),
            float(1 - row.max()),
            True,
        )
        for i, row in enumerate(probabilities)
    ]
    ept_economic = _neural_predictions(
        lockbox, calibration["checkpoint"], calibration["temperature"], calibration
    )
    neutral = [Prediction(0, 0, 0, 0.5, 0.5, 0, True)] * len(lockbox)
    spread = frozen["controller_parameters"]["common_base_spread"]
    penalty = frozen["controller_parameters"]["common_inventory_penalty"]
    economics = {
        "NO_TRADE": _simulate(lockbox, neutral, spread, penalty, "no_trade"),
        "GLFT": _simulate(lockbox, neutral, spread, penalty, "glft"),
        "XGBOOST_GPU": _simulate(lockbox, xgb_economic, spread, penalty, "learned"),
        "M3T_EPT_CAL": _simulate(lockbox, ept_economic, spread, penalty, "learned"),
    }
    payload = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_count": 1,
        "freeze_manifest_content_hash": seal,
        "lockbox_rows": len(lockbox),
        "predictive": predictive,
        "economics": economics,
        "return_scale_train_only": return_scale,
        "claim_scope": frozen["claim_scope"],
        "complexity_justified": economics["M3T_EPT_CAL"]["net_pnl"]
        > economics["XGBOOST_GPU"]["net_pnl"],
        "post_evaluation_tuning_performed": False,
    }
    atomic_json(FINAL, payload)
    return FINAL
