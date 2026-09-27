from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import data_root, sha256_file
from .grouped_windows import GROUP_COLUMNS
from .model import SSLHeads
from .model_v2 import MarketForgeM3TV2
from .probe import atomic_json
from .state import ROOT, content_hash
from .training import BOOK_COLUMNS, EVENT_CONT, GLOBAL_STATE, gpu_lease

SEQUENCE_LENGTH = 32
SEED = 1701


@dataclass(frozen=True, slots=True)
class TargetScale:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


class V2WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Spawn-safe v2 dataset with explicit market, modality, and target masks."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_scaler: dict[str, dict[str, float]],
        target_scaler: dict[str, TargetScale],
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.action = self.frame["event_action_id"].to_numpy(np.int64)
        self.side = self.frame["event_side_id"].to_numpy(np.int64)
        self.event = self._scaled(EVENT_CONT, feature_scaler)
        self.state = self._scaled(GLOBAL_STATE, feature_scaler)
        self.book = self._scaled(BOOK_COLUMNS, feature_scaler).reshape(-1, 10, 6)
        self.source = self.frame["source_id"].to_numpy(np.int64)
        self.instrument = self.frame["instrument_embedding_id"].to_numpy(np.int64)
        self.modality = self.frame[["modality_l1", "modality_l2", "modality_l3"]].to_numpy(
            np.float32
        )
        self.direction = self.frame["direction_1000ms"].to_numpy(np.int64) + 1
        self.targets = {}
        for name in (
            "return_1000ms",
            "bid_quote_markout_1000ms",
            "ask_quote_markout_1000ms",
            "future_realized_vol_1000ms",
        ):
            values = self.frame[name].to_numpy(np.float32)
            if "markout" in name:
                values = values / self.frame["tick_size"].to_numpy(np.float32)
            self.targets[name] = target_scaler[name].transform(values).astype(np.float32)
        self.targets.update(
            {
                name: self.frame[name].fillna(0).to_numpy(np.float32)
                for name in (
                    "bid_fill",
                    "ask_fill",
                    "bid_fill_conditional_adverse",
                    "ask_fill_conditional_adverse",
                    "bid_adverse_available",
                    "ask_adverse_available",
                    "l3_targets_available",
                )
            }
        )
        self.sample_weight = self.frame["source_balance_weight"].to_numpy(np.float64)
        self.valid_ends: list[int] = []
        boundary_columns = GROUP_COLUMNS + ["split"]
        for positions in self.frame.groupby(
            boundary_columns, sort=False, dropna=False
        ).indices.values():
            ordered = np.sort(positions)
            breaks = np.flatnonzero(np.diff(ordered) != 1)
            for segment in np.split(ordered, breaks + 1):
                self.valid_ends.extend(int(index) for index in segment[sequence_length - 1 :])

    def _scaled(self, columns: list[str], scaler: dict[str, dict[str, float]]) -> np.ndarray:
        values = self.frame[columns].to_numpy(np.float32)
        means = np.asarray([scaler[column]["mean"] for column in columns], dtype=np.float32)
        stds = np.asarray([scaler[column]["std"] for column in columns], dtype=np.float32)
        return np.nan_to_num((values - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self) -> int:
        return len(self.valid_ends)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target = self.valid_ends[index]
        start = target - self.sequence_length + 1
        return {
            "action": torch.from_numpy(self.action[start : target + 1]),
            "side": torch.from_numpy(self.side[start : target + 1]),
            "event": torch.from_numpy(self.event[start : target + 1]),
            "state": torch.from_numpy(self.state[start : target + 1]),
            "book": torch.from_numpy(self.book[start : target + 1]),
            "source": torch.tensor(self.source[target]),
            "instrument": torch.tensor(self.instrument[target]),
            "modality": torch.from_numpy(self.modality[target]),
            "direction": torch.tensor(self.direction[target]),
            **{name: torch.tensor(values[target]) for name, values in self.targets.items()},
        }

    def sampler_weights(self) -> torch.Tensor:
        return torch.as_tensor(
            [self.sample_weight[target] for target in self.valid_ends], dtype=torch.double
        )


def _load_scalers() -> tuple[dict[str, dict[str, float]], dict[str, TargetScale]]:
    feature_payload = json.loads(
        (data_root() / "manifests/train_scaler_v4_multi_market.json").read_text(encoding="utf-8")
    )
    train = pd.read_parquet(data_root() / "splits_v4_multi_market/train.parquet")
    names = (
        "return_1000ms",
        "bid_quote_markout_1000ms",
        "ask_quote_markout_1000ms",
        "future_realized_vol_1000ms",
    )
    target_scaler = {}
    for name in names:
        values = train[name].to_numpy(np.float64)
        if "markout" in name:
            values = values / train["tick_size"].to_numpy(np.float64)
        target_scaler[name] = TargetScale(
            0.0 if name == "future_realized_vol_1000ms" else float(np.nanmean(values)),
            float(max(np.nanstd(values), 1e-8)),
        )
    return feature_payload["columns"], target_scaler


def _datasets() -> tuple[dict[str, V2WindowDataset], dict[str, TargetScale]]:
    feature_scaler, target_scaler = _load_scalers()
    root = data_root() / "splits_v4_multi_market"
    datasets = {
        split: V2WindowDataset(
            pd.read_parquet(root / f"{split.lower()}.parquet"),
            feature_scaler,
            target_scaler,
        )
        for split in ("TRAIN", "MODEL_VALIDATION", "STRATEGY_VALIDATION")
    }
    return datasets, target_scaler


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for v2 M3T; CPU fallback forbidden")
    device = torch.device("cuda:0")
    probe = torch.randn(32, 32, device=device, requires_grad=True)
    (probe.square().mean()).backward()
    if probe.grad is None or not torch.isfinite(probe.grad).all():
        raise RuntimeError("CUDA forward/backward probe failed")
    return device


def _model() -> MarketForgeM3TV2:
    return MarketForgeM3TV2(
        len(GLOBAL_STATE), d_model=48, layers=2, heads=4, num_sources=2, num_instruments=2
    )


def _forward(model: MarketForgeM3TV2, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        batch["action"],
        batch["side"],
        batch["event"],
        batch["state"],
        batch["book"],
        batch["source"],
        batch["instrument"],
        batch["modality"],
    )


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _predict(
    model: MarketForgeM3TV2, dataset: V2WindowDataset, device: torch.device
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    output: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.inference_mode():
        for batch_cpu in loader:
            batch = _to_device(batch_cpu, device)
            predictions = _forward(model, batch)
            for name, values in predictions.items():
                output.setdefault(name, []).append(values.float().cpu().numpy())
            for name in (
                "direction",
                "return_1000ms",
                "bid_quote_markout_1000ms",
                "ask_quote_markout_1000ms",
                "future_realized_vol_1000ms",
                "bid_fill",
                "ask_fill",
                "bid_fill_conditional_adverse",
                "ask_fill_conditional_adverse",
                "bid_adverse_available",
                "ask_adverse_available",
                "l3_targets_available",
            ):
                output.setdefault(f"target_{name}", []).append(batch_cpu[name].numpy())
    return {name: np.concatenate(parts) for name, parts in output.items()}


def _metrics(prediction: dict[str, np.ndarray]) -> dict[str, float | None]:
    direction = prediction["direction_logits"].argmax(axis=1)
    target = prediction["target_direction"]
    result: dict[str, float | None] = {
        "direction_macro_f1": float(f1_score(target, direction, average="macro")),
        "return_mae_scaled": float(
            mean_absolute_error(prediction["target_return_1000ms"], prediction["return_mean"])
        ),
        "bid_markout_mae_scaled": float(
            mean_absolute_error(
                prediction["target_bid_quote_markout_1000ms"],
                prediction["bid_quote_markout"],
            )
        ),
        "ask_markout_mae_scaled": float(
            mean_absolute_error(
                prediction["target_ask_quote_markout_1000ms"],
                prediction["ask_quote_markout"],
            )
        ),
        "volatility_mae_scaled": float(
            mean_absolute_error(
                prediction["target_future_realized_vol_1000ms"], prediction["volatility"]
            )
        ),
    }
    for side in ("bid", "ask"):
        l3_available = prediction["target_l3_targets_available"].astype(bool)
        probability = 1 / (1 + np.exp(-prediction[f"{side}_fill_logit"]))
        fill_target = prediction[f"target_{side}_fill"][l3_available]
        result[f"{side}_fill_brier"] = float(
            brier_score_loss(fill_target, probability[l3_available])
        )
        available = prediction[f"target_{side}_adverse_available"].astype(bool)
        result[f"{side}_adverse_brier"] = (
            float(
                brier_score_loss(
                    prediction[f"target_{side}_fill_conditional_adverse"][available],
                    1 / (1 + np.exp(-prediction[f"{side}_adverse_logit"][available])),
                )
            )
            if available.any()
            and len(np.unique(prediction[f"target_{side}_fill_conditional_adverse"][available])) > 1
            else None
        )
    return result


def _save_checkpoint(
    name: str,
    model: MarketForgeM3TV2,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    parent: Path | None,
    split_fingerprint: str,
) -> Path:
    path = ROOT / "artifacts/checkpoints_v2" / f"{name.lower()}_seed{SEED}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": 2,
            "stage": name,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "seed": SEED,
            "split_fingerprint": split_fingerprint,
            "parent_checkpoint": str(parent) if parent else None,
            "parent_sha256": sha256_file(parent) if parent else None,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)
    return path


def _predictive_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    direction = nn.functional.cross_entropy(output["direction_logits"], batch["direction"])
    return_loss = nn.functional.gaussian_nll_loss(
        output["return_mean"],
        batch["return_1000ms"],
        output["return_log_scale"].mul(2).exp().clamp_min(1e-6),
    )
    volatility = nn.functional.smooth_l1_loss(
        output["volatility"], batch["future_realized_vol_1000ms"]
    )
    return direction + 0.5 * return_loss + 0.25 * volatility


def _ept_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = _predictive_loss(output, batch)
    loss = loss + 0.5 * nn.functional.smooth_l1_loss(
        output["bid_quote_markout"], batch["bid_quote_markout_1000ms"]
    )
    loss = loss + 0.5 * nn.functional.smooth_l1_loss(
        output["ask_quote_markout"], batch["ask_quote_markout_1000ms"]
    )
    availability = batch["l3_targets_available"]
    for side in ("bid", "ask"):
        fill = nn.functional.binary_cross_entropy_with_logits(
            output[f"{side}_fill_logit"], batch[f"{side}_fill"], reduction="none"
        )
        loss = loss + 0.25 * (fill * availability).sum() / availability.sum().clamp_min(1)
        adverse_mask = batch[f"{side}_adverse_available"]
        adverse = nn.functional.binary_cross_entropy_with_logits(
            output[f"{side}_adverse_logit"],
            batch[f"{side}_fill_conditional_adverse"],
            reduction="none",
        )
        loss = loss + 0.25 * (adverse * adverse_mask).sum() / adverse_mask.sum().clamp_min(1)
    return loss


def _train_stage(
    name: str,
    dataset: V2WindowDataset,
    validation: V2WindowDataset,
    device: torch.device,
    split_fingerprint: str,
    parent: Path | None = None,
    ept: bool = False,
    epochs: int = 5,
) -> tuple[Path, dict[str, object]]:
    model = _model().to(device)
    if parent is not None:
        payload = torch.load(parent, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    sampler = WeightedRandomSampler(dataset.sampler_weights(), len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=0, pin_memory=True)
    started = time.perf_counter()
    losses: list[float] = []
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_cpu in loader:
            batch = _to_device(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = _forward(model, batch)
                loss = _ept_loss(output, batch) if ept else _predictive_loss(output, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    checkpoint = _save_checkpoint(name, model, optimizer, epochs, parent, split_fingerprint)
    prediction = _predict(model, validation, device)
    metrics: dict[str, object] = {
        "stage": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parent_checkpoint": str(parent) if parent else None,
        "parent_sha256": sha256_file(parent) if parent else None,
        "epochs": epochs,
        "final_train_loss": losses[-1],
        "wall_seconds": time.perf_counter() - started,
        "device": str(next(model.parameters()).device),
        "precision": "bfloat16",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation": _metrics(prediction),
    }
    return checkpoint, metrics


def _ssl_pretrain(
    dataset: V2WindowDataset, device: torch.device, split_fingerprint: str, epochs: int = 3
) -> tuple[Path, dict[str, object]]:
    model = _model().to(device)
    heads = SSLHeads(48, len(GLOBAL_STATE)).to(device)
    optimizer = torch.optim.AdamW([*model.parameters(), *heads.parameters()], lr=2e-4)
    sampler = WeightedRandomSampler(dataset.sampler_weights(), len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=0, pin_memory=True)
    started = time.perf_counter()
    final_loss = math.nan
    for _epoch in range(epochs):
        model.train()
        for batch_cpu in loader:
            batch = _to_device(batch_cpu, device)
            mask = torch.rand(batch["action"].shape, device=device) < 0.15
            masked = batch["action"].clone()
            masked[mask] = 7
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.encode(
                    masked,
                    batch["side"],
                    batch["event"],
                    batch["state"],
                    batch["book"],
                    batch["source"],
                    batch["instrument"],
                    batch["modality"],
                )
                mask_loss = nn.functional.cross_entropy(
                    heads.masked_action(hidden[mask]), batch["action"][mask]
                )
                next_loss = nn.functional.cross_entropy(
                    heads.next_action(hidden[:, :-1]).flatten(0, 1),
                    batch["action"][:, 1:].flatten(),
                )
                state_loss = nn.functional.smooth_l1_loss(
                    heads.future_state(hidden[:, :-1]), batch["state"][:, 1:]
                )
                loss = mask_loss + next_loss + state_loss
            loss.backward()
            nn.utils.clip_grad_norm_([*model.parameters(), *heads.parameters()], 1.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    torch.cuda.synchronize()
    path = ROOT / "artifacts/checkpoints_v2/m3t-ssl-pretrain_seed1701.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": 2,
            "stage": "M3T-SSL-PRETRAIN",
            "model": model.state_dict(),
            "ssl_heads": heads.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs,
            "seed": SEED,
            "split_fingerprint": split_fingerprint,
            "final_lockbox_v2_used": False,
        },
        temporary,
    )
    os.replace(temporary, path)
    return path, {
        "stage": "M3T-SSL-PRETRAIN",
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "epochs": epochs,
        "final_train_loss": final_loss,
        "wall_seconds": time.perf_counter() - started,
        "device": str(next(model.parameters()).device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _temperature(logits: np.ndarray, target: np.ndarray) -> float:
    candidates = np.geomspace(0.25, 4.0, 81)
    losses = []
    for temperature in candidates:
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        probability = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        losses.append(log_loss(target, probability, labels=[0, 1, 2]))
    return float(candidates[int(np.argmin(losses))])


def _calibrate_branch(
    name: str,
    checkpoint: Path,
    validation: V2WindowDataset,
    strategy: V2WindowDataset,
    device: torch.device,
) -> dict[str, object]:
    model = _model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    valid = _predict(model, validation, device)
    held = _predict(model, strategy, device)
    temperature = _temperature(valid["direction_logits"], valid["target_direction"])
    scaled = held["direction_logits"] / temperature
    probability = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    calibration: dict[str, object] = {
        "direction_temperature": temperature,
        "strategy_direction_nll": float(
            log_loss(held["target_direction"], probability, labels=[0, 1, 2])
        ),
    }
    for side in ("bid", "ask"):
        valid_l3 = valid["target_l3_targets_available"].astype(bool)
        held_l3 = held["target_l3_targets_available"].astype(bool)
        target = valid[f"target_{side}_fill"][valid_l3]
        if len(np.unique(target)) > 1:
            calibrator = LogisticRegression(random_state=SEED).fit(
                valid[f"{side}_fill_logit"][valid_l3].reshape(-1, 1), target
            )
            held_probability = calibrator.predict_proba(
                held[f"{side}_fill_logit"][held_l3].reshape(-1, 1)
            )[:, 1]
            calibration[f"{side}_fill_platt"] = {
                "slope": float(calibrator.coef_[0, 0]),
                "intercept": float(calibrator.intercept_[0]),
                "strategy_brier": float(
                    brier_score_loss(held[f"target_{side}_fill"][held_l3], held_probability)
                ),
            }
        for target_name, output_name in (
            (f"{side}_quote_markout_1000ms", f"{side}_quote_markout"),
        ):
            slope, intercept = np.polyfit(valid[output_name], valid[f"target_{target_name}"], 1)
            calibrated = slope * held[output_name] + intercept
            calibration[f"{side}_markout_affine"] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "strategy_mae_scaled": float(
                    mean_absolute_error(held[f"target_{target_name}"], calibrated)
                ),
            }
    output_path = data_root() / "manifests" / f"{name.lower()}_v2.json"
    atomic_json(
        output_path,
        {
            "schema_version": 2,
            "stage": name,
            "parent_checkpoint": str(checkpoint),
            "parent_sha256": sha256_file(checkpoint),
            "fit_split": "MODEL_VALIDATION",
            "evaluation_split": "STRATEGY_VALIDATION",
            "calibration": calibration,
            "test_used": False,
            "final_lockbox_v2_used": False,
        },
    )
    return {"stage": name, "artifact": str(output_path), "calibration": calibration}


def run_factorial_ladder_v2() -> Path:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = _cuda_device()
    split_manifest = json.loads(
        (data_root() / "manifests/splits_v4_multi_market.json").read_text(encoding="utf-8")
    )
    split_fingerprint = split_manifest["split_fingerprint"]
    datasets, target_scaler = _datasets()
    with gpu_lease():
        torch.cuda.reset_peak_memory_stats(device)
        sup_checkpoint, sup = _train_stage(
            "M3T-SUP",
            datasets["TRAIN"],
            datasets["MODEL_VALIDATION"],
            device,
            split_fingerprint,
        )
        ssl_pretrain_checkpoint, ssl_pretrain = _ssl_pretrain(
            datasets["TRAIN"], device, split_fingerprint
        )
        ssl_checkpoint, ssl = _train_stage(
            "M3T-SSL",
            datasets["TRAIN"],
            datasets["MODEL_VALIDATION"],
            device,
            split_fingerprint,
            parent=ssl_pretrain_checkpoint,
        )
        sup_ept_checkpoint, sup_ept = _train_stage(
            "M3T-SUP-EPT",
            datasets["TRAIN"],
            datasets["MODEL_VALIDATION"],
            device,
            split_fingerprint,
            parent=sup_checkpoint,
            ept=True,
        )
        ssl_ept_checkpoint, ssl_ept = _train_stage(
            "M3T-SSL-EPT",
            datasets["TRAIN"],
            datasets["MODEL_VALIDATION"],
            device,
            split_fingerprint,
            parent=ssl_checkpoint,
            ept=True,
        )
        sup_cal = _calibrate_branch(
            "M3T-SUP-EPT-CAL",
            sup_ept_checkpoint,
            datasets["MODEL_VALIDATION"],
            datasets["STRATEGY_VALIDATION"],
            device,
        )
        ssl_cal = _calibrate_branch(
            "M3T-SSL-EPT-CAL",
            ssl_ept_checkpoint,
            datasets["MODEL_VALIDATION"],
            datasets["STRATEGY_VALIDATION"],
            device,
        )
        peak_vram = torch.cuda.max_memory_allocated(device)
    output = data_root() / "manifests/model_ladder_v2.json"
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "split_fingerprint": split_fingerprint,
        "architecture": {
            "d_model": 48,
            "layers": 2,
            "heads": 4,
            "sequence_length": SEQUENCE_LENGTH,
            "identical_scale_across_branches": True,
            "side_specific_markout_heads": True,
            "source_and_instrument_embeddings": True,
            "modality_masks": True,
        },
        "target_scaler": {
            name: {"mean": scale.mean, "std": scale.std} for name, scale in target_scaler.items()
        },
        "branches": {
            "M3T-SUP": sup,
            "M3T-SSL-PRETRAIN": ssl_pretrain,
            "M3T-SSL": ssl,
            "M3T-SUP-EPT": sup_ept,
            "M3T-SSL-EPT": ssl_ept,
            "M3T-SUP-EPT-CAL": sup_cal,
            "M3T-SSL-EPT-CAL": ssl_cal,
        },
        "ancestry_checks": {
            "SUP_EPT_descends_from_SUP": sup_ept["parent_sha256"] == sup["checkpoint_sha256"],
            "SSL_descends_from_SSL_pretrain": ssl["parent_sha256"]
            == ssl_pretrain["checkpoint_sha256"],
            "SSL_EPT_descends_from_SSL": ssl_ept["parent_sha256"] == ssl["checkpoint_sha256"],
        },
        "selection_splits": ["TRAIN", "MODEL_VALIDATION", "STRATEGY_VALIDATION"],
        "test_used": False,
        "final_lockbox_v2_used": False,
        "device": torch.cuda.get_device_name(0),
        "peak_vram_bytes": peak_vram,
    }
    payload["config_hash"] = content_hash(payload["architecture"])
    if not all(payload["ancestry_checks"].values()):
        raise RuntimeError("v2 checkpoint ancestry check failed")
    atomic_json(output, payload)
    return output
