from __future__ import annotations

import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import data_root
from .model import MarketForgeM3T, SSLHeads
from .state import ROOT, content_hash

GLOBAL_STATE = [
    "spread",
    "relative_spread",
    "microprice_minus_mid",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
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
]
BOOK_COLUMNS = [
    f"{side}_{kind}_{level:02d}"
    for level in range(1, 11)
    for side in ("bid", "ask")
    for kind in ("px", "sz", "ct")
]
EVENT_CONT = ["event_relative_price", "event_size_log", "delta_time_log"]


class WindowDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self, frame: pd.DataFrame, sequence_length: int = 32, limit: int | None = None
    ) -> None:
        frame = frame.iloc[:limit] if limit else frame
        self.action = frame["event_action_id"].to_numpy(np.int64)
        self.side = frame["event_side_id"].to_numpy(np.int64)
        self.event = np.nan_to_num(frame[EVENT_CONT].to_numpy(np.float32))
        self.state = np.nan_to_num(frame[GLOBAL_STATE].to_numpy(np.float32))
        self.book = np.nan_to_num(frame[BOOK_COLUMNS].to_numpy(np.float32)).reshape(-1, 10, 6)
        self.direction = frame["direction_10"].to_numpy(np.int64) + 1
        self.future_return = frame["future_return_10"].to_numpy(np.float32)
        self.markout = (
            (frame["counterfactual_bid_markout_10"] + frame["counterfactual_ask_markout_10"]) / 2
        ).to_numpy(np.float32)
        bid_markout = frame["counterfactual_bid_markout_10"].to_numpy(np.float32)
        ask_markout = frame["counterfactual_ask_markout_10"].to_numpy(np.float32)
        self.fill = ((bid_markout <= 0) | (ask_markout <= 0)).astype(np.float32)
        self.adverse = ((bid_markout < 0) | (ask_markout < 0)).astype(np.float32)
        self.volatility = frame["future_volatility_10"].to_numpy(np.float32)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return max(0, len(self.action) - self.sequence_length + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        end = index + self.sequence_length
        target = end - 1
        return (
            torch.from_numpy(self.action[index:end]),
            torch.from_numpy(self.side[index:end]),
            torch.from_numpy(self.event[index:end]),
            torch.from_numpy(self.state[index:end]),
            torch.from_numpy(self.book[index:end]),
            torch.tensor(self.direction[target]),
            torch.tensor(self.future_return[target]),
            torch.tensor(self.markout[target]),
            torch.tensor(self.volatility[target]),
            torch.tensor(self.fill[target]),
            torch.tensor(self.adverse[target]),
        )


@contextmanager
def gpu_lease() -> object:
    lock = ROOT / "artifacts" / "locks" / "gpu0.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def train_tiny(max_rows: int = 4096, epochs: int = 2) -> tuple[MarketForgeM3T, dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for M3T; CPU fallback forbidden")
    torch.manual_seed(1701)
    split = data_root() / "splits" / "development"
    train_ds = WindowDataset(pd.read_parquet(split / "train.parquet"), limit=max_rows)
    valid_ds = WindowDataset(pd.read_parquet(split / "model_validation.parquet"), limit=2048)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=128, num_workers=0, pin_memory=True)
    device = torch.device("cuda:0")
    model = MarketForgeM3T(len(GLOBAL_STATE), d_model=48, layers=2, heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    classification = nn.CrossEntropyLoss()
    regression = nn.SmoothL1Loss()
    started = time.perf_counter()
    examples = 0
    losses: list[float] = []
    with gpu_lease():
        torch.cuda.reset_peak_memory_stats(device)
        for _epoch in range(epochs):
            model.train()
            for batch in train_loader:
                action, side, event, state, book, direction, future_return, markout, volatility = [
                    item.to(device, non_blocking=True) for item in batch[:9]
                ]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(action, side, event, state, book)
                    loss = classification(output["direction_logits"], direction)
                    loss = (
                        loss
                        + regression(output["expected_return"], future_return)
                        + 0.1 * regression(output["expected_markout"], markout)
                        + regression(output["volatility"], volatility)
                    )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                examples += len(action)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in valid_loader:
                action, side, event, state, book = [
                    item.to(device, non_blocking=True) for item in batch[:5]
                ]
                output = model(action, side, event, state, book)
                predictions.append(output["direction_logits"].argmax(-1).cpu().numpy())
                targets.append(batch[5].numpy())
        prediction, target = np.concatenate(predictions), np.concatenate(targets)
        checkpoint = ROOT / "artifacts" / "checkpoints" / "m3t_tiny_smoke.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epochs,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "config_hash": content_hash({"rows": max_rows, "epochs": epochs}),
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
        metrics = {
            "epochs": epochs,
            "train_examples": examples,
            "final_loss": losses[-1],
            "validation_macro_f1": float(f1_score(target, prediction, average="macro")),
            "validation_balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
            "wall_seconds": elapsed,
            "examples_per_second": examples / elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(next(model.parameters()).device),
            "precision": "bfloat16",
            "checkpoint": str(checkpoint),
        }
    return model, metrics


def pretrain_ssl(max_rows: int = 12000, epochs: int = 3, seed: int = 1701) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for M3T SSL; CPU fallback forbidden")
    torch.manual_seed(seed)
    train = pd.read_parquet(data_root() / "splits" / "development" / "train.parquet")
    dataset = WindowDataset(train, limit=max_rows)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    device = torch.device("cuda:0")
    model = MarketForgeM3T(len(GLOBAL_STATE), d_model=64, layers=2, heads=4).to(device)
    heads = SSLHeads(64, len(GLOBAL_STATE)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=3e-4)
    started = time.perf_counter()
    steps = tokens = 0
    final_components: dict[str, float] = {}
    with gpu_lease():
        torch.cuda.reset_peak_memory_stats(device)
        for _epoch in range(epochs):
            model.train()
            for batch in loader:
                action, side, event, state, book = [
                    item.to(device, non_blocking=True) for item in batch[:5]
                ]
                mask = torch.rand(action.shape, device=device) < 0.15
                masked_action = action.clone()
                masked_action[mask] = 7
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model.encode(masked_action, side, event, state, book)
                    mask_loss = nn.functional.cross_entropy(
                        heads.masked_action(hidden[mask]), action[mask]
                    )
                    next_loss = nn.functional.cross_entropy(
                        heads.next_action(hidden[:, :-1]).flatten(0, 1), action[:, 1:].flatten()
                    )
                    state_loss = nn.functional.smooth_l1_loss(
                        heads.future_state(hidden[:, :-1]), state[:, 1:]
                    )
                    view2 = model.encode(action, side, event, state, book)[:, -1]
                    z1 = nn.functional.normalize(heads.contrastive(hidden[:, -1]), dim=-1)
                    z2 = nn.functional.normalize(heads.contrastive(view2), dim=-1)
                    logits = z1 @ z2.T / 0.1
                    similarity = nn.functional.cosine_similarity(
                        state[:, -1, None, :], state[None, :, -1, :], dim=-1
                    )
                    false_negatives = (similarity > 0.995) & ~torch.eye(
                        len(state), dtype=torch.bool, device=device
                    )
                    logits = logits.masked_fill(false_negatives, -torch.inf)
                    contrast_loss = nn.functional.cross_entropy(
                        logits, torch.arange(len(state), device=device)
                    )
                    loss = mask_loss + next_loss + state_loss + 0.1 * contrast_loss
                loss.backward()
                optimizer.step()
                steps += 1
                tokens += action.numel()
                final_components = {
                    "mask": float(mask_loss.detach()),
                    "next": float(next_loss.detach()),
                    "state": float(state_loss.detach()),
                    "contrast": float(contrast_loss.detach()),
                    "combined": float(loss.detach()),
                }
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        checkpoint = ROOT / "artifacts" / "checkpoints" / f"m3t_ssl_seed{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "ssl_heads": heads.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epochs,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "config_hash": content_hash({"max_rows": max_rows, "epochs": epochs, "seed": seed}),
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
        return {
            "stage": "SSL",
            "seed": seed,
            "epochs": epochs,
            "steps": steps,
            "tokens": tokens,
            "losses": final_components,
            "wall_seconds": elapsed,
            "tokens_per_second": tokens / elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "checkpoint": str(checkpoint),
            "device": str(next(model.parameters()).device),
            "precision": "bfloat16",
        }


def train_supervised(
    stage: str,
    seed: int = 1701,
    initial_checkpoint: str | None = None,
    max_rows: int = 20000,
    epochs: int = 6,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for M3T supervised training; CPU fallback forbidden")
    torch.manual_seed(seed)
    split = data_root() / "splits" / "development"
    train_ds = WindowDataset(pd.read_parquet(split / "train.parquet"), limit=max_rows)
    valid_ds = WindowDataset(pd.read_parquet(split / "model_validation.parquet"))
    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True
    )
    valid_loader = DataLoader(valid_ds, batch_size=256, num_workers=0, pin_memory=True)
    device = torch.device("cuda:0")
    model = MarketForgeM3T(len(GLOBAL_STATE), d_model=64, layers=2, heads=4).to(device)
    if initial_checkpoint:
        checkpoint_data = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    class_counts = np.bincount(train_ds.direction, minlength=3).astype(np.float32)
    class_weights = torch.tensor(
        class_counts.sum() / np.maximum(class_counts, 1) / 3, device=device
    )
    started = time.perf_counter()
    examples = 0
    final_loss = 0.0
    with gpu_lease():
        torch.cuda.reset_peak_memory_stats(device)
        for _epoch in range(epochs):
            model.train()
            for batch in train_loader:
                action, side, event, state, book, direction, future_return, markout, volatility = [
                    item.to(device, non_blocking=True) for item in batch[:9]
                ]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(action, side, event, state, book)
                    loss = nn.functional.cross_entropy(
                        output["direction_logits"], direction, weight=class_weights
                    )
                    loss = loss + nn.functional.smooth_l1_loss(
                        output["expected_return"], future_return
                    )
                    loss = loss + 0.1 * nn.functional.smooth_l1_loss(
                        output["expected_markout"], markout
                    )
                    loss = loss + nn.functional.smooth_l1_loss(output["volatility"], volatility)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                final_loss = float(loss.detach().cpu())
                examples += len(action)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        model.eval()
        predictions, targets = [], []
        with torch.inference_mode():
            for batch in valid_loader:
                output = model(*[item.to(device, non_blocking=True) for item in batch[:5]])
                predictions.append(output["direction_logits"].argmax(-1).cpu().numpy())
                targets.append(batch[5].numpy())
        prediction, target = np.concatenate(predictions), np.concatenate(targets)
        checkpoint = ROOT / "artifacts" / "checkpoints" / f"{stage.lower()}_seed{seed}.pt"
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epochs,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "config_hash": content_hash({"stage": stage, "seed": seed, "epochs": epochs}),
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
        prediction_counts = np.bincount(prediction, minlength=3).tolist()
        target_counts = np.bincount(target, minlength=3).tolist()
        return {
            "stage": stage,
            "seed": seed,
            "epochs": epochs,
            "train_examples": examples,
            "final_loss": final_loss,
            "validation_macro_f1": float(f1_score(target, prediction, average="macro")),
            "validation_balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
            "wall_seconds": elapsed,
            "examples_per_second": examples / elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "prediction_class_counts": prediction_counts,
            "target_class_counts": target_counts,
            "device": str(next(model.parameters()).device),
            "precision": "bfloat16",
            "checkpoint": str(checkpoint),
        }


def economic_posttrain(
    initial_checkpoint: str, seed: int = 1701, epochs: int = 4
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for economic post-training")
    torch.manual_seed(seed)
    train = pd.read_parquet(data_root() / "splits" / "development" / "train.parquet")
    dataset = WindowDataset(train, limit=20000)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    device = torch.device("cuda:0")
    model = MarketForgeM3T(len(GLOBAL_STATE), d_model=64, layers=2, heads=4).to(device)
    model.load_state_dict(
        torch.load(initial_checkpoint, map_location=device, weights_only=False)["model"]
    )
    return_log_scale = nn.Parameter(torch.tensor(-5.0, device=device))
    optimizer = torch.optim.AdamW(list(model.parameters()) + [return_log_scale], lr=1e-4)
    started = time.perf_counter()
    final: dict[str, float] = {}
    with gpu_lease():
        torch.cuda.reset_peak_memory_stats(device)
        for _epoch in range(epochs):
            model.train()
            for batch in loader:
                (
                    action,
                    side,
                    event,
                    state,
                    book,
                    direction,
                    future_return,
                    markout,
                    volatility,
                    fill,
                    adverse,
                ) = [item.to(device, non_blocking=True) for item in batch]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(action, side, event, state, book)
                    scale = return_log_scale.exp().clamp_min(1e-6)
                    return_nll = (
                        0.5 * ((future_return - output["expected_return"]) / scale) ** 2
                        + return_log_scale
                    ).mean()
                    markout_loss = nn.functional.smooth_l1_loss(output["expected_markout"], markout)
                    volatility_loss = nn.functional.smooth_l1_loss(output["volatility"], volatility)
                    fill_loss = nn.functional.binary_cross_entropy_with_logits(
                        output["fill_logit"], fill
                    )
                    adverse_loss = nn.functional.binary_cross_entropy_with_logits(
                        output["adverse_logit"], adverse
                    )
                    direction_loss = nn.functional.cross_entropy(
                        output["direction_logits"], direction
                    )
                    loss = (
                        return_nll
                        + markout_loss
                        + volatility_loss
                        + fill_loss
                        + adverse_loss
                        + 0.2 * direction_loss
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                final = {
                    "return_nll": float(return_nll.detach()),
                    "markout": float(markout_loss.detach()),
                    "volatility": float(volatility_loss.detach()),
                    "fill": float(fill_loss.detach()),
                    "adverse": float(adverse_loss.detach()),
                    "direction": float(direction_loss.detach()),
                    "combined": float(loss.detach()),
                }
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        checkpoint = ROOT / "artifacts" / "checkpoints" / f"m3t-ept_seed{seed}.pt"
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "return_log_scale": return_log_scale.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "epoch": epochs,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "config_hash": content_hash({"stage": "EPT", "seed": seed, "epochs": epochs}),
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
        return {
            "stage": "M3T-EPT",
            "seed": seed,
            "epochs": epochs,
            "losses": final,
            "return_scale": float(return_log_scale.exp().detach()),
            "wall_seconds": elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "checkpoint": str(checkpoint),
            "device": str(next(model.parameters()).device),
            "precision": "bfloat16",
        }
