from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
)
from sklearn.tree import DecisionTreeRegressor, export_text
from torch.utils.data import DataLoader

from .data import data_root, sha256_file
from .probe import atomic_json
from .specs import InstrumentSpec, load_registry
from .state import ROOT, content_hash
from .v2_baselines import XGB_FEATURES, _matrix, _predict_booster
from .v2_posthoc_governance import create_posthoc_charter, protected_inventory
from .v2_training import V2WindowDataset, _load_scalers, _model, _predict, _to_device

matplotlib.use("Agg")

ARTIFACT_ROOT = ROOT / "artifacts/posthoc_v2"
RESULT_ROOT = ROOT / "results/posthoc_v2"
REPORT_ROOT = ROOT / "reports/posthoc_v2"
TABLE_ROOT = REPORT_ROOT / "tables"
FIGURE_ROOT = REPORT_ROOT / "figures"
LABEL = "MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS"
MIN_SLICE_SUPPORT = 30
SEED = 1701
EPS = 1e-7
DIAGNOSTIC_REVISION = 4
PREDICTION_CACHE_REVISION = 6


def _frozen_economic_assumptions() -> dict[str, float]:
    payload = json.loads(
        (data_root() / "manifests/development_economics_v2.json").read_text(encoding="utf-8")
    )["controller"]
    return {
        "liquidation_cost_ticks": float(payload["liquidation_cost_ticks"]),
        "slippage_usd": float(payload["slippage_usd"]),
        "hedge_cost_usd": float(payload["hedge_cost_usd"]),
    }


def registered_round_trip_cost_ticks(
    spec: InstrumentSpec, frozen_assumptions: dict[str, float] | None = None
) -> float:
    """Return the frozen round-trip cost in instrument-native ticks."""
    assumptions = frozen_assumptions or _frozen_economic_assumptions()
    usd_cost = 2.0 * spec.fee_value + assumptions["slippage_usd"] + assumptions["hedge_cost_usd"]
    return usd_cost / spec.tick_value + assumptions["liquidation_cost_ticks"]


def _registered_instrument_spec(
    registry: dict[str, InstrumentSpec], source_key: str, instrument: str
) -> InstrumentSpec:
    exact = registry.get(f"{source_key}:{instrument}")
    if exact is not None:
        return exact
    candidates = [spec for spec in registry.values() if spec.symbol == instrument]
    if len(candidates) != 1:
        raise KeyError(
            f"instrument registry lookup is ambiguous or absent: {source_key}:{instrument}"
        )
    return candidates[0]


def _registered_cost_series(frame: pd.DataFrame) -> pd.Series:
    registry = load_registry()
    assumptions = _frozen_economic_assumptions()
    values = []
    for record in frame[["source_key", "instrument"]].itertuples(index=False):
        spec = _registered_instrument_spec(registry, record.source_key, record.instrument)
        values.append(registered_round_trip_cost_ticks(spec, assumptions))
    return pd.Series(values, index=frame.index, dtype=float)


def _l3_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["l3_targets_available"].fillna(False).astype(bool)


def _available_mask(frame: pd.DataFrame, side: str, target: str) -> pd.Series:
    mask = _l3_mask(frame)
    if target == "adverse":
        mask &= frame[f"{side}_adverse_available"].fillna(False).astype(bool)
    elif target in {"post_fill_markout", "time_to_fill"}:
        mask &= frame[f"{side}_fill"].fillna(0).astype(bool)
    return mask


def _adverse_output_column(model: str, side: str) -> str:
    if model == "xgboost":
        return f"xgboost_{side}_markout_derived_adverse_score"
    return f"m3t_{side}_adverse_probability"


def _adverse_error_observations(frame: pd.DataFrame, model: str = "m3t") -> pd.DataFrame:
    observations = []
    for side in ("bid", "ask"):
        mask = _available_mask(frame, side, "adverse")
        selected = frame.loc[mask].copy()
        if selected.empty:
            continue
        selected["side"] = side
        selected["adverse_target"] = selected[f"{side}_fill_conditional_adverse"]
        selected["adverse_prediction"] = selected[_adverse_output_column(model, side)]
        selected["adverse_absolute_error"] = (
            selected["adverse_prediction"] - selected["adverse_target"]
        ).abs()
        observations.append(selected)
    if not observations:
        return pd.DataFrame(
            columns=["side", "adverse_target", "adverse_prediction", "adverse_absolute_error"]
        )
    return pd.concat(observations, ignore_index=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, _json_safe(payload))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-clipped))


def _bin_edges(values: np.ndarray, bins: int = 10) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.asarray([-np.inf, np.inf])
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return np.asarray([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _labels_from_edges(values: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = [f"Q{index + 1}" for index in range(len(edges) - 1)]
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True).astype("string")


def _binary_calibration(
    target: np.ndarray, probability: np.ndarray, *, bins: int = 10, adaptive: bool = False
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    y = np.asarray(target, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if not len(y):
        return {"support": 0}, pd.DataFrame()
    if adaptive:
        edges = _bin_edges(p, bins)
    else:
        edges = np.linspace(0, 1, bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    gaps = []
    for index in range(len(edges) - 1):
        selected = ids == index
        if not selected.any():
            continue
        observed = float(y[selected].mean())
        predicted = float(p[selected].mean())
        gap = abs(observed - predicted)
        gaps.append((int(selected.sum()), gap))
        rows.append(
            {
                "bin": index + 1,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "support": int(selected.sum()),
                "positive": int(y[selected].sum()),
                "negative": int(selected.sum() - y[selected].sum()),
                "mean_probability": predicted,
                "empirical_rate": observed,
                "gap": gap,
            }
        )
    ece = sum(count * gap for count, gap in gaps) / len(y)
    mce = max((gap for _, gap in gaps), default=float("nan"))
    uncertainty = float(y.mean() * (1 - y.mean()))
    reliability = sum(
        row["support"] * (row["mean_probability"] - row["empirical_rate"]) ** 2 for row in rows
    ) / len(y)
    resolution = sum(
        row["support"] * (row["empirical_rate"] - y.mean()) ** 2 for row in rows
    ) / len(y)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = intercept = None
    if len(np.unique(y)) == 2:
        fitted = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED).fit(logit, y)
        slope = float(fitted.coef_[0, 0])
        intercept = float(fitted.intercept_[0])
    brier = float(brier_score_loss(y, p))
    decomposition_rhs = reliability - resolution + uncertainty
    decomposition_residual = brier - decomposition_rhs
    expected_binning_residual_tolerance = 0.05
    metrics: dict[str, float | int | None] = {
        "support": len(y),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "brier": brier,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": float(ece),
        "mce": float(mce),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "sharpness_std": float(p.std()),
        "mean_probability": float(p.mean()),
        "empirical_rate": float(y.mean()),
        "brier_reliability": float(reliability),
        "brier_resolution": float(resolution),
        "brier_uncertainty": uncertainty,
        "brier_decomposition_rhs": float(decomposition_rhs),
        "brier_decomposition_residual": float(decomposition_residual),
        "brier_decomposition_absolute_residual": float(abs(decomposition_residual)),
        "brier_decomposition_expected_binning_tolerance": expected_binning_residual_tolerance,
        "brier_decomposition_within_expected_binning_residual": bool(
            abs(decomposition_residual) <= expected_binning_residual_tolerance
        ),
    }
    return metrics, pd.DataFrame(rows)


def _multiclass_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    p /= p.sum(axis=1, keepdims=True)
    one_hot = np.eye(3)[y]
    predicted = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = (predicted == y).astype(float)
    top, _ = _binary_calibration(correct, confidence)
    adaptive, _ = _binary_calibration(correct, confidence, adaptive=True)
    return {
        "support": len(y),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "top_label_ece": float(top["ece"]),
        "top_label_mce": float(top["mce"]),
        "adaptive_ece": float(adaptive["ece"]),
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": float((-p * np.log(p)).sum(axis=1).mean()),
    }


def _final_frame() -> pd.DataFrame:
    feature_path = data_root() / "final_lockbox_v2/features_v4_multi_market/xnas_nvda.parquet"
    label_path = data_root() / "final_lockbox_v2/labels_v2_l3/xnas_nvda.parquet"
    features = pd.read_parquet(feature_path)
    labels = pd.read_parquet(label_path)
    if len(features) != 3238 or len(labels) != 3238:
        raise RuntimeError("immutable final row count changed")
    if not np.array_equal(features["prediction_time"], labels["prediction_time"]):
        raise RuntimeError("immutable final rowwise timestamps changed")
    features.insert(0, "capture_id", np.arange(len(features), dtype=np.int64))
    labels.insert(0, "capture_id", np.arange(len(labels), dtype=np.int64))
    frame = features.merge(
        labels,
        on="capture_id",
        how="inner",
        sort=False,
        validate="one_to_one",
        suffixes=("", "_label"),
    )
    if not np.array_equal(frame["capture_id"], np.arange(3238)):
        raise RuntimeError("post-hoc ordinal alignment changed row order")
    frame = frame.drop(columns=["prediction_time_label"])
    frame["split"] = "FINAL_POSTHOC_ONLY"
    frame["source_key"] = "databento:XNAS.ITCH"
    frame["source_id"] = 2
    frame["instrument_embedding_id"] = 2
    frame["source_balance_weight"] = 1.0 / len(frame)
    return frame


def _m3t_inference(
    frame: pd.DataFrame, partition: str
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for frozen M3T post-hoc inference")
    feature_scaler, target_scaler = _load_scalers()
    dataset = V2WindowDataset(frame, feature_scaler, target_scaler)
    rows = dataset.frame.iloc[dataset.valid_ends].reset_index(drop=True).copy()
    ladder = json.loads((data_root() / "manifests/model_ladder_v2.json").read_text())
    branch = ladder["branches"]["M3T-SSL-EPT"]
    calibration = ladder["branches"]["M3T-SSL-EPT-CAL"]["calibration"]
    model = _model().to("cuda:0")
    model.load_state_dict(
        torch.load(branch["checkpoint"], map_location="cuda:0", weights_only=False)["model"]
    )
    model.eval()
    prediction = _predict(model, dataset, torch.device("cuda:0"))
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    latent_parts: list[np.ndarray] = []
    occlusion: dict[str, list[float]] = {
        name: []
        for name in (
            "event_stream",
            "state_stream",
            "book_stream",
            "depth_level_1",
            "depth_levels_6_10",
            "modality_context",
        )
    }
    with torch.inference_mode():
        for batch_cpu in loader:
            batch = _to_device(batch_cpu, torch.device("cuda:0"))
            base = model(
                batch["action"],
                batch["side"],
                batch["event"],
                batch["state"],
                batch["book"],
                batch["source"],
                batch["instrument"],
                batch["modality"],
            )
            hidden_seq = model.encode(
                batch["action"],
                batch["side"],
                batch["event"],
                batch["state"],
                batch["book"],
                batch["source"],
                batch["instrument"],
                batch["modality"],
            )
            hidden = hidden_seq[:, -1]
            latent_parts.append(hidden.float().cpu().numpy())
            base_probability = torch.softmax(base["direction_logits"], dim=1)
            base_fill = torch.stack(
                [torch.sigmoid(base["bid_fill_logit"]), torch.sigmoid(base["ask_fill_logit"])],
                dim=1,
            )
            for name, values in occlusion.items():
                changed = {key: value.clone() for key, value in batch.items()}
                if name == "event_stream":
                    changed["action"].zero_()
                    changed["side"].zero_()
                    changed["event"].zero_()
                elif name == "state_stream":
                    changed["state"].zero_()
                elif name == "book_stream":
                    changed["book"].zero_()
                elif name == "depth_level_1":
                    changed["book"][:, :, 0, :].zero_()
                elif name == "depth_levels_6_10":
                    changed["book"][:, :, 5:, :].zero_()
                else:
                    changed["modality"].zero_()
                altered = model(
                    changed["action"],
                    changed["side"],
                    changed["event"],
                    changed["state"],
                    changed["book"],
                    changed["source"],
                    changed["instrument"],
                    changed["modality"],
                )
                probability_delta = torch.mean(
                    torch.abs(torch.softmax(altered["direction_logits"], dim=1) - base_probability)
                )
                altered_fill = torch.stack(
                    [
                        torch.sigmoid(altered["bid_fill_logit"]),
                        torch.sigmoid(altered["ask_fill_logit"]),
                    ],
                    dim=1,
                )
                fill_delta = torch.mean(torch.abs(altered_fill - base_fill))
                values.append(float((probability_delta + fill_delta).cpu()))
    logits = prediction["direction_logits"]
    raw = _softmax(logits)
    frozen = _softmax(logits / float(calibration["direction_temperature"]))
    output = rows.copy()
    output["partition"] = partition
    for index, name in enumerate(("down", "flat", "up")):
        output[f"m3t_direction_raw_{name}"] = raw[:, index]
        output[f"m3t_direction_frozen_{name}"] = frozen[:, index]
    output["m3t_direction_prediction"] = frozen.argmax(axis=1) - 1
    output["m3t_confidence"] = frozen.max(axis=1)
    output["m3t_entropy"] = (-frozen * np.log(np.clip(frozen, EPS, 1))).sum(axis=1)
    return_scale = target_scaler["return_1000ms"]
    output["m3t_expected_return"] = prediction["return_mean"] * return_scale.std + return_scale.mean
    for side in ("bid", "ask"):
        raw_fill = _sigmoid(prediction[f"{side}_fill_logit"])
        platt = calibration[f"{side}_fill_platt"]
        output[f"m3t_{side}_fill_raw"] = raw_fill
        with np.errstate(over="ignore"):
            output[f"m3t_{side}_fill_frozen"] = 1.0 / (
                1.0
                + np.exp(-(platt["slope"] * prediction[f"{side}_fill_logit"] + platt["intercept"]))
            )
        affine = calibration[f"{side}_markout_affine"]
        scale = target_scaler[f"{side}_quote_markout_1000ms"]
        output[f"m3t_{side}_markout_ticks"] = (
            affine["slope"] * prediction[f"{side}_quote_markout"] + affine["intercept"]
        ) * scale.std + scale.mean
        output[f"m3t_{side}_adverse_probability"] = _sigmoid(prediction[f"{side}_adverse_logit"])
    occlusion_rows = pd.DataFrame(
        [
            {
                "partition": partition,
                "feature_group": name,
                "mean_absolute_probability_impact": float(np.mean(values)),
                "batches": len(values),
            }
            for name, values in occlusion.items()
        ]
    )
    return output, np.concatenate(latent_parts), occlusion_rows


def _add_xgboost_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = _matrix(frame)
    root = ROOT / "artifacts/xgboost_v2"
    output = frame.copy()
    direction_model = xgb.Booster(model_file=str(root / "direction.json"))
    direction, warnings = _predict_booster(direction_model, matrix)
    for index, name in enumerate(("down", "flat", "up")):
        output[f"xgboost_direction_{name}"] = direction[:, index]
    output["xgboost_direction_prediction"] = direction.argmax(axis=1) - 1
    output["xgboost_confidence"] = direction.max(axis=1)
    output["xgboost_entropy"] = (-direction * np.log(np.clip(direction, EPS, 1))).sum(axis=1)
    for name in ("return", "bid_quote_markout", "ask_quote_markout", "volatility"):
        values, caught = _predict_booster(
            xgb.Booster(model_file=str(root / f"{name}.json")), matrix
        )
        warnings.extend(caught)
        output[f"xgboost_{name}"] = values
    for side in ("bid", "ask"):
        values, caught = _predict_booster(
            xgb.Booster(model_file=str(root / f"{side}_fill.json")), matrix
        )
        warnings.extend(caught)
        output[f"xgboost_{side}_fill_frozen"] = values
        output[f"xgboost_{side}_markout_ticks"] = output[f"xgboost_{side}_quote_markout"]
        with np.errstate(over="ignore"):
            output[f"xgboost_{side}_markout_derived_adverse_score"] = 1.0 / (
                1.0 + np.exp(output[f"xgboost_{side}_quote_markout"].to_numpy())
            )
    contributions = direction_model.predict(xgb.DMatrix(matrix), pred_contribs=True)
    importance = np.mean(np.abs(contributions[..., :-1]), axis=tuple(range(contributions.ndim - 1)))
    shap = pd.DataFrame(
        {
            "feature": XGB_FEATURES,
            "mean_abs_tree_shap": importance,
            "warnings": " | ".join(warnings),
        }
    ).sort_values("mean_abs_tree_shap", ascending=False)
    return output, shap


def _prediction_frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    cached = (
        RESULT_ROOT / "development_diagnostics.parquet",
        RESULT_ROOT / "final_posthoc_diagnostics.parquet",
        RESULT_ROOT / "development_m3t_latent.npy",
        RESULT_ROOT / "final_m3t_latent.npy",
        TABLE_ROOT / "m3t_occlusion.csv",
    )
    cache_marker = ARTIFACT_ROOT / "prediction_cache_revision.json"
    cache_revision = (
        json.loads(cache_marker.read_text(encoding="utf-8")).get("revision")
        if cache_marker.is_file()
        else None
    )
    if cache_revision == PREDICTION_CACHE_REVISION and all(path.is_file() for path in cached):
        development = pd.read_parquet(cached[0])
        final = pd.read_parquet(cached[1])
        dev_latent = np.load(cached[2])
        final_latent = np.load(cached[3])
        occlusion = pd.read_csv(cached[4])
        if len(development) == len(dev_latent) == 384 and len(final) == len(final_latent) == 3207:
            return development, final, dev_latent, final_latent, occlusion
    development = pd.read_parquet(
        data_root() / "splits_v4_multi_market/strategy_validation.parquet"
    )
    development.insert(0, "capture_id", np.arange(len(development), dtype=np.int64))
    final = _final_frame()
    dev_m3t, dev_latent, dev_occ = _m3t_inference(development, "DEVELOPMENT")
    final_m3t, final_latent, final_occ = _m3t_inference(final, "FINAL_POSTHOC_ONLY")
    dev, dev_shap = _add_xgboost_predictions(dev_m3t)
    final_output, final_shap = _add_xgboost_predictions(final_m3t)
    dev.to_parquet(RESULT_ROOT / "development_diagnostics.parquet", index=False)
    final_output.to_parquet(RESULT_ROOT / "final_posthoc_diagnostics.parquet", index=False)
    np.save(RESULT_ROOT / "development_m3t_latent.npy", dev_latent)
    np.save(RESULT_ROOT / "final_m3t_latent.npy", final_latent)
    pd.concat([dev_occ, final_occ], ignore_index=True).to_csv(
        TABLE_ROOT / "m3t_occlusion.csv", index=False
    )
    pd.concat(
        [
            dev_shap.assign(partition="DEVELOPMENT"),
            final_shap.assign(partition="FINAL_POSTHOC_ONLY"),
        ]
    ).to_csv(TABLE_ROOT / "xgboost_tree_shap.csv", index=False)
    _write_json(
        cache_marker,
        {
            "revision": PREDICTION_CACHE_REVISION,
            "source": "frozen checkpoints and immutable replay artifacts",
            "training_performed": False,
        },
    )
    return dev, final_output, dev_latent, final_latent, pd.concat([dev_occ, final_occ])


def _calibration_analysis(development: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[pd.DataFrame] = []
    class_names = ("down", "flat", "up")
    for partition, frame in (("DEVELOPMENT", development), ("FINAL_POSTHOC_ONLY", final)):
        target = frame["direction_1000ms"].to_numpy(np.int64) + 1
        for model in ("m3t", "xgboost"):
            variants = ("raw", "frozen") if model == "m3t" else ("frozen",)
            for variant in variants:
                prefix = f"m3t_direction_{variant}" if model == "m3t" else "xgboost_direction"
                probabilities = frame[[f"{prefix}_{name}" for name in class_names]].to_numpy()
                metrics = _multiclass_metrics(target, probabilities)
                metric_rows.append(
                    {
                        "partition": partition,
                        "model": model.upper(),
                        "task": "direction_multiclass",
                        "variant": variant,
                        **metrics,
                    }
                )
                confidence = probabilities.max(axis=1)
                correct = (probabilities.argmax(axis=1) == target).astype(int)
                for adaptive in (False, True):
                    _, curve = _binary_calibration(correct, confidence, adaptive=adaptive)
                    curve_rows.append(
                        curve.assign(
                            partition=partition,
                            model=model.upper(),
                            task="direction_top_label",
                            variant=variant,
                            binning="adaptive" if adaptive else "equal_width",
                        )
                    )
                for class_index, class_name in enumerate(class_names):
                    class_metrics, curve = _binary_calibration(
                        (target == class_index).astype(int), probabilities[:, class_index]
                    )
                    metric_rows.append(
                        {
                            "partition": partition,
                            "model": model.upper(),
                            "task": f"direction_{class_name}",
                            "variant": variant,
                            **class_metrics,
                        }
                    )
                    curve_rows.append(
                        curve.assign(
                            partition=partition,
                            model=model.upper(),
                            task=f"direction_{class_name}",
                            variant=variant,
                            binning="equal_width",
                        )
                    )
            for side in ("bid", "ask"):
                mask = frame["l3_targets_available"].astype(bool).to_numpy()
                target_fill = frame.loc[mask, f"{side}_fill"].to_numpy()
                variants = ("raw", "frozen") if model == "m3t" else ("frozen",)
                for variant in variants:
                    column = f"{model}_{side}_fill_{variant}"
                    probability = frame.loc[mask, column].to_numpy()
                    equal_metrics, equal_curve = _binary_calibration(target_fill, probability)
                    adaptive_metrics, adaptive_curve = _binary_calibration(
                        target_fill, probability, adaptive=True
                    )
                    metric_rows.append(
                        {
                            "partition": partition,
                            "model": model.upper(),
                            "task": f"{side}_fill",
                            "variant": variant,
                            **equal_metrics,
                            "adaptive_ece": adaptive_metrics.get("ece"),
                            "adaptive_mce": adaptive_metrics.get("mce"),
                        }
                    )
                    for binning, curve in (
                        ("equal_width", equal_curve),
                        ("adaptive", adaptive_curve),
                    ):
                        curve_rows.append(
                            curve.assign(
                                partition=partition,
                                model=model.upper(),
                                task=f"{side}_fill",
                                variant=variant,
                                binning=binning,
                            )
                        )
    metrics = pd.DataFrame(metric_rows)
    curves = pd.concat(curve_rows, ignore_index=True)
    decomposition_rows = metrics[metrics["task"].isin(["bid_fill", "ask_fill"])]
    if not decomposition_rows["brier_decomposition_within_expected_binning_residual"].all():
        failed = decomposition_rows.loc[
            ~decomposition_rows["brier_decomposition_within_expected_binning_residual"].astype(bool)
        ]
        raise RuntimeError(
            "Brier decomposition residual exceeds registered binning tolerance: "
            f"{failed.to_dict(orient='records')}"
        )
    metrics.to_csv(TABLE_ROOT / "calibration_metrics.csv", index=False)
    curves.to_csv(TABLE_ROOT / "reliability_curves.csv", index=False)
    confidence_rows = []
    for partition, frame in (("DEVELOPMENT", development), ("FINAL_POSTHOC_ONLY", final)):
        for model in ("m3t", "xgboost"):
            confidence_rows.append(
                pd.DataFrame(
                    {
                        "partition": partition,
                        "model": model.upper(),
                        "confidence": frame[f"{model}_confidence"],
                        "entropy": frame[f"{model}_entropy"],
                    }
                )
            )
    confidence = pd.concat(confidence_rows, ignore_index=True)
    confidence.to_parquet(RESULT_ROOT / "confidence_entropy.parquet", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    selected = curves[
        (curves["partition"] == "FINAL_POSTHOC_ONLY")
        & (curves["variant"] == "frozen")
        & (curves["binning"] == "equal_width")
        & curves["task"].isin(["bid_fill", "ask_fill"])
    ]
    for (model, task), group in selected.groupby(["model", "task"]):
        axes[0].plot(
            group["mean_probability"], group["empirical_rate"], marker="o", label=f"{model} {task}"
        )
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(title="Final fill reliability (post-hoc)", xlabel="Predicted", ylabel="Empirical")
    axes[0].legend(fontsize=8)
    for model, group in confidence[confidence["partition"] == "FINAL_POSTHOC_ONLY"].groupby(
        "model"
    ):
        axes[1].hist(group["confidence"], bins=20, alpha=0.55, label=model)
    axes[1].set(title="Final direction confidence", xlabel="Top-label probability", ylabel="Count")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "calibration_reliability_and_confidence.png", dpi=160)
    plt.close(fig)
    final_fill = metrics[
        (metrics["partition"] == "FINAL_POSTHOC_ONLY")
        & (metrics["variant"] == "frozen")
        & metrics["task"].isin(["bid_fill", "ask_fill"])
    ]
    advantage = {}
    for side in ("bid", "ask"):
        selected = final_fill[final_fill["task"] == f"{side}_fill"].set_index("model")
        reliability_advantage = float(
            selected.loc["M3T", "brier_reliability"] - selected.loc["XGBOOST", "brier_reliability"]
        )
        resolution_advantage = float(
            selected.loc["XGBOOST", "brier_resolution"] - selected.loc["M3T", "brier_resolution"]
        )
        total_advantage = float(selected.loc["M3T", "brier"] - selected.loc["XGBOOST", "brier"])
        material_resolution = resolution_advantage > max(1e-4, 0.01 * total_advantage)
        material_reliability = reliability_advantage > max(1e-4, 0.01 * total_advantage)
        if material_reliability and material_resolution:
            driver = "BOTH_MAINLY_RELIABILITY"
        elif material_reliability:
            driver = "PRIMARILY_BETTER_RELIABILITY"
        elif material_resolution:
            driver = "PRIMARILY_BETTER_RESOLUTION"
        else:
            driver = "NEITHER_COMPONENT_IN_ISOLATION"
        advantage[side] = {
            "xgboost_total_brier_advantage": total_advantage,
            "xgboost_reliability_advantage": reliability_advantage,
            "xgboost_resolution_advantage": resolution_advantage,
            "shared_outcome_uncertainty": float(selected.loc["M3T", "brier_uncertainty"]),
            "interpretation": driver,
        }
    return {
        "metrics_rows": len(metrics),
        "curve_rows": len(curves),
        "brier_note": "Brier combines calibration (reliability), resolution, and outcome uncertainty; it is not a pure calibration metric.",
        "fill_brier_advantage_decomposition": advantage,
    }


def _development_boundaries(development: pd.DataFrame) -> dict[str, list[float]]:
    frame = development.copy()
    frame["top_depth"] = frame["depth_bid_1"] + frame["depth_ask_1"]
    frame["top5_depth"] = frame["depth_bid_5"] + frame["depth_ask_5"]
    frame["queue_ahead"] = frame[["bid_queue_ahead_at_entry", "ask_queue_ahead_at_entry"]].mean(
        axis=1
    )
    boundaries = {}
    for column in (
        "spread",
        "top_depth",
        "top5_depth",
        "depth_slope",
        "imbalance_1",
        "signed_volume",
        "microprice_minus_mid",
        "event_rate",
        "realized_vol_50",
        "queue_ahead",
        "session_position",
        "m3t_confidence",
        "m3t_entropy",
        "xgboost_confidence",
        "xgboost_entropy",
    ):
        boundaries[column] = _bin_edges(frame[column].to_numpy(), 4).tolist()
    _write_json(
        ARTIFACT_ROOT / "development_slice_boundaries.json",
        {
            "schema_version": 1,
            "label": "V3_CANDIDATE_BOUNDARIES_FIXED_ON_DEVELOPMENT",
            "boundaries": boundaries,
        },
    )
    return boundaries


def _decorate(frame: pd.DataFrame, boundaries: dict[str, list[float]]) -> pd.DataFrame:
    output = frame.copy()
    output["top_depth"] = output["depth_bid_1"] + output["depth_ask_1"]
    output["top5_depth"] = output["depth_bid_5"] + output["depth_ask_5"]
    output["queue_ahead"] = output[["bid_queue_ahead_at_entry", "ask_queue_ahead_at_entry"]].mean(
        axis=1
    )
    output["modality"] = (
        "L1="
        + output["modality_l1"].astype(str)
        + "/L2="
        + output["modality_l2"].astype(str)
        + "/L3="
        + output["modality_l3"].astype(str)
    )
    output["timestamp_multiplicity"] = output.groupby("prediction_time")[
        "prediction_time"
    ].transform("size")
    output["repeated_exchange_timestamp"] = np.where(
        output["timestamp_multiplicity"] > 1, "REPEATED", "UNIQUE"
    )
    for column, edge_values in boundaries.items():
        if column in output:
            output[f"{column}_bucket"] = _labels_from_edges(
                output[column], np.asarray(edge_values, dtype=float)
            )
    output["clear_proximity"] = np.where(
        output.groupby("reset_generation", sort=False).cumcount() < 64,
        "WITHIN_64_ROWS_OF_RESET",
        "OTHER",
    )
    return output


def _slice_metrics(group: pd.DataFrame, model: str) -> dict[str, Any]:
    target = group["direction_1000ms"].to_numpy(np.int64) + 1
    probability = group[
        [f"{model}_direction_{name}" for name in ("down", "flat", "up")]
        if model == "xgboost"
        else [f"m3t_direction_frozen_{name}" for name in ("down", "flat", "up")]
    ].to_numpy()
    prediction = probability.argmax(axis=1)
    fill_targets = np.concatenate([group["bid_fill"].to_numpy(), group["ask_fill"].to_numpy()])
    fill_probability = np.concatenate(
        [
            group[f"{model}_bid_fill_frozen"].to_numpy(),
            group[f"{model}_ask_fill_frozen"].to_numpy(),
        ]
    )
    fill_mask = np.concatenate([group["l3_targets_available"].to_numpy(bool)] * 2) & np.isfinite(
        fill_targets
    )
    result: dict[str, Any] = {
        "support": len(group),
        "sufficient_support": len(group) >= MIN_SLICE_SUPPORT,
        "direction_accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "nll": float(log_loss(target, probability, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.sum((probability - np.eye(3)[target]) ** 2, axis=1))),
        "mean_confidence": float(probability.max(axis=1).mean()),
    }
    if fill_mask.any():
        fill_metrics, _ = _binary_calibration(fill_targets[fill_mask], fill_probability[fill_mask])
        result.update(
            {
                "fill_support": int(fill_mask.sum()),
                "fill_positive": int(fill_targets[fill_mask].sum()),
                "fill_negative": int(fill_mask.sum() - fill_targets[fill_mask].sum()),
                "empirical_fill_rate": float(fill_targets[fill_mask].mean()),
                "predicted_fill_rate": float(fill_probability[fill_mask].mean()),
                "fill_calibration_gap": float(
                    fill_probability[fill_mask].mean() - fill_targets[fill_mask].mean()
                ),
                "fill_brier": fill_metrics["brier"],
                "fill_ece": fill_metrics["ece"],
            }
        )
    valid_markout = pd.concat(
        [
            group.loc[
                _available_mask(group, "bid", "post_fill_markout"),
                "bid_fill_conditional_markout_100ms",
            ],
            group.loc[
                _available_mask(group, "ask", "post_fill_markout"),
                "ask_fill_conditional_markout_100ms",
            ],
        ]
    ).dropna()
    result["mean_fill_conditional_markout_points"] = (
        float(valid_markout.mean()) if len(valid_markout) else None
    )
    adverse = pd.concat(
        [
            group.loc[_available_mask(group, "bid", "adverse"), "bid_fill_conditional_adverse"],
            group.loc[_available_mask(group, "ask", "adverse"), "ask_fill_conditional_adverse"],
        ]
    ).dropna()
    result["adverse_selection_rate"] = float(adverse.mean()) if len(adverse) else None
    return result


def _slice_analysis(development: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    boundaries = _development_boundaries(development)
    dev, fin = _decorate(development, boundaries), _decorate(final, boundaries)
    dimensions = [
        "spread_bucket",
        "top_depth_bucket",
        "top5_depth_bucket",
        "depth_slope_bucket",
        "imbalance_1_bucket",
        "signed_volume_bucket",
        "microprice_minus_mid_bucket",
        "event_rate_bucket",
        "realized_vol_50_bucket",
        "queue_ahead_bucket",
        "session_position_bucket",
        "m3t_confidence_bucket",
        "m3t_entropy_bucket",
        "xgboost_confidence_bucket",
        "xgboost_entropy_bucket",
        "source",
        "venue",
        "instrument",
        "modality",
        "reset_generation",
        "clear_proximity",
        "repeated_exchange_timestamp",
    ]
    rows: list[dict[str, Any]] = []
    for partition, frame in (("DEVELOPMENT", dev), ("FINAL_POSTHOC_ONLY", fin)):
        for model in ("m3t", "xgboost"):
            for dimension in dimensions:
                for value, group in frame.groupby(dimension, observed=True, dropna=False):
                    rows.append(
                        {
                            "partition": partition,
                            "interpretation": "EXPLORATORY_POSTHOC"
                            if partition.startswith("FINAL")
                            else "DEVELOPMENT_DIAGNOSTIC",
                            "model": model.upper(),
                            "slice_dimension": dimension,
                            "slice_value": str(value),
                            **_slice_metrics(group, model),
                        }
                    )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_ROOT / "operational_slices.csv", index=False)
    execution_rows = []
    disagreement_rows = []
    for partition, frame in (("DEVELOPMENT", dev), ("FINAL_POSTHOC_ONLY", fin)):
        actual = frame["direction_1000ms"].to_numpy()
        disagreement_class = np.select(
            [
                (frame["m3t_direction_prediction"] == actual)
                & (frame["xgboost_direction_prediction"] == actual),
                (frame["m3t_direction_prediction"] == actual)
                & (frame["xgboost_direction_prediction"] != actual),
                (frame["m3t_direction_prediction"] != actual)
                & (frame["xgboost_direction_prediction"] == actual),
            ],
            ["BOTH_CORRECT", "M3T_CORRECT_XGB_WRONG", "M3T_WRONG_XGB_CORRECT"],
            default="BOTH_WRONG",
        )
        frame = frame.assign(model_disagreement_class=disagreement_class)
        for model in ("m3t", "xgboost"):
            for side in ("bid", "ask"):
                side_frame = frame.loc[_l3_mask(frame)].copy()
                if side_frame.empty:
                    execution_rows.append(
                        {
                            "partition": partition,
                            "interpretation": "EXPLORATORY_POSTHOC_ONLY",
                            "model": model.upper(),
                            "side": side,
                            "slice_dimension": "L3_AVAILABILITY",
                            "slice_value": "NOT_AVAILABLE",
                            "support": 0,
                            "positive": 0,
                            "negative": 0,
                            "sufficient_support": False,
                        }
                    )
                    continue
                side_frame["side"] = side
                side_frame["fill_status"] = np.where(
                    side_frame[f"{side}_fill"].astype(bool), "FILL", "NO_FILL"
                )
                derived = {
                    "predicted_fill_probability": side_frame[f"{model}_{side}_fill_frozen"],
                    "simulated_time_to_fill_ns": side_frame[f"{side}_time_to_first_fill_ns"],
                    "predicted_adverse_score": side_frame[_adverse_output_column(model, side)],
                    "expected_markout_ticks": side_frame[f"{model}_{side}_markout_ticks"],
                    "predicted_economic_edge_ticks": side_frame[f"{model}_{side}_fill_frozen"]
                    * (
                        side_frame[f"{model}_{side}_markout_ticks"]
                        - _registered_cost_series(side_frame)
                    ),
                }
                for name, values in derived.items():
                    side_frame[name] = values
                    development_l3 = development.loc[_l3_mask(development)]
                    development_values = development_l3[
                        f"{model}_{side}_fill_frozen"
                        if name == "predicted_fill_probability"
                        else f"{side}_time_to_first_fill_ns"
                        if name == "simulated_time_to_fill_ns"
                        else _adverse_output_column(model, side)
                        if name == "predicted_adverse_score"
                        else f"{model}_{side}_markout_ticks"
                    ]
                    if name == "predicted_economic_edge_ticks":
                        development_values = development_l3[f"{model}_{side}_fill_frozen"] * (
                            development_l3[f"{model}_{side}_markout_ticks"]
                            - _registered_cost_series(development_l3)
                        )
                    side_frame[f"{name}_bucket"] = _labels_from_edges(
                        side_frame[name], _bin_edges(development_values.to_numpy(), 4)
                    )
                dimensions = [
                    "side",
                    "fill_status",
                    "predicted_fill_probability_bucket",
                    "simulated_time_to_fill_ns_bucket",
                    "predicted_adverse_score_bucket",
                    "expected_markout_ticks_bucket",
                    "predicted_economic_edge_ticks_bucket",
                    "model_disagreement_class",
                ]
                for dimension in dimensions:
                    for value, group in side_frame.groupby(dimension, observed=True, dropna=False):
                        fill = group[f"{side}_fill"].to_numpy(float)
                        probability = group[f"{model}_{side}_fill_frozen"].to_numpy(float)
                        fraction = group[f"{side}_fill_fraction"].to_numpy(float)
                        filled_mask = _available_mask(group, side, "post_fill_markout")
                        markout = np.where(
                            filled_mask,
                            group[f"{side}_fill_conditional_markout_100ms"].to_numpy(float),
                            0.0,
                        )
                        fraction = np.where(filled_mask, fraction, 0.0)
                        multiplier = group["contract_multiplier"].to_numpy(float)
                        gross = markout * multiplier * fraction
                        tick_value = group["tick_value"].to_numpy(float)
                        costs = _registered_cost_series(group).to_numpy() * tick_value * fraction
                        execution_rows.append(
                            {
                                "partition": partition,
                                "interpretation": "EXPLORATORY_POSTHOC"
                                if partition.startswith("FINAL")
                                else "DEVELOPMENT_DIAGNOSTIC",
                                "model": model.upper(),
                                "side": side,
                                "slice_dimension": dimension,
                                "slice_value": str(value),
                                "support": len(group),
                                "positive": int(fill.sum()),
                                "negative": int(len(fill) - fill.sum()),
                                "sufficient_support": len(group) >= MIN_SLICE_SUPPORT,
                                "empirical_fill_rate": float(fill.mean()),
                                "predicted_fill_rate": float(probability.mean()),
                                "fill_calibration_gap": float((probability - fill).mean()),
                                "mean_fill_conditional_markout_points": float(
                                    group.loc[
                                        _available_mask(group, side, "post_fill_markout"),
                                        f"{side}_fill_conditional_markout_100ms",
                                    ].mean()
                                ),
                                "adverse_selection_rate": float(
                                    group.loc[
                                        _available_mask(group, side, "adverse"),
                                        f"{side}_fill_conditional_adverse",
                                    ].mean()
                                ),
                                "diagnostic_gross_edge_usd": float(gross.sum()),
                                "diagnostic_fees_and_liquidation_usd": float(costs.sum()),
                                "diagnostic_net_pnl_usd": float((gross - costs).sum()),
                            }
                        )
        for model in ("m3t", "xgboost"):
            correct = frame[f"{model}_direction_prediction"] == frame["direction_1000ms"]
            row_weighted = float(correct.mean())
            timestamp_weighted = float(
                pd.DataFrame({"prediction_time": frame["prediction_time"], "correct": correct})
                .groupby("prediction_time")["correct"]
                .mean()
                .mean()
            )
            disagreement_rows.append(
                {
                    "partition": partition,
                    "model": model.upper(),
                    "authoritative_row_preserving_accuracy": row_weighted,
                    "exploratory_timestamp_group_weighted_accuracy": timestamp_weighted,
                    "difference": timestamp_weighted - row_weighted,
                    "replaces_authoritative_metric": False,
                }
            )
    execution_table = pd.DataFrame(execution_rows)
    execution_table.to_csv(TABLE_ROOT / "execution_behavior_slices.csv", index=False)
    pd.DataFrame(disagreement_rows).to_csv(
        TABLE_ROOT / "repeated_timestamp_weighting_sensitivity.csv", index=False
    )
    return {
        "rows": len(table),
        "execution_rows": len(execution_table),
        "insufficient_support_rows": int((~table["sufficient_support"]).sum()),
        "minimum_support_rule": MIN_SLICE_SUPPORT,
    }


def _error_cohorts(development: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    features = [
        "spread",
        "realized_vol_50",
        "imbalance_1",
        "signed_volume",
        "depth_bid_5",
        "depth_ask_5",
        "bid_queue_ahead_at_entry",
        "ask_queue_ahead_at_entry",
        "event_rate",
        "session_position",
        "m3t_confidence",
        "m3t_entropy",
    ]
    development_l3 = development.loc[_l3_mask(development)].copy()
    final_l3 = final.loc[_l3_mask(final)].copy()
    adverse_dev = _adverse_error_observations(development)
    adverse_final = _adverse_error_observations(final)
    adverse_dev["side_indicator"] = (adverse_dev["side"] == "ask").astype(float)
    adverse_final["side_indicator"] = (adverse_final["side"] == "ask").astype(float)
    economic_dev = development_l3.copy()
    economic_final = final_l3.copy()
    for frame in (economic_dev, economic_final):
        realized = np.zeros(len(frame), dtype=float)
        for side in ("bid", "ask"):
            valid = _available_mask(frame, side, "post_fill_markout").to_numpy()
            realized += np.where(
                valid,
                frame[f"{side}_fill_fraction"].to_numpy(float)
                * frame[f"{side}_fill_conditional_markout_100ms"].fillna(0).to_numpy(float),
                0.0,
            )
        frame["economic_diagnostic_loss"] = -realized
    targets = {
        "direction_error": (
            development,
            final,
            (development["m3t_direction_prediction"] != development["direction_1000ms"]).astype(
                float
            ),
            features,
        ),
        "fill_probability_absolute_error": (
            development_l3,
            final_l3,
            0.5
            * (
                np.abs(development_l3["m3t_bid_fill_frozen"] - development_l3["bid_fill"])
                + np.abs(development_l3["m3t_ask_fill_frozen"] - development_l3["ask_fill"])
            ),
            features,
        ),
        "adverse_selection_absolute_error": (
            adverse_dev,
            adverse_final,
            adverse_dev["adverse_absolute_error"],
            [*features, "side_indicator"],
        ),
        "economic_diagnostic_loss": (
            economic_dev,
            economic_final,
            economic_dev["economic_diagnostic_loss"],
            features,
        ),
    }
    rows = []
    rules = {}
    for task, (dev_frame, final_frame, target, task_features) in targets.items():
        if len(dev_frame) < 30:
            rows.append(
                {
                    "task": task,
                    "cohort_leaf": "NOT_AVAILABLE",
                    "development_support": len(dev_frame),
                    "development_mean_error": None,
                    "final_support": len(final_frame),
                    "final_application": "INSUFFICIENT_VALID_TARGET_SUPPORT",
                }
            )
            rules[task] = "NOT_AVAILABLE"
            continue
        x_dev = np.nan_to_num(dev_frame[task_features].to_numpy(float))
        x_final = np.nan_to_num(final_frame[task_features].to_numpy(float))
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=SEED)
        tree.fit(x_dev, target)
        dev_leaf = tree.apply(x_dev)
        final_leaf = tree.apply(x_final)
        rules[task] = export_text(tree, feature_names=task_features)
        for leaf in np.unique(dev_leaf):
            dev_selected = dev_leaf == leaf
            final_selected = final_leaf == leaf
            rows.append(
                {
                    "task": task,
                    "cohort_leaf": int(leaf),
                    "development_support": int(dev_selected.sum()),
                    "development_mean_error": float(np.asarray(target)[dev_selected].mean()),
                    "final_support": int(final_selected.sum()),
                    "final_application": "FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY",
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_ROOT / "error_cohorts.csv", index=False)
    _write_json(
        ARTIFACT_ROOT / "development_error_cohort_rules.json",
        {
            "label": "V3_CANDIDATE_DEVELOPMENT_ONLY_DISCOVERY",
            "features": features,
            "adverse_target_policy": "SIDE_LONG; L3 AND SIDE_ADVERSE_AVAILABLE ONLY; UNDEFINED TARGETS EXCLUDED",
            "adverse_development_support": len(adverse_dev),
            "adverse_final_support": len(adverse_final),
            "rules": rules,
            "final_rules_optimized": False,
        },
    )
    return {"tasks": len(targets), "cohorts": len(rows)}


def _disagreement_and_selective(development: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    disagreement_rows = []
    selective_rows = []
    thresholds: dict[str, dict[str, float]] = {}
    for model in ("m3t", "xgboost"):
        thresholds[model] = {
            str(coverage): float(
                np.quantile(development[f"{model}_confidence"], 1 - coverage / 100)
            )
            for coverage in (100, 80, 60, 40, 20)
        }
    for partition, frame in (("DEVELOPMENT", development), ("FINAL_POSTHOC_ONLY", final)):
        actual = frame["direction_1000ms"].to_numpy()
        m_correct = frame["m3t_direction_prediction"].to_numpy() == actual
        x_correct = frame["xgboost_direction_prediction"].to_numpy() == actual
        groups = np.select(
            [m_correct & x_correct, m_correct & ~x_correct, ~m_correct & x_correct],
            ["BOTH_CORRECT", "M3T_CORRECT_XGB_WRONG", "M3T_WRONG_XGB_CORRECT"],
            default="BOTH_WRONG",
        )
        for group_name in np.unique(groups):
            selected = frame[groups == group_name]
            disagreement_rows.append(
                {
                    "partition": partition,
                    "group": group_name,
                    "support": len(selected),
                    "spread_mean": float(selected["spread"].mean()),
                    "volatility_mean": float(selected["realized_vol_50"].mean()),
                    "absolute_imbalance_mean": float(selected["imbalance_1"].abs().mean()),
                    "event_intensity_mean": float(selected["event_rate"].mean()),
                    "top5_depth_mean": float(
                        (selected["depth_bid_5"] + selected["depth_ask_5"]).mean()
                    ),
                    "queue_ahead_mean": float(
                        selected[["bid_queue_ahead_at_entry", "ask_queue_ahead_at_entry"]]
                        .mean(axis=1)
                        .mean()
                    ),
                }
            )
        for side in ("bid", "ask"):
            y = frame[f"{side}_fill"].to_numpy()
            m = frame[f"m3t_{side}_fill_frozen"].to_numpy()
            x = frame[f"xgboost_{side}_fill_frozen"].to_numpy()
            residual = pd.DataFrame(
                {
                    "partition": partition,
                    "side": side,
                    "absolute_probability_disagreement": np.abs(m - x),
                    "m3t_calibration_residual": m - y,
                    "xgboost_calibration_residual": x - y,
                    "brier_contribution_difference_m3t_minus_xgb": (m - y) ** 2 - (x - y) ** 2,
                }
            )
            residual.to_parquet(
                RESULT_ROOT / f"fill_disagreement_{partition.lower()}_{side}.parquet", index=False
            )
        for model in ("m3t", "xgboost"):
            for coverage, threshold in thresholds[model].items():
                selected = frame[f"{model}_confidence"] >= threshold
                if not selected.any():
                    continue
                actual_selected = actual[selected]
                prediction = frame.loc[selected, f"{model}_direction_prediction"].to_numpy()
                probability = frame.loc[
                    selected,
                    [
                        f"{model}_direction_{name}"
                        if model == "xgboost"
                        else f"m3t_direction_frozen_{name}"
                        for name in ("down", "flat", "up")
                    ],
                ].to_numpy()
                selective_rows.append(
                    {
                        "partition": partition,
                        "model": model.upper(),
                        "development_fixed_coverage_target": int(coverage),
                        "development_fixed_threshold": threshold,
                        "actual_coverage": float(selected.mean()),
                        "support": int(selected.sum()),
                        "accuracy": float(accuracy_score(actual_selected, prediction)),
                        "macro_f1": float(
                            f1_score(actual_selected, prediction, average="macro", zero_division=0)
                        ),
                        "nll": float(log_loss(actual_selected + 1, probability, labels=[0, 1, 2])),
                        "interpretation": "FINAL_POSTHOC_ONLY"
                        if partition.startswith("FINAL")
                        else "V3_CANDIDATE_DEVELOPMENT",
                    }
                )
    pd.DataFrame(disagreement_rows).to_csv(TABLE_ROOT / "direction_disagreement.csv", index=False)
    pd.DataFrame(selective_rows).to_csv(TABLE_ROOT / "risk_coverage.csv", index=False)
    _write_json(
        ARTIFACT_ROOT / "development_confidence_thresholds.json",
        {
            "label": "V3_CANDIDATE_DEVELOPMENT_ONLY",
            "thresholds": thresholds,
            "changes_v2_no_trade": False,
        },
    )
    return {"disagreement_groups": len(disagreement_rows), "selective_rows": len(selective_rows)}


def _confidence_monotonicity(development: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    rows = []
    conclusions: dict[str, dict[str, Any]] = {}
    for model in ("m3t", "xgboost"):
        edges = _bin_edges(development[f"{model}_confidence"].to_numpy(), 4)
        conclusions[model.upper()] = {}
        for partition, frame in (("DEVELOPMENT", development), ("FINAL_POSTHOC_ONLY", final)):
            buckets = _labels_from_edges(frame[f"{model}_confidence"], edges)
            partition_rows = []
            for bucket, group in frame.groupby(buckets, observed=True):
                target = group["direction_1000ms"].to_numpy(np.int64) + 1
                probability = group[
                    [
                        f"{model}_direction_{name}"
                        if model == "xgboost"
                        else f"m3t_direction_frozen_{name}"
                        for name in ("down", "flat", "up")
                    ]
                ].to_numpy()
                prediction = probability.argmax(axis=1)
                l3 = _l3_mask(group)
                fill_target = np.concatenate(
                    [group.loc[l3, "bid_fill"], group.loc[l3, "ask_fill"]]
                ).astype(float)
                fill_probability = np.concatenate(
                    [
                        group.loc[l3, f"{model}_bid_fill_frozen"],
                        group.loc[l3, f"{model}_ask_fill_frozen"],
                    ]
                ).astype(float)
                post_fill = pd.concat(
                    [
                        group.loc[
                            _available_mask(group, "bid", "post_fill_markout"),
                            "bid_fill_conditional_markout_100ms",
                        ],
                        group.loc[
                            _available_mask(group, "ask", "post_fill_markout"),
                            "ask_fill_conditional_markout_100ms",
                        ],
                    ]
                ).dropna()
                row = {
                    "partition": partition,
                    "model": model.upper(),
                    "development_confidence_quartile": str(bucket),
                    "bin_rank": int(str(bucket).removeprefix("Q")),
                    "support": len(group),
                    "accuracy": float(accuracy_score(target, prediction)),
                    "macro_f1": float(
                        f1_score(target, prediction, average="macro", zero_division=0)
                    ),
                    "nll": float(log_loss(target, probability, labels=[0, 1, 2])),
                    "error_rate": float(np.mean(prediction != target)),
                    "fill_support": len(fill_target),
                    "fill_brier": float(np.mean((fill_probability - fill_target) ** 2))
                    if len(fill_target)
                    else None,
                    "post_fill_markout_support": len(post_fill),
                    "mean_post_fill_markout": float(post_fill.mean()) if len(post_fill) else None,
                    "slice_definition_source": "DEVELOPMENT_FIXED_CONFIDENCE_QUARTILES",
                    "interpretation": "EXPLORATORY_POSTHOC_ONLY"
                    if partition.startswith("FINAL")
                    else "DEVELOPMENT_DIAGNOSTIC",
                }
                rows.append(row)
                partition_rows.append(row)
            ordered = pd.DataFrame(partition_rows).sort_values("bin_rank")
            correlation = (
                float(spearmanr(ordered["bin_rank"], ordered["accuracy"]).statistic)
                if len(ordered) >= 3 and ordered["accuracy"].nunique() > 1
                else None
            )
            violations = int((ordered["accuracy"].diff().dropna() < 0).sum())
            if correlation is None:
                conclusion = "NOT_ESTABLISHED"
            elif correlation >= 0.8 and violations == 0:
                conclusion = "MONOTONICALLY_USEFUL"
            elif correlation < 0 or violations >= 2:
                conclusion = "NON_MONOTONIC"
            else:
                conclusion = "WEAK"
            conclusions[model.upper()][partition] = {
                "classification": conclusion,
                "spearman_confidence_rank_vs_accuracy": correlation,
                "monotonicity_violations": violations,
                "accuracy_q4_minus_q1": float(
                    ordered.iloc[-1]["accuracy"] - ordered.iloc[0]["accuracy"]
                )
                if len(ordered) >= 2
                else None,
            }
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_ROOT / "confidence_monotonicity.csv", index=False)
    _write_json(
        ARTIFACT_ROOT / "confidence_monotonicity_summary.json",
        {
            "thresholds_fit_on_final": False,
            "conclusions": conclusions,
            "interpretation": "DESCRIPTIVE_ONLY_NOT_A_SELECTION_RULE",
        },
    )
    return conclusions


def _economic_rows(frame: pd.DataFrame, model: str, partition: str) -> pd.DataFrame:
    registry = load_registry()
    assumptions = _frozen_economic_assumptions()
    rows = []
    for side in ("bid", "ask"):
        for record in frame.itertuples(index=False):
            if not bool(record.l3_targets_available):
                continue
            spec = _registered_instrument_spec(registry, record.source_key, record.instrument)
            probability = float(getattr(record, f"{model}_{side}_fill_frozen"))
            markout_ticks = float(getattr(record, f"{model}_{side}_markout_ticks"))
            adverse_column = _adverse_output_column(model, side)
            adverse = float(getattr(record, adverse_column))
            cost_ticks = registered_round_trip_cost_ticks(spec, assumptions)
            expected_edge_ticks = probability * (markout_ticks - cost_ticks)
            active = expected_edge_ticks > 0 and adverse < 0.7
            fraction_value = getattr(record, f"{side}_fill_fraction")
            fill_fraction = float(fraction_value) if pd.notna(fraction_value) else 0.0
            markout = getattr(record, f"{side}_fill_conditional_markout_100ms")
            filled = bool(active and fill_fraction > 0 and pd.notna(markout))
            total_gross = (
                float(markout) * spec.contract_multiplier * fill_fraction if filled else 0.0
            )
            spread = (
                float(record.mid - record.bid_px_01)
                if side == "bid"
                else float(record.ask_px_01 - record.mid)
            )
            spread_capture = spread * spec.contract_multiplier * fill_fraction if filled else 0.0
            subsequent_price_movement = total_gross - spread_capture
            fees = 2 * spec.fee_value * fill_fraction if filled else 0.0
            liquidation = (
                spec.ticks_to_usd(assumptions["liquidation_cost_ticks"], fill_fraction)
                if filled
                else 0.0
            )
            slippage = assumptions["slippage_usd"] * fill_fraction if filled else 0.0
            hedge = assumptions["hedge_cost_usd"] * fill_fraction if filled else 0.0
            net = total_gross - fees - liquidation - slippage - hedge
            rows.append(
                {
                    "partition": partition,
                    "interpretation": "EXPLORATORY_POSTHOC"
                    if partition.startswith("FINAL")
                    else "DEVELOPMENT_DIAGNOSTIC",
                    "model": model.upper(),
                    "capture_id": int(record.capture_id),
                    "side": side,
                    "predicted_fill_probability": probability,
                    "predicted_markout_ticks": markout_ticks,
                    "adverse_controller_input": adverse,
                    "adverse_output_semantics": "DETERMINISTIC_MARKOUT_DERIVED_SCORE"
                    if model == "xgboost"
                    else "LEARNED_UNCALIBRATED_PROBABILITY",
                    "registered_round_trip_cost_ticks": cost_ticks,
                    "expected_edge_ticks": expected_edge_ticks,
                    "active_diagnostic": active,
                    "simulated_l3_fill": filled,
                    "gross_spread_price_edge_usd": total_gross,
                    "total_gross_pnl_usd": total_gross,
                    "spread_capture_usd": spread_capture,
                    "subsequent_markout_price_movement_usd": subsequent_price_movement,
                    "adverse_selection_component_usd": min(subsequent_price_movement, 0.0),
                    "fees_usd": fees,
                    "liquidation_cost_usd": liquidation,
                    "slippage_usd": slippage,
                    "hedge_cost_usd": hedge,
                    "realized_pnl_usd": net,
                    "unrealized_pnl_usd": 0.0,
                    "net_pnl_usd": net,
                    "realized_edge_ticks": (
                        net / spec.tick_value if filled and spec.tick_value else 0.0
                    ),
                    "accounting_identity_error_usd": abs(
                        total_gross - (spread_capture + subsequent_price_movement)
                    )
                    + abs(net - (total_gross - fees - liquidation - slippage - hedge)),
                }
            )
    return pd.DataFrame(rows)


def _economic_analysis(development: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    final_xgboost, _ = _add_xgboost_predictions(_final_frame())
    final_xgboost.to_parquet(RESULT_ROOT / "final_xgboost_all_rows.parquet", index=False)
    tables = [
        _economic_rows(development, "m3t", "DEVELOPMENT"),
        _economic_rows(development, "xgboost", "DEVELOPMENT"),
        _economic_rows(final, "m3t", "FINAL_POSTHOC_ONLY"),
        _economic_rows(final_xgboost, "xgboost", "FINAL_POSTHOC_ONLY"),
    ]
    detail = pd.concat(tables, ignore_index=True)
    detail.to_parquet(RESULT_ROOT / "economic_attribution_detail.parquet", index=False)
    summary = detail.groupby(["partition", "model"], as_index=False).agg(
        quotes=("active_diagnostic", "sum"),
        fills=("simulated_l3_fill", "sum"),
        gross_spread_price_edge_usd=("gross_spread_price_edge_usd", "sum"),
        total_gross_pnl_usd=("total_gross_pnl_usd", "sum"),
        spread_capture_usd=("spread_capture_usd", "sum"),
        subsequent_markout_price_movement_usd=(
            "subsequent_markout_price_movement_usd",
            "sum",
        ),
        adverse_selection_component_usd=("adverse_selection_component_usd", "sum"),
        fees_usd=("fees_usd", "sum"),
        liquidation_cost_usd=("liquidation_cost_usd", "sum"),
        slippage_usd=("slippage_usd", "sum"),
        hedge_cost_usd=("hedge_cost_usd", "sum"),
        realized_pnl_usd=("realized_pnl_usd", "sum"),
        unrealized_pnl_usd=("unrealized_pnl_usd", "sum"),
        net_pnl_usd=("net_pnl_usd", "sum"),
        accounting_identity_error_usd=("accounting_identity_error_usd", "sum"),
    )
    summary["authoritative_policy"] = "NO_TRADE"
    summary.to_csv(TABLE_ROOT / "economic_attribution.csv", index=False)
    if float(summary["accounting_identity_error_usd"].max()) > 1e-9:
        raise RuntimeError("post-hoc economic attribution accounting identity failed")

    authoritative = json.loads(
        (ROOT / "artifacts/final_evaluation_v2.json").read_text(encoding="utf-8")
    )["economic"]
    reconciliation_rows = []
    for model, key in (
        ("M3T", "m3t_common_controller_diagnostic"),
        ("XGBOOST", "xgboost_common_controller_diagnostic"),
    ):
        observed = summary[
            (summary["partition"] == "FINAL_POSTHOC_ONLY") & (summary["model"] == model)
        ].iloc[0]
        expected = authoritative[key]
        fields = {
            "quotes": int(expected["quotes"]),
            "fills": int(expected["fills"]),
            "gross_price_markout_pnl_usd": float(expected["price_pnl_usd"]),
            "fees_usd": float(expected["fees_usd"]),
            "liquidation_cost_usd": float(expected["liquidation_cost_usd"]),
            "slippage_usd": float(expected["slippage_usd"]),
            "hedge_cost_usd": float(expected["hedge_cost_usd"]),
            "net_pnl_usd": float(expected["realized_pnl_usd"]),
        }
        reconstructed = {
            "quotes": int(observed["quotes"]),
            "fills": int(observed["fills"]),
            "gross_price_markout_pnl_usd": float(observed["total_gross_pnl_usd"]),
            "fees_usd": float(observed["fees_usd"]),
            "liquidation_cost_usd": float(observed["liquidation_cost_usd"]),
            "slippage_usd": float(observed["slippage_usd"]),
            "hedge_cost_usd": float(observed["hedge_cost_usd"]),
            "net_pnl_usd": float(observed["net_pnl_usd"]),
        }
        for row_type, values in (
            ("AUTHORITATIVE", fields),
            ("RECONSTRUCTED_POSTHOC", reconstructed),
        ):
            reconciliation_rows.append(
                {
                    "model": model,
                    "row_type": row_type,
                    **values,
                    "reconciliation_status": "REFERENCE"
                    if row_type == "AUTHORITATIVE"
                    else (
                        "MATCH"
                        if values["quotes"] == fields["quotes"]
                        and values["fills"] == fields["fills"]
                        and all(
                            abs(values[name] - fields[name]) <= 1e-9
                            for name in (
                                "gross_price_markout_pnl_usd",
                                "fees_usd",
                                "liquidation_cost_usd",
                                "slippage_usd",
                                "hedge_cost_usd",
                                "net_pnl_usd",
                            )
                        )
                        else "MISMATCH"
                    ),
                }
            )
    reconciliation_rows.append(
        {
            "model": "NO_TRADE",
            "row_type": "AUTHORITATIVE",
            "quotes": 0,
            "fills": 0,
            "gross_price_markout_pnl_usd": 0.0,
            "fees_usd": 0.0,
            "liquidation_cost_usd": 0.0,
            "slippage_usd": 0.0,
            "hedge_cost_usd": 0.0,
            "net_pnl_usd": float(authoritative["selected_NO_TRADE"]["realized_pnl_usd"]),
            "reconciliation_status": "REFERENCE",
        }
    )
    reconciliation = pd.DataFrame(reconciliation_rows)
    reconciliation.to_csv(TABLE_ROOT / "authoritative_economic_reconciliation.csv", index=False)
    mismatches = reconciliation["reconciliation_status"] == "MISMATCH"
    if mismatches.any():
        raise RuntimeError(
            "post-hoc economic reconstruction does not match authoritative diagnostic: "
            f"{reconciliation.loc[mismatches].to_dict(orient='records')}"
        )
    edge_rows = []
    dev_edges: dict[str, list[float]] = {}
    for model in ("M3T", "XGBOOST"):
        dev_values = detail.loc[detail["model"] == model, "expected_edge_ticks"]
        dev_edges[model] = _bin_edges(dev_values.to_numpy(), 10).tolist()
    for (partition, model, side), group in detail.groupby(["partition", "model", "side"]):
        edges = np.asarray(dev_edges[model], dtype=float)
        buckets = _labels_from_edges(group["expected_edge_ticks"], edges)
        for bucket, selected in group.groupby(buckets, observed=True):
            correlation_inputs_vary = (
                selected["expected_edge_ticks"].nunique() > 1
                and selected["realized_edge_ticks"].nunique() > 1
            )
            corr = (
                spearmanr(
                    selected["expected_edge_ticks"], selected["realized_edge_ticks"]
                ).statistic
                if correlation_inputs_vary
                else float("nan")
            )
            edge_rows.append(
                {
                    "partition": partition,
                    "model": model,
                    "side": side,
                    "development_edge_bucket": str(bucket),
                    "support": len(selected),
                    "predicted_edge_mean_ticks": float(selected["expected_edge_ticks"].mean()),
                    "realized_edge_mean_ticks": float(selected["realized_edge_ticks"].mean()),
                    "signed_bias_ticks": float(
                        (selected["expected_edge_ticks"] - selected["realized_edge_ticks"]).mean()
                    ),
                    "mae_ticks": float(
                        mean_absolute_error(
                            selected["realized_edge_ticks"], selected["expected_edge_ticks"]
                        )
                    ),
                    "spearman_rank": None if np.isnan(corr) else float(corr),
                    "interpretation": "ECONOMIC_CALIBRATION_DIAGNOSTIC_ONLY",
                }
            )
    edge = pd.DataFrame(edge_rows)
    edge.to_csv(TABLE_ROOT / "predicted_vs_realized_edge.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    final_summary = summary[summary["partition"] == "FINAL_POSTHOC_ONLY"]
    components = ["total_gross_pnl_usd", "fees_usd", "liquidation_cost_usd"]
    for index, row in final_summary.reset_index(drop=True).iterrows():
        axes[0].bar(
            np.arange(3) + index * 0.3,
            [row[components[0]], -row[components[1]], -row[components[2]]],
            width=0.28,
            label=row["model"],
        )
    axes[0].set_xticks(np.arange(3) + 0.15, ["Total gross PnL", "Fees", "Liquidation"])
    axes[0].set_ylabel("USD")
    axes[0].set_title("Final diagnostic economic waterfall components")
    axes[0].legend()
    for (model, side), group in edge[edge["partition"] == "FINAL_POSTHOC_ONLY"].groupby(
        ["model", "side"]
    ):
        axes[1].plot(
            group["predicted_edge_mean_ticks"],
            group["realized_edge_mean_ticks"],
            marker="o",
            label=f"{model} {side}",
        )
    axes[1].set(
        title="Expected vs realized edge (post-hoc)",
        xlabel="Expected ticks",
        ylabel="Realized ticks",
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "economic_waterfall_and_edge_calibration.png", dpi=160)
    plt.close(fig)
    return {
        "detail_rows": len(detail),
        "summary": summary.to_dict(orient="records"),
        "authoritative_reconciliation": "MATCH",
    }


def _markout_and_queue(development: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    boundaries = _development_boundaries(development)
    dev, fin = _decorate(development, boundaries), _decorate(final, boundaries)
    markout_rows = []
    queue_rows = []
    feasible = {
        "QUOTE_MARKOUT_100MS": ("{side}_quote_markout_100ms", False),
        "QUOTE_MARKOUT_1000MS": ("{side}_quote_markout_1000ms", False),
        "SIMULATED_L3_POST_FILL_MARKOUT_100MS": (
            "{side}_fill_conditional_markout_100ms",
            True,
        ),
    }
    requested = ["10ms", "25ms", "50ms", "100ms", "250ms", "500ms", "1s", "5s"]
    unsupported = [item for item in requested if item not in {"100ms", "1s"}]
    for partition, frame in (("DEVELOPMENT", dev), ("FINAL_POSTHOC_ONLY", fin)):
        for model in ("m3t", "xgboost"):
            for side in ("bid", "ask"):
                model_frame = frame.copy()
                development_l3 = development.loc[_l3_mask(development)].copy()
                l3_frame = model_frame.loc[_l3_mask(model_frame)].copy()
                for candidate in (development_l3, l3_frame):
                    candidate["queue_ahead_ratio_at_price"] = candidate[
                        f"{side}_queue_ahead_at_entry"
                    ] / candidate[f"{side}_sz_01"].replace(0, np.nan)
                    candidate["queue_ahead_ratio_top5"] = candidate[
                        f"{side}_queue_ahead_at_entry"
                    ] / candidate[f"depth_{side}_5"].replace(0, np.nan)
                model_frame["predicted_fill_bucket"] = _labels_from_edges(
                    model_frame[f"{model}_{side}_fill_frozen"],
                    _bin_edges(development_l3[f"{model}_{side}_fill_frozen"].to_numpy(), 4),
                )
                model_frame["predicted_adverse_bucket"] = _labels_from_edges(
                    model_frame[_adverse_output_column(model, side)],
                    _bin_edges(development_l3[_adverse_output_column(model, side)].to_numpy(), 4),
                )
                model_frame["time_to_fill_bucket"] = _labels_from_edges(
                    model_frame[f"{side}_time_to_first_fill_ns"],
                    _bin_edges(
                        development_l3.loc[
                            _available_mask(development_l3, side, "time_to_fill"),
                            f"{side}_time_to_first_fill_ns",
                        ].to_numpy(),
                        4,
                    ),
                )
                dimensions = [
                    "spread_bucket",
                    "realized_vol_50_bucket",
                    "queue_ahead_bucket",
                    f"{model}_confidence_bucket",
                    "predicted_adverse_bucket",
                    "predicted_fill_bucket",
                    "time_to_fill_bucket",
                ]
                for markout_type, (pattern, fill_conditioned) in feasible.items():
                    column = pattern.format(side=side)
                    for dimension in dimensions:
                        execution_dimension = dimension in {
                            "queue_ahead_bucket",
                            "predicted_adverse_bucket",
                            "predicted_fill_bucket",
                            "time_to_fill_bucket",
                        }
                        eligible_frame = (
                            model_frame.loc[_l3_mask(model_frame)]
                            if execution_dimension or fill_conditioned
                            else model_frame
                        )
                        for value, group in eligible_frame.groupby(
                            dimension, observed=True, dropna=False
                        ):
                            values = group[column].dropna()
                            if fill_conditioned:
                                values = group.loc[
                                    _available_mask(group, side, "post_fill_markout"), column
                                ].dropna()
                            if not len(values):
                                continue
                            markout_rows.append(
                                {
                                    "partition": partition,
                                    "model": model.upper(),
                                    "markout_type": markout_type,
                                    "side": side,
                                    "horizon": "100ms"
                                    if markout_type.endswith("100MS")
                                    else "1000ms",
                                    "slice_dimension": dimension,
                                    "slice_value": str(value),
                                    "support": len(values),
                                    "l3_required": fill_conditioned or execution_dimension,
                                    "l3_valid_only": bool(fill_conditioned or execution_dimension),
                                    "mean": float(values.mean()),
                                    "median": float(values.median()),
                                    "q10": float(values.quantile(0.1)),
                                    "q25": float(values.quantile(0.25)),
                                    "q75": float(values.quantile(0.75)),
                                    "q90": float(values.quantile(0.9)),
                                    "interpretation": "EXPLORATORY_POSTHOC"
                                    if partition.startswith("FINAL")
                                    else "DEVELOPMENT_DIAGNOSTIC",
                                }
                            )
                queue_dimensions = [
                    f"{side}_queue_ahead_at_entry",
                    "queue_ahead_ratio_at_price",
                    "queue_ahead_ratio_top5",
                    "spread",
                    "realized_vol_50",
                    "event_rate",
                    "imbalance_1",
                ]
                for dimension in queue_dimensions:
                    if l3_frame.empty:
                        continue
                    edges = _bin_edges(development_l3[dimension].to_numpy(), 4)
                    buckets = _labels_from_edges(l3_frame[dimension], edges)
                    for bucket, group in l3_frame.groupby(buckets, observed=True):
                        target = group[f"{side}_fill"].to_numpy()
                        probability = group[f"{model}_{side}_fill_frozen"].to_numpy()
                        filled_ttf = group.loc[
                            _available_mask(group, side, "time_to_fill"),
                            f"{side}_time_to_first_fill_ns",
                        ].dropna()
                        queue_rows.append(
                            {
                                "partition": partition,
                                "model": model.upper(),
                                "side": side,
                                "condition": dimension,
                                "development_bucket": str(bucket),
                                "support": len(group),
                                "instruments": "|".join(
                                    sorted(group["instrument"].astype(str).unique())
                                ),
                                "l3_valid_only": True,
                                "positive": int(target.sum()),
                                "negative": int(len(target) - target.sum()),
                                "simulated_l3_fill_rate": float(target.mean()),
                                "predicted_fill_probability": float(probability.mean()),
                                "calibration_residual": float((probability - target).mean()),
                                "mean_brier_contribution": float(
                                    np.mean((probability - target) ** 2)
                                ),
                                "time_to_fill_support": len(filled_ttf),
                                "time_to_fill_q25_ns": float(filled_ttf.quantile(0.25))
                                if len(filled_ttf)
                                else None,
                                "median_time_to_fill_ns": float(filled_ttf.median())
                                if len(filled_ttf)
                                else None,
                                "time_to_fill_q75_ns": float(filled_ttf.quantile(0.75))
                                if len(filled_ttf)
                                else None,
                            }
                        )
    markouts = pd.DataFrame(markout_rows)
    queue = pd.DataFrame(queue_rows)
    markouts.to_csv(TABLE_ROOT / "markout_curves.csv", index=False)
    queue.to_csv(TABLE_ROOT / "queue_fill_calibration.csv", index=False)
    _write_json(
        ARTIFACT_ROOT / "markout_horizon_availability.json",
        {
            "available": [
                "QUOTE_MARKOUT_100MS",
                "QUOTE_MARKOUT_1000MS",
                "SIMULATED_L3_POST_FILL_MARKOUT_100MS",
            ],
            "semantic_rule": "Only fill-conditioned replay-derived markouts use SIMULATED_L3_POST_FILL_MARKOUT",
            "unavailable_requested_clock_horizons": unsupported,
            "event_count_horizons": "UNAVAILABLE_IN_EXISTING_FROZEN_LABELS",
            "replay_performed": False,
        },
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    selected = markouts[
        (markouts["partition"] == "FINAL_POSTHOC_ONLY")
        & (markouts["slice_dimension"] == "predicted_fill_bucket")
    ]
    for (model, side, markout_type), group in selected.groupby(["model", "side", "markout_type"]):
        ax.plot(
            group["slice_value"],
            group["mean"],
            marker="o",
            label=f"{model} {side} {markout_type}",
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        title="Side-consistent markouts by predicted fill bucket",
        xlabel="Development-derived bucket",
        ylabel="Price-point markout",
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "markout_tca_curves.png", dpi=160)
    plt.close(fig)
    final_queue = queue[queue["partition"] == "FINAL_POSTHOC_ONLY"]
    comparison = final_queue.pivot_table(
        index=["side", "condition", "development_bucket"],
        columns="model",
        values="mean_brier_contribution",
    ).dropna()
    comparison["m3t_minus_xgboost_brier"] = comparison["M3T"] - comparison["XGBOOST"]
    largest = comparison.sort_values("m3t_minus_xgboost_brier", ascending=False).head(1)
    execution_difference = (
        {
            "side": largest.index[0][0],
            "condition": largest.index[0][1],
            "development_bucket": largest.index[0][2],
            "m3t_minus_xgboost_brier": float(largest.iloc[0]["m3t_minus_xgboost_brier"]),
        }
        if len(largest)
        else {"status": "NOT_ESTABLISHED"}
    )
    return {
        "markout_rows": len(markouts),
        "queue_rows": len(queue),
        "unsupported": unsupported,
        "largest_xgboost_fill_advantage_state": execution_difference,
    }


def _shift_metrics(
    name: str, development: pd.Series, final: pd.Series, family: str
) -> dict[str, Any]:
    dev = development.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    fin = final.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if not len(dev) or not len(fin):
        return {
            "feature": name,
            "drift_family": family,
            "development_support": len(dev),
            "final_support": len(fin),
            "status": "NOT_AVAILABLE",
        }
    pooled = math.sqrt((np.var(dev) + np.var(fin)) / 2)
    standardized = (np.mean(fin) - np.mean(dev)) / pooled if pooled > 0 else 0.0
    bins = np.unique(np.quantile(dev, np.linspace(0, 1, 11)))
    if len(bins) > 1:
        bins[0], bins[-1] = -np.inf, np.inf
        dev_hist = np.histogram(dev, bins=bins)[0] / len(dev)
        fin_hist = np.histogram(fin, bins=bins)[0] / len(fin)
        psi = float(np.sum((fin_hist - dev_hist) * np.log((fin_hist + EPS) / (dev_hist + EPS))))
    else:
        psi = 0.0
    return {
        "feature": name,
        "drift_family": family,
        "development_support": len(dev),
        "final_support": len(fin),
        "development_mean": float(np.mean(dev)),
        "final_mean": float(np.mean(fin)),
        "standardized_mean_difference": float(standardized),
        "wasserstein_distance": float(wasserstein_distance(dev, fin)),
        "ks_statistic": float(ks_2samp(dev, fin).statistic),
        "psi": psi,
        "population_p_value_claim": False,
        "status": "AVAILABLE",
    }


def _normalized_structural_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    groups = frame.groupby(["source_key", "instrument"], sort=False, dropna=False)

    def relative_to_domain_median(values: pd.Series) -> pd.Series:
        median = values.groupby([frame["source_key"], frame["instrument"]]).transform("median")
        return values / median.replace(0, np.nan)

    output["spread_ticks"] = frame["spread"] / frame["tick_size"]
    output["relative_spread"] = frame["spread"] / frame["mid"].replace(0, np.nan)
    depth1 = frame["depth_bid_1"] + frame["depth_ask_1"]
    depth5 = frame["depth_bid_5"] + frame["depth_ask_5"]
    output["depth1_domain_median_ratio"] = relative_to_domain_median(depth1)
    output["depth5_domain_median_ratio"] = relative_to_domain_median(depth5)
    output["top1_share_of_top5_depth"] = depth1 / depth5.replace(0, np.nan)
    output["top1_bid_ask_depth_imbalance"] = (
        frame["depth_bid_1"] - frame["depth_ask_1"]
    ) / depth1.replace(0, np.nan)
    output["top5_bid_ask_depth_imbalance"] = (
        frame["depth_bid_5"] - frame["depth_ask_5"]
    ) / depth5.replace(0, np.nan)
    output["relative_volatility_bps"] = frame["realized_vol_50"] * 10_000.0
    output["event_intensity_domain_median_ratio"] = relative_to_domain_median(frame["event_rate"])
    log_interarrival = np.log1p(frame["interarrival_ns"].clip(lower=0))
    mean = groups["interarrival_ns"].transform(lambda values: np.log1p(values.clip(lower=0)).mean())
    std = groups["interarrival_ns"].transform(
        lambda values: max(float(np.log1p(values.clip(lower=0)).std()), EPS)
    )
    output["log_interarrival_domain_zscore"] = (log_interarrival - mean) / std
    l3 = _l3_mask(frame)
    output["queue_ahead_to_displayed_at_price"] = np.where(
        l3,
        0.5
        * (
            frame["bid_queue_ahead_at_entry"] / frame["bid_sz_01"].replace(0, np.nan)
            + frame["ask_queue_ahead_at_entry"] / frame["ask_sz_01"].replace(0, np.nan)
        ),
        np.nan,
    )
    output["queue_ahead_to_top5_depth"] = np.where(
        l3,
        0.5
        * (
            frame["bid_queue_ahead_at_entry"] / frame["depth_bid_5"].replace(0, np.nan)
            + frame["ask_queue_ahead_at_entry"] / frame["depth_ask_5"].replace(0, np.nan)
        ),
        np.nan,
    )
    return output


def _drift_and_ood(
    development: pd.DataFrame,
    final: pd.DataFrame,
    development_latent: np.ndarray,
    final_latent: np.ndarray,
) -> dict[str, Any]:
    features = [
        "spread",
        "depth_bid_1",
        "depth_ask_1",
        "depth_bid_5",
        "depth_ask_5",
        "imbalance_1",
        "signed_volume",
        "realized_vol_50",
        "event_rate",
        "interarrival_ns",
        "bid_queue_ahead_at_entry",
        "ask_queue_ahead_at_entry",
    ]
    rows = [
        _shift_metrics(feature, development[feature], final[feature], "RAW_UNIT_DRIFT")
        for feature in features
    ]
    drift = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)
    drift.to_csv(TABLE_ROOT / "feature_drift.csv", index=False)
    normalized_development = _normalized_structural_features(development)
    normalized_final = _normalized_structural_features(final)
    normalized = pd.DataFrame(
        [
            _shift_metrics(
                feature,
                normalized_development[feature],
                normalized_final[feature],
                "NORMALIZED_STRUCTURAL_DRIFT",
            )
            for feature in normalized_development.columns
        ]
    ).sort_values("ks_statistic", ascending=False)
    normalized.to_csv(TABLE_ROOT / "normalized_structural_drift.csv", index=False)
    _write_json(
        ARTIFACT_ROOT / "normalized_drift_semantics.json",
        {
            "label": "NORMALIZED_STRUCTURAL_DRIFT",
            "unit_policy": "dimensionless ratios, ticks, bps, or within-domain standardized values",
            "domain_normalization": "performed separately within source/instrument domain; uses features only and no outcomes",
            "population_p_values": False,
        },
    )
    mean = development_latent.mean(axis=0)
    covariance = np.cov(development_latent, rowvar=False)
    regularization = np.trace(covariance) / covariance.shape[0] * 0.1 + 1e-5
    inverse = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * regularization)
    dev_centered = development_latent - mean
    final_centered = final_latent - mean
    dev_distance = np.sqrt(np.einsum("ij,jk,ik->i", dev_centered, inverse, dev_centered))
    final_distance = np.sqrt(np.einsum("ij,jk,ik->i", final_centered, inverse, final_centered))
    edges = _bin_edges(dev_distance, 4)
    latent_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "partition": "DEVELOPMENT",
                    "distance": dev_distance,
                    "development_distance_bucket": _labels_from_edges(
                        pd.Series(dev_distance), edges
                    ),
                    "m3t_correct": development["m3t_direction_prediction"].to_numpy()
                    == development["direction_1000ms"].to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "partition": "FINAL_POSTHOC_ONLY",
                    "distance": final_distance,
                    "development_distance_bucket": _labels_from_edges(
                        pd.Series(final_distance), edges
                    ),
                    "m3t_correct": final["m3t_direction_prediction"].to_numpy()
                    == final["direction_1000ms"].to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )
    latent_table.to_parquet(RESULT_ROOT / "latent_ood_distance.parquet", index=False)
    latent_summary = (
        latent_table.groupby(["partition", "development_distance_bucket"], observed=True)
        .agg(
            support=("distance", "size"),
            mean_distance=("distance", "mean"),
            direction_accuracy=("m3t_correct", "mean"),
        )
        .reset_index()
    )
    latent_summary.to_csv(TABLE_ROOT / "latent_ood_performance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    top = drift.head(8).sort_values("ks_statistic")
    axes[0].barh(top["feature"], top["ks_statistic"])
    axes[0].set_title("Largest development-to-final feature shifts")
    for partition, group in latent_table.groupby("partition"):
        axes[1].hist(group["distance"], bins=20, alpha=0.55, label=partition)
    axes[1].set_title("Frozen M3T latent OOD distance")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "drift_and_latent_ood.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    normalized_top = normalized.head(8).sort_values("ks_statistic")
    ax.barh(normalized_top["feature"], normalized_top["ks_statistic"])
    ax.set_title("Normalized structural drift (descriptive KS)")
    ax.set_xlabel("KS statistic; no population p-value claim")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "normalized_structural_drift.png", dpi=160)
    plt.close(fig)
    return {
        "largest_raw_unit_shift": drift.iloc[0].to_dict(),
        "largest_normalized_structural_shift": normalized.iloc[0].to_dict(),
        "development_latent_distance_mean": float(dev_distance.mean()),
        "final_latent_distance_mean": float(final_distance.mean()),
        "detector_label": "DIAGNOSTIC_ONLY_FIT_ON_DEVELOPMENT",
    }


def _interpretability(
    development: pd.DataFrame,
    final: pd.DataFrame,
    development_latent: np.ndarray,
    final_latent: np.ndarray,
) -> dict[str, Any]:
    direction = xgb.Booster(model_file=str(ROOT / "artifacts/xgboost_v2/direction.json"))
    sample = development.iloc[:: max(1, len(development) // 128)]
    interactions = direction.predict(xgb.DMatrix(_matrix(sample)), pred_interactions=True)
    interaction_strength = np.mean(
        np.abs(interactions[..., :-1, :-1]), axis=tuple(range(interactions.ndim - 2))
    )
    interaction_strength = (
        interaction_strength.mean(axis=0)
        if interaction_strength.ndim == 3
        else interaction_strength
    )
    pairs = []
    for first in range(len(XGB_FEATURES)):
        for second in range(first + 1, len(XGB_FEATURES)):
            pairs.append(
                {
                    "feature_1": XGB_FEATURES[first],
                    "feature_2": XGB_FEATURES[second],
                    "mean_abs_tree_interaction": float(interaction_strength[first, second]),
                }
            )
    pd.DataFrame(pairs).sort_values("mean_abs_tree_interaction", ascending=False).head(50).to_csv(
        TABLE_ROOT / "xgboost_interactions.csv", index=False
    )
    slice_rows = []
    volatility_edges = _bin_edges(development["realized_vol_50"].to_numpy(), 2)
    for partition, frame in (("DEVELOPMENT", development), ("FINAL_POSTHOC_ONLY", final)):
        contributions = direction.predict(xgb.DMatrix(_matrix(frame)), pred_contribs=True)
        buckets = _labels_from_edges(frame["realized_vol_50"], volatility_edges)
        for bucket in buckets.dropna().unique():
            selected = buckets == bucket
            values = np.mean(
                np.abs(contributions[selected.to_numpy(), ..., :-1]),
                axis=tuple(range(contributions.ndim - 1)),
            )
            for feature, importance in zip(XGB_FEATURES, values, strict=True):
                slice_rows.append(
                    {
                        "partition": partition,
                        "development_volatility_bucket": str(bucket),
                        "support": int(selected.sum()),
                        "feature": feature,
                        "mean_abs_tree_shap": float(importance),
                    }
                )
    pd.DataFrame(slice_rows).to_csv(TABLE_ROOT / "xgboost_slice_shap.csv", index=False)
    probe = LogisticRegression(max_iter=1000, random_state=SEED).fit(
        development_latent, development["direction_1000ms"].to_numpy() + 1
    )
    probe_rows = [
        {
            "partition": "DEVELOPMENT",
            "macro_f1": float(
                f1_score(
                    development["direction_1000ms"] + 1,
                    probe.predict(development_latent),
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        {
            "partition": "FINAL_POSTHOC_ONLY",
            "macro_f1": float(
                f1_score(
                    final["direction_1000ms"] + 1,
                    probe.predict(final_latent),
                    average="macro",
                    zero_division=0,
                )
            ),
        },
    ]
    probe_table = pd.DataFrame(probe_rows)
    probe_table["label"] = "V3_CANDIDATE_DEVELOPMENT_FIT_LATENT_PROBE"
    probe_table.to_csv(TABLE_ROOT / "m3t_latent_linear_probe.csv", index=False)
    ladder = json.loads((data_root() / "manifests/model_ladder_v2.json").read_text())
    model = _model()
    model.load_state_dict(
        torch.load(
            ladder["branches"]["M3T-SSL-EPT"]["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )["model"]
    )
    embedding_rows = []
    for name, weights in (
        ("source", model.source_embedding.weight.detach().numpy()),
        ("instrument", model.instrument_embedding.weight.detach().numpy()),
    ):
        embedding_rows.append(
            {
                "embedding_family": name,
                "fitted_vectors": len(weights),
                "mean_norm": float(np.linalg.norm(weights, axis=1).mean()),
                "pairwise_distance": float(np.linalg.norm(weights[0] - weights[1])),
                "unseen_rule": "MEAN_OF_FITTED_EMBEDDINGS",
            }
        )
    pd.DataFrame(embedding_rows).to_csv(TABLE_ROOT / "m3t_embedding_diagnostics.csv", index=False)
    return {
        "tree_interactions_sample": len(sample),
        "latent_probe": probe_rows,
        "attention_causal_claim_made": False,
    }


def _robustness_surfaces(development: pd.DataFrame) -> dict[str, Any]:
    development = development.loc[_l3_mask(development)].copy()
    rows = []
    queue = development[["bid_queue_ahead_at_entry", "ask_queue_ahead_at_entry"]].mean(axis=1)
    queue_thresholds = [float(value) for value in np.quantile(queue.dropna(), [0.25, 0.5, 0.75])]
    registered_base = _registered_cost_series(development)
    registered_liquidation = _frozen_economic_assumptions()["liquidation_cost_ticks"]
    registered_non_liquidation = registered_base - registered_liquidation
    for model in ("m3t", "xgboost"):
        for additional_cost_ticks in (0.0, 0.25, 0.5, 1.0, 2.0):
            for liquidation_ticks in (0.0, 0.25, 0.5, 1.0):
                for confidence_threshold in (0.0, 0.4, 0.6, 0.8):
                    for edge_threshold in (0.0, 0.05, 0.10):
                        for queue_threshold in queue_thresholds:
                            for inventory_penalty in (0.0, 0.05, 0.10, 0.25):
                                expected = []
                                for side in ("bid", "ask"):
                                    probability = development[f"{model}_{side}_fill_frozen"]
                                    markout = development[f"{model}_{side}_markout_ticks"]
                                    confidence = development[f"{model}_confidence"]
                                    side_queue = development[f"{side}_queue_ahead_at_entry"]
                                    raw_edge = probability * (
                                        markout
                                        - registered_non_liquidation
                                        - liquidation_ticks
                                        - additional_cost_ticks
                                    )
                                    eligible = (
                                        (confidence >= confidence_threshold)
                                        & (raw_edge >= edge_threshold)
                                        & (side_queue <= queue_threshold)
                                    )
                                    expected.append(
                                        (raw_edge - inventory_penalty).where(eligible, 0.0)
                                    )
                                values = pd.concat(expected, ignore_index=True)
                                rows.append(
                                    {
                                        "label": "V3_CANDIDATE_DEVELOPMENT_ONLY_SURFACE",
                                        "model": model.upper(),
                                        "registered_non_liquidation_cost_ticks": float(
                                            registered_non_liquidation.iloc[0]
                                        ),
                                        "additional_cost_ticks": additional_cost_ticks,
                                        "liquidation_cost_ticks": liquidation_ticks,
                                        "confidence_threshold": confidence_threshold,
                                        "edge_threshold_ticks": edge_threshold,
                                        "maximum_queue_ahead": queue_threshold,
                                        "latency_ns": 1_000_000,
                                        "latency_variation": "UNSUPPORTED_WITHOUT_REPLAY",
                                        "inventory_penalty": inventory_penalty,
                                        "mean_expected_edge_ticks": float(values.mean()),
                                        "positive_edge_fraction": float((values > 0).mean()),
                                        "winner_selected": False,
                                    }
                                )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_ROOT / "robustness_surfaces.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    selected = table[
        (table["liquidation_cost_ticks"] == registered_liquidation)
        & (table["confidence_threshold"] == 0.0)
        & (table["edge_threshold_ticks"] == 0.0)
        & (table["inventory_penalty"] == 0.0)
        & (table["maximum_queue_ahead"] == queue_thresholds[1])
    ]
    for model, group in selected.groupby("model"):
        ax.plot(
            group["additional_cost_ticks"],
            group["mean_expected_edge_ticks"],
            marker="o",
            label=model,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        title="Development-only expected-edge cost surface",
        xlabel="Additional cost (ticks)",
        ylabel="Mean expected edge (ticks)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "development_robustness_surface.png", dpi=160)
    plt.close(fig)
    return {"rows": len(table), "final_used_for_parameter_exploration": False}


def _monitoring_scorecard(development: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    dev_probability = development[
        [f"m3t_direction_frozen_{name}" for name in ("down", "flat", "up")]
    ].to_numpy()
    final_probability = final[
        [f"m3t_direction_frozen_{name}" for name in ("down", "flat", "up")]
    ].to_numpy()
    dev_target = development["direction_1000ms"].to_numpy(np.int64) + 1
    final_target = final["direction_1000ms"].to_numpy(np.int64) + 1
    development_l3 = development.loc[_l3_mask(development)]
    final_l3 = final.loc[_l3_mask(final)]
    dev_fill = development_l3[["bid_fill", "ask_fill"]].mean(axis=1)
    final_fill = final_l3[["bid_fill", "ask_fill"]].mean(axis=1)
    dev_fill_probability = development_l3[["m3t_bid_fill_frozen", "m3t_ask_fill_frozen"]].mean(
        axis=1
    )
    final_fill_probability = final_l3[["m3t_bid_fill_frozen", "m3t_ask_fill_frozen"]].mean(axis=1)
    metrics = {
        "spread": (development["spread"], final["spread"], "data_health"),
        "top5_depth": (
            development["depth_bid_5"] + development["depth_ask_5"],
            final["depth_bid_5"] + final["depth_ask_5"],
            "data_health",
        ),
        "event_rate": (development["event_rate"], final["event_rate"], "data_health"),
        "missingness": (development.isna().mean(axis=1), final.isna().mean(axis=1), "data_health"),
        "stale_data_rate": (
            (development["interarrival_ns"] > development["interarrival_ns"].quantile(0.99)).astype(
                float
            ),
            (final["interarrival_ns"] > development["interarrival_ns"].quantile(0.99)).astype(
                float
            ),
            "data_health",
        ),
        "m3t_confidence": (development["m3t_confidence"], final["m3t_confidence"], "model"),
        "m3t_entropy": (development["m3t_entropy"], final["m3t_entropy"], "model"),
        "m3t_error": (
            (development["m3t_direction_prediction"] != development["direction_1000ms"]).astype(
                float
            ),
            (final["m3t_direction_prediction"] != final["direction_1000ms"]).astype(float),
            "model",
        ),
        "m3t_nll": (
            -np.log(np.clip(dev_probability[np.arange(len(dev_target)), dev_target], EPS, 1)),
            -np.log(np.clip(final_probability[np.arange(len(final_target)), final_target], EPS, 1)),
            "model",
        ),
        "m3t_multiclass_brier": (
            np.sum((dev_probability - np.eye(3)[dev_target]) ** 2, axis=1),
            np.sum((final_probability - np.eye(3)[final_target]) ** 2, axis=1),
            "model",
        ),
        "fill_calibration_gap": (
            dev_fill_probability - dev_fill,
            final_fill_probability - final_fill,
            "model",
        ),
        "fill_rate": (
            dev_fill,
            final_fill,
            "execution",
        ),
        "queue_ahead": (
            development_l3[["bid_queue_ahead_at_entry", "ask_queue_ahead_at_entry"]].mean(axis=1),
            final_l3[["bid_queue_ahead_at_entry", "ask_queue_ahead_at_entry"]].mean(axis=1),
            "execution",
        ),
        "adverse_selection": (
            _adverse_error_observations(development)["adverse_target"],
            _adverse_error_observations(final)["adverse_target"],
            "execution",
        ),
        "post_fill_markout": (
            pd.concat(
                [
                    development.loc[
                        _available_mask(development, side, "post_fill_markout"),
                        f"{side}_fill_conditional_markout_100ms",
                    ]
                    for side in ("bid", "ask")
                ],
                ignore_index=True,
            ),
            pd.concat(
                [
                    final.loc[
                        _available_mask(final, side, "post_fill_markout"),
                        f"{side}_fill_conditional_markout_100ms",
                    ]
                    for side in ("bid", "ask")
                ],
                ignore_index=True,
            ),
            "execution",
        ),
        "time_to_fill_ns": (
            pd.concat(
                [
                    development.loc[
                        _available_mask(development, side, "time_to_fill"),
                        f"{side}_time_to_first_fill_ns",
                    ]
                    for side in ("bid", "ask")
                ],
                ignore_index=True,
            ),
            pd.concat(
                [
                    final.loc[
                        _available_mask(final, side, "time_to_fill"),
                        f"{side}_time_to_first_fill_ns",
                    ]
                    for side in ("bid", "ask")
                ],
                ignore_index=True,
            ),
            "execution",
        ),
    }
    economics = pd.read_parquet(RESULT_ROOT / "economic_attribution_detail.parquet")
    for name, column in (
        ("pnl", "net_pnl_usd"),
        ("fees", "fees_usd"),
        ("turnover", "simulated_l3_fill"),
    ):
        metrics[name] = (
            economics.loc[economics["partition"] == "DEVELOPMENT", column].astype(float),
            economics.loc[economics["partition"] == "FINAL_POSTHOC_ONLY", column].astype(float),
            "risk_economics",
        )
    metrics["inventory"] = (pd.Series([0.0]), pd.Series([0.0]), "risk_economics")
    drawdowns = {}
    for partition in ("DEVELOPMENT", "FINAL_POSTHOC_ONLY"):
        pnl = economics.loc[economics["partition"] == partition, "net_pnl_usd"].to_numpy(float)
        nav = np.cumsum(pnl)
        drawdowns[partition] = float(np.max(np.maximum.accumulate(nav) - nav)) if len(nav) else 0.0
    metrics["drawdown"] = (
        pd.Series([drawdowns["DEVELOPMENT"]]),
        pd.Series([drawdowns["FINAL_POSTHOC_ONLY"]]),
        "risk_economics",
    )
    rows = []
    rank = {"GREEN": 0, "WATCH": 1, "REVIEW": 2, "DISABLE": 3}
    for name, (dev, fin, category) in metrics.items():
        dev = pd.Series(dev).replace([np.inf, -np.inf], np.nan).dropna()
        fin = pd.Series(fin).replace([np.inf, -np.inf], np.nan).dropna()
        if len(dev) < MIN_SLICE_SUPPORT or len(fin) < MIN_SLICE_SUPPORT:
            rows.append(
                {
                    "category": category,
                    "metric": name,
                    "development_support": len(dev),
                    "final_support": len(fin),
                    "development_median": None,
                    "development_iqr": None,
                    "final_median": None,
                    "normalized_shift": None,
                    "status": "NOT_AVAILABLE",
                    "threshold_origin": "INSUFFICIENT_VALID_AVAILABLE_TARGET_SUPPORT",
                    "production_deployment_implied": False,
                }
            )
            continue
        median = float(dev.median())
        scale = float(max(dev.quantile(0.75) - dev.quantile(0.25), EPS))
        shift = abs(float(fin.median()) - median) / scale
        status = (
            "GREEN"
            if shift <= 1
            else "WATCH"
            if shift <= 2
            else "REVIEW"
            if shift <= 4
            else "DISABLE"
        )
        rows.append(
            {
                "category": category,
                "metric": name,
                "development_support": len(dev),
                "final_support": len(fin),
                "development_median": median,
                "development_iqr": scale,
                "final_median": float(fin.median()),
                "normalized_shift": shift,
                "status": status,
                "threshold_origin": "DEVELOPMENT_IQR_HEURISTIC",
                "production_deployment_implied": False,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_ROOT / "monitoring_scorecard.csv", index=False)
    available_statuses = [value for value in table["status"] if value in rank]
    overall = (
        max(available_statuses, key=lambda value: rank[value])
        if available_statuses
        else "NOT_AVAILABLE"
    )
    return {"overall_status": overall, "rows": len(table), "actual_production": False}


def _table_markdown(frame: pd.DataFrame, columns: list[str], limit: int = 12) -> str:
    shown = frame.loc[:, [column for column in columns if column in frame]].head(limit).copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.5g}" if pd.notna(value) else "NA")
    headers = [str(column) for column in shown.columns]
    rows = [[str(value) for value in row] for row in shown.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(name: str, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Analysis layer: `{LABEL}` — **NON-AUTHORITATIVE / POST-HOC ONLY**.",
        "",
        "The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.",
        "",
    ]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body, ""])
    (REPORT_ROOT / name).write_text("\n".join(lines), encoding="utf-8")


def _reports(summaries: dict[str, Any]) -> dict[str, Any]:
    calibration = pd.read_csv(TABLE_ROOT / "calibration_metrics.csv")
    slices = pd.read_csv(TABLE_ROOT / "operational_slices.csv")
    execution_slices = pd.read_csv(TABLE_ROOT / "execution_behavior_slices.csv")
    cohorts = pd.read_csv(TABLE_ROOT / "error_cohorts.csv")
    disagreement = pd.read_csv(TABLE_ROOT / "direction_disagreement.csv")
    markouts = pd.read_csv(TABLE_ROOT / "markout_curves.csv")
    queue = pd.read_csv(TABLE_ROOT / "queue_fill_calibration.csv")
    economics = pd.read_csv(TABLE_ROOT / "economic_attribution.csv")
    reconciliation = pd.read_csv(TABLE_ROOT / "authoritative_economic_reconciliation.csv")
    edge = pd.read_csv(TABLE_ROOT / "predicted_vs_realized_edge.csv")
    drift = pd.read_csv(TABLE_ROOT / "feature_drift.csv")
    normalized_drift = pd.read_csv(TABLE_ROOT / "normalized_structural_drift.csv")
    confidence_monotonicity = pd.read_csv(TABLE_ROOT / "confidence_monotonicity.csv")
    shap = pd.read_csv(TABLE_ROOT / "xgboost_tree_shap.csv")
    occlusion = pd.read_csv(TABLE_ROOT / "m3t_occlusion.csv")
    robustness = pd.read_csv(TABLE_ROOT / "robustness_surfaces.csv")
    monitoring = pd.read_csv(TABLE_ROOT / "monitoring_scorecard.csv")

    final_cal = calibration[
        (calibration["partition"] == "FINAL_POSTHOC_ONLY")
        & (calibration["variant"] == "frozen")
        & calibration["task"].isin(["bid_fill", "ask_fill"])
    ]
    fill_pivot = final_cal.pivot(index="task", columns="model", values="brier")
    strongest_cal = (
        f"Frozen XGBoost had lower final post-hoc total fill Brier on bid and ask: "
        f"bid {fill_pivot.loc['bid_fill', 'XGBOOST']:.4f} vs M3T {fill_pivot.loc['bid_fill', 'M3T']:.4f}; "
        f"ask {fill_pivot.loc['ask_fill', 'XGBOOST']:.4f} vs M3T {fill_pivot.loc['ask_fill', 'M3T']:.4f}. "
        "This total score is not, by itself, proof of better calibration."
    )
    final_valid = slices[
        (slices["partition"] == "FINAL_POSTHOC_ONLY") & slices["sufficient_support"]
    ]
    comparison = final_valid.pivot_table(
        index=["slice_dimension", "slice_value"], columns="model", values="macro_f1"
    ).dropna()
    comparison["M3T_minus_XGB"] = comparison["M3T"] - comparison["XGBOOST"]
    comparison_display = comparison.reset_index().merge(
        final_valid.groupby(["slice_dimension", "slice_value"], as_index=False)["support"].min(),
        on=["slice_dimension", "slice_value"],
        how="left",
    )
    comparison_display["slice_definition_source"] = "DEVELOPMENT_FIXED_OR_PREDEFINED"
    comparison_display["interpretation"] = "EXPLORATORY_POSTHOC_ONLY"
    best_m3t = comparison.sort_values("M3T_minus_XGB", ascending=False).iloc[0]
    best_xgb = comparison.sort_values("M3T_minus_XGB").iloc[0]
    best_m3t_name = comparison.sort_values("M3T_minus_XGB", ascending=False).index[0]
    best_xgb_name = comparison.sort_values("M3T_minus_XGB").index[0]
    best_m3t_support = int(
        final_valid[
            (final_valid["slice_dimension"] == best_m3t_name[0])
            & (final_valid["slice_value"].astype(str) == str(best_m3t_name[1]))
        ]["support"].min()
    )
    best_xgb_support = int(
        final_valid[
            (final_valid["slice_dimension"] == best_xgb_name[0])
            & (final_valid["slice_value"].astype(str) == str(best_xgb_name[1]))
        ]["support"].min()
    )
    economics_final = economics[economics["partition"] == "FINAL_POSTHOC_ONLY"]
    raw_drift_top = drift.iloc[0]
    normalized_drift_top = normalized_drift.iloc[0]
    confidence_final = summaries["confidence_monotonicity"]
    queue_failure = summaries["markout_queue"]["largest_xgboost_fill_advantage_state"]
    development_positive = execution_slices[
        (execution_slices["partition"] == "DEVELOPMENT")
        & execution_slices["sufficient_support"].astype(bool)
        & (execution_slices["diagnostic_net_pnl_usd"] > 0)
    ].sort_values("diagnostic_net_pnl_usd", ascending=False)
    positive_slice = development_positive.iloc[0].to_dict() if len(development_positive) else None
    brier_components = summaries["calibration"]["fill_brier_advantage_decomposition"]
    direction_final = calibration[
        (calibration["partition"] == "FINAL_POSTHOC_ONLY")
        & (calibration["task"] == "direction_multiclass")
        & (calibration["variant"] == "frozen")
    ].set_index("model")
    m3t_economics = economics_final[economics_final["model"] == "M3T"].iloc[0]
    xgb_economics = economics_final[economics_final["model"] == "XGBOOST"].iloc[0]
    positive_slice_text = (
        f"Yes: `{positive_slice['slice_dimension']}={positive_slice['slice_value']}` for "
        f"{positive_slice['model']} had ${positive_slice['diagnostic_net_pnl_usd']:.4f} "
        f"diagnostic net PnL on {int(positive_slice['support'])} development events. It remains "
        "a V3_CANDIDATE because it is a multiple-slice discovery, not an independently registered strategy."
        if positive_slice
        else "No sufficiently supported development-defined slice had positive diagnostic economics."
    )
    executive_questions = "\n".join(
        [
            f"1. **Why did M3T lead directionally?** M3T final macro-F1 was {direction_final.loc['M3T', 'macro_f1']:.4f} versus {direction_final.loc['XGBOOST', 'macro_f1']:.4f}. The descriptive disagreement/occlusion evidence is consistent with transfer from sequential state/event representations; one unseen day cannot establish a causal mechanism.",
            "2. **Why did XGBoost lead fill Brier?** Explicit queue/depth features aligned better with the counterfactual fill process; total bid/ask Brier was lower. XGBoost's adverse input is only a deterministic markout-derived score, not a learned adverse classifier.",
            f"3. **Reliability, resolution, or both?** Bid: `{brier_components['bid']['interpretation']}`; ask: `{brier_components['ask']['interpretation']}`. The full reliability, resolution, uncertainty, and binning residuals are reported separately.",
            f"4. **Is confidence monotonic/useful?** M3T is `{confidence_final['M3T']['FINAL_POSTHOC_ONLY']['classification']}` and XGBoost is `{confidence_final['XGBOOST']['FINAL_POSTHOC_ONLY']['classification']}` on development-fixed final quartiles. No final threshold was tuned.",
            f"5. **Largest normalized shift?** `{normalized_drift_top['feature']}` with descriptive KS {normalized_drift_top['ks_statistic']:.4f}. Raw-unit `{raw_drift_top['feature']}` KS {raw_drift_top['ks_statistic']:.4f} is reported separately and is not called structural OOD.",
            f"6. **Largest M3T queue/fill failure?** `{queue_failure}` is the largest observed XGBoost Brier-contribution advantage among predefined L3 execution buckets.",
            f"7. **Why did after-cost economics fail?** M3T spread capture ${m3t_economics['spread_capture_usd']:.4f} was overcome by subsequent markout movement ${m3t_economics['subsequent_markout_price_movement_usd']:.4f}, leaving gross ${m3t_economics['total_gross_pnl_usd']:.4f}; XGBoost was ${xgb_economics['spread_capture_usd']:.4f}, ${xgb_economics['subsequent_markout_price_movement_usd']:.4f}, and ${xgb_economics['total_gross_pnl_usd']:.4f}, respectively. Registered costs then deepened both losses.",
            f"8. **Costs or adverse markouts?** M3T fees plus liquidation were ${m3t_economics['fees_usd'] + m3t_economics['liquidation_cost_usd']:.4f} versus ${abs(m3t_economics['subsequent_markout_price_movement_usd']):.4f} of adverse subsequent movement, so costs were slightly larger. XGBoost costs were ${xgb_economics['fees_usd'] + xgb_economics['liquidation_cost_usd']:.4f} versus ${abs(xgb_economics['subsequent_markout_price_movement_usd']):.4f}, so adverse movement was slightly larger. Neither mechanism alone explains both models.",
            f"9. **Any positive development-defined slice?** {positive_slice_text}",
            "10. **Highest-priority V3 experiment?** Acquire multiple independent MBO instrument-days, preregister the comparison, and validate a queue-specialized hybrid M3T/XGBoost fill head with cross-fitted calibration.",
        ]
    )

    _write_report(
        "POSTHOC_EXECUTIVE_SUMMARY.md",
        "MarketForge-QTR v2 Post-hoc Executive Summary",
        [
            (
                "Governance",
                "The v2 lockbox remained at `evaluation_count = 1`. This layer changes no authoritative claim or artifact.",
            ),
            (
                "Strongest calibration finding",
                strongest_cal
                + " Brier is not a pure calibration metric; reliability, resolution, and uncertainty are reported separately.",
            ),
            (
                "Largest observed M3T advantage among predefined post-hoc slices",
                f"M3T's largest observed advantage was `{best_m3t_name[0]}={best_m3t_name[1]}` with support {best_m3t_support}, M3T {best_m3t['M3T']:.4f}, XGBoost {best_m3t['XGBOOST']:.4f}, difference {best_m3t['M3T_minus_XGB']:.4f}. XGBoost's largest observed advantage was `{best_xgb_name[0]}={best_xgb_name[1]}` with support {best_xgb_support}, M3T {best_xgb['M3T']:.4f}, XGBoost {best_xgb['XGBOOST']:.4f}, M3T-minus-XGBoost {best_xgb['M3T_minus_XGB']:.4f}. Definitions were fixed from development where applicable; all findings are `EXPLORATORY_POSTHOC_ONLY`, correlated across related slices, and carry no event-row p-values.",
            ),
            (
                "Economic mechanism",
                "Both active diagnostic controllers remained after-cost negative while the frozen authoritative policy remained NO_TRADE. Transaction costs plus adverse/weak post-fill edge prevented predictive gains from becoming an economic edge.",
            ),
            (
                "Raw-unit and normalized OOD findings",
                f"Raw-unit `{raw_drift_top['feature']}` had KS {raw_drift_top['ks_statistic']:.4f}; normalized structural `{normalized_drift_top['feature']}` had KS {normalized_drift_top['ks_statistic']:.4f}. Only the latter is used as the structural headline. Neither carries a population p-value claim.",
            ),
            ("Ten executive questions", executive_questions),
            (
                "V3 priority",
                "Acquire multiple independent MBO instrument-days, then cross-fit side/queue-aware fill calibration and a hybrid representation-plus-queue model. It is a `V3_CANDIDATE`, not a v2 revision.",
            ),
            (
                "Primary limitation",
                "Only one final and two development instrument-day units exist; event counts are not independent population sample size.",
            ),
        ],
    )
    _write_report(
        "CALIBRATION_DEEP_DIVE.md",
        "Calibration Deep Dive",
        [
            ("Finding", strongest_cal),
            (
                "Metrics",
                _table_markdown(
                    final_cal,
                    [
                        "model",
                        "task",
                        "support",
                        "positive",
                        "negative",
                        "brier",
                        "log_loss",
                        "ece",
                        "mce",
                        "adaptive_ece",
                        "calibration_slope",
                        "calibration_intercept",
                        "brier_reliability",
                        "brier_resolution",
                        "brier_uncertainty",
                        "brier_decomposition_residual",
                        "brier_decomposition_within_expected_binning_residual",
                    ],
                ),
            ),
            (
                "Interpretation",
                f"Raw and frozen M3T curves are reported on development; only the already-frozen calibration is evaluated on final. XGBoost has no separate frozen calibrator, so its frozen entry is its trained probability. Brier includes reliability, resolution, and uncertainty and is not a pure calibration metric. Bid decomposition: `{brier_components['bid']['interpretation']}`; ask: `{brier_components['ask']['interpretation']}`. The identity Brier ≈ reliability - resolution + uncertainty passes the registered binning-residual tolerance for every fill row. No final-data calibrator was fitted.",
            ),
        ],
    )
    _write_report(
        "SLICE_ANALYSIS.md",
        "Operational Slice Analysis",
        [
            (
                "Scope",
                f"{len(slices)} direction/calibration rows and {len(execution_slices)} side-long execution/economic rows were produced. Every row contains support; binary fill rows contain positive and negative counts. Support below {MIN_SLICE_SUPPORT} is flagged insufficient.",
            ),
            (
                "Largest supported differences",
                _table_markdown(
                    comparison_display.sort_values("M3T_minus_XGB", ascending=False),
                    [
                        "slice_dimension",
                        "slice_value",
                        "support",
                        "M3T",
                        "XGBOOST",
                        "M3T_minus_XGB",
                        "slice_definition_source",
                        "interpretation",
                    ],
                ),
            ),
            (
                "Discipline",
                "Final slices are exploratory/post-hoc only. The companion execution table includes fill/no-fill, side, time-to-fill, predicted fill/adverse/markout/edge, disagreement, gross edge, costs, and diagnostic net PnL. Repeated timestamps remain present; the timestamp-group weighting sensitivity is explicitly exploratory and does not replace row-preserving authoritative metrics.",
            ),
        ],
    )
    _write_report(
        "ERROR_COHORT_ANALYSIS.md",
        "Development-discovered Error Cohorts",
        [
            (
                "Method",
                "Depth-3 regression trees with minimum 30 development observations per leaf discovered direction, fill, adverse-selection, and economic-loss cohorts. Fixed rules were then applied to final descriptively; no rule was optimized on final.",
            ),
            (
                "Cohorts",
                _table_markdown(
                    cohorts.sort_values("development_mean_error", ascending=False),
                    [
                        "task",
                        "cohort_leaf",
                        "development_support",
                        "development_mean_error",
                        "final_support",
                        "final_application",
                    ],
                ),
            ),
        ],
    )
    _write_report(
        "MODEL_DISAGREEMENT.md",
        "M3T versus XGBoost Disagreement",
        [
            (
                "Direction groups",
                _table_markdown(
                    disagreement,
                    [
                        "partition",
                        "group",
                        "support",
                        "spread_mean",
                        "volatility_mean",
                        "absolute_imbalance_mean",
                        "event_intensity_mean",
                        "top5_depth_mean",
                        "queue_ahead_mean",
                    ],
                ),
            ),
            (
                "Conclusion",
                f"The largest observed M3T advantage among predefined post-hoc slices was `{best_m3t_name[0]}={best_m3t_name[1]}` (support {best_m3t_support}). The analogous XGBoost observation was `{best_xgb_name[0]}={best_xgb_name[1]}` (support {best_xgb_support}). These correlated exploratory slices are not independent discoveries, carry no event-row p-values, and do not select a hybrid or alter v2.",
            ),
        ],
    )
    _write_report(
        "MARKOUT_TCA_ANALYSIS.md",
        "Multi-horizon Markout and TCA Analysis",
        [
            (
                "Availability",
                "Existing frozen labels support side-consistent 100 ms and 1 s quote markouts and 100 ms post-fill markouts. 10/25/50/250/500 ms and 5 s, plus event-count horizons, are unavailable without replay and were not invented.",
            ),
            (
                "Selected curves",
                _table_markdown(
                    markouts,
                    [
                        "partition",
                        "model",
                        "side",
                        "markout_type",
                        "horizon",
                        "slice_dimension",
                        "slice_value",
                        "support",
                        "mean",
                        "median",
                        "q10",
                        "q90",
                    ],
                ),
            ),
            (
                "Semantics",
                "Quote-time forward markouts are explicitly `QUOTE_MARKOUT_100MS` or `QUOTE_MARKOUT_1000MS`; they are not fills. Only fill-conditioned replay outcomes are `SIMULATED_L3_POST_FILL_MARKOUT_100MS`. Simulated fills are never described as observed, and the small-order/no-endogenous-impact limitation remains.",
            ),
        ],
    )
    _write_report(
        "QUEUE_FILL_ANALYSIS.md",
        "Queue and Fill Diagnostics",
        [
            (
                "Conditional fill behavior",
                _table_markdown(
                    queue,
                    [
                        "partition",
                        "model",
                        "side",
                        "condition",
                        "development_bucket",
                        "support",
                        "positive",
                        "negative",
                        "simulated_l3_fill_rate",
                        "predicted_fill_probability",
                        "calibration_residual",
                        "mean_brier_contribution",
                        "time_to_fill_support",
                        "time_to_fill_q25_ns",
                        "median_time_to_fill_ns",
                        "time_to_fill_q75_ns",
                    ],
                ),
            ),
            (
                "Validation status",
                f"Every row is L3-valid. Historical resting-order evidence remains calibration/error evidence only. Counterfactual fills remain `SIMULATED_L3`; no perfection or observed-fill claim is made. Largest observed XGBoost Brier advantage state: `{queue_failure}`.",
            ),
        ],
    )
    _write_report(
        "ECONOMIC_ATTRIBUTION.md",
        "Economic Attribution",
        [
            (
                "Frozen policy",
                "`NO_TRADE` remains the authoritative frozen policy with zero trading PnL. Active results below are diagnostics and cannot become a selected strategy.",
            ),
            (
                "Attribution",
                _table_markdown(
                    economics_final,
                    [
                        "model",
                        "quotes",
                        "fills",
                        "gross_spread_price_edge_usd",
                        "spread_capture_usd",
                        "subsequent_markout_price_movement_usd",
                        "adverse_selection_component_usd",
                        "fees_usd",
                        "liquidation_cost_usd",
                        "slippage_usd",
                        "hedge_cost_usd",
                        "realized_pnl_usd",
                        "net_pnl_usd",
                    ],
                ),
            ),
            (
                "Authoritative reconciliation",
                _table_markdown(
                    reconciliation,
                    [
                        "model",
                        "row_type",
                        "quotes",
                        "fills",
                        "gross_price_markout_pnl_usd",
                        "fees_usd",
                        "liquidation_cost_usd",
                        "slippage_usd",
                        "hedge_cost_usd",
                        "net_pnl_usd",
                        "reconciliation_status",
                    ],
                ),
            ),
            (
                "Accounting identity",
                "Passive spread capture is fill-to-contemporaneous-mid. Subsequent markout/price movement is contemporaneous-mid-to-future-mid. Their sum equals total gross fill-to-future-mid PnL; spread capture is diagnostic and is never added twice. Net PnL = total gross PnL - fees - liquidation - slippage - hedge cost.",
            ),
            (
                "Expected versus realized edge",
                _table_markdown(
                    edge[edge["partition"] == "FINAL_POSTHOC_ONLY"],
                    [
                        "model",
                        "side",
                        "development_edge_bucket",
                        "support",
                        "predicted_edge_mean_ticks",
                        "realized_edge_mean_ticks",
                        "signed_bias_ticks",
                        "mae_ticks",
                        "spearman_rank",
                    ],
                ),
            ),
        ],
    )
    _write_report(
        "DRIFT_OOD_ANALYSIS.md",
        "Drift and OOD Analysis",
        [
            (
                "Raw-unit drift",
                _table_markdown(
                    drift,
                    [
                        "feature",
                        "development_support",
                        "final_support",
                        "standardized_mean_difference",
                        "wasserstein_distance",
                        "ks_statistic",
                        "psi",
                        "population_p_value_claim",
                    ],
                ),
            ),
            (
                "Normalized structural drift",
                _table_markdown(
                    normalized_drift,
                    [
                        "feature",
                        "development_support",
                        "final_support",
                        "standardized_mean_difference",
                        "wasserstein_distance",
                        "ks_statistic",
                        "psi",
                        "population_p_value_claim",
                    ],
                ),
            ),
            (
                "Latent distance",
                f"The M3T distance reference was fitted on development only. Mean development distance was {summaries['drift']['development_latent_distance_mean']:.4f}; mean final distance was {summaries['drift']['final_latent_distance_mean']:.4f}. This detector is diagnostic only.",
            ),
        ],
    )
    _write_report(
        "INTERPRETABILITY.md",
        "Model Interpretability Diagnostics",
        [
            (
                "XGBoost",
                _table_markdown(
                    shap.sort_values("mean_abs_tree_shap", ascending=False),
                    ["partition", "feature", "mean_abs_tree_shap"],
                ),
            ),
            (
                "M3T",
                _table_markdown(
                    occlusion,
                    ["partition", "feature_group", "mean_absolute_probability_impact", "batches"],
                ),
            ),
            (
                "Limits",
                "TreeSHAP describes the fitted trees. XGBoost fill probabilities are learned classifier outputs, while its adverse controller input is a deterministic `markout_derived_adverse_score`, not an independently trained adverse probability. M3T adverse outputs are learned, uncalibrated probabilities. Stream/depth occlusion measures prediction sensitivity, not causality. The development-fitted latent probe is a `V3_CANDIDATE`; attention maps are not causal explanations.",
            ),
        ],
    )
    _write_report(
        "ROBUSTNESS_SURFACES.md",
        "Development-only Robustness Surfaces",
        [
            (
                "Scope",
                "Cost, liquidation cost, and confidence surfaces use development predictions only and are labeled `V3_CANDIDATE`. Existing labels fix entry latency at 1 ms; alternative latency surfaces are unavailable without replay. No final winner was selected.",
            ),
            (
                "Surface sample",
                _table_markdown(
                    robustness,
                    [
                        "model",
                        "registered_non_liquidation_cost_ticks",
                        "additional_cost_ticks",
                        "liquidation_cost_ticks",
                        "confidence_threshold",
                        "edge_threshold_ticks",
                        "maximum_queue_ahead",
                        "inventory_penalty",
                        "latency_ns",
                        "latency_variation",
                        "mean_expected_edge_ticks",
                        "positive_edge_fraction",
                        "winner_selected",
                    ],
                ),
            ),
        ],
    )
    _write_report(
        "MONITORING_SCORECARD.md",
        "Production-style Monitoring Scorecard",
        [
            (
                "Not deployment",
                "This is a post-hoc production-style scorecard; it does not imply live deployment. Thresholds use a transparent development-IQR heuristic.",
            ),
            (
                "Scorecard",
                _table_markdown(
                    monitoring,
                    [
                        "category",
                        "metric",
                        "development_support",
                        "final_support",
                        "development_median",
                        "development_iqr",
                        "final_median",
                        "normalized_shift",
                        "status",
                        "threshold_origin",
                    ],
                ),
            ),
            (
                "Statuses",
                f"GREEN = within 1 development IQR; WATCH = 1–2; REVIEW = 2–4; DISABLE = above 4. Metrics with fewer than {MIN_SLICE_SUPPORT} valid available observations in either domain are `NOT_AVAILABLE`. Fill, queue, adverse, post-fill markout, and time-to-fill metrics use only L3/side-available targets. These are heuristic diagnostics.",
            ),
        ],
    )
    _write_report(
        "CONFIDENCE_MONOTONICITY.md",
        "Confidence Monotonicity Diagnostics",
        [
            (
                "Development-fixed quartiles",
                _table_markdown(
                    confidence_monotonicity,
                    [
                        "partition",
                        "model",
                        "development_confidence_quartile",
                        "support",
                        "accuracy",
                        "macro_f1",
                        "nll",
                        "error_rate",
                        "fill_support",
                        "fill_brier",
                        "post_fill_markout_support",
                        "mean_post_fill_markout",
                        "slice_definition_source",
                        "interpretation",
                    ],
                    limit=20,
                ),
            ),
            (
                "Conclusion",
                f"M3T final confidence ranking: `{confidence_final['M3T']['FINAL_POSTHOC_ONLY']['classification']}` with Spearman {confidence_final['M3T']['FINAL_POSTHOC_ONLY']['spearman_confidence_rank_vs_accuracy']}; XGBoost: `{confidence_final['XGBOOST']['FINAL_POSTHOC_ONLY']['classification']}` with Spearman {confidence_final['XGBOOST']['FINAL_POSTHOC_ONLY']['spearman_confidence_rank_vs_accuracy']}. Monotonicity violations and risk-coverage results are descriptive; no final threshold was fit or selected.",
            ),
        ],
    )
    _write_report(
        "V3_CANDIDATES.md",
        "V3 Candidate Backlog",
        [
            (
                "Governance",
                "Every item is explicitly `V3_CANDIDATE`, non-authoritative, and was not applied to v2.",
            ),
            (
                "Ranked backlog",
                "| Rank | Candidate | Evidence | Expected impact | Effort | New data | Overfit risk |\n|---:|---|---|---|---|---|---|\n| 1 | `V3_CANDIDATE`: acquire multiple MBO instrument-days and rerun independent-day validation | Primary-unit limitation dominates all inference | High | High | Required | Low |\n| 2 | `V3_CANDIDATE`: queue-specialized XGBoost/M3T hybrid fill head | XGBoost final fill Brier is lower while M3T direction is stronger | High | Medium | Helpful | Medium |\n| 3 | `V3_CANDIDATE`: cross-fitted beta/vector or isotonic fill calibration | Reliability residuals vary by side and queue | Medium | Medium | Helpful | High |\n| 4 | `V3_CANDIDATE`: queue-aware SSL objective | Queue/depth cohorts expose execution errors | Medium | High | Required | Medium |\n| 5 | `V3_CANDIDATE`: pre-registered slice-specific controller | Some development-derived cohorts differ materially | Medium | Medium | Required | Very high |\n| 6 | `V3_CANDIDATE`: richer legal 10 ms–5 s labels | Current frozen labels expose only 100 ms/1 s | Medium | High | Required | Low |",
            ),
        ],
    )
    _write_report(
        "POSTHOC_LIMITATIONS.md",
        "Post-hoc Limitations",
        [
            (
                "Inference",
                "There are only two development and one final instrument-day units. Event rows are dependent and are not population sample size. No statistical significance across markets, White/SPA/PBO/DSR resurrection, or population p-value claim is made.",
            ),
            (
                "Post-hoc bias",
                "Final slicing is exploratory and may identify unstable patterns. Within-day support counts quantify events, not independent days. Development-derived thresholds and cohorts remain candidates for a separately preregistered v3.",
            ),
            (
                "Execution",
                "Fills are `SIMULATED_L3` under small-order/no-endogenous-impact assumptions. Existing horizons and latency are limited. Counterfactual execution is not observed execution.",
            ),
            (
                "Authority",
                "Nothing here changes v2 models, calibration, controller, no-trade choice, claims, resume evidence, or evaluation count.",
            ),
        ],
    )
    _write_report(
        "PROVENANCE_ERRATA.md",
        "Post-hoc Provenance Errata",
        [
            (
                "Observed chronology inconsistency",
                "The immutable supporting artifact `artifacts/final_evaluation_v2_idempotency.json` records `verified_at_utc = 2026-09-27T18:42:00Z`, while amendment 002 and 003 embed later seal times (`18:45:00Z` and `18:50:00Z`). The final evaluation embeds `18:40:12.756441+00:00`. This makes the supporting timestamp inconsistent with the full transcript/amendment narrative if read as a total ordering.",
            ),
            (
                "Controlling provenance evidence",
                "Execution transcript ordering, immutable SHA-256 identities, filesystem mtimes, the amendment chain, and the final result identity are the meaningful evidence. The idempotency artifact still binds identical before/after final hashes and `evaluation_count = 1`. Embedded timestamps and mtimes do not form one internally consistent clock sequence.",
            ),
            (
                "Disposition",
                "No authoritative file was rewritten, and the inconsistency does not alter data, predictions, metrics, economics, or scientific results. Examination did not establish a benign clock-skew or manual-recording explanation, so none is asserted. This is a post-hoc provenance erratum only.",
            ),
        ],
    )
    return {
        "strongest_calibration_finding": strongest_cal,
        "largest_observed_m3t_advantage_among_predefined_posthoc_slices": (
            f"{best_m3t_name}; support={best_m3t_support}; M3T-minus-XGBoost macro-F1 "
            f"{best_m3t['M3T_minus_XGB']:.4f}; EXPLORATORY_POSTHOC_ONLY"
        ),
        "where_m3t_beats_xgboost": str(best_m3t_name),
        "where_xgboost_beats_m3t": str(best_xgb_name),
        "economic_failure_reason": "weak/adverse post-fill edge plus fees and liquidation costs",
        "largest_raw_unit_drift": (
            f"{raw_drift_top['feature']} KS={raw_drift_top['ks_statistic']:.4f}"
        ),
        "largest_normalized_structural_drift": (
            f"{normalized_drift_top['feature']} KS={normalized_drift_top['ks_statistic']:.4f}"
        ),
        "confidence_monotonicity": confidence_final,
        "largest_xgboost_fill_advantage_state": queue_failure,
        "authoritative_economic_reconciliation": "MATCH",
        "strongest_v3_candidate": "acquire multiple independent MBO instrument-days, then validate a queue-specialized hybrid",
        "remaining_data_limitation": "two development and one final instrument-day primary units",
    }


def _posthoc_audit() -> Path:
    charter = json.loads((ARTIFACT_ROOT / "POSTHOC_CHARTER.json").read_text(encoding="utf-8"))
    current = protected_inventory()
    reports = sorted(REPORT_ROOT.glob("*.md"))
    final = json.loads((ROOT / "artifacts/final_evaluation_v2.json").read_text(encoding="utf-8"))
    resume = (ROOT / "reports/final_v2/RESUME_EVIDENCE.md").read_text(encoding="utf-8")
    markouts = pd.read_csv(TABLE_ROOT / "markout_curves.csv")
    queue = pd.read_csv(TABLE_ROOT / "queue_fill_calibration.csv")
    reconciliation = pd.read_csv(TABLE_ROOT / "authoritative_economic_reconciliation.csv")
    normalized_drift = pd.read_csv(TABLE_ROOT / "normalized_structural_drift.csv")
    old_xgb_adverse_name = "xgboost_bid_adverse_probability"
    prediction_columns = set(
        pd.read_parquet(RESULT_ROOT / "final_posthoc_diagnostics.parquet").columns
    )
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in reports)
    checks = {
        "authoritative_final_evaluation_hash_unchanged": current[
            "artifacts/final_evaluation_v2.json"
        ]
        == charter["protected_inventory"]["artifacts/final_evaluation_v2.json"],
        "original_freeze_and_amendments_unchanged": all(
            current[path] == charter["protected_inventory"][path]
            for path in current
            if "final_freeze" in path
        ),
        "evaluation_count_one": final.get("evaluation_count") == 1,
        "model_checkpoints_unchanged": all(
            current[path] == charter["protected_inventory"][path]
            for path in current
            if "checkpoints_v2" in path or "xgboost_v2" in path
        ),
        "frozen_calibrators_unchanged": all(
            current[path] == charter["protected_inventory"][path]
            for path in current
            if "-cal_v2" in path or path.endswith("model_ladder_v2.json")
        ),
        "frozen_controller_unchanged": current["data/manifests/development_economics_v2.json"]
        == charter["protected_inventory"]["data/manifests/development_economics_v2.json"],
        "posthoc_outputs_separate": all(
            path.is_relative_to(ARTIFACT_ROOT)
            or path.is_relative_to(RESULT_ROOT)
            or path.is_relative_to(REPORT_ROOT)
            for root in (ARTIFACT_ROOT, RESULT_ROOT, REPORT_ROOT)
            for path in root.rglob("*")
            if path.is_file()
        ),
        "resume_evidence_unchanged_and_no_posthoc_copy": current[
            "reports/final_v2/RESUME_EVIDENCE.md"
        ]
        == charter["protected_inventory"]["reports/final_v2/RESUME_EVIDENCE.md"]
        and "POSTHOC" not in resume.upper(),
        "final_reports_label_posthoc": all(
            "POST-HOC" in path.read_text(encoding="utf-8").upper() for path in reports
        ),
        "v3_candidates_non_authoritative": "NON-AUTHORITATIVE"
        in (REPORT_ROOT / "V3_CANDIDATES.md").read_text(encoding="utf-8").upper(),
        "final_slices_exploratory": "EXPLORATORY_POSTHOC"
        in (TABLE_ROOT / "operational_slices.csv").read_text(encoding="utf-8"),
        "markout_semantics_explicit": set(markouts["markout_type"].unique())
        == {
            "QUOTE_MARKOUT_100MS",
            "QUOTE_MARKOUT_1000MS",
            "SIMULATED_L3_POST_FILL_MARKOUT_100MS",
        }
        and not any(
            value.startswith("QUOTE_") and "SIMULATED_L3" in value
            for value in markouts["markout_type"].astype(str)
        ),
        "l3_execution_tables_masked": bool(queue["l3_valid_only"].all())
        and not queue["instruments"].str.contains("BRN", regex=False).any(),
        "adverse_proxy_semantics_explicit": old_xgb_adverse_name not in prediction_columns
        and (
            "MARKOUT-DERIVED SCORE" in report_text.upper()
            or "MARKOUT_DERIVED_ADVERSE_SCORE" in report_text.upper()
        ),
        "economic_reconciliation_matches": not (
            reconciliation["reconciliation_status"] == "MISMATCH"
        ).any()
        and (
            reconciliation.loc[
                reconciliation["row_type"] == "RECONSTRUCTED_POSTHOC",
                "reconciliation_status",
            ]
            == "MATCH"
        ).all(),
        "normalized_structural_drift_present": not normalized_drift.empty
        and set(normalized_drift["drift_family"].unique()) == {"NORMALIZED_STRUCTURAL_DRIFT"},
        "confidence_monotonicity_reported": (REPORT_ROOT / "CONFIDENCE_MONOTONICITY.md").is_file(),
        "provenance_erratum_preserved": (REPORT_ROOT / "PROVENANCE_ERRATA.md").is_file(),
        "prior_revision_preserved": sha256_file(
            ARTIFACT_ROOT / "revisions/revision_003/POSTHOC_RESULTS_MANIFEST.json"
        )
        == "5d302dcece7d74432799229cc24bb8520ba070fa4a59f08546febb095a6f8133"
        and sha256_file(ARTIFACT_ROOT / "revisions/revision_003/POSTHOC_AUDIT.json")
        == "7928c14c233eea5c7b8ad10874a79dd14ac7eb1f83f311e497351d812c92593a",
    }
    output = ARTIFACT_ROOT / "POSTHOC_AUDIT.json"
    _write_json(
        output,
        {
            "schema_version": 1,
            "analysis_layer": LABEL,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "status": "SUCCEEDED" if all(checks.values()) else "FAILED",
            "checks": checks,
            "passed": sum(checks.values()),
            "total": len(checks),
            "authoritative_evaluation_count": final.get("evaluation_count"),
            "protected_inventory": current,
        },
    )
    if not all(checks.values()):
        raise RuntimeError(f"post-hoc audit failed: {checks}")
    return output


def _manifest(summary: dict[str, Any]) -> Path:
    output = REPORT_ROOT / "POSTHOC_RESULTS_MANIFEST.json"
    paths = []
    for root in (ARTIFACT_ROOT, RESULT_ROOT, REPORT_ROOT):
        paths.extend(path for path in root.rglob("*") if path.is_file() and path != output)
    files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    }
    payload = {
        "schema_version": 1,
        "diagnostic_revision": DIAGNOSTIC_REVISION,
        "analysis_layer": LABEL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "authoritative": False,
        "authoritative_v2_unchanged": True,
        "evaluation_count": 1,
        "previous_posthoc_manifest_sha256": "5d302dcece7d74432799229cc24bb8520ba070fa4a59f08546febb095a6f8133",
        "previous_posthoc_audit_sha256": "7928c14c233eea5c7b8ad10874a79dd14ac7eb1f83f311e497351d812c92593a",
        "prior_revision_preserved_at": "artifacts/posthoc_v2/revisions/revision_003",
        "authoritative_final_sha256": sha256_file(ROOT / "artifacts/final_evaluation_v2.json"),
        "revision_type": "POSTHOC_ONLY_CORRECTION_NOT_A_V2_AMENDMENT",
        "correction_list": json.loads(
            (ARTIFACT_ROOT / "POSTHOC_REVISION_004_PRECHECK.json").read_text(encoding="utf-8")
        )["corrections"],
        "markout_semantics": [
            "QUOTE_MARKOUT_100MS",
            "QUOTE_MARKOUT_1000MS",
            "SIMULATED_L3_POST_FILL_MARKOUT_100MS",
        ],
        "final_data_interpretation": "DESCRIPTIVE_EXPLORATORY_POSTHOC_ONLY",
        "summary": summary,
        "files": files,
        "file_count_excluding_manifest": len(files),
    }
    payload["manifest_content_hash"] = content_hash(payload)
    _write_json(output, payload)
    return output


def _existing_valid() -> Path | None:
    manifest_path = REPORT_ROOT / "POSTHOC_RESULTS_MANIFEST.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("analysis_layer") != LABEL or manifest.get("evaluation_count") != 1:
        return None
    if manifest.get("diagnostic_revision") != DIAGNOSTIC_REVISION:
        return None
    charter = json.loads((ARTIFACT_ROOT / "POSTHOC_CHARTER.json").read_text(encoding="utf-8"))
    if protected_inventory() != charter["protected_inventory"]:
        raise RuntimeError("authoritative v2 changed after post-hoc completion")
    for relative, metadata in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            return None
    return manifest_path


def run_v2_posthoc() -> Path:
    """Run the idempotent, strictly non-authoritative v2 diagnostic layer."""
    create_posthoc_charter()
    existing = _existing_valid()
    if existing is not None:
        return existing
    for path in (ARTIFACT_ROOT, RESULT_ROOT, REPORT_ROOT, TABLE_ROOT, FIGURE_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    before = protected_inventory()
    development, final, dev_latent, final_latent, _ = _prediction_frames()
    summaries = {
        "calibration": _calibration_analysis(development, final),
        "slices": _slice_analysis(development, final),
        "error_cohorts": _error_cohorts(development, final),
        "disagreement": _disagreement_and_selective(development, final),
        "confidence_monotonicity": _confidence_monotonicity(development, final),
        "economics": _economic_analysis(development, final),
        "markout_queue": _markout_and_queue(development, final),
        "drift": _drift_and_ood(development, final, dev_latent, final_latent),
        "interpretability": _interpretability(development, final, dev_latent, final_latent),
        "robustness": _robustness_surfaces(development),
        "monitoring": _monitoring_scorecard(development, final),
    }
    executive = _reports(summaries)
    summaries["executive"] = executive
    _write_json(
        ARTIFACT_ROOT / "POSTHOC_RUN_SUMMARY.json",
        {
            "schema_version": 1,
            "diagnostic_revision": DIAGNOSTIC_REVISION,
            "analysis_layer": LABEL,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "authoritative": False,
            "evaluation_count": 1,
            "summaries": summaries,
        },
    )
    after = protected_inventory()
    if before != after:
        raise RuntimeError("protected v2 inventory changed during post-hoc analysis")
    _posthoc_audit()
    return _manifest(executive)
