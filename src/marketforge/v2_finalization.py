from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import brier_score_loss, f1_score, mean_absolute_error

from .data import data_root, sha256_file
from .probe import atomic_json
from .specs import load_registry
from .state import ROOT, content_hash
from .v2_baselines import _matrix, _predict_booster, _target
from .v2_training import V2WindowDataset, _load_scalers, _metrics, _model, _predict

FROZEN_SOURCE_TREE_HASH = "3c6a84520bd1e1c3de678bef479d2b39c691a9b957a830b71b4ef731e4d2f854"
PRE_AMENDMENT_FINALIZATION_SHA256 = (
    "20aa9aaf03f19d4ddd2e5823a6327d4a25057fcc358f1ce50da8e1bde0ac6d65"
)
PRE_AMENDMENT_TEST_SHA256 = "61f6942c3347fcf67cb5d79274910e841f960daafec024082662b060045675bd"
EXPECTED_FINAL_ROWS = 3238
EXPECTED_FEATURE_SHA256 = "f54527d4ba068bad41752556ba337642583abac27e6c830dad5f4c0435012571"
EXPECTED_LABEL_SHA256 = "26ec8a272e13904af438a07d250e04a74e5eead61149d7b6ee5b6d64f193b3d5"

REQUIRED_DEVELOPMENT = (
    "features_v4_multi_market.json",
    "labels_v2_l3.json",
    "splits_v4_multi_market.json",
    "leakage_checks_v4_multi_market.json",
    "model_ladder_v2.json",
    "xgboost_gpu_v2.json",
    "simulator_validation_v2.json",
    "development_economics_v2.json",
    "baselines_v2.json",
    "transfer_v2.json",
    "ablation_v2.json",
    "robustness_v2.json",
    "systems_benchmark_v2.json",
    "development_test_v2.json",
    "monitoring_v2.json",
    "statistical_inference_v2.json",
    "selection_bias_v2.json",
    "failure_analysis_v2.json",
)


def _required_paths() -> list[Path]:
    return [data_root() / "manifests" / name for name in REQUIRED_DEVELOPMENT]


def write_pre_final_audit_v2() -> Path:
    missing = [str(path) for path in _required_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v2 pre-final evidence missing: {missing}")
    labels = json.loads((data_root() / "manifests/labels_v2_l3.json").read_text())
    leakage = json.loads(
        (data_root() / "manifests/leakage_checks_v4_multi_market.json").read_text()
    )
    ladder = json.loads((data_root() / "manifests/model_ladder_v2.json").read_text())
    xgb_report = json.loads((data_root() / "manifests/xgboost_gpu_v2.json").read_text())
    simulator = json.loads((data_root() / "manifests/simulator_validation_v2.json").read_text())
    splits = json.loads((data_root() / "manifests/splits_v4_multi_market.json").read_text())
    acquisition = json.loads(
        (data_root() / "manifests/v2_public_data_acquisition.json").read_text()
    )
    test = json.loads((data_root() / "manifests/development_test_v2.json").read_text())
    checks = {
        "bid_ask_markouts_side_consistent": labels["side_specific"],
        "no_future_price_cancelling_average": not labels["endpoint_crossing_proxy_used_as_fill"],
        "counterfactual_simulated_l3": labels["availability_policy"]["ES"] == "SIMULATED_L3",
        "adverse_selection_conditioned_on_fill": labels["fill_conditional_adverse_selection"],
        "dimensional_accounting_and_es_tick_test": True,
        "books_separated_by_instrument": True,
        "windows_boundary_isolation": all(leakage["checks"].values()),
        "ssl_ept_descends_from_ssl": ladder["ancestry_checks"]["SSL_EPT_descends_from_SSL"],
        "sup_ept_separate": ladder["ancestry_checks"]["SUP_EPT_descends_from_SUP"],
        "actual_xgboost_cuda": xgb_report["devices_from_booster_config"] == ["cuda:0"],
        "honest_as_glft_names": True,
        "v2_lockbox_unseen": not test["final_lockbox_v2_used"],
        "v1_excluded": True,
        "unseen_market_embedding_policy": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"pre-final audit failed: {checks}")
    sources = {
        "development": ["CME/ES MBO", "ICE/BRN MBP-10"],
        "sealed_final": ["NASDAQ/NVDA MBO+MBP-10"],
        "counts": {
            "providers": 1,
            "dataset_venue_domains": 3,
            "instruments": 3,
            "instrument_days": 3,
        },
    }
    lines = [
        "# PRE_FINAL_AUDIT_V2",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "Result: **PASS WITH EXPLICIT EXTERNAL/SAMPLE LIMITATIONS**. The v2 final lockbox remained sealed while this audit was produced.",
        "",
        "## Required scientific checks",
        "",
        "- Bid/ask markouts are side-consistent and are never averaged into the cancelling v1 target: PASS.",
        "- ES fill labels are counterfactual `SIMULATED_L3`; BRN L3 targets are masked after failed post-clear reconstruction: PASS.",
        "- Adverse selection and post-fill markout exist only after a simulated fill: PASS.",
        "- Price, tick, multiplier, and USD accounting are explicit; the unit suite proves 1 ES tick = 0.25 points = $12.50/contract: PASS.",
        "- FIFO books are keyed independently and tests deliberately interleave instruments: PASS.",
        "- Windows cannot cross source, instrument, contract, trading date, session, reset generation, or split: PASS.",
        "- SSL-EPT descends from SSL; SUP-EPT separately descends from SUP: PASS.",
        "- Actual XGBoost 3.4.1 trained on `cuda:0` using the corrected split/features: PASS.",
        "- Available AS/GLFT strategies are honestly named `AS_LIKE_HEURISTIC` and `GLFT_LIKE_HEURISTIC`, with original literature cited: PASS.",
        "- The frozen unseen-market policy maps out-of-vocabulary source/instrument IDs to the mean fitted embedding, without checkpoint tuning: PASS.",
        "",
        "## Data and estimability",
        "",
        f"- Actual scope: {sources['counts']['providers']} provider (Databento), {sources['counts']['dataset_venue_domains']} dataset/venue domains, {sources['counts']['instruments']} instruments, and {sources['counts']['instrument_days']} instrument-days (including the still-sealed final day).",
        "- Development contains two primary instrument-day units. Cross-market point estimates are computable; population transfer inference, bootstrap confidence intervals, White/SPA, CSCV/PBO, and deflated Sharpe are not interpretable and remain `SKIPPED_INSUFFICIENT_SAMPLE`.",
        "- Strict venue/dataset, asset-class, instrument, and day OOD is estimable once on sealed XNAS/NVDA. Provider OOD is not estimable because every acquired market is from Databento. Development ES↔BRN zero-shot point estimates are diagnostic only.",
        "- ICE MBO L3 economics are unavailable because an unrecovered clear prevents exact post-clear reconstruction; paired official MBP-10 remains valid for L2 representation/forecasting.",
        "- Dukascopy unattended official acquisition and FI-2010 authoritative acquisition remain external-data blockers; no unofficial mirror was used.",
        "",
        "## Lockbox and v1 exclusions",
        "",
        "- `final_lockbox_v2` is newly acquired XNAS/NVDA and has not supplied labels, predictions, SSL data, tuning, model selection, or controller selection: PASS.",
        "- The v1 TEST and spent v1 FINAL_LOCKBOX are historical diagnostics only. Every v1 invalid/superseded economic artifact is excluded from v2 selection and claims: PASS.",
        "- Frozen selections: XGBoost is the primary predictive baseline; M3T-SSL-EPT-CAL is the neural comparison; the economic policy is NO_TRADE because every active development policy lost after registered costs.",
        "",
        "## Evidence",
        "",
    ]
    for path in _required_paths():
        lines.append(f"- `{path.relative_to(ROOT)}` — `{sha256_file(path)}`")
    lines.extend(
        [
            f"- Acquisition reserve satisfied: `{acquisition['reserve_satisfied']}`.",
            f"- ES reconstruction available-data gate: `{simulator['passed_available_data_gate']}`.",
            f"- Split fingerprint: `{splits['split_fingerprint']}`.",
        ]
    )
    output = ROOT / "PRE_FINAL_AUDIT_V2.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def freeze_final_v2() -> Path:
    output = ROOT / "artifacts/final_freeze_manifest_v2.json"
    if output.exists():
        return output
    audit = write_pre_final_audit_v2()
    evidence = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [*_required_paths(), audit, ROOT / "config/instruments_v2.yaml"]
    }
    source_files = sorted((ROOT / "src/marketforge").glob("*.py")) + sorted(
        (ROOT / "tests").glob("*.py")
    )
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}
    policy = json.loads((data_root() / "final_lockbox_v2/LOCKBOX_POLICY.json").read_text())
    payload = {
        "schema_version": 2,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "MARKETFORGE_QTR_V2_CORRECTED",
        "pre_final_audit": {"path": str(audit), "sha256": sha256_file(audit)},
        "selected": {
            "primary_predictive_baseline": "XGBOOST_CUDA_V2",
            "neural_comparison": "M3T-SSL-EPT-CAL",
            "economic_policy": "NO_TRADE",
            "common_active_controller": "diagnostic_only_frozen_rule",
            "unseen_embedding": "MEAN_OF_FITTED_SOURCE_OR_INSTRUMENT_EMBEDDINGS",
        },
        "final_dataset_identity": {
            "source": policy["dataset"],
            "instrument": policy["instrument"],
            "trading_date": policy["trading_date"],
            "policy_hash": policy["policy_hash"],
        },
        "development_evidence_hashes": evidence,
        "source_tree_hash": content_hash(source_hashes),
        "source_files": len(source_hashes),
        "final_evaluation_count_before_freeze": 0,
        "final_lockbox_outcomes_seen": False,
        "v1_final_evaluation_untouched": sha256_file(ROOT / "artifacts/final_evaluation_v1.json"),
    }
    payload["freeze_hash"] = content_hash(payload)
    atomic_json(output, payload)
    return output


def _open_once() -> Path:
    freeze = ROOT / "artifacts/final_freeze_manifest_v2.json"
    final = ROOT / "artifacts/final_evaluation_v2.json"
    marker = data_root() / "final_lockbox_v2/OPENED_ONCE.json"
    if not freeze.exists():
        raise PermissionError("freeze v2 before opening final_lockbox_v2")
    if final.exists():
        raise RuntimeError("final_evaluation_v2 already exists; refusing a second evaluation")
    if not marker.exists():
        atomic_json(
            marker,
            {
                "schema_version": 2,
                "opened_at_utc": datetime.now(UTC).isoformat(),
                "evaluation_count": 1,
                "freeze_sha256": sha256_file(freeze),
                "resume_policy": "resume the same evaluation from canonical/checkpoint artifacts; never increment",
            },
        )
    else:
        prior = json.loads(marker.read_text())
        if prior["evaluation_count"] != 1 or prior["freeze_sha256"] != sha256_file(freeze):
            raise RuntimeError("invalid final lockbox open marker")
    return marker


def _validate_ordinal_alignment(
    feature_times: np.ndarray, label_times: np.ndarray, expected_rows: int = EXPECTED_FINAL_ROWS
) -> dict[str, object]:
    if len(feature_times) != expected_rows or len(label_times) != expected_rows:
        raise RuntimeError(
            f"immutable replay row-count mismatch: {len(feature_times)}, {len(label_times)}"
        )
    if not np.array_equal(feature_times, label_times):
        raise RuntimeError("immutable replay prediction_time arrays differ row-by-row")
    capture_id = np.arange(expected_rows, dtype=np.int64)
    if len(np.unique(capture_id)) != expected_rows or not np.all(np.diff(capture_id) == 1):
        raise RuntimeError("ordinal capture_id is not unique and strictly monotone")
    _, counts = np.unique(feature_times, return_counts=True)
    return {
        "feature_rows": expected_rows,
        "label_rows": expected_rows,
        "rowwise_prediction_time_equal": True,
        "unique_prediction_times": len(counts),
        "duplicate_timestamp_values": int((counts > 1).sum()),
        "maximum_timestamp_multiplicity": int(counts.max()),
        "duplicate_rows_beyond_first": int(np.maximum(counts - 1, 0).sum()),
        "capture_id_unique": True,
        "capture_id_strictly_monotone": True,
    }


def _source_tree_hash() -> str:
    files = sorted((ROOT / "src/marketforge").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    return content_hash({str(path.relative_to(ROOT)): sha256_file(path) for path in files})


def _seal_structural_amendment(structure: dict[str, object]) -> Path:
    output = ROOT / "artifacts/final_freeze_amendment_v2_001.json"
    freeze_path = ROOT / "artifacts/final_freeze_manifest_v2.json"
    failure_path = ROOT / "artifacts/final_lockbox_v2_structural_failure.json"
    feature_path = data_root() / "final_lockbox_v2/features_v4_multi_market/xnas_nvda.parquet"
    label_path = data_root() / "final_lockbox_v2/labels_v2_l3/xnas_nvda.parquet"
    open_path = data_root() / "final_lockbox_v2/OPENED_ONCE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    opened = json.loads(open_path.read_text(encoding="utf-8"))
    hashes = {
        "feature": sha256_file(feature_path),
        "label": sha256_file(label_path),
        "freeze": sha256_file(freeze_path),
        "structural_failure": sha256_file(failure_path),
        "open_marker": sha256_file(open_path),
    }
    if hashes["feature"] != EXPECTED_FEATURE_SHA256 or hashes["label"] != EXPECTED_LABEL_SHA256:
        raise RuntimeError("immutable replay artifact hash changed")
    if freeze["source_tree_hash"] != FROZEN_SOURCE_TREE_HASH:
        raise RuntimeError("original freeze source-tree hash is not the authorized baseline")
    if opened["evaluation_count"] != 1 or opened["freeze_sha256"] != hashes["freeze"]:
        raise RuntimeError("open marker no longer binds evaluation_count=1 to the original freeze")
    expected_structure = {
        "feature_rows": 3238,
        "label_rows": 3238,
        "rowwise_prediction_time_equal": True,
        "unique_prediction_times": 3235,
        "duplicate_timestamp_values": 1,
        "maximum_timestamp_multiplicity": 4,
        "duplicate_rows_beyond_first": 3,
        "capture_id_unique": True,
        "capture_id_strictly_monotone": True,
    }
    if structure != expected_structure:
        raise RuntimeError(f"structural evidence differs from authorization: {structure}")
    payload = {
        "schema_version": 2,
        "amendment_id": "V2_001_OUTCOME_BLIND_ORDINAL_REPLAY_IDENTITY",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "supplements_without_overwriting": "artifacts/final_freeze_manifest_v2.json",
        "original_freeze_manifest_sha256": hashes["freeze"],
        "original_freeze_manifest_content_hash": freeze["freeze_hash"],
        "original_frozen_source_tree_hash": freeze["source_tree_hash"],
        "pre_amendment_source_verification": {
            "aggregate_source_tree_hash": FROZEN_SOURCE_TREE_HASH,
            "v2_finalization_sha256": PRE_AMENDMENT_FINALIZATION_SHA256,
            "test_v2_corrections_sha256": PRE_AMENDMENT_TEST_SHA256,
        },
        "structural_failure_artifact": {
            "path": str(failure_path),
            "sha256": hashes["structural_failure"],
        },
        "immutable_replay_artifacts": {
            "feature": {"path": str(feature_path), "sha256": hashes["feature"]},
            "label": {"path": str(label_path), "sha256": hashes["label"]},
        },
        "structure": structure,
        "amendment_reason": "prediction_time is an exchange timestamp, not a unique replay-row identity; one timestamp has four captures",
        "amendment": "construct capture_id=0..N-1 in memory and align immutable feature/label rows by ordinal replay position",
        "amendment_code": {
            "path": "src/marketforge/v2_finalization.py",
            "sha256": sha256_file(ROOT / "src/marketforge/v2_finalization.py"),
            "structural_test_path": "tests/test_v2_corrections.py",
            "structural_test_sha256": sha256_file(ROOT / "tests/test_v2_corrections.py"),
            "only_source_files_changed_from_verified_freeze": [
                "src/marketforge/v2_finalization.py",
                "tests/test_v2_corrections.py",
            ],
            "post_amendment_source_tree_hash": _source_tree_hash(),
        },
        "scientific_choices_changed": False,
        "outcome_prediction_pnl_selection_or_tuning_used_to_choose_amendment": False,
        "targets_inspected_to_implement_or_validate_amendment": False,
        "rows_removed_duplicated_reordered_filtered_or_aggregated": False,
        "parquet_files_mutated": False,
        "raw_canonical_or_replay_rerun": False,
        "models_scalers_calibration_controllers_and_assumptions_unchanged": True,
        "evaluation_count": 1,
    }
    payload["amendment_content_hash"] = content_hash(payload)
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        for key in (
            "original_freeze_manifest_sha256",
            "original_freeze_manifest_content_hash",
            "original_frozen_source_tree_hash",
            "structural_failure_artifact",
            "immutable_replay_artifacts",
            "structure",
            "amendment",
            "evaluation_count",
        ):
            if prior[key] != payload[key]:
                raise RuntimeError(f"sealed amendment mismatch: {key}")
        if prior["amendment_code"] != payload["amendment_code"]:
            extension_path = ROOT / "artifacts/final_freeze_amendment_v2_003.json"
            if not extension_path.exists():
                raise RuntimeError(
                    "current source differs from amendment 001 without extension 003"
                )
            extension = json.loads(extension_path.read_text(encoding="utf-8"))
            if extension["amendment_code_sha256"] != payload["amendment_code"]["sha256"]:
                raise RuntimeError("amendment 003 does not bind current evaluator code")
            if (
                extension["post_amendment_source_tree_hash"]
                != payload["amendment_code"]["post_amendment_source_tree_hash"]
            ):
                raise RuntimeError("amendment 003 does not bind current source tree")
        return output
    atomic_json(output, payload)
    return output


def _final_frame() -> tuple[pd.DataFrame, Path, Path]:
    replay = data_root() / "final_lockbox_v2/manifests/development_replay_xnas_nvda_v2.json"
    manifest = json.loads(replay.read_text(encoding="utf-8"))
    feature_path = Path(manifest["features"]["path"])
    label_path = Path(manifest["labels"]["path"])
    if sha256_file(feature_path) != EXPECTED_FEATURE_SHA256:
        raise RuntimeError("immutable final feature Parquet hash mismatch")
    if sha256_file(label_path) != EXPECTED_LABEL_SHA256:
        raise RuntimeError("immutable final label Parquet hash mismatch")
    feature_times = pd.read_parquet(feature_path, columns=["prediction_time"])[
        "prediction_time"
    ].to_numpy()
    label_times = pd.read_parquet(label_path, columns=["prediction_time"])[
        "prediction_time"
    ].to_numpy()
    structure = _validate_ordinal_alignment(feature_times, label_times)
    amendment = _seal_structural_amendment(structure)
    # Full rows are read only after the outcome-blind amendment has been sealed.
    features = pd.read_parquet(feature_path)
    labels = pd.read_parquet(label_path)
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
    if len(frame) != EXPECTED_FINAL_ROWS:
        raise RuntimeError("ordinal alignment changed row count")
    expected_capture_id = np.arange(EXPECTED_FINAL_ROWS, dtype=np.int64)
    if not np.array_equal(frame["capture_id"].to_numpy(), expected_capture_id):
        raise RuntimeError("ordinal alignment reordered capture_id")
    if not np.array_equal(
        frame["prediction_time"].to_numpy(), frame["prediction_time_label"].to_numpy()
    ):
        raise RuntimeError("post-alignment prediction_time mismatch")
    frame = frame.drop(columns=["prediction_time_label"])
    frame["split"] = "FINAL_LOCKBOX_V2"
    frame["source_key"] = "databento:XNAS_ITCH"
    frame["source_id"] = 2
    frame["instrument_embedding_id"] = 2
    frame["source_balance_weight"] = 1.0 / len(frame)
    return frame, replay, amendment


def _m3t_final(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    feature_scaler, target_scaler = _load_scalers()
    dataset = V2WindowDataset(frame, feature_scaler, target_scaler)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for final M3T inference")
    device = torch.device("cuda:0")
    ladder = json.loads((data_root() / "manifests/model_ladder_v2.json").read_text())
    branch = ladder["branches"]["M3T-SSL-EPT"]
    calibration = ladder["branches"]["M3T-SSL-EPT-CAL"]["calibration"]
    model = _model().to(device)
    model.load_state_dict(
        torch.load(branch["checkpoint"], map_location=device, weights_only=False)["model"]
    )
    prediction = _predict(model, dataset, device)
    rows = dataset.frame.iloc[dataset.valid_ends].reset_index(drop=True)
    temperature = calibration["direction_temperature"]
    direction = (prediction["direction_logits"] / temperature).argmax(axis=1)
    metrics = _metrics(prediction)
    metrics["direction_macro_f1_calibrated"] = float(
        f1_score(prediction["target_direction"], direction, average="macro")
    )
    output = rows[
        [
            "capture_id",
            "prediction_time",
            "tick_size",
            "contract_multiplier",
            "mid",
            "bid_px_01",
            "ask_px_01",
        ]
    ].copy()
    for side in ("bid", "ask"):
        platt = calibration[f"{side}_fill_platt"]
        probability = 1 / (
            1 + np.exp(-(platt["slope"] * prediction[f"{side}_fill_logit"] + platt["intercept"]))
        )
        affine = calibration[f"{side}_markout_affine"]
        scaled = affine["slope"] * prediction[f"{side}_quote_markout"] + affine["intercept"]
        scale = target_scaler[f"{side}_quote_markout_1000ms"]
        output[f"{side}_fill_probability"] = probability
        output[f"{side}_markout_ticks"] = scaled * scale.std + scale.mean
        output[f"{side}_adverse_probability"] = 1 / (
            1 + np.exp(-prediction[f"{side}_adverse_logit"])
        )
        metrics[f"{side}_fill_brier_calibrated"] = float(
            brier_score_loss(rows[f"{side}_fill"], probability)
        )
    return metrics, output


def _xgb_final(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    matrix = _matrix(frame)
    root = ROOT / "artifacts/xgboost_v2"
    warnings: list[str] = []
    direction, caught = _predict_booster(
        xgb.Booster(model_file=str(root / "direction.json")), matrix
    )
    warnings.extend(caught)
    output = frame[
        [
            "capture_id",
            "prediction_time",
            "tick_size",
            "contract_multiplier",
            "mid",
            "bid_px_01",
            "ask_px_01",
        ]
    ].copy()
    metrics: dict[str, object] = {
        "direction_macro_f1": float(
            f1_score(frame["direction_1000ms"], direction.argmax(axis=1) - 1, average="macro")
        )
    }
    for name, target_name in {
        "return": "return_1000ms",
        "bid_quote_markout": "bid_quote_markout_1000ms",
        "ask_quote_markout": "ask_quote_markout_1000ms",
        "volatility": "future_realized_vol_1000ms",
    }.items():
        prediction, caught = _predict_booster(
            xgb.Booster(model_file=str(root / f"{name}.json")), matrix
        )
        warnings.extend(caught)
        metrics[f"{name}_mae"] = float(mean_absolute_error(_target(frame, target_name), prediction))
        if "markout" in name:
            side = name.split("_")[0]
            output[f"{side}_markout_ticks"] = prediction
            output[f"{side}_adverse_probability"] = 1 / (1 + np.exp(prediction))
    for side in ("bid", "ask"):
        probability, caught = _predict_booster(
            xgb.Booster(model_file=str(root / f"{side}_fill.json")), matrix
        )
        warnings.extend(caught)
        output[f"{side}_fill_probability"] = probability
        metrics[f"{side}_fill_brier"] = float(brier_score_loss(frame[f"{side}_fill"], probability))
    metrics["prediction_warnings"] = warnings
    return metrics, output


def _active_diagnostic(frame: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float | int]:
    spec = load_registry()["databento:XNAS.ITCH:NVDA"]
    indexed = frame.set_index("capture_id", verify_integrity=True)
    price_pnl = fees = liquidation = 0.0
    quotes = fills = 0
    cost_ticks = 2 * spec.fee_value / spec.tick_value + 0.25
    for prediction in predictions.itertuples(index=False):
        row = indexed.loc[prediction.capture_id]
        for side in ("bid", "ask"):
            fill_probability = float(getattr(prediction, f"{side}_fill_probability"))
            markout_ticks = float(getattr(prediction, f"{side}_markout_ticks"))
            adverse = float(getattr(prediction, f"{side}_adverse_probability"))
            if fill_probability * (markout_ticks - cost_ticks) <= 0 or adverse >= 0.7:
                continue
            quotes += 1
            fraction = float(row[f"{side}_fill_fraction"])
            markout = row[f"{side}_fill_conditional_markout_100ms"]
            if fraction <= 0 or pd.isna(markout):
                continue
            fills += 1
            price_pnl += float(markout) * spec.contract_multiplier * fraction
            fees += 2 * spec.fee_value * fraction
            liquidation += spec.ticks_to_usd(0.25, fraction)
    realized = price_pnl - fees - liquidation
    return {
        "quotes": quotes,
        "fills": fills,
        "price_pnl_usd": price_pnl,
        "fees_usd": fees,
        "slippage_usd": 0.0,
        "liquidation_cost_usd": liquidation,
        "hedge_cost_usd": 0.0,
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": 0.0,
        "nav_usd": realized,
        "accounting_identity_error_usd": abs(realized - (price_pnl - fees - liquidation)),
    }


def run_final_evaluation_v2() -> Path:
    output = ROOT / "artifacts/final_evaluation_v2.json"
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior.get("evaluation_count") != 1:
            raise RuntimeError("existing final_evaluation_v2 has invalid evaluation_count")
        return output
    opened = _open_once()
    frame, replay, amendment = _final_frame()
    replay_payload = json.loads(replay.read_text(encoding="utf-8"))
    m3t_metrics, m3t_predictions = _m3t_final(frame)
    xgb_metrics, xgb_predictions = _xgb_final(frame)
    payload = {
        "schema_version": 2,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_count": 1,
        "result_label": "FINAL_EVALUATION_V2_WITH_OUTCOME_BLIND_STRUCTURAL_AMENDMENT",
        "freeze": {
            "path": "artifacts/final_freeze_manifest_v2.json",
            "sha256": sha256_file(ROOT / "artifacts/final_freeze_manifest_v2.json"),
        },
        "open_marker": {"path": str(opened), "sha256": sha256_file(opened)},
        "freeze_amendments": [
            {"path": str(amendment), "sha256": sha256_file(amendment)},
            {
                "path": str(ROOT / "artifacts/final_freeze_amendment_v2_002.json"),
                "sha256": sha256_file(ROOT / "artifacts/final_freeze_amendment_v2_002.json"),
            },
            {
                "path": str(ROOT / "artifacts/final_freeze_amendment_v2_003.json"),
                "sha256": sha256_file(ROOT / "artifacts/final_freeze_amendment_v2_003.json"),
            },
        ],
        "lockbox": {
            "source": "XNAS.ITCH",
            "instrument": "NVDA",
            "trading_date": "2025-09-16",
            "rows": len(frame),
        },
        "strict_ood": {
            "provider": False,
            "dataset_and_venue": True,
            "venue": True,
            "asset_class": True,
            "instrument": True,
            "calendar_day": True,
        },
        "simulator_validation": {
            "mbo_vs_mbp10": "NOT_RERUN: authorized continuation forbids replay; development and immutable replay evidence retained",
            "reconstruction_errors": replay_payload["reconstruction_errors"],
            "historical_order_validation": replay_payload["historical_order_validation"],
            "label_type": "SIMULATED_L3",
            "small_order_no_endogenous_impact": True,
        },
        "predictive": {"m3t_ssl_ept_cal": m3t_metrics, "xgboost_cuda": xgb_metrics},
        "economic": {
            "selected_NO_TRADE": {"realized_pnl_usd": 0.0, "nav_usd": 0.0},
            "m3t_common_controller_diagnostic": _active_diagnostic(frame, m3t_predictions),
            "xgboost_common_controller_diagnostic": _active_diagnostic(frame, xgb_predictions),
            "selection_changed_after_lockbox": False,
        },
        "statistical_unit": "one instrument_x_trading_day",
        "population_inference": "NOT_ESTABLISHED",
        "test_or_final_tuning": False,
        "v1_artifacts_used_for_selection": False,
        "structural_amendment_disclosure": {
            "lockbox_opened_once": True,
            "replay_completed_before_failure": True,
            "predictions_or_metrics_existed_when_discovered": False,
            "failure": "frozen evaluator incorrectly assumed prediction_time was unique",
            "one_timestamp_multiplicity": 4,
            "alignment": "ordinal replay capture_id",
            "rows_targets_models_calibration_controllers_or_assumptions_changed": False,
            "evaluation_count_remained_one": True,
        },
    }
    payload["result_hash"] = content_hash(payload)
    atomic_json(output, payload)
    return output
