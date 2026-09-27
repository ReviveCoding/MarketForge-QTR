from __future__ import annotations

import json

from .data import data_root
from .probe import atomic_json
from .results import append_result, dataset_fingerprint
from .training import economic_posttrain, pretrain_ssl, train_supervised


def _warehouse(metrics: dict[str, object]) -> None:
    append_result(
        {
            "stage": metrics["stage"],
            "status": "SUCCEEDED",
            "dataset_fingerprint": dataset_fingerprint(),
            "seed": metrics["seed"],
            "hardware": "RTX 4090 Laptop GPU",
            "precision": metrics["precision"],
            "model": metrics["stage"],
            "metrics_json": json.dumps(metrics, sort_keys=True),
            "wall_seconds": metrics["wall_seconds"],
            "artifact_paths_json": json.dumps([metrics["checkpoint"]]),
        }
    )


def run_pretraining_ladder() -> object:
    supervised = train_supervised("M3T-SUP", max_rows=16000, epochs=5)
    _warehouse(supervised)
    ssl = pretrain_ssl(max_rows=16000, epochs=3)
    _warehouse(ssl)
    fine_tuned = train_supervised(
        "M3T-SSL", initial_checkpoint=str(ssl["checkpoint"]), max_rows=16000, epochs=5
    )
    _warehouse(fine_tuned)
    report = data_root() / "manifests" / "model_ladder_pretrain_v1.json"
    atomic_json(
        report,
        {
            "tier": 1,
            "selection_data": "MODEL_VALIDATION",
            "test_used": False,
            "final_lockbox_used": False,
            "m3t_sup": supervised,
            "ssl_pretraining": ssl,
            "m3t_ssl_finetuned": fine_tuned,
            "ssl_macro_f1_delta": fine_tuned["validation_macro_f1"]
            - supervised["validation_macro_f1"],
        },
    )
    return report


def run_supervised_diagnostic() -> object:
    metrics = train_supervised("M3T-SUP-WEIGHTED", max_rows=20000, epochs=8)
    _warehouse(metrics)
    baseline_path = data_root() / "manifests" / "baseline_results_v1.json"
    baselines = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    )
    best_baseline = max(
        (entry for entry in baselines.get("results", []) if "macro_f1" in entry["metrics"]),
        key=lambda entry: entry["metrics"]["macro_f1"],
        default=None,
    )
    report = data_root() / "manifests" / "m3t_supervised_failure_diagnostic_v1.json"
    atomic_json(
        report,
        {
            "trigger": "M3T-SUP collapsed to majority class in Tier-1 unweighted screening",
            "corrective_change": "TRAIN-only inverse-frequency cross-entropy weights",
            "weighted_result": metrics,
            "best_cpu_baseline": best_baseline,
            "test_used": False,
            "final_lockbox_used": False,
            "complexity_justified_at_this_stage": bool(
                best_baseline
                and metrics["validation_macro_f1"] > best_baseline["metrics"]["macro_f1"]
            ),
        },
    )
    return report


def run_posttraining() -> object:
    initial = data_root().parents[0] / "artifacts" / "checkpoints" / "m3t-sup-weighted_seed1701.pt"
    metrics = economic_posttrain(str(initial))
    _warehouse(metrics)
    report = data_root() / "manifests" / "m3t_ept_v1.json"
    atomic_json(
        report,
        {
            "initial_checkpoint": str(initial),
            "result": metrics,
            "test_used": False,
            "final_lockbox_used": False,
        },
    )
    return report
