from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .data import sha256_file
from .probe import atomic_json
from .state import ROOT, content_hash

CLASSIFICATIONS: dict[str, tuple[str, ...]] = {
    "REUSABLE": (
        "data/raw/databento/glbx-mdp3-futures-mbo.csv",
        "data/raw/databento/glbx-mdp3-futures-mbp10.csv",
        "data/manifests/databento_GLBX.MDP3_mbo.json",
        "data/manifests/databento_GLBX.MDP3_mbp-10.json",
        "data/manifests/simulator_validation_v1.json",
        "artifacts/system/environment_runtime.json",
        "artifacts/system/cuda_runtime.json",
        "data/manifests/systems_benchmark_v1.json",
    ),
    "REUSABLE_SUBJECT_TO_AUDIT": (
        "data/manifests/canonical_databento_mbo_v2.json",
        "data/manifests/features_databento_mbo_v3.json",
        "data/manifests/leakage_checks_v3.json",
    ),
    "HISTORICAL_DIAGNOSTIC_EVIDENCE": (
        "data/manifests/baseline_results_v1.json",
        "data/manifests/model_ladder_pretrain_v1.json",
        "data/manifests/m3t_supervised_failure_diagnostic_v1.json",
        "artifacts/checkpoints/m3t-sup_seed1701.pt",
        "artifacts/checkpoints/m3t-sup-weighted_seed1701.pt",
        "artifacts/checkpoints/m3t_ssl_seed1701.pt",
        "artifacts/checkpoints/m3t-ssl_seed1701.pt",
        "artifacts/checkpoints/xgboost_gpu_v1.json",
    ),
    "SUPERSEDED_INVALID_ECONOMIC_TARGETS": (
        "data/manifests/m3t_ept_v1.json",
        "data/manifests/calibration_m3t_ept_v1.json",
        "artifacts/checkpoints/m3t-ept_seed1701.pt",
    ),
    "SUPERSEDED_INVALID_ACCOUNTING_AND_FILL_MODEL": (
        "data/manifests/development_economics_v1.json",
        "data/manifests/robustness_v1.json",
        "artifacts/final_evaluation_v1.json",
    ),
    "SPENT_DO_NOT_REUSE": ("data/final_lockbox/final_lockbox.parquet",),
    "HISTORICAL_COMPLETED_PROTOCOL_SUPERSEDED": ("artifacts/final_freeze_manifest.json",),
    "HISTORICAL_AUDIT_ONLY": (
        "reports/final/MARKETFORGE_QTR_TECHNICAL_REPORT.md",
        "reports/final/EXECUTIVE_SUMMARY.md",
        "reports/final/RESUME_EVIDENCE.md",
        "reports/final/RESULTS_MANIFEST.json",
    ),
}


def create_v1_superseded_manifest() -> Path:
    output = ROOT / "artifacts" / "v1_superseded_manifest.json"
    if output.exists():
        return output
    items = []
    for classification, relative_paths in CLASSIFICATIONS.items():
        for relative in relative_paths:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            items.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "classification": classification,
                }
            )
    source_files = sorted(
        path
        for directory in (ROOT / "src", ROOT / "tests", ROOT / "scripts")
        for path in directory.rglob("*")
        if path.is_file()
    )
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "notice": "docs/V1_SUPERSEDED_NOTICE.md",
        "v1_final_evaluation_must_not_be_overwritten": True,
        "v1_final_lockbox_spent": True,
        "items": items,
        "systems_infrastructure": {
            "classification": "REUSABLE",
            "source_tree_fingerprint": content_hash(source_hashes),
            "files_hashed": len(source_hashes),
        },
        "gate_reclassification": {
            "gate_13_development_economics": "SUPERSEDED",
            "gate_17_statistical_inference": "SKIPPED_INSUFFICIENT_SAMPLE",
            "gate_18_selection_bias": "SKIPPED_INSUFFICIENT_SAMPLE",
            "gate_20_final_freeze": "HISTORICAL_COMPLETED_PROTOCOL_SUPERSEDED",
            "gate_21_final_evaluation": "HISTORICAL_COMPLETED_PROTOCOL_SUPERSEDED_SPENT_LOCKBOX",
        },
    }
    payload["manifest_content_hash"] = content_hash(payload)
    atomic_json(output, payload)
    return output
