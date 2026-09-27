from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .data import data_root, sha256_file
from .probe import atomic_json
from .state import ROOT, content_hash

POSTHOC_ROOTS = (
    ROOT / "artifacts/posthoc_v2",
    ROOT / "results/posthoc_v2",
    ROOT / "reports/posthoc_v2",
)


def _protected_paths() -> list[Path]:
    fixed = [
        ROOT / "artifacts/final_evaluation_v2.json",
        ROOT / "artifacts/final_freeze_manifest_v2.json",
        ROOT / "artifacts/final_freeze_amendment_v2_001.json",
        ROOT / "artifacts/final_freeze_amendment_v2_002.json",
        ROOT / "artifacts/final_freeze_amendment_v2_003.json",
        ROOT / "reports/final_v2/RESULTS_MANIFEST.json",
        ROOT / "reports/final_v2/RESUME_EVIDENCE.md",
        data_root() / "final_lockbox_v2/OPENED_ONCE.json",
        data_root() / "final_lockbox_v2/features_v4_multi_market/xnas_nvda.parquet",
        data_root() / "final_lockbox_v2/labels_v2_l3/xnas_nvda.parquet",
        data_root() / "manifests/model_ladder_v2.json",
        data_root() / "manifests/development_economics_v2.json",
    ]
    patterns = (
        "artifacts/checkpoints_v2/*.pt",
        "artifacts/xgboost_v2/*.json",
        "data/manifests/m3t-*-cal_v2.json",
    )
    paths = list(fixed)
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protected v2 files missing: {missing}")
    return sorted(set(paths), key=lambda path: str(path.relative_to(ROOT)).lower())


def protected_inventory() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in _protected_paths()
    }


def create_posthoc_charter() -> Path:
    """Seal the non-authoritative boundary without opening outcome columns."""
    audit_path = ROOT / "artifacts/v2_completion_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    final_path = ROOT / "artifacts/final_evaluation_v2.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if audit.get("status") != "SUCCEEDED" or audit.get("passed") != audit.get("total"):
        raise RuntimeError("authoritative v2 completion audit does not pass")
    if final.get("evaluation_count") != 1:
        raise RuntimeError("authoritative v2 evaluation_count is not one")
    inventory = protected_inventory()
    output = POSTHOC_ROOTS[0] / "POSTHOC_CHARTER.json"
    payload = {
        "schema_version": 1,
        "analysis_layer": "MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authoritative_v2_unchanged": True,
        "posthoc_only": True,
        "no_posthoc_model_selection": True,
        "no_final_recalibration": True,
        "no_threshold_tuning_on_final": True,
        "no_population_inference": True,
        "final_data_use": "DESCRIPTIVE_EXPLORATORY_DIAGNOSTICS_ONLY",
        "authoritative_evaluation_count": 1,
        "completion_audit": {
            "path": "artifacts/v2_completion_audit.json",
            "status": audit["status"],
            "passed": audit["passed"],
            "total": audit["total"],
            "sha256": sha256_file(audit_path),
        },
        "authoritative_final_evaluation": {
            "path": "artifacts/final_evaluation_v2.json",
            "sha256": inventory["artifacts/final_evaluation_v2.json"],
        },
        "final_report_manifest": {
            "path": "reports/final_v2/RESULTS_MANIFEST.json",
            "sha256": inventory["reports/final_v2/RESULTS_MANIFEST.json"],
        },
        "protected_inventory": inventory,
        "permitted_output_roots": [
            "artifacts/posthoc_v2",
            "results/posthoc_v2",
            "reports/posthoc_v2",
        ],
        "prohibited_actions": [
            "retraining",
            "recalibration_on_final",
            "threshold_tuning_on_final",
            "controller_or_model_reselection",
            "authoritative_claim_revision",
            "lockbox_replay_or_second_opening",
        ],
    }
    payload["charter_content_hash"] = content_hash(payload)
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior["protected_inventory"] != inventory:
            raise RuntimeError("protected v2 inventory changed since post-hoc charter")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return output


def verify_posthoc_boundary() -> dict[str, object]:
    charter_path = POSTHOC_ROOTS[0] / "POSTHOC_CHARTER.json"
    charter = json.loads(charter_path.read_text(encoding="utf-8"))
    current = protected_inventory()
    matches = current == charter["protected_inventory"]
    final = json.loads((ROOT / "artifacts/final_evaluation_v2.json").read_text(encoding="utf-8"))
    return {
        "protected_inventory_unchanged": matches,
        "evaluation_count_one": final.get("evaluation_count") == 1,
        "current_inventory": current,
        "charter_hash": sha256_file(charter_path),
    }
