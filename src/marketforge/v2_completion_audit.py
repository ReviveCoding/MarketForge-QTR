from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .data import sha256_file
from .probe import atomic_json
from .state import ROOT
from .v2_reporting import REPORT_NAMES

PROTECTED_HASHES = {
    "artifacts/final_evaluation_v1.json": "d7720cd2d0884538145e33ac54758ff97712f13da3ae413cdb5c8fc78c19a5f2",
    "artifacts/final_freeze_manifest_v2.json": "44f269ec24c6f9480b6f2c06cba94d85c7adf7418af60cee16346e46e0c919b6",
    "data/final_lockbox_v2/OPENED_ONCE.json": "b3bfd22986db37ed8c436b687d77acf5fcba604ce283bdd2fc4234d9bc009f7c",
    "data/final_lockbox_v2/features_v4_multi_market/xnas_nvda.parquet": "f54527d4ba068bad41752556ba337642583abac27e6c830dad5f4c0435012571",
    "data/final_lockbox_v2/labels_v2_l3/xnas_nvda.parquet": "26ec8a272e13904af438a07d250e04a74e5eead61149d7b6ee5b6d64f193b3d5",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_v2_completion_audit() -> Path:
    checks: dict[str, bool] = {}
    observed_hashes = {}
    for relative, expected in PROTECTED_HASHES.items():
        path = ROOT / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        observed_hashes[relative] = observed
        checks[f"protected_hash::{relative}"] = observed == expected

    final_path = ROOT / "artifacts/final_evaluation_v2.json"
    final = _json(final_path)
    checks["final_result_label"] = (
        final["result_label"] == "FINAL_EVALUATION_V2_WITH_OUTCOME_BLIND_STRUCTURAL_AMENDMENT"
    )
    checks["evaluation_count_one"] = final["evaluation_count"] == 1
    checks["no_post_lockbox_selection_change"] = not final["economic"][
        "selection_changed_after_lockbox"
    ]
    checks["population_inference_not_established"] = (
        final["population_inference"] == "NOT_ESTABLISHED"
    )
    checks["no_test_or_final_tuning"] = not final["test_or_final_tuning"]
    checks["v1_excluded_from_selection"] = not final["v1_artifacts_used_for_selection"]
    checks["final_accounting_m3t"] = (
        final["economic"]["m3t_common_controller_diagnostic"]["accounting_identity_error_usd"] == 0
    )
    checks["final_accounting_xgboost"] = (
        final["economic"]["xgboost_common_controller_diagnostic"]["accounting_identity_error_usd"]
        == 0
    )
    idempotency = _json(ROOT / "artifacts/final_evaluation_v2_idempotency.json")
    checks["final_evaluation_idempotent"] = bool(
        idempotency["before_sha256"] == sha256_file(final_path)
        and idempotency["after_sha256"] == sha256_file(final_path)
        and idempotency["mtime_unchanged"]
        and idempotency["evaluation_count"] == 1
    )

    amendments = final["freeze_amendments"]
    checks["three_amendments_recorded"] = len(amendments) == 3
    for index, item in enumerate(amendments, start=1):
        path = Path(item["path"])
        checks[f"amendment_{index}_hash"] = path.is_file() and sha256_file(path) == item["sha256"]
    amendment_1 = _json(ROOT / "artifacts/final_freeze_amendment_v2_001.json")
    checks["ordinal_structure_exact"] = amendment_1["structure"] == {
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
    checks["amendment_outcome_blind"] = not amendment_1[
        "outcome_prediction_pnl_selection_or_tuning_used_to_choose_amendment"
    ]
    checks["no_row_change"] = not amendment_1[
        "rows_removed_duplicated_reordered_filtered_or_aggregated"
    ]
    checks["no_replay_rerun"] = not amendment_1["raw_canonical_or_replay_rerun"]

    report_root = ROOT / "reports/final_v2"
    report_manifest = _json(report_root / "RESULTS_MANIFEST.json")
    checks["report_evaluation_count_one"] = report_manifest["evaluation_count"] == 1
    for name in REPORT_NAMES:
        path = report_root / name
        item = report_manifest["reports"].get(name)
        checks[f"report_hash::{name}"] = (
            path.is_file() and item is not None and sha256_file(path) == item["sha256"]
        )
    for name, item in report_manifest["figures"].items():
        path = Path(item["path"])
        checks[f"figure_hash::{name}"] = path.is_file() and sha256_file(path) == item["sha256"]

    run_state = _json(ROOT / "RUN_STATE.json")
    checks["run_state_complete"] = run_state["project_status"].startswith(
        "V2_CORRECTED_PROTOCOL_COMPLETE"
    )
    checks["statistical_gate_skipped"] = (
        run_state["v2_stages"]["v2_gate_10_statistics_selection_bias"]["status"]
        == "SKIPPED_INSUFFICIENT_SAMPLE"
    )
    cuda = _json(ROOT / "artifacts/system/cuda_runtime.json")
    checks["cuda_forward_backward_verified"] = bool(
        cuda["cuda_available"]
        and cuda["backward_gradient_finite"]
        and cuda["tensor_device"] == "cuda:0"
        and cuda["model_device"] == "cuda:0"
    )
    free_bytes = shutil.disk_usage(ROOT).free
    checks["storage_reserve_40_gib"] = free_bytes >= 40 * 1024**3
    checks["pre_final_audit_exists"] = (ROOT / "PRE_FINAL_AUDIT_V2.md").is_file()
    checks["original_freeze_exists"] = (ROOT / "artifacts/final_freeze_manifest_v2.json").is_file()

    output = ROOT / "artifacts/v2_completion_audit.json"
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "SUCCEEDED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "protected_hashes": observed_hashes,
        "final_evaluation_sha256": sha256_file(final_path),
        "report_manifest_sha256": sha256_file(report_root / "RESULTS_MANIFEST.json"),
        "free_bytes": free_bytes,
        "external_blockers": [
            "FI-2010 authoritative download disabled",
            "official unattended Dukascopy acquisition not established",
        ],
        "non_success_states_preserved": [
            "SKIPPED_INSUFFICIENT_SAMPLE for population inference/selection-bias diagnostics",
            "ICE MBO L3 excluded after unrecovered clear",
            "Medium/Large scaling stopped by development funnel",
        ],
    }
    atomic_json(output, payload)
    if payload["status"] != "SUCCEEDED":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"v2 completion audit failed: {failed}")
    return output
