from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

from .advanced import (
    run_inference_and_selection_bias,
    run_monitoring,
    run_robustness,
    run_systems_benchmark,
    run_transfer_study,
    train_gpu_xgboost,
)
from .audit import audit_mbo, read_quality
from .baselines import run_baselines
from .calibration import calibrate_checkpoint
from .canonical import canonicalize_mbo
from .data import CME_MBO_SAMPLE, acquire, acquisition_summary, data_root
from .economics import run_development_economics, validate_risk_gate
from .features import reconstruct_features
from .finalization import freeze_final_protocol, run_final_evaluation
from .ladder import run_posttraining, run_pretraining_ladder, run_supervised_diagnostic
from .probe import atomic_json, cuda_probe, environment_probe
from .reporting import generate_reports
from .simulator_validation import validate_simulator
from .smoke import run_smoke
from .splits import build_splits, verify_leakage
from .state import content_hash, record, rows
from .v2_completion_audit import run_v2_completion_audit
from .v2_finalization import freeze_final_v2, run_final_evaluation_v2, write_pre_final_audit_v2
from .v2_posthoc import run_v2_posthoc
from .v2_reporting import generate_v2_reports

COMMANDS = [
    "bootstrap",
    "probe",
    "data",
    "audit",
    "canonicalize",
    "study",
    "features",
    "splits",
    "smoke",
    "baselines",
    "pretrain",
    "posttrain",
    "calibrate",
    "simulator-validation",
    "backtest",
    "experiments",
    "analyze",
    "report",
    "full-dev",
    "freeze-final",
    "final-evaluation",
    "full",
    "status",
    "resume",
    "v2-audit",
    "v2-freeze-final",
    "v2-final-evaluation",
    "v2-report",
    "v2-completion-audit",
    "v2-posthoc",
]


def run_probe() -> None:
    digest = content_hash({"stage": "probe", "version": 1})
    record("gate_00_environment", "RUNNING", digest, [])
    env = environment_probe()
    record("gate_00_environment", "SUCCEEDED", digest, [str(env)])
    record("gate_01_cuda", "RUNNING", digest, [])
    cuda = cuda_probe()
    record("gate_01_cuda", "SUCCEEDED", digest, [str(cuda)])
    print(json.dumps({"environment": str(env), "cuda": str(cuda)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="marketforge")
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args()
    if args.command in {"bootstrap", "probe"}:
        run_probe()
    elif args.command == "data":
        digest = content_hash({"stage": "data", "spec": CME_MBO_SAMPLE.url})
        record("gate_02_licenses_data", "RUNNING", digest, [])
        path = acquire(CME_MBO_SAMPLE)
        evidence = [str(path), str(path.parents[2] / "manifests" / "databento_GLBX.MDP3_mbo.json")]
        record("gate_02_licenses_data", "SUCCEEDED", digest, evidence)
        print(acquisition_summary(path))
    elif args.command == "audit":
        raw = acquire(CME_MBO_SAMPLE)
        digest = content_hash({"stage": "audit", "sha": acquisition_summary(raw)})
        record("gate_03_raw_qa", "RUNNING", digest, [])
        report = audit_mbo(raw)
        quality = read_quality(report)
        if quality["quality_status"] == "ERROR":
            record("gate_03_raw_qa", "FAILED", digest, [str(report)], "fatal raw QA errors")
            raise SystemExit("Raw QA failed; see report")
        record("gate_03_raw_qa", "SUCCEEDED", digest, [str(report)])
        print(json.dumps(quality, indent=2))
    elif args.command == "canonicalize":
        raw = acquire(CME_MBO_SAMPLE)
        digest = content_hash(
            {"stage": "canonicalize", "raw": acquisition_summary(raw), "parser": 2}
        )
        record("gate_04_canonicalization", "RUNNING", digest, [])
        marker = canonicalize_mbo(raw)
        record("gate_04_canonicalization", "SUCCEEDED", digest, [str(marker)])
        print(marker.read_text(encoding="utf-8"))
    elif args.command == "features":
        canonical_marker = canonicalize_mbo(acquire(CME_MBO_SAMPLE))
        canonical = json.loads(canonical_marker.read_text(encoding="utf-8"))
        marker = reconstruct_features(Path(canonical["canonical_root"]))
        print(marker.read_text(encoding="utf-8"))
    elif args.command == "splits":
        canonical_marker = canonicalize_mbo(acquire(CME_MBO_SAMPLE))
        canonical = json.loads(canonical_marker.read_text(encoding="utf-8"))
        feature_marker = reconstruct_features(Path(canonical["canonical_root"]))
        features = json.loads(feature_marker.read_text(encoding="utf-8"))
        digest = content_hash({"stage": "splits", "features": features, "version": 3})
        record("gate_05_leakage", "RUNNING", digest, [])
        marker = build_splits(Path(features["feature_file"]))
        checks = verify_leakage(marker)
        if not checks["passed"]:
            record("gate_05_leakage", "FAILED", digest, [str(marker)], json.dumps(checks))
            raise SystemExit("Leakage checks failed")
        report = marker.parent / "leakage_checks_v3.json"
        atomic_json(report, checks)
        record("gate_05_leakage", "SUCCEEDED", digest, [str(marker), str(report)])
        print(json.dumps({"manifest": str(marker), "checks": checks}, indent=2))
    elif args.command == "baselines":
        report = run_baselines()
        print(report.read_text(encoding="utf-8"))
    elif args.command == "smoke":
        report = run_smoke()
        print(report.read_text(encoding="utf-8"))
    elif args.command == "pretrain":
        report = run_pretraining_ladder()
        payload = json.loads(report.read_text(encoding="utf-8"))
        gate7 = content_hash(payload["m3t_sup"])
        gate8 = content_hash(payload["ssl_pretraining"])
        record(
            "gate_07_supervised_m3t",
            "SUCCEEDED",
            gate7,
            [payload["m3t_sup"]["checkpoint"], str(report)],
        )
        record(
            "gate_08_ssl",
            "SUCCEEDED",
            gate8,
            [
                payload["ssl_pretraining"]["checkpoint"],
                payload["m3t_ssl_finetuned"]["checkpoint"],
                str(report),
            ],
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "experiments":
        report = run_supervised_diagnostic()
        print(report.read_text(encoding="utf-8"))
    elif args.command == "posttrain":
        report = run_posttraining()
        payload = json.loads(report.read_text(encoding="utf-8"))
        record(
            "gate_09_economic_posttraining",
            "SUCCEEDED",
            content_hash(payload),
            [payload["result"]["checkpoint"], str(report)],
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "calibrate":
        ept = json.loads(
            (Path(data_root()) / "manifests" / "m3t_ept_v1.json").read_text(encoding="utf-8")
        )
        report = calibrate_checkpoint(ept["result"]["checkpoint"])
        payload = json.loads(report.read_text(encoding="utf-8"))
        record("gate_10_calibration", "SUCCEEDED", content_hash(payload), [str(report)])
        print(json.dumps(payload, indent=2))
    elif args.command == "simulator-validation":
        canonical_marker = canonicalize_mbo(acquire(CME_MBO_SAMPLE))
        canonical = json.loads(canonical_marker.read_text(encoding="utf-8"))
        report = validate_simulator(Path(canonical["canonical_root"]))
        payload = json.loads(report.read_text(encoding="utf-8"))
        status = "SUCCEEDED" if payload["passed_available_data_gate"] else "FAILED"
        record("gate_11_simulator_validation", status, content_hash(payload), [str(report)])
        if status == "FAILED":
            raise SystemExit("Simulator validation failed")
        print(json.dumps(payload, indent=2))
    elif args.command == "backtest":
        risk = validate_risk_gate()
        risk_payload = json.loads(risk.read_text(encoding="utf-8"))
        if not risk_payload["passed"]:
            record("gate_12_risk_gate", "FAILED", content_hash(risk_payload), [str(risk)])
            raise SystemExit("Risk gate validation failed")
        record("gate_12_risk_gate", "SUCCEEDED", content_hash(risk_payload), [str(risk)])
        report = run_development_economics()
        payload = json.loads(report.read_text(encoding="utf-8"))
        record("gate_13_development_economics", "SUCCEEDED", content_hash(payload), [str(report)])
        print(json.dumps(payload, indent=2))
    elif args.command == "analyze":
        model, xgb_report = train_gpu_xgboost()
        transfer = run_transfer_study(model)
        transfer_payload = json.loads(transfer.read_text(encoding="utf-8"))
        record(
            "gate_14_transfer",
            "BLOCKED_EXTERNAL_DATA",
            content_hash(transfer_payload),
            [str(transfer), str(xgb_report)],
            transfer_payload["reason"],
        )
        robustness = run_robustness(model)
        record(
            "gate_15_robustness",
            "SUCCEEDED",
            content_hash(json.loads(robustness.read_text(encoding="utf-8"))),
            [str(robustness)],
        )
        systems = run_systems_benchmark()
        record(
            "gate_16_systems",
            "SUCCEEDED",
            content_hash(json.loads(systems.read_text(encoding="utf-8"))),
            [str(systems)],
        )
        inference, bias = run_inference_and_selection_bias()
        record(
            "gate_17_statistical_inference",
            "SUCCEEDED",
            content_hash(json.loads(inference.read_text(encoding="utf-8"))),
            [str(inference)],
        )
        record(
            "gate_18_selection_bias",
            "SUCCEEDED",
            content_hash(json.loads(bias.read_text(encoding="utf-8"))),
            [str(bias)],
        )
        monitoring = run_monitoring(model)
        record(
            "gate_19_monitoring",
            "SUCCEEDED",
            content_hash(json.loads(monitoring.read_text(encoding="utf-8"))),
            [str(monitoring)],
        )
        print(
            json.dumps(
                {
                    "xgboost": str(xgb_report),
                    "transfer": str(transfer),
                    "robustness": str(robustness),
                    "systems": str(systems),
                    "inference": str(inference),
                    "selection_bias": str(bias),
                    "monitoring": str(monitoring),
                },
                indent=2,
            )
        )
    elif args.command == "freeze-final":
        record(
            "gate_06_baseline_reproduction",
            "BLOCKED_EXTERNAL_DATA",
            content_hash({"source": "FI-2010", "download": "disabled"}),
            [str(Path("BLOCKERS.md"))],
            "authoritative Fairdata download is not enabled",
        )
        report = freeze_final_protocol()
        payload = json.loads(report.read_text(encoding="utf-8"))
        record("gate_20_final_freeze", "SUCCEEDED", content_hash(payload), [str(report)])
        print(json.dumps(payload, indent=2))
    elif args.command == "final-evaluation":
        report = run_final_evaluation()
        payload = json.loads(report.read_text(encoding="utf-8"))
        record("gate_21_final_evaluation", "SUCCEEDED", content_hash(payload), [str(report)])
        print(json.dumps(payload, indent=2))
    elif args.command == "report":
        report = generate_reports()
        manifest = report.parent / "RESULTS_MANIFEST.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        record(
            "gate_22_report_audit", "SUCCEEDED", content_hash(payload), [str(report), str(manifest)]
        )
        print(json.dumps({"technical_report": str(report), "manifest": str(manifest)}, indent=2))
    elif args.command == "study":
        print((Path("docs") / "desktop_study.md").read_text(encoding="utf-8"))
    elif args.command in {"full-dev", "full"}:
        stages = [
            ("probe", Path("artifacts/system/cuda_runtime.json")),
            ("data", Path("data/manifests/databento_GLBX.MDP3_mbo.json")),
            ("audit", Path("data/manifests/qa_databento_GLBX.MDP3_mbo.json")),
            ("canonicalize", Path("data/manifests/canonical_databento_mbo_v2.json")),
            ("features", Path("data/manifests/features_databento_mbo_v3.json")),
            ("splits", Path("data/manifests/leakage_checks_v3.json")),
            ("smoke", Path("data/manifests/smoke_v1.json")),
            ("baselines", Path("data/manifests/baseline_results_v1.json")),
            ("pretrain", Path("data/manifests/model_ladder_pretrain_v1.json")),
            ("experiments", Path("data/manifests/m3t_supervised_failure_diagnostic_v1.json")),
            ("posttrain", Path("data/manifests/m3t_ept_v1.json")),
            ("calibrate", Path("data/manifests/calibration_m3t_ept_v1.json")),
            ("simulator-validation", Path("data/manifests/simulator_validation_v1.json")),
            ("backtest", Path("data/manifests/development_economics_v1.json")),
            ("analyze", Path("data/manifests/monitoring_v1.json")),
        ]
        if args.command == "full":
            stages.extend(
                [
                    ("freeze-final", Path("artifacts/final_freeze_manifest.json")),
                    ("final-evaluation", Path("artifacts/final_evaluation_v1.json")),
                    ("report", Path("reports/final/RESULTS_MANIFEST.json")),
                ]
            )
        for stage, marker in stages:
            if marker.exists():
                print(f"SKIP {stage}: {marker} exists")
                continue
            subprocess.run([sys.executable, "-m", "marketforge.cli", stage], check=True)
    elif args.command in {"status", "resume"}:
        run_state = Path("RUN_STATE.json")
        if run_state.exists():
            payload = json.loads(run_state.read_text(encoding="utf-8"))
            payload["pipeline_state_rows"] = rows()
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(rows(), indent=2))
    elif args.command == "v2-audit":
        report = write_pre_final_audit_v2()
        print(report.read_text(encoding="utf-8"))
    elif args.command == "v2-freeze-final":
        report = freeze_final_v2()
        print(report.read_text(encoding="utf-8"))
    elif args.command == "v2-final-evaluation":
        report = run_final_evaluation_v2()
        print(report.read_text(encoding="utf-8"))
    elif args.command == "v2-report":
        report = generate_v2_reports()
        print(report)
    elif args.command == "v2-completion-audit":
        report = run_v2_completion_audit()
        print(report.read_text(encoding="utf-8"))
    elif args.command == "v2-posthoc":
        report = run_v2_posthoc()
        print(report.read_text(encoding="utf-8"))
    else:
        raise SystemExit(
            f"Stage {args.command!r} is registered but not implemented yet; refusing false success"
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
