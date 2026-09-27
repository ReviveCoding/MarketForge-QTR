from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from .data import data_root, sha256_file
from .probe import atomic_json
from .state import ROOT

REPORT_NAMES = (
    "MARKETFORGE_QTR_TECHNICAL_REPORT.md",
    "EXECUTIVE_SUMMARY.md",
    "DATASET_CARD.md",
    "DATA_QUALITY_REPORT.md",
    "MODEL_CARD.md",
    "TRAINING_REPORT.md",
    "CALIBRATION_REPORT.md",
    "SIMULATOR_VALIDATION_REPORT.md",
    "STRATEGY_SPECIFICATION.md",
    "RISK_CONTROL_SPECIFICATION.md",
    "ABLATION_REPORT.md",
    "ROBUSTNESS_REPORT.md",
    "BACKTEST_OVERFIT_REPORT.md",
    "SYSTEM_BENCHMARK_REPORT.md",
    "MONITORING_REPORT.md",
    "FAILURE_ANALYSIS.md",
    "LIMITATIONS.md",
    "FUTURE_WORK.md",
    "RESUME_EVIDENCE.md",
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    return path


def _metric(value: object, digits: int = 4) -> str:
    return "NOT ESTABLISHED" if value is None else f"{float(value):.{digits}f}"


def _figures(root: Path, final: dict[str, object]) -> list[Path]:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    predictive = final["predictive"]
    names = ["M3T-SSL-EPT-CAL", "XGBoost CUDA"]
    f1 = [
        predictive["m3t_ssl_ept_cal"]["direction_macro_f1_calibrated"],
        predictive["xgboost_cuda"]["direction_macro_f1"],
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(names, f1, color=["#4c78a8", "#f58518"])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Strict venue/asset/instrument/day OOD final lockbox")
    ax.set_ylim(0, max(f1) * 1.25)
    fig.tight_layout()
    predictive_path = figures / "final_predictive_comparison.png"
    fig.savefig(predictive_path, dpi=160)
    plt.close(fig)

    economics = final["economic"]
    pnl = [
        economics["selected_NO_TRADE"]["realized_pnl_usd"],
        economics["m3t_common_controller_diagnostic"]["realized_pnl_usd"],
        economics["xgboost_common_controller_diagnostic"]["realized_pnl_usd"],
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(["NO_TRADE", "M3T diagnostic", "XGBoost diagnostic"], pnl, color="#e45756")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Realized PnL (USD)")
    ax.set_title("Frozen final economic policy and active diagnostics")
    fig.tight_layout()
    economic_path = figures / "final_economic_diagnostics.png"
    fig.savefig(economic_path, dpi=160)
    plt.close(fig)
    return [predictive_path, economic_path]


def generate_v2_reports() -> Path:
    root = ROOT / "reports/final_v2"
    root.mkdir(parents=True, exist_ok=True)
    manifests = data_root() / "manifests"
    final_path = ROOT / "artifacts/final_evaluation_v2.json"
    final = _json(final_path)
    if final["evaluation_count"] != 1:
        raise RuntimeError("v2 report refuses invalid final evaluation count")
    if final["result_label"] != "FINAL_EVALUATION_V2_WITH_OUTCOME_BLIND_STRUCTURAL_AMENDMENT":
        raise RuntimeError("v2 report requires the disclosed amended result label")
    ladder = _json(manifests / "model_ladder_v2.json")
    xgb = _json(manifests / "xgboost_gpu_v2.json")
    simulator = _json(manifests / "simulator_validation_v2.json")
    ablation = _json(manifests / "ablation_v2.json")
    robustness = _json(manifests / "robustness_v2.json")
    systems = _json(manifests / "systems_benchmark_v2.json")
    monitoring = _json(manifests / "monitoring_v2.json")
    inference = _json(manifests / "statistical_inference_v2.json")
    bias = _json(manifests / "selection_bias_v2.json")
    failure = _json(manifests / "failure_analysis_v2.json")
    splits = _json(manifests / "splits_v4_multi_market.json")
    acquisition = _json(manifests / "v2_public_data_acquisition.json")
    cuda = _json(ROOT / "artifacts/system/cuda_runtime.json")
    amendment_1 = _json(ROOT / "artifacts/final_freeze_amendment_v2_001.json")
    amendment_2 = _json(ROOT / "artifacts/final_freeze_amendment_v2_002.json")

    m3t_final = final["predictive"]["m3t_ssl_ept_cal"]
    xgb_final = final["predictive"]["xgboost_cuda"]
    final_econ = final["economic"]
    amendment_disclosure = """## Outcome-blind structural amendment disclosure

- The final lockbox was opened exactly once; `evaluation_count` remained 1.
- Exact XNAS/NVDA replay completed before the first structural failure.
- No predictions or final metrics existed when that first failure was discovered.
- The frozen evaluator incorrectly assumed `prediction_time` was unique.
- Exactly one timestamp had multiplicity four: 3,238 rows but 3,235 unique timestamps.
- Amendment 001 replaced timestamp-only feature/label joining with ordinal replay identity; no row was removed, duplicated, reordered, filtered, or aggregated.
- A second occurrence of the same defect appeared in the economic diagnostic after predictions were transiently computed; no result artifact or metric output was persisted or used. Amendment 002 propagated the same `capture_id` into prediction tables and diagnostic lookup. Amendment 003 cryptographically bound the execution chain.
- No targets, feature definitions, models, checkpoints, scalers, embeddings, calibration parameters, controllers, fees, ticks, multipliers, latency, risk limits, assumptions, or selection decisions changed.
- The original freeze and all structural-failure/amendment artifacts remain preserved. This result is explicitly `FINAL_EVALUATION_V2_WITH_OUTCOME_BLIND_STRUCTURAL_AMENDMENT`, not an unamended freeze execution.
"""
    technical = f"""# MarketForge-QTR v2 Technical Report

Generated {datetime.now(UTC).isoformat()}.

## Status and protocol separation

V1 remains preserved engineering/audit history. Its economics and economic targets are scientifically superseded, its final lockbox is spent, and `final_evaluation_v1.json` was not changed. V2 is the current corrected protocol.

{amendment_disclosure}

## Data actually used

One provider (Databento), three dataset/venue domains, three instruments, and three instrument-days were available: CME ES MBO for development execution labels; ICE Brent MBP-10 for L2 representation after MBO reconstruction failed following an unrecovered clear; and XNAS NVDA MBO as the one-time final lockbox. This supports strict venue/dataset, asset-class, instrument, and calendar-day OOD point estimates, not provider OOD or population inference.

## Corrected targets and accounting

V2 uses side-specific bid/ask markouts, counterfactual FIFO `SIMULATED_L3` fills, fill-conditional adverse selection, modality availability masks, grouped windows, instrument metadata, and dimensionally explicit USD accounting. The test suite proves one ES tick is 0.25 points and $12.50 per contract. Endpoint crossing is not used as an L3 fill label.

## Development findings

The six-branch factorial ladder was executed with identical model scale. M3T-SSL-EPT descended from SSL and M3T-SUP-EPT independently descended from SUP. XGBoost trained on CUDA and won the development predictive comparison. Every active development market-making strategy lost after registered costs, so Medium/Large scaling stopped and `NO_TRADE` was frozen.

## Final one-time result

| Metric | M3T-SSL-EPT-CAL | XGBoost CUDA |
|---|---:|---:|
| Direction macro-F1 | {_metric(m3t_final["direction_macro_f1_calibrated"])} | {_metric(xgb_final["direction_macro_f1"])} |
| Bid fill Brier | {_metric(m3t_final["bid_fill_brier_calibrated"])} | {_metric(xgb_final["bid_fill_brier"])} |
| Ask fill Brier | {_metric(m3t_final["ask_fill_brier_calibrated"])} | {_metric(xgb_final["ask_fill_brier"])} |

The final selected `NO_TRADE` policy realized $0 by construction. Frozen active diagnostics remained negative: M3T `${final_econ["m3t_common_controller_diagnostic"]["realized_pnl_usd"]:.3f}` and XGBoost `${final_econ["xgboost_common_controller_diagnostic"]["realized_pnl_usd"]:.3f}`. Selection did not change after the lockbox.

## Simulator evidence

The immutable final replay reports zero reconstruction errors. Historical real-order queue validation observed 30 fills, with 28 queue-eligible at fill (93.33%) and mean queue-ahead error 49.13 shares. This is calibration/error evidence, not a claim of perfect observed fills. Paired MBO→MBP validation was not rerun after authorization because replay was expressly prohibited.

## Statistical interpretation

The primary unit is instrument × trading day. Development has two units and final evaluation one unit. Confidence intervals, White/SPA, CSCV/PBO, deflated Sharpe, and population transfer inference are `NOT ESTABLISHED` / `SKIPPED_INSUFFICIENT_SAMPLE`. Row-level counts are not presented as independent day-level evidence.

## Conclusion

V2 corrects the v1 scientific defects and produces an honest negative economic conclusion. The complex neural system does not establish an economic edge; XGBoost is competitive or stronger predictively, and no-trade remains the only supported economic selection.
"""
    _write(root, "MARKETFORGE_QTR_TECHNICAL_REPORT.md", technical)
    _write(
        root,
        "EXECUTIVE_SUMMARY.md",
        f"""# Executive Summary

MarketForge-QTR v2 completed the corrected protocol and a one-time amended final evaluation. The final XNAS/NVDA lockbox was opened once. An outcome-blind ordinal row-identity amendment fixed a non-unique timestamp assumption without changing or removing data, models, calibration, controller choices, or assumptions.

M3T final direction macro-F1 was {_metric(m3t_final["direction_macro_f1_calibrated"])}; XGBoost was {_metric(xgb_final["direction_macro_f1"])}. Both frozen active economic diagnostics lost money after costs (M3T `${final_econ["m3t_common_controller_diagnostic"]["realized_pnl_usd"]:.3f}`, XGBoost `${final_econ["xgboost_common_controller_diagnostic"]["realized_pnl_usd"]:.3f}`). The preselected `NO_TRADE` policy remained unchanged. Population claims are not established from three instrument-days.
""",
    )
    _write(
        root,
        "DATASET_CARD.md",
        """# Dataset Card

- Provider: Databento official public samples.
- Development: CME/ES MBO (`SIMULATED_L3`) and ICE/BRN MBP-10 (L2 only; L3 masked).
- Final: XNAS/NVDA MBO, 2025-09-16, 3,238 immutable sampled replay rows.
- Scope: one provider, three dataset/venue domains, three instruments, three instrument-days.
- FX: not acquired because a documented official unattended Dukascopy path was not established.
- License/access: official public sources only; no purchase, mirror, bypass, or redistribution claim.
- Final data were excluded from tuning and SSL pretraining.
""",
    )
    _write(
        root,
        "DATA_QUALITY_REPORT.md",
        f"""# Data Quality Report

Development leakage checks all passed under split fingerprint `{splits["split_fingerprint"]}`. ES reconstructed-book validation passed the available-data gate. ICE MBO failed after an unrecovered clear and was excluded from L3 labels rather than repaired by invention. Final replay had zero reconstruction errors and retained all 3,238 rows. Storage reserve after acquisition was {acquisition["free_bytes_after_acquisition"] / 1024**3:.2f} GiB against a 40 GiB minimum.
""",
    )
    branches = ladder["branches"]
    _write(
        root,
        "MODEL_CARD.md",
        f"""# Model Card

Primary neural comparison: M3T-SSL-EPT-CAL, {branches["M3T-SSL-EPT"]["parameters"]:,} parameters, side-specific economic heads, source/instrument embeddings, and modality masks. Unseen identifiers use the frozen mean-fitted-embedding rule. Primary tabular comparator: actual XGBoost {xgb["xgboost_version"]} trained on `{xgb["devices_from_booster_config"][0]}`. Intended use is public-sample research, not live trading.
""",
    )
    _write(
        root,
        "TRAINING_REPORT.md",
        f"""# Training Report

The executed ladder was M3T-SUP, M3T-SSL, M3T-SUP-EPT, M3T-SSL-EPT, M3T-SUP-EPT-CAL, and M3T-SSL-EPT-CAL. Ancestry checks all passed. Training used CUDA/BF16 on `{ladder["device"]}` with peak recorded VRAM {ladder["peak_vram_bytes"] / 1024**2:.1f} MiB. Final-lockbox data were absent from training and pretraining. Medium/Large scaling stopped after XGBoost won development validation and all active economics were negative.
""",
    )
    _write(
        root,
        "CALIBRATION_REPORT.md",
        f"""# Calibration Report

Calibration was fit on MODEL_VALIDATION and evaluated on STRATEGY_VALIDATION before the final lockbox. Frozen SSL-EPT temperature was {branches["M3T-SSL-EPT-CAL"]["calibration"]["direction_temperature"]:.6f}. Final calibrated bid/ask fill Brier scores were {_metric(m3t_final["bid_fill_brier_calibrated"])} and {_metric(m3t_final["ask_fill_brier_calibrated"])}. Calibration was not refit after final access.
""",
    )
    queue = final["simulator_validation"]["historical_order_validation"]
    _write(
        root,
        "SIMULATOR_VALIDATION_REPORT.md",
        f"""# Simulator Validation Report

Development available-data gate: `{simulator["passed_available_data_gate"]}`. Final immutable XNAS replay reconstruction errors: 0. Historical-order diagnostic: {queue["actual_fills_observed"]} actual fills, {queue["queue_eligible_at_actual_fill"]} eligible, rate {queue["eligibility_rate"]:.4f}, mean queue-ahead error {queue["mean_queue_ahead_error_at_actual_fill"]:.2f} shares. Labels are `SIMULATED_L3`, never OBSERVED, and assume a small order with no endogenous impact. Final MBO replay/paired-MBP comparison was not rerun after the amendment authorization prohibited replay.
""",
    )
    _write(
        root,
        "STRATEGY_SPECIFICATION.md",
        """# Strategy Specification

Frozen selection: `NO_TRADE`. Frozen diagnostic controller quotes a side only when fill probability × (side-specific predicted markout in ticks − registered cost in ticks) is positive and adverse probability is below 0.7. AS/GLFT approximations remain honestly named `AS_LIKE_HEURISTIC` and `GLFT_LIKE_HEURISTIC`; faithful published calibration was not identified by the limited public samples.
""",
    )
    _write(
        root,
        "RISK_CONTROL_SPECIFICATION.md",
        """# Risk Control Specification

The authoritative final policy places no orders. Diagnostic fills use one unit, explicit entry/exit fees, registered tick metadata, a 0.25-tick liquidation cost, zero assumed entry slippage/hedge cost, immediate markout liquidation, and zero ending inventory. All price PnL, fees, slippage, liquidation, hedge cost, realized/unrealized PnL, and NAV are tracked separately in USD.
""",
    )
    _write(
        root,
        "ABLATION_REPORT.md",
        f"""# Ablation Report

Base development macro-F1: {_metric(ablation["base"]["direction_macro_f1"])}. Depth dropout: {_metric(ablation["depth_dropout"]["direction_macro_f1"])}; L3 modality mask off: {_metric(ablation["l3_modality_mask_off"]["direction_macro_f1"])}; source embedding swap: {_metric(ablation["source_embedding_swap"]["direction_macro_f1"])}. Architecture retraining ablations stopped after the resource-aware funnel criterion failed.
""",
    )
    _write(
        root,
        "ROBUSTNESS_REPORT.md",
        f"""# Robustness Report

No active strategy was profitable at registered base cost. Liquidation-cost stress remained negative, and the report's robust-edge flag is `{robustness["robust_edge"]}`. Latency resimulation is not established beyond the registered 1 ms label latency. ICE execution evidence is excluded after reconstruction failure.
""",
    )
    _write(
        root,
        "BACKTEST_OVERFIT_REPORT.md",
        f"""# Backtest Overfit Report

Statistical inference status: `{inference["status"]}`. Selection-bias status: `{bias["status"]}`. White Reality Check, SPA, CSCV/PBO, and deflated Sharpe are `NOT ESTABLISHED` because only two development instrument-day units exist. The final lockbox was evaluated once, and no choice changed afterward.
""",
    )
    _write(
        root,
        "SYSTEM_BENCHMARK_REPORT.md",
        f"""# System Benchmark Report

GPU: `{systems["device"]}`. M3T p50 inference was {systems["m3t_inference"][0]["p50_ms"]:.3f} ms at batch 1 and {systems["m3t_inference"][1]["p50_ms"]:.3f} ms at batch 64. The Windows-safe benchmark used DataLoader workers={systems["dataloader"]["workers"]}. GPU training was serialized through the project lock.
""",
    )
    _write(
        root,
        "MONITORING_REPORT.md",
        f"""# Monitoring Report

Shadow monitoring produced {monitoring["rows"]} rows with states `{monitoring["states"]}`. Execution remained NO_TRADE. The monitoring log is `{monitoring["log"]["path"]}` with SHA-256 `{monitoring["log"]["sha256"]}`.
""",
    )
    _write(
        root,
        "FAILURE_ANALYSIS.md",
        f"""# Failure Analysis

MarketForge's SSL-EPT branch improved over SUP-EPT on development direction/fill calibration, but XGBoost won the development primary comparison and every active economic policy lost after costs. Final M3T direction macro-F1 exceeded XGBoost ({_metric(m3t_final["direction_macro_f1_calibrated"])} vs {_metric(xgb_final["direction_macro_f1"])}), while both active economic diagnostics lost. Complexity is not economically justified: `{failure["complexity_justified"]}`.

The final evaluator also failed twice on the same non-unique timestamp assumption. Both failures and the complete 001→003 outcome-blind amendment chain are preserved; neither was hidden or relabeled as an unamended freeze.
""",
    )
    _write(
        root,
        "LIMITATIONS.md",
        """# Limitations

- Only three instrument-days and one provider; no population inference or provider OOD.
- Public-sample, small-order, no-impact `SIMULATED_L3`; not observed fills or live PnL.
- ICE L3 excluded after an unrecovered clear; FX unavailable.
- Final paired MBO/MBP validation was not rerun because the authorized continuation prohibited replay.
- No faithful AS/GLFT intensity calibration; heuristics are explicitly named.
- Outcome-blind structural amendments were necessary after lockbox open and are fully disclosed.
""",
    )
    _write(
        root,
        "FUTURE_WORK.md",
        """# Future Work

Acquire lawful multi-day, multi-provider data; validate FIFO fills across more instruments; add faithful AS/GLFT estimation where identifiable; implement capture IDs at replay creation before any future freeze; add multi-day dependence-aware inference; and revisit model scaling only if corrected development economics become positive after costs.
""",
    )
    _write(
        root,
        "RESUME_EVIDENCE.md",
        f"""# Resume Evidence

- Final label: `{final["result_label"]}`
- Evaluation count: `{final["evaluation_count"]}`
- Final result SHA-256: `{sha256_file(final_path)}`
- Original freeze SHA-256: `{final["freeze"]["sha256"]}`
- Amendment 001 content hash: `{amendment_1["amendment_content_hash"]}`
- Amendment 002 ID: `{amendment_2["amendment_id"]}`
- CUDA available/verified: `{cuda.get("cuda_available", cuda.get("torch_cuda_available"))}`
- Exact status: `python -m marketforge.cli status`
- Exact report regeneration: `python -m marketforge.cli v2-report`
- Further `v2-final-evaluation` calls return the existing immutable result and do not recompute.
""",
    )
    figures = _figures(root, final)
    report_files = [root / name for name in REPORT_NAMES]
    input_paths = [
        final_path,
        ROOT / "artifacts/final_freeze_manifest_v2.json",
        ROOT / "artifacts/final_freeze_amendment_v2_001.json",
        ROOT / "artifacts/final_freeze_amendment_v2_002.json",
        ROOT / "artifacts/final_freeze_amendment_v2_003.json",
        ROOT / "artifacts/final_lockbox_v2_structural_failure.json",
        ROOT / "artifacts/final_lockbox_v2_structural_failure_002.json",
    ]
    manifest = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "result_label": final["result_label"],
        "evaluation_count": 1,
        "reports": {
            path.name: {"path": str(path), "sha256": sha256_file(path)} for path in report_files
        },
        "figures": {
            path.name: {"path": str(path), "sha256": sha256_file(path)} for path in figures
        },
        "authoritative_inputs": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths
        },
        "v1_preserved_separate": True,
        "population_inference_established": False,
        "headline_live_pnl_permitted": False,
    }
    atomic_json(root / "RESULTS_MANIFEST.json", manifest)
    return root / "MARKETFORGE_QTR_TECHNICAL_REPORT.md"
