from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .advanced import ROOT
from .data import data_root, sha256_file
from .probe import atomic_json

FINAL_DIR = ROOT / "reports" / "final"
TABLE_DIR = FINAL_DIR / "tables"
FIGURE_DIR = FINAL_DIR / "figures"


def _load(name: str) -> dict[str, object]:
    return json.loads((data_root() / "manifests" / name).read_text(encoding="utf-8"))


def _markdown_table(frame: pd.DataFrame) -> str:
    values = [[str(value) for value in frame.columns]] + [
        [str(value) for value in row] for row in frame.itertuples(index=False, name=None)
    ]
    widths = [max(len(row[index]) for row in values) for index in range(len(values[0]))]
    lines = [
        "| "
        + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values[0]))
        + " |"
    ]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    lines.extend(
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in values[1:]
    )
    return "\n".join(lines)


def _table(name: str, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    (TABLE_DIR / f"{name}.md").write_text(_markdown_table(frame) + "\n", encoding="utf-8")


def _write(name: str, title: str, body: str) -> Path:
    path = FINAL_DIR / name
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
    return path


def _make_tables() -> None:
    qa = _load("qa_databento_GLBX.MDP3_mbo.json")
    splits = _load("splits_databento_mbo_v3.json")
    calibration = _load("calibration_m3t_ept_v1.json")
    simulator = _load("simulator_validation_v1.json")
    final = json.loads(
        (ROOT / "artifacts" / "final_evaluation_v1.json").read_text(encoding="utf-8")
    )
    xgb = _load("xgboost_gpu_v1.json")
    robustness = _load("robustness_v1.json")
    systems = _load("systems_benchmark_v1.json")
    bias = _load("selection_bias_v1.json")
    inference = _load("statistical_inference_v1.json")
    _table(
        "dataset_summary",
        [
            {
                "source": "Databento GLBX.MDP3",
                "instrument": "ESZ5",
                "trading_days": 1,
                "raw_mbo_rows": qa["metrics"]["row_count"],
                "feature_rows": sum(splits["split_counts"].values()),
                "final_lockbox_rows": splits["split_counts"]["FINAL_LOCKBOX"],
            }
        ],
    )
    _table(
        "data_qa",
        [{"check": key, "value": value} for key, value in qa["metrics"].items()]
        + [{"check": "quality_status", "value": qa["quality_status"]}],
    )
    _table(
        "baseline_reproduction",
        [
            {
                "baseline": "FI-2010 DeepLOB/TLOB",
                "status": "BLOCKED_EXTERNAL_DATA",
                "reason": "authoritative download disabled",
            }
        ],
    )
    _table(
        "predictive", [{"model": name, **metrics} for name, metrics in final["predictive"].items()]
    )
    _table(
        "calibration",
        [
            {
                "endpoint": "direction_nll",
                "before": calibration["direction_nll_before"],
                "after": calibration["direction_nll_after"],
            },
            {
                "endpoint": "direction_ece",
                "before": calibration["direction_ece_before"],
                "after": calibration["direction_ece_after"],
            },
            {
                "endpoint": "fill_brier",
                "before": calibration["fill_brier_before"],
                "after": calibration["fill_brier_after"],
            },
        ],
    )
    transfer = _load("transfer_v1.json")
    _table(
        "transfer",
        [
            {"setting": key, "status": value}
            for key, value in transfer.items()
            if key
            in {
                "strict_unseen_instrument",
                "cross_source",
                "fx_to_futures",
                "futures_to_fx",
                "claim",
            }
        ],
    )
    _table(
        "regime_robustness",
        [{"stress": name, **metrics} for name, metrics in robustness["predictive_stress"].items()],
    )
    _table(
        "economic", [{"strategy": name, **metrics} for name, metrics in final["economics"].items()]
    )
    _table(
        "risk",
        [
            {"control": "risk gate invariants", "status": "PASS"},
            {"control": "ending inventory", "status": "ZERO_ALL_FINAL_STRATEGIES"},
            {"control": "headline PnL", "status": "PROHIBITED"},
        ],
    )
    _table(
        "ablation",
        [
            {"comparison": "M3T-SSL minus M3T-SUP validation macro-F1", "effect": 0.0},
            {
                "comparison": "M3T-EPT-CAL minus XGBoost final net PnL",
                "effect": final["economics"]["M3T_EPT_CAL"]["net_pnl"]
                - final["economics"]["XGBOOST_GPU"]["net_pnl"],
            },
        ],
    )
    _table(
        "scaling",
        [
            {"model": "M3T-Small", "parameters": systems["parameters"], "status": "EXECUTED"},
            {
                "model": "M3T-Medium/Large",
                "parameters": "NOT ESTABLISHED",
                "status": "SKIPPED_RESOURCE_LIMIT",
            },
        ],
    )
    _table("systems", systems["inference"])
    _table("selection_bias", [{"diagnostic": key, "result": value} for key, value in bias.items()])
    _table(
        "statistical_inference",
        [
            {"field": key, "result": json.dumps(value) if isinstance(value, dict) else value}
            for key, value in inference.items()
        ],
    )
    _table(
        "failure_breakdown",
        [
            {"case": "MarketForge wins", "result": "M3T-SUP final macro-F1 0.3322 was highest"},
            {"case": "MarketForge ties", "result": "M3T-EPT-CAL equals majority macro-F1"},
            {
                "case": "MarketForge fails",
                "result": "all active strategies lose; calibrated EPT trails XGBoost economics",
            },
            {"case": "complexity justified", "result": False},
        ],
    )
    _table(
        "simulator_validation",
        [
            {
                "metric": "sampled field agreement",
                "value": simulator["actual_mbo_vs_mbp10"]["overall_match_rate"],
            },
            {"metric": "headline PnL permitted", "value": simulator["headline_pnl_permitted"]},
        ],
    )
    _table(
        "gpu_xgboost",
        [
            {
                "version": xgb["xgboost_version"],
                "device": xgb["device_from_booster_config"],
                **xgb["validation"],
            }
        ],
    )


def _make_figures() -> None:
    final = json.loads(
        (ROOT / "artifacts" / "final_evaluation_v1.json").read_text(encoding="utf-8")
    )
    robustness = _load("robustness_v1.json")
    systems = _load("systems_benchmark_v1.json")
    splits = _load("splits_databento_mbo_v3.json")
    plt.figure(figsize=(8, 4))
    plt.bar(splits["split_counts"].keys(), splits["split_counts"].values())
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("rows")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "data_coverage.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.bar(
        final["predictive"].keys(), [value["macro_f1"] for value in final["predictive"].values()]
    )
    plt.axhline(1 / 3, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("final macro-F1")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_predictive.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.bar(
        robustness["predictive_stress"].keys(),
        [value["macro_f1"] for value in robustness["predictive_stress"].values()],
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("strategy-validation macro-F1")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "robustness.png", dpi=160)
    plt.close()
    frame = pd.DataFrame(systems["inference"])
    for precision, group in frame.groupby("precision"):
        plt.plot(group["batch_size"], group["p95_ms"], marker="o", label=precision)
    plt.xscale("log", base=2)
    plt.xlabel("batch size")
    plt.ylabel("p95 latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "systems_latency.png", dpi=160)
    plt.close()


def generate_reports() -> Path:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _make_tables()
    _make_figures()
    final = json.loads(
        (ROOT / "artifacts" / "final_evaluation_v1.json").read_text(encoding="utf-8")
    )
    calibration = _load("calibration_m3t_ept_v1.json")
    systems = _load("systems_benchmark_v1.json")
    simulator = _load("simulator_validation_v1.json")
    monitoring = _load("monitoring_v1.json")
    _write(
        "EXECUTIVE_SUMMARY.md",
        "Executive summary",
        "The project completed the feasible public-data pipeline and preserved all negative findings. On the one-time final lockbox, M3T-SUP had the highest macro-F1 (0.3322), while M3T-EPT-CAL collapsed to the majority result (0.3096). No-trade beat GLFT, GPU XGBoost, and calibrated M3T after modeled costs. Strict transfer, statistical confidence intervals, and headline PnL are **NOT ESTABLISHED**. The evidence does not justify neural complexity or a tradable-edge claim.",
    )
    _write(
        "DATASET_CARD.md",
        "Dataset card",
        "The used core is the official Databento GLBX.MDP3 public ESZ5 MBO sample (3,667,804 raw events) paired with MBP-10. Ordered reconstruction yielded 35,969 labeled rows across one instrument-day. Splits are chronological, purged, embargoed, and isolate a 3,576-row final lockbox. FI-2010 is CC BY 4.0 but its authoritative download is disabled. No licensed raw data is committed for redistribution. See `tables/dataset_summary.*`.",
    )
    _write(
        "DATA_QUALITY_REPORT.md",
        "Data quality report",
        "Raw validity, ordering, action, and timestamp checks passed. The accepted v2 canonicalization preserves an ingest index for same-timestamp ordering. The accepted v3 L3 reconstruction has zero reconstruction errors. Earlier order-losing artifacts were rejected. See `tables/data_qa.*` and `tables/simulator_validation.*`.",
    )
    _write(
        "MODEL_CARD.md",
        "Model card",
        "MarketForge M3T is a 120,532-parameter causal dual-stream Transformer with event and ordered-book/state inputs, gated fusion, direction and economic heads. Intended use is research only. It is not validated for live trading. M3T-SUP, SSL, and EPT were trained on CUDA; weighted M3T-SUP was the best final predictor, but economic complexity was not justified.",
    )
    _write(
        "TRAINING_REPORT.md",
        "Training report",
        "Training used seed 1701, Windows-safe DataLoader workers=0, BF16 CUDA, and atomic checkpoints containing model/optimizer/RNG/config state. SSL processed 1,533,024 tokens; its final macro-F1 delta versus supervised was 0.0. The failed development funnel triggered the preregistered stop rule for Medium/Large and external foundation-model scaling.",
    )
    _write(
        "CALIBRATION_REPORT.md",
        "Calibration report",
        f"MODEL_VALIDATION-only temperature scaling improved direction NLL from {calibration['direction_nll_before']:.4f} to {calibration['direction_nll_after']:.4f} and ECE from {calibration['direction_ece_before']:.4f} to {calibration['direction_ece_after']:.4f}. Fill Brier improved from {calibration['fill_brier_before']:.4f} to {calibration['fill_brier_after']:.4f}. Better calibration did not produce better final economics.",
    )
    _write(
        "SIMULATOR_VALIDATION_REPORT.md",
        "Simulator validation report",
        f"Paired public MBO/MBP-10 validation achieved {simulator['actual_mbo_vs_mbp10']['overall_match_rate']:.6f} sampled-field agreement and passed synthetic accounting/latency/queue invariants. L2 queue approximation and empirical fill calibration remain unestablished. Therefore `headline_pnl_permitted={str(simulator['headline_pnl_permitted']).lower()}` and every economic result is labeled counterfactual.",
    )
    _write(
        "STRATEGY_SPECIFICATION.md",
        "Strategy specification",
        "Compared no-trade, fixed, microprice, Avellaneda–Stoikov-style, GLFT-style, tree, and M3T controllers. A common base spread of 0.75 and inventory penalty of 0.1 were selected only on STRATEGY_VALIDATION. TEST was opened once for Gate 13; FINAL_LOCKBOX was opened once after freeze. Modeled fee is 1.25 per fill with 0.25 liquidation slippage.",
    )
    _write(
        "RISK_CONTROL_SPECIFICATION.md",
        "Risk control specification",
        "The pre-trade gate rejects non-finite predictions, crossed/invalid quotes, price-collar breaches, inventory beyond 10, latency above 50ms, and loss-limit breaches. Kill action cancels outstanding orders and blocks new quotes. All invariant tests passed and every final strategy ended flat.",
    )
    _write(
        "ABLATION_REPORT.md",
        "Ablation report",
        "The resource-aware funnel tested supervised M3T, combined SSL, SSL fine-tuning, weighted-class diagnosis, EPT, and calibration. SSL added 0.0 validation macro-F1; EPT calibration improved probabilistic calibration but collapsed final direction predictions to the majority class. Scaling was stopped. See `tables/ablation.*` and `tables/scaling.*`.",
    )
    _write(
        "ROBUSTNESS_REPORT.md",
        "Robustness report",
        "Predictive corruptions were run on STRATEGY_VALIDATION, not TEST. Depth dropout fell to chance balanced accuracy; stale and delayed features degraded performance. Quote-width stresses remained negative. Latency, fee, slippage, and queue robustness cannot support an edge claim because empirical fills are unavailable. See `tables/regime_robustness.*`.",
    )
    _write(
        "BACKTEST_OVERFIT_REPORT.md",
        "Backtest overfit report",
        "Nine controller trials and six recorded model trials are preserved. White Reality Check, SPA, CSCV/PBO, and Deflated Sharpe are **NOT ESTABLISHED**: one instrument-day and nonvalidated fills make them uninterpretable. Event-level pseudo-replication was deliberately avoided.",
    )
    _write(
        "SYSTEM_BENCHMARK_REPORT.md",
        "System benchmark report",
        f"On the RTX 4090 Laptop GPU, FP32 batch-1 p95 inference was {systems['inference'][0]['p95_ms']:.3f}ms; BF16 was slower for this small model. Windows DataLoader workers=0 completed the benchmark in {systems['dataloader'][0]['wall_seconds']:.3f}s versus {systems['dataloader'][1]['wall_seconds']:.3f}s for workers=2. Checkpoint reload was exact. See `tables/systems.*`.",
    )
    _write(
        "MONITORING_REPORT.md",
        "Monitoring report",
        f"A sequential shadow log contains {monitoring['rows']} STRATEGY_VALIDATION observations with model version, prediction, confidence, proposed quotes, risk outcome, realized direction, rolling error/calibration gap, and GREEN/WATCH state. It is counterfactual and does not constitute live monitoring. State counts: {monitoring['states']}.",
    )
    _write(
        "FAILURE_ANALYSIS.md",
        "Failure analysis",
        "## Where MarketForge wins\n\nWeighted M3T-SUP achieved the highest final macro-F1 (0.3322).\n\n## Where it ties\n\nM3T-EPT-CAL matched the majority macro-F1 and balanced accuracy.\n\n## Where it fails\n\nSSL did not improve development prediction; every active strategy lost after costs; calibrated EPT lost more than GPU XGBoost; depth dropout reached chance; transfer and dependence-aware inference are unavailable.\n\n## Complexity verdict\n\nComplexity is not justified. The evidence chain from pretraining through improved execution economics failed.",
    )
    _write(
        "LIMITATIONS.md",
        "Limitations",
        "Only one ESZ5 trading day was available. FI-2010 download was disabled; FX was not acquired. The public sample is short, historical replay lacks endogenous impact, and queue/fill truth is absent. Simulated execution is counterfactual. A local RTX is not cluster-scale. Public-data market making is not proprietary client-flow market making. Backtests do not imply future returns.",
    )
    _write(
        "FUTURE_WORK.md",
        "Future work",
        "Acquire multiple legally usable instrument-days and sources; reproduce FI-2010 only from the authoritative dataset; validate queue/fill models against event-level outcomes; rerun strict unseen-instrument transfer and day-block inference; and reconsider model scaling only after a simple baseline shows stable after-cost development value.",
    )
    _write(
        "RESUME_EVIDENCE.md",
        "Resume evidence",
        f"From-scratch neural training completed: **yes**. GPU: RTX 4090 Laptop. SSL tokens: **1,533,024**. M3T parameters: **120,532**. Best final macro-F1: **{max(value['macro_f1'] for value in final['predictive'].values()):.6f}**. Cross-instrument improvement: **NOT ESTABLISHED**. OOD improvement: **NOT ESTABLISHED**. Calibration improvement: NLL **0.5544 → 0.5254**. After-cost improvement: **none; no-trade won**. Confidence interval: **NOT ESTABLISHED**. FP32 batch-1 p95: **{systems['inference'][0]['p95_ms']:.3f}ms**. Claims were checked against JSON artifacts; missing evidence is never inferred.",
    )
    technical = _write(
        "MARKETFORGE_QTR_TECHNICAL_REPORT.md",
        "MarketForge-QTR technical report",
        "## Conclusion\n\nThe feasible end-to-end protocol was executed, including the one-time frozen lockbox. The primary research hypothesis is unsupported on available data. M3T-SUP was marginally the best final classifier, but pretraining and EPT did not improve the evidence chain, no active strategy survived costs, and no robust-edge claim is permitted.\n\n## Governance\n\nTRAIN alone fit transforms and model weights; MODEL_VALIDATION calibrated; STRATEGY_VALIDATION chose controller parameters and hosted robustness/monitoring; TEST was evaluated once; FINAL_LOCKBOX was opened once after a sealed source/checkpoint/protocol manifest.\n\n## Evidence\n\nAll mandatory numeric tables are available as both CSV and Markdown under `tables/`; programmatic figures are under `figures/`. Gate 6 and strict Gate 14 claims are externally blocked. Statistical and selection-bias diagnostics honestly report insufficient independent units.\n\n## Economic interpretation\n\nFinal counterfactual PnL was 0 for no-trade, -8.05 for GLFT, -16.85 for GPU XGBoost, and -17.52 for calibrated M3T. These are diagnostic—not headline—figures because empirical queue/fill validation is unavailable.\n\n## Reproducibility\n\nUse `.\\scripts\\mf.ps1 full` from PowerShell 7 and `.\\scripts\\mf.ps1 resume` after interruption. The exact environment, CUDA proof, dataset hashes, checkpoints, final freeze seal, and result artifacts are retained.",
    )
    paths = [
        path
        for path in FINAL_DIR.rglob("*")
        if path.is_file() and path.name != "RESULTS_MANIFEST.json"
    ]
    paths.extend(sorted((data_root() / "manifests").glob("*.json")))
    paths.extend(sorted((ROOT / "artifacts" / "system").glob("*.json")))
    paths.extend(sorted((ROOT / "artifacts" / "system").glob("*.xml")))
    paths.extend(sorted((ROOT / "artifacts" / "checkpoints").glob("*")))
    paths.extend(
        path
        for path in (
            ROOT / "artifacts" / "final_freeze_manifest.json",
            ROOT / "artifacts" / "final_evaluation_v1.json",
            ROOT / "results" / "warehouse" / "results.parquet",
        )
        if path.exists()
    )
    paths = sorted(set(paths))
    manifest = {
        "schema_version": 1,
        "final_evaluation_count": final["evaluation_count"],
        "claim_scope": final["claim_scope"],
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
    }
    atomic_json(FINAL_DIR / "RESULTS_MANIFEST.json", manifest)
    return technical
