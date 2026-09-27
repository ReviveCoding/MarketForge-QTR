# MarketForge-QTR status

PROJECT STATUS: V2_CORRECTED_PROTOCOL_COMPLETE_WITH_OUTCOME_BLIND_STRUCTURAL_AMENDMENT

## Completed gates

V1 preservation/supersession; corrected side-specific targets; FIFO `SIMULATED_L3`; multi-instrument/session/reset isolation; instrument registry; dimensional USD accounting; lawful data acquisition; leakage-safe splits; six-branch M3T ladder; actual CUDA XGBoost; simulator validation; development economics; transfer/ablation/robustness/systems/monitoring/failure analysis; pre-final audit; v2 freeze; one-time final evaluation; final v2 reports; full artifact audit.

Statistical inference and selection-bias procedures are `SKIPPED_INSUFFICIENT_SAMPLE`, not scientific successes. FI-2010 and unattended official Dukascopy acquisition remain external-data blockers.

## Protocol separation

V1 is preserved but scientifically superseded for economic/model-mechanism claims. `artifacts/final_evaluation_v1.json` remains SHA-256 `d7720cd2d0884538145e33ac54758ff97712f13da3ae413cdb5c8fc78c19a5f2`; its final lockbox is spent and was not reused.

V2 is authoritative. The final result is `FINAL_EVALUATION_V2_WITH_OUTCOME_BLIND_STRUCTURAL_AMENDMENT`. The original freeze remains immutable. Amendments 001–003 document ordinal replay identity and its propagation; no row or scientific choice changed.

## Datasets actually used

- Development: CME/ES MBO and paired MBP-10; ICE/BRN paired MBP-10 for L2 only after MBO reconstruction failed following a clear.
- Final: XNAS/NVDA MBO, 2025-09-16; 3,238 immutable replay rows.
- Scope: one provider, three dataset/venue domains, three instruments, three instrument-days.
- FX was not acquired because no documented official unattended Dukascopy path was established.

## Models actually trained

M3T-SUP, M3T-SSL pretraining/fine-tuning, M3T-SUP-EPT, M3T-SSL-EPT, both calibration branches, and actual XGBoost 3.4.1. Medium/Large and external foundation models were stopped by the resource-aware funnel after XGBoost won development validation and every active strategy lost after costs.

## GPU usage verified

`artifacts/system/cuda_runtime.json` records a real CUDA forward/backward pass on the NVIDIA RTX 4090 Laptop GPU. Neural training used CUDA/BF16; XGBoost booster configuration recorded `cuda:0`; GPU-heavy work was serialized.

## Key verified results

- Final M3T calibrated direction macro-F1: 0.256400.
- Final XGBoost direction macro-F1: 0.190259.
- Final M3T calibrated bid/ask fill Brier: 0.303238 / 0.279194.
- Final XGBoost bid/ask fill Brier: 0.257059 / 0.218421.
- Frozen `NO_TRADE` remained selected; no post-lockbox choice changed.
- Final exact replay reconstruction errors: 0; real-order eligibility diagnostic 28/30 (93.33%).

## Key null/negative results

Final active diagnostics lost after costs: M3T −$6.537 and XGBoost −$1.891. Population transfer/economic inference is not established. Complexity is not economically justified. These are public-sample research diagnostics, not observed fills, live PnL, or future-return claims.

## Outcome-blind amendment

The lockbox opened once and replay completed before the first failure. No predictions/final metrics existed when the frozen evaluator's timestamp-uniqueness assumption first failed. Exactly one timestamp had multiplicity four. Amendment 001 aligned all rows by ordinal `capture_id`. A second occurrence in diagnostic lookup arose after transient predictions but before any persisted result/metric output; amendment 002 propagated the same identity, and amendment 003 bound the execution chain. Evaluation count remained 1.

FINAL REPORT PATH: `reports/final_v2/MARKETFORGE_QTR_TECHNICAL_REPORT.md`

RESUME EVIDENCE PATH: `reports/final_v2/RESUME_EVIDENCE.md`

EXACT REPRODUCE COMMAND: `.\.venv\Scripts\python.exe -m pytest -q; .\scripts\mf.ps1 v2-final-evaluation; .\scripts\mf.ps1 v2-report` (uses immutable completed replay evidence; final evaluation is idempotent)

EXACT RESUME COMMAND: `.\scripts\mf.ps1 resume`
