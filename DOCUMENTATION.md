# Execution record

## 2026-09-27 repository reconciliation

Read `MASTER_PROMPT.md`, `AGENTS.md`, and `artifacts/system/BOOTSTRAP_PRECHECK.json` completely. The repository had no implementation, virtual environment, datasets, continuity state, or Git commits. Python 3.12 is installed and selected because current official PyTorch Windows documentation does not support the bootstrap-default Python 3.13. Host `nvidia-smi` sees one RTX 4090 Laptop GPU with 16,376 MiB VRAM. No experiment result has yet been claimed.

The measured C: free space was 101,894,500,352 bytes. The specification's safety reserve evaluates to 40 GiB, leaving roughly 54.9 GiB as the initial maximum discretionary budget; downloads will be smaller and progressive.

## Gates 0-1

Created `.venv` with Python 3.12.10 and installed PyTorch 2.14.0+cu130 plus the core research stack. `artifacts/system/environment_runtime.json` recorded 94,816,399,360 free bytes after installation. `artifacts/system/cuda_runtime.json` proves a real CUDA forward/backward computation on the RTX 4090 Laptop GPU: CUDA available, one device, BF16 supported, tensors and model on `cuda:0`, finite gradients. Gate state is durably recorded in `artifacts/pipeline_state.sqlite3`.

## Gates 2–13

Acquired the legal Databento public CME MBO and MBP-10 samples with immutable hashes. QA covered 3,667,804 MBO rows. Ordered L3 reconstruction produced 35,969 feature rows with zero reconstruction errors; chronological purged splits isolate TRAIN, MODEL_VALIDATION, STRATEGY_VALIDATION, TEST, and FINAL_LOCKBOX. Development commands have not opened FINAL_LOCKBOX.

The official FI-2010 Fairdata record is CC BY 4.0 but currently reports that download is not enabled, so published baseline reproduction is `BLOCKED_EXTERNAL_DATA`. The project continued on the independent Databento path.

All neural stages ran on CUDA under the single-GPU lease. M3T-SUP, SSL, SSL fine-tuning, a class-weighted diagnostic, and economic post-training were executed. HGB remained the predictive winner (macro-F1 0.3290 vs weighted M3T 0.3253); SSL did not improve macro-F1. Validation-only calibration materially improved NLL and ECE.

Simulator validation against paired public MBP-10 achieved 99.9655% sampled-field agreement, but L2 queue realism and empirical fill calibration are not established. Consequently, economic results are explicitly non-headline counterfactual-crossing diagnostics. Gate 13 used STRATEGY_VALIDATION for nine controller settings and opened development TEST once; no-trade won and every active strategy lost after costs.

## Gates 14–22

Actual XGBoost 3.4.1 trained with booster device `cuda:0`; its MODEL_VALIDATION macro-F1 was 0.3199. Strict transfer was blocked by the one-instrument/one-day public core, while within-day liquidity and volatility regime diagnostics were retained. Corruption stresses found depth dropout at chance. Systems measurements found FP32 faster than BF16 for the 120,532-parameter model and Windows DataLoader workers=0 dramatically faster than workers=2.

Dependence-aware inference and CSCV/DSR-style diagnostics were evaluated for eligibility but not fabricated: one instrument-day provides one valid primary unit. Sequential STRATEGY_VALIDATION shadow logs were written for 3,576 rows.

Gate 20 froze source/config fingerprints, dataset identity, checkpoint hashes, calibration, controller/strategy settings, and primary comparisons. Gate 21 then opened 3,576 FINAL_LOCKBOX rows exactly once. Weighted M3T-SUP was the best final classifier (macro-F1 0.33217), but calibrated EPT matched majority behavior. No-trade beat all active strategies after costs; the simulator scope prohibits headline PnL. Gate 22 generated the complete report/table/figure suite and a 104-artifact hash manifest spanning reports, evidence, checkpoints, and the warehouse. Final Ruff and 14 tests passed; the exact `full` command subsequently skipped every completion marker.

## V2 correction (2026-09-27)

V1 is now a preserved but scientifically superseded protocol. `docs/V1_SUPERSEDED_NOTICE.md` and `artifacts/v1_superseded_manifest.json` hash and classify its evidence. The v1 final evaluation remains immutable and its lockbox is permanently spent.

V2 implements side-specific quote and post-fill markouts, independent instrument/session/reset books, grouped model windows, an authoritative instrument registry, and explicit price-point/tick/USD accounting. Algebraic, interleaved-instrument, grouping, queue, model-head, and ES dimensional tests pass.

Official public Databento samples added ICE Brent MBO/MBP-10 and Nasdaq NVDA MBO/MBP-10. ES and ICE are development sources. XNAS NVDA was isolated immediately after metadata-only coverage inspection as the sealed whole-instrument-day v2 final lockbox; development outcomes, labels, and predictions have not been inspected. The free-space reserve remains satisfied. Development canonicalization of ES and ICE preserves exact ingest order, nanosecond timestamps, independent sessions, reset generations, and instrument-specific units.

## V2 completion and outcome-blind amendment (2026-09-27)

The corrected development protocol executed the six-branch M3T factorial ladder, actual CUDA XGBoost, corrected economics, transfer/ablation/robustness/systems/monitoring analyses, and a pre-final audit. XGBoost won development prediction and every active strategy lost after costs, so the frozen policy was `NO_TRADE` and neural scaling stopped.

`PRE_FINAL_AUDIT_V2.md` passed before `artifacts/final_freeze_manifest_v2.json` was created. The new XNAS/NVDA lockbox then opened exactly once. Replay completed with 3,238 feature/label rows and zero reconstruction errors. One exchange timestamp appeared four times, exposing a timestamp-uniqueness assumption before inference. The user authorized an outcome-blind amendment: in-memory ordinal `capture_id` alignment, no Parquet mutation, no row change, and no scientific-choice change. A second use of timestamp indexing in the economic diagnostic was corrected by propagating the same identity. Amendments 001–003 and both structural-failure records preserve the full audit trail. Evaluation count remained one.

Final M3T/XGBoost direction macro-F1 point estimates were 0.256400 and 0.190259. The frozen no-trade policy remained unchanged; active M3T and XGBoost diagnostics were −$6.537 and −$1.891 after registered costs. Population inference remains not established. The authoritative report suite is under `reports/final_v2/`.
