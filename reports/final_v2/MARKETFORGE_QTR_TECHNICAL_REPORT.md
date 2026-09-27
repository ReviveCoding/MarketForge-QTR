# MarketForge-QTR v2 Technical Report

Generated 2026-09-27T18:46:54.051385+00:00.

## Status and protocol separation

V1 remains preserved engineering/audit history. Its economics and economic targets are scientifically superseded, its final lockbox is spent, and `final_evaluation_v1.json` was not changed. V2 is the current corrected protocol.

## Outcome-blind structural amendment disclosure

- The final lockbox was opened exactly once; `evaluation_count` remained 1.
- Exact XNAS/NVDA replay completed before the first structural failure.
- No predictions or final metrics existed when that first failure was discovered.
- The frozen evaluator incorrectly assumed `prediction_time` was unique.
- Exactly one timestamp had multiplicity four: 3,238 rows but 3,235 unique timestamps.
- Amendment 001 replaced timestamp-only feature/label joining with ordinal replay identity; no row was removed, duplicated, reordered, filtered, or aggregated.
- A second occurrence of the same defect appeared in the economic diagnostic after predictions were transiently computed; no result artifact or metric output was persisted or used. Amendment 002 propagated the same `capture_id` into prediction tables and diagnostic lookup. Amendment 003 cryptographically bound the execution chain.
- No targets, feature definitions, models, checkpoints, scalers, embeddings, calibration parameters, controllers, fees, ticks, multipliers, latency, risk limits, assumptions, or selection decisions changed.
- The original freeze and all structural-failure/amendment artifacts remain preserved. This result is explicitly `FINAL_EVALUATION_V2_WITH_OUTCOME_BLIND_STRUCTURAL_AMENDMENT`, not an unamended freeze execution.


## Data actually used

One provider (Databento), three dataset/venue domains, three instruments, and three instrument-days were available: CME ES MBO for development execution labels; ICE Brent MBP-10 for L2 representation after MBO reconstruction failed following an unrecovered clear; and XNAS NVDA MBO as the one-time final lockbox. This supports strict venue/dataset, asset-class, instrument, and calendar-day OOD point estimates, not provider OOD or population inference.

## Corrected targets and accounting

V2 uses side-specific bid/ask markouts, counterfactual FIFO `SIMULATED_L3` fills, fill-conditional adverse selection, modality availability masks, grouped windows, instrument metadata, and dimensionally explicit USD accounting. The test suite proves one ES tick is 0.25 points and $12.50 per contract. Endpoint crossing is not used as an L3 fill label.

## Development findings

The six-branch factorial ladder was executed with identical model scale. M3T-SSL-EPT descended from SSL and M3T-SUP-EPT independently descended from SUP. XGBoost trained on CUDA and won the development predictive comparison. Every active development market-making strategy lost after registered costs, so Medium/Large scaling stopped and `NO_TRADE` was frozen.

## Final one-time result

| Metric | M3T-SSL-EPT-CAL | XGBoost CUDA |
|---|---:|---:|
| Direction macro-F1 | 0.2564 | 0.1903 |
| Bid fill Brier | 0.3032 | 0.2571 |
| Ask fill Brier | 0.2792 | 0.2184 |

The final selected `NO_TRADE` policy realized $0 by construction. Frozen active diagnostics remained negative: M3T `$-6.537` and XGBoost `$-1.891`. Selection did not change after the lockbox.

## Simulator evidence

The immutable final replay reports zero reconstruction errors. Historical real-order queue validation observed 30 fills, with 28 queue-eligible at fill (93.33%) and mean queue-ahead error 49.13 shares. This is calibration/error evidence, not a claim of perfect observed fills. Paired MBO→MBP validation was not rerun after authorization because replay was expressly prohibited.

## Statistical interpretation

The primary unit is instrument × trading day. Development has two units and final evaluation one unit. Confidence intervals, White/SPA, CSCV/PBO, deflated Sharpe, and population transfer inference are `NOT ESTABLISHED` / `SKIPPED_INSUFFICIENT_SAMPLE`. Row-level counts are not presented as independent day-level evidence.

## Conclusion

V2 corrects the v1 scientific defects and produces an honest negative economic conclusion. The complex neural system does not establish an economic edge; XGBoost is competitive or stronger predictively, and no-trade remains the only supported economic selection.
