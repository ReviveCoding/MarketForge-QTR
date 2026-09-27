# Production-style Monitoring Scorecard

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Not deployment

This is a post-hoc production-style scorecard; it does not imply live deployment. Thresholds use a transparent development-IQR heuristic.

## Scorecard

| category | metric | development_support | final_support | development_median | development_iqr | final_median | normalized_shift | status | threshold_origin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data_health | spread | 384 | 3207 | 0.25 | 0.24 | 0.01 | 1 | WATCH | DEVELOPMENT_IQR_HEURISTIC |
| data_health | top5_depth | 384 | 3207 | 1021 | 797.5 | 9167 | 10.214 | DISABLE | DEVELOPMENT_IQR_HEURISTIC |
| data_health | event_rate | 384 | 3207 | 202.35 | 218.17 | 372.72 | 0.78088 | GREEN | DEVELOPMENT_IQR_HEURISTIC |
| data_health | missingness | 384 | 3207 | 0.052083 | 0.067708 | 0.026178 | 0.3826 | GREEN | DEVELOPMENT_IQR_HEURISTIC |
| data_health | stale_data_rate | 384 | 3207 | 0 | 1e-07 | 0 | 0 | GREEN | DEVELOPMENT_IQR_HEURISTIC |
| model | m3t_confidence | 384 | 3207 | 0.52651 | 0.27076 | 0.72816 | 0.74477 | GREEN | DEVELOPMENT_IQR_HEURISTIC |
| model | m3t_entropy | 384 | 3207 | 0.99361 | 0.26746 | 0.68553 | 1.1519 | WATCH | DEVELOPMENT_IQR_HEURISTIC |
| model | m3t_error | 384 | 3207 | 0 | 1 | 1 | 1 | GREEN | DEVELOPMENT_IQR_HEURISTIC |
| model | m3t_nll | 384 | 3207 | 0.78808 | 0.64832 | 1.5377 | 1.1562 | WATCH | DEVELOPMENT_IQR_HEURISTIC |
| model | m3t_multiclass_brier | 384 | 3207 | 0.45655 | 0.52966 | 1.0559 | 1.1315 | WATCH | DEVELOPMENT_IQR_HEURISTIC |
| model | fill_calibration_gap | 243 | 3207 | -0.11354 | 0.49339 | 0.097506 | 0.42775 | GREEN | DEVELOPMENT_IQR_HEURISTIC |
| execution | fill_rate | 243 | 3207 | 0.5 | 0.5 | 0.5 | 0 | GREEN | DEVELOPMENT_IQR_HEURISTIC |

## Statuses

GREEN = within 1 development IQR; WATCH = 1–2; REVIEW = 2–4; DISABLE = above 4. Metrics with fewer than 30 valid available observations in either domain are `NOT_AVAILABLE`. Fill, queue, adverse, post-fill markout, and time-to-fill metrics use only L3/side-available targets. These are heuristic diagnostics.
