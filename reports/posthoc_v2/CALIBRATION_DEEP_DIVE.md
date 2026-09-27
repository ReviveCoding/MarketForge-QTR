# Calibration Deep Dive

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Finding

Frozen XGBoost had lower final post-hoc total fill Brier on bid and ask: bid 0.2592 vs M3T 0.3032; ask 0.2196 vs M3T 0.2792. This total score is not, by itself, proof of better calibration.

## Metrics

| model | task | support | positive | negative | brier | log_loss | ece | mce | adaptive_ece | calibration_slope | calibration_intercept | brier_reliability | brier_resolution | brier_uncertainty | brier_decomposition_residual | brier_decomposition_within_expected_binning_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3T | bid_fill | 3207 | 1007 | 2200 | 0.30324 | 1.2221 | 0.24466 | 0.67754 | 0.24165 | -0.0068231 | -0.78721 | 0.088097 | 0.00032297 | 0.2154 | 5.9368e-05 | True |
| M3T | ask_fill | 3207 | 984 | 2223 | 0.27919 | 1.0772 | 0.20014 | 0.73393 | 0.20057 | -0.02186 | -0.83672 | 0.066406 | 0.00042181 | 0.21268 | 0.0005252 | True |
| XGBOOST | bid_fill | 3207 | 1007 | 2200 | 0.25916 | 0.81904 | 0.21559 | 0.259 | 0.21248 | 0.33872 | 0.009321 | 0.046792 | 0.0027687 | 0.2154 | -0.00026312 | True |
| XGBOOST | ask_fill | 3207 | 984 | 2223 | 0.2196 | 0.63715 | 0.077125 | 0.26862 | 0.08041 | 0.2424 | -0.54098 | 0.0079166 | 0.0004248 | 0.21268 | -0.00057743 | True |

## Interpretation

Raw and frozen M3T curves are reported on development; only the already-frozen calibration is evaluated on final. XGBoost has no separate frozen calibrator, so its frozen entry is its trained probability. Brier includes reliability, resolution, and uncertainty and is not a pure calibration metric. Bid decomposition: `BOTH_MAINLY_RELIABILITY`; ask: `PRIMARILY_BETTER_RELIABILITY`. The identity Brier ≈ reliability - resolution + uncertainty passes the registered binning-residual tolerance for every fill row. No final-data calibrator was fitted.
