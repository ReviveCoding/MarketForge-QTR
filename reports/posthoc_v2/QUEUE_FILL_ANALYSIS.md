# Queue and Fill Diagnostics

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Conditional fill behavior

| partition | model | side | condition | development_bucket | support | positive | negative | simulated_l3_fill_rate | predicted_fill_probability | calibration_residual | mean_brier_contribution | time_to_fill_support | time_to_fill_q25_ns | median_time_to_fill_ns | time_to_fill_q75_ns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEVELOPMENT | M3T | bid | bid_queue_ahead_at_entry | Q1 | 62 | 38 | 24 | 0.6129 | 0.52552 | -0.087382 | 0.23646 | 38 | 8.4777e+06 | 2.0181e+08 | 4.8721e+08 |
| DEVELOPMENT | M3T | bid | bid_queue_ahead_at_entry | Q2 | 60 | 12 | 48 | 0.2 | 0.26285 | 0.062852 | 0.16256 | 12 | 2.411e+08 | 3.761e+08 | 4.5552e+08 |
| DEVELOPMENT | M3T | bid | bid_queue_ahead_at_entry | Q3 | 62 | 3 | 59 | 0.048387 | 0.17345 | 0.12506 | 0.077336 | 3 | 2.901e+08 | 5.725e+08 | 7.1043e+08 |
| DEVELOPMENT | M3T | bid | bid_queue_ahead_at_entry | Q4 | 59 | 5 | 54 | 0.084746 | 0.12745 | 0.042704 | 0.09015 | 5 | 1.8839e+08 | 5.1468e+08 | 5.7984e+08 |
| DEVELOPMENT | M3T | bid | queue_ahead_ratio_at_price | Q1 | 61 | 33 | 28 | 0.54098 | 0.43892 | -0.10206 | 0.22215 | 33 | 7.4668e+06 | 1.2265e+08 | 4.5668e+08 |
| DEVELOPMENT | M3T | bid | queue_ahead_ratio_at_price | Q2 | 110 | 12 | 98 | 0.10909 | 0.20476 | 0.095671 | 0.11164 | 12 | 2.3412e+08 | 3.5915e+08 | 4.5552e+08 |
| DEVELOPMENT | M3T | bid | queue_ahead_ratio_at_price | Q3 | 11 | 2 | 9 | 0.18182 | 0.16338 | -0.018434 | 0.18585 | 2 | 5.2914e+08 | 5.4359e+08 | 5.5804e+08 |
| DEVELOPMENT | M3T | bid | queue_ahead_ratio_at_price | Q4 | 61 | 11 | 50 | 0.18033 | 0.25462 | 0.074289 | 0.10906 | 11 | 2.3478e+08 | 4.2894e+08 | 7.8927e+08 |
| DEVELOPMENT | M3T | bid | queue_ahead_ratio_top5 | Q1 | 61 | 36 | 25 | 0.59016 | 0.52264 | -0.067522 | 0.23719 | 36 | 6.4711e+06 | 1.9592e+08 | 4.3587e+08 |
| DEVELOPMENT | M3T | bid | queue_ahead_ratio_top5 | Q2 | 61 | 12 | 49 | 0.19672 | 0.26333 | 0.06661 | 0.14779 | 12 | 2.6401e+08 | 3.7976e+08 | 5.639e+08 |
| DEVELOPMENT | M3T | bid | queue_ahead_ratio_top5 | Q3 | 60 | 4 | 56 | 0.066667 | 0.17728 | 0.11062 | 0.079823 | 4 | 1.3665e+08 | 3.1562e+08 | 4.6737e+08 |
| DEVELOPMENT | M3T | bid | queue_ahead_ratio_top5 | Q4 | 61 | 6 | 55 | 0.098361 | 0.13189 | 0.033532 | 0.10254 | 6 | 2.8442e+08 | 5.7617e+08 | 5.9376e+08 |

## Validation status

Every row is L3-valid. Historical resting-order evidence remains calibration/error evidence only. Counterfactual fills remain `SIMULATED_L3`; no perfection or observed-fill claim is made. Largest observed XGBoost Brier advantage state: `{'side': 'ask', 'condition': 'realized_vol_50', 'development_bucket': 'Q1', 'm3t_minus_xgboost_brier': 0.4678657170026985}`.
