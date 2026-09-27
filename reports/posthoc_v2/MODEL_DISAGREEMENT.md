# M3T versus XGBoost Disagreement

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Direction groups

| partition | group | support | spread_mean | volatility_mean | absolute_imbalance_mean | event_intensity_mean | top5_depth_mean | queue_ahead_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEVELOPMENT | BOTH_CORRECT | 204 | 0.15049 | 0.00015262 | 0.37403 | 186.37 | 773.84 | 70.889 |
| DEVELOPMENT | BOTH_WRONG | 100 | 0.1725 | 0.00014073 | 0.47173 | 206.72 | 795.3 | 68.911 |
| DEVELOPMENT | M3T_CORRECT_XGB_WRONG | 37 | 0.17946 | 0.00013511 | 0.60928 | 380.54 | 823.05 | 68.979 |
| DEVELOPMENT | M3T_WRONG_XGB_CORRECT | 43 | 0.29163 | 6.5901e-05 | 0.57016 | 334.28 | 1053.3 | 74.3 |
| FINAL_POSTHOC_ONLY | BOTH_CORRECT | 154 | 0.014351 | 0.00021747 | 0.41935 | 1554.2 | 11935 | 611.23 |
| FINAL_POSTHOC_ONLY | BOTH_WRONG | 1334 | 0.015 | 0.00020691 | 0.39572 | 4.554e+09 | 10454 | 587.47 |
| FINAL_POSTHOC_ONLY | M3T_CORRECT_XGB_WRONG | 865 | 0.014081 | 0.00019991 | 0.40797 | 25652 | 11308 | 566.73 |
| FINAL_POSTHOC_ONLY | M3T_WRONG_XGB_CORRECT | 854 | 0.01123 | 0.00018829 | 0.3548 | 909.31 | 13473 | 531.86 |

## Conclusion

The largest observed M3T advantage among predefined post-hoc slices was `m3t_confidence_bucket=Q1` (support 225). The analogous XGBoost observation was `event_rate_bucket=Q2` (support 847). These correlated exploratory slices are not independent discoveries, carry no event-row p-values, and do not select a hybrid or alter v2.
