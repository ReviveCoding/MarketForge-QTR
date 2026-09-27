# Operational Slice Analysis

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Scope

272 direction/calibration rows and 224 side-long execution/economic rows were produced. Every row contains support; binary fill rows contain positive and negative counts. Support below 30 is flagged insufficient.

## Largest supported differences

| slice_dimension | slice_value | support | M3T | XGBOOST | M3T_minus_XGB | slice_definition_source | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m3t_confidence_bucket | Q1 | 225 | 0.3193 | 0.14644 | 0.17285 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| m3t_entropy_bucket | Q4 | 265 | 0.3129 | 0.14277 | 0.17013 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| session_position_bucket | Q4 | 189 | 0.2653 | 0.11702 | 0.14827 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| xgboost_confidence_bucket | Q2 | 926 | 0.28911 | 0.14279 | 0.14632 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| event_rate_bucket | Q1 | 38 | 0.34603 | 0.20988 | 0.13616 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| realized_vol_50_bucket | Q1 | 57 | 0.26944 | 0.14501 | 0.12444 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| signed_volume_bucket | Q3 | 88 | 0.20287 | 0.08165 | 0.12122 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| xgboost_entropy_bucket | Q3 | 777 | 0.2727 | 0.15374 | 0.11896 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| m3t_entropy_bucket | Q2 | 839 | 0.26598 | 0.14928 | 0.1167 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| imbalance_1_bucket | Q1 | 615 | 0.29416 | 0.17754 | 0.11662 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| event_rate_bucket | Q4 | 1950 | 0.2771 | 0.16672 | 0.11038 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |
| realized_vol_50_bucket | Q4 | 186 | 0.31621 | 0.20879 | 0.10742 | DEVELOPMENT_FIXED_OR_PREDEFINED | EXPLORATORY_POSTHOC_ONLY |

## Discipline

Final slices are exploratory/post-hoc only. The companion execution table includes fill/no-fill, side, time-to-fill, predicted fill/adverse/markout/edge, disagreement, gross edge, costs, and diagnostic net PnL. Repeated timestamps remain present; the timestamp-group weighting sensitivity is explicitly exploratory and does not replace row-preserving authoritative metrics.
