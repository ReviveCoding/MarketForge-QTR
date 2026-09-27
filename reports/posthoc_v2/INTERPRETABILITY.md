# Model Interpretability Diagnostics

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## XGBoost

| partition | feature | mean_abs_tree_shap |
| --- | --- | --- |
| DEVELOPMENT | ewma_vol | 0.18784 |
| DEVELOPMENT | microprice_minus_mid | 0.15051 |
| FINAL_POSTHOC_ONLY | depth_bid_1 | 0.14438 |
| DEVELOPMENT | depth_bid_1 | 0.12822 |
| FINAL_POSTHOC_ONLY | signed_volume | 0.11544 |
| FINAL_POSTHOC_ONLY | microprice_minus_mid | 0.11528 |
| DEVELOPMENT | imbalance_1 | 0.10761 |
| DEVELOPMENT | relative_spread | 0.10462 |
| DEVELOPMENT | spread | 0.099194 |
| FINAL_POSTHOC_ONLY | depth_convexity | 0.095596 |
| DEVELOPMENT | cancel_rate | 0.093815 |
| FINAL_POSTHOC_ONLY | depth_ask_1 | 0.092282 |

## M3T

| partition | feature_group | mean_absolute_probability_impact | batches |
| --- | --- | --- | --- |
| DEVELOPMENT | event_stream | 0.073365 | 2 |
| DEVELOPMENT | state_stream | 0.055083 | 2 |
| DEVELOPMENT | book_stream | 0.1094 | 2 |
| DEVELOPMENT | depth_level_1 | 0.010307 | 2 |
| DEVELOPMENT | depth_levels_6_10 | 0.056323 | 2 |
| DEVELOPMENT | modality_context | 0.07957 | 2 |
| FINAL_POSTHOC_ONLY | event_stream | 0.20989 | 13 |
| FINAL_POSTHOC_ONLY | state_stream | 0.36901 | 13 |
| FINAL_POSTHOC_ONLY | book_stream | 0.089083 | 13 |
| FINAL_POSTHOC_ONLY | depth_level_1 | 0.0084102 | 13 |
| FINAL_POSTHOC_ONLY | depth_levels_6_10 | 0.043819 | 13 |
| FINAL_POSTHOC_ONLY | modality_context | 0.1296 | 13 |

## Limits

TreeSHAP describes the fitted trees. XGBoost fill probabilities are learned classifier outputs, while its adverse controller input is a deterministic `markout_derived_adverse_score`, not an independently trained adverse probability. M3T adverse outputs are learned, uncalibrated probabilities. Stream/depth occlusion measures prediction sensitivity, not causality. The development-fitted latent probe is a `V3_CANDIDATE`; attention maps are not causal explanations.
