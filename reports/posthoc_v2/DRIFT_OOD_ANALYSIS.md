# Drift and OOD Analysis

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Raw-unit drift

| feature | development_support | final_support | standardized_mean_difference | wasserstein_distance | ks_statistic | psi | population_p_value_claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| depth_ask_5 | 384 | 3207 | 1.4201 | 4801.6 | 0.99065 | 8.9881 | False |
| depth_bid_5 | 384 | 3207 | 0.88994 | 5942.3 | 0.98753 | 8.4873 | False |
| depth_ask_1 | 384 | 3207 | 0.58284 | 514.64 | 0.82836 | 3.4036 | False |
| depth_bid_1 | 384 | 3207 | 0.21743 | 662.53 | 0.81829 | 3.3151 | False |
| ask_queue_ahead_at_entry | 243 | 3207 | 0.56142 | 477.86 | 0.78978 | 3.0934 | False |
| bid_queue_ahead_at_entry | 243 | 3207 | 0.21656 | 517.04 | 0.77612 | 3.1039 | False |
| spread | 384 | 3207 | -1.6784 | 0.1611 | 0.63281 | 10.631 | False |
| realized_vol_50 | 384 | 3207 | 0.61948 | 0.00010236 | 0.61504 | 9.493 | False |
| signed_volume | 384 | 3207 | -0.14383 | 16667 | 0.50452 | 3.9292 | False |
| interarrival_ns | 384 | 3207 | -0.099866 | 1.811e+10 | 0.37967 | 1.5781 | False |
| event_rate | 384 | 3207 | 0.043275 | 1.8943e+09 | 0.3778 | 1.5817 | False |
| imbalance_1 | 384 | 3207 | 0.018184 | 0.059499 | 0.066685 | 0.087779 | False |

## Normalized structural drift

| feature | development_support | final_support | standardized_mean_difference | wasserstein_distance | ks_statistic | psi | population_p_value_claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| relative_volatility_bps | 384 | 3207 | 0.61948 | 1.0236 | 0.61504 | 9.493 | False |
| relative_spread | 384 | 3207 | -0.16556 | 3.6695e-05 | 0.58333 | 11.115 | False |
| queue_ahead_to_top5_depth | 243 | 3207 | -0.33458 | 0.040125 | 0.46418 | 1.3319 | False |
| top1_share_of_top5_depth | 384 | 3207 | -0.21965 | 0.042267 | 0.40981 | 1.159 | False |
| spread_ticks | 384 | 3207 | 0.34977 | 0.26492 | 0.38851 | 9.7379 | False |
| depth5_domain_median_ratio | 384 | 3207 | 0.29661 | 0.52656 | 0.35647 | 1.8117 | False |
| depth1_domain_median_ratio | 384 | 3207 | 0.14479 | 0.85588 | 0.25303 | 0.96291 | False |
| event_intensity_domain_median_ratio | 384 | 3207 | 0.043275 | 5.0825e+06 | 0.2513 | 0.78228 | False |
| top5_bid_ask_depth_imbalance | 384 | 3207 | -0.019971 | 0.15035 | 0.24818 | 0.76477 | False |
| log_interarrival_domain_zscore | 384 | 3207 | -9.971e-18 | 0.23895 | 0.13758 | 0.38744 | False |
| queue_ahead_to_displayed_at_price | 243 | 3207 | 0.036014 | 0.35281 | 0.077985 | 0.22693 | False |
| top1_bid_ask_depth_imbalance | 384 | 3207 | 0.018184 | 0.059499 | 0.066685 | 0.087779 | False |

## Latent distance

The M3T distance reference was fitted on development only. Mean development distance was 2.6710; mean final distance was 69.8824. This detector is diagnostic only.
