# Development-only Robustness Surfaces

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Scope

Cost, liquidation cost, and confidence surfaces use development predictions only and are labeled `V3_CANDIDATE`. Existing labels fix entry latency at 1 ms; alternative latency surfaces are unavailable without replay. No final winner was selected.

## Surface sample

| model | registered_non_liquidation_cost_ticks | additional_cost_ticks | liquidation_cost_ticks | confidence_threshold | edge_threshold_ticks | maximum_queue_ahead | inventory_penalty | latency_ns | latency_variation | mean_expected_edge_ticks | positive_edge_fraction | winner_selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 59.5 | 0 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | 0.030688 | 0.35185 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 59.5 | 0.05 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | 0.013095 | 0.28395 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 59.5 | 0.1 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | -0.0044976 | 0.14609 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 59.5 | 0.25 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | -0.057275 | 0 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 66.5 | 0 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | 0.036566 | 0.41358 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 66.5 | 0.05 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | 0.015887 | 0.34362 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 66.5 | 0.1 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | -0.0047922 | 0.17901 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 66.5 | 0.25 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | -0.066829 | 0 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 77.5 | 0 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | 0.044416 | 0.50823 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 77.5 | 0.05 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | 0.019005 | 0.4321 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 77.5 | 0.1 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | -0.0064067 | 0.2037 | False |
| M3T | 0.2 | 0 | 0 | 0 | 0 | 77.5 | 0.25 | 1000000 | UNSUPPORTED_WITHOUT_REPLAY | -0.082641 | 0 | False |
