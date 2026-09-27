# Development-discovered Error Cohorts

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Method

Depth-3 regression trees with minimum 30 development observations per leaf discovered direction, fill, adverse-selection, and economic-loss cohorts. Fixed rules were then applied to final descriptively; no rule was optimized on final.

## Cohorts

| task | cohort_leaf | development_support | development_mean_error | final_support | final_application |
| --- | --- | --- | --- | --- | --- |
| direction_error | 7 | 35 | 0.74286 | 948 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| direction_error | 2 | 37 | 0.59459 | 118 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| adverse_selection_absolute_error | 5 | 34 | 0.54014 | 0 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| adverse_selection_absolute_error | 2 | 33 | 0.51027 | 2 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| adverse_selection_absolute_error | 6 | 36 | 0.49799 | 1987 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| direction_error | 8 | 35 | 0.48571 | 1780 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| adverse_selection_absolute_error | 3 | 31 | 0.46356 | 2 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| fill_probability_absolute_error | 6 | 40 | 0.41104 | 1506 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| fill_probability_absolute_error | 5 | 30 | 0.38257 | 1549 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| fill_probability_absolute_error | 4 | 81 | 0.32852 | 99 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| direction_error | 5 | 244 | 0.30738 | 245 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
| fill_probability_absolute_error | 3 | 92 | 0.29341 | 53 | FIXED_DEVELOPMENT_DISCOVERED_RULE_POSTHOC_ONLY |
