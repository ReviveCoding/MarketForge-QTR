# MarketForge-QTR v2 Post-hoc Executive Summary

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Governance

The v2 lockbox remained at `evaluation_count = 1`. This layer changes no authoritative claim or artifact.

## Strongest calibration finding

Frozen XGBoost had lower final post-hoc total fill Brier on bid and ask: bid 0.2592 vs M3T 0.3032; ask 0.2196 vs M3T 0.2792. This total score is not, by itself, proof of better calibration. Brier is not a pure calibration metric; reliability, resolution, and uncertainty are reported separately.

## Largest observed M3T advantage among predefined post-hoc slices

M3T's largest observed advantage was `m3t_confidence_bucket=Q1` with support 225, M3T 0.3193, XGBoost 0.1464, difference 0.1729. XGBoost's largest observed advantage was `event_rate_bucket=Q2` with support 847, M3T 0.2018, XGBoost 0.2191, M3T-minus-XGBoost -0.0173. Definitions were fixed from development where applicable; all findings are `EXPLORATORY_POSTHOC_ONLY`, correlated across related slices, and carry no event-row p-values.

## Economic mechanism

Both active diagnostic controllers remained after-cost negative while the frozen authoritative policy remained NO_TRADE. Transaction costs plus adverse/weak post-fill edge prevented predictive gains from becoming an economic edge.

## Raw-unit and normalized OOD findings

Raw-unit `depth_ask_5` had KS 0.9906; normalized structural `relative_volatility_bps` had KS 0.6150. Only the latter is used as the structural headline. Neither carries a population p-value claim.

## Ten executive questions

1. **Why did M3T lead directionally?** M3T final macro-F1 was 0.2564 versus 0.1886. The descriptive disagreement/occlusion evidence is consistent with transfer from sequential state/event representations; one unseen day cannot establish a causal mechanism.
2. **Why did XGBoost lead fill Brier?** Explicit queue/depth features aligned better with the counterfactual fill process; total bid/ask Brier was lower. XGBoost's adverse input is only a deterministic markout-derived score, not a learned adverse classifier.
3. **Reliability, resolution, or both?** Bid: `BOTH_MAINLY_RELIABILITY`; ask: `PRIMARILY_BETTER_RELIABILITY`. The full reliability, resolution, uncertainty, and binning residuals are reported separately.
4. **Is confidence monotonic/useful?** M3T is `NON_MONOTONIC` and XGBoost is `WEAK` on development-fixed final quartiles. No final threshold was tuned.
5. **Largest normalized shift?** `relative_volatility_bps` with descriptive KS 0.6150. Raw-unit `depth_ask_5` KS 0.9906 is reported separately and is not called structural OOD.
6. **Largest M3T queue/fill failure?** `{'side': 'ask', 'condition': 'realized_vol_50', 'development_bucket': 'Q1', 'm3t_minus_xgboost_brier': 0.4678657170026985}` is the largest observed XGBoost Brier-contribution advantage among predefined L3 execution buckets.
7. **Why did after-cost economics fail?** M3T spread capture $4.5900 was overcome by subsequent markout movement $-5.3300, leaving gross $-0.7400; XGBoost was $1.5700, $-1.7950, and $-0.2250, respectively. Registered costs then deepened both losses.
8. **Costs or adverse markouts?** M3T fees plus liquidation were $5.7970 versus $5.3300 of adverse subsequent movement, so costs were slightly larger. XGBoost costs were $1.6660 versus $1.7950, so adverse movement was slightly larger. Neither mechanism alone explains both models.
9. **Any positive development-defined slice?** Yes: `predicted_fill_probability_bucket=Q1` for M3T had $3.1250 diagnostic net PnL on 61 development events. It remains a V3_CANDIDATE because it is a multiple-slice discovery, not an independently registered strategy.
10. **Highest-priority V3 experiment?** Acquire multiple independent MBO instrument-days, preregister the comparison, and validate a queue-specialized hybrid M3T/XGBoost fill head with cross-fitted calibration.

## V3 priority

Acquire multiple independent MBO instrument-days, then cross-fit side/queue-aware fill calibration and a hybrid representation-plus-queue model. It is a `V3_CANDIDATE`, not a v2 revision.

## Primary limitation

Only one final and two development instrument-day units exist; event counts are not independent population sample size.
