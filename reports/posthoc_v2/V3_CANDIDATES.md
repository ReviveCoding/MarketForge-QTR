# V3 Candidate Backlog

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Governance

Every item is explicitly `V3_CANDIDATE`, non-authoritative, and was not applied to v2.

## Ranked backlog

| Rank | Candidate | Evidence | Expected impact | Effort | New data | Overfit risk |
|---:|---|---|---|---|---|---|
| 1 | `V3_CANDIDATE`: acquire multiple MBO instrument-days and rerun independent-day validation | Primary-unit limitation dominates all inference | High | High | Required | Low |
| 2 | `V3_CANDIDATE`: queue-specialized XGBoost/M3T hybrid fill head | XGBoost final fill Brier is lower while M3T direction is stronger | High | Medium | Helpful | Medium |
| 3 | `V3_CANDIDATE`: cross-fitted beta/vector or isotonic fill calibration | Reliability residuals vary by side and queue | Medium | Medium | Helpful | High |
| 4 | `V3_CANDIDATE`: queue-aware SSL objective | Queue/depth cohorts expose execution errors | Medium | High | Required | Medium |
| 5 | `V3_CANDIDATE`: pre-registered slice-specific controller | Some development-derived cohorts differ materially | Medium | Medium | Required | Very high |
| 6 | `V3_CANDIDATE`: richer legal 10 ms–5 s labels | Current frozen labels expose only 100 ms/1 s | Medium | High | Required | Low |
