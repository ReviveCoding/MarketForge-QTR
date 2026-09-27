# Decision log

## D-001 — Python 3.12 environment

Use the installed Python 3.12 for `.venv`, not the launcher default 3.13. Current official PyTorch Windows documentation supports Python 3.9-3.12.

## D-002 — Storage reserve

Maintain 40 GiB free on C: (`max(25, min(40, 5% of 951.6 GiB))`). Acquire smoke then public-core data progressively.

## D-003 — No premature final lockbox

Do not materialize or inspect final-lockbox rows until the development protocol and artifact manifest are frozen. Public data will be partitioned chronologically and the development commands will enforce denial.

## D-004 — Evidence discipline

Only artifact-derived metrics enter reports. Missing evidence is `NOT ESTABLISHED`; unavailable paid/authenticated sources become `BLOCKED_EXTERNAL_DATA` without blocking unaffected stages.

## D-005 — CUDA runtime

Use official PyTorch 2.14.0+cu130 on Python 3.12. The measured RTX supports BF16. Every neural trainer must assert CUDA placement and share the single-GPU lease.

## D-006 — Public-core substitution and blocker discipline

Use the official Databento public CME samples as the feasible public core. Mark FI-2010 reproduction `BLOCKED_EXTERNAL_DATA` because its authoritative record disables download; do not source an unofficial mirror.

## D-007 — Preserve event order

Canonical and reconstructed datasets retain a monotonic ingest index in addition to exact nanosecond timestamps. Earlier v1 artifacts that lost within-timestamp order are rejected and never used.

## D-008 — Stop neural scaling after failed funnel

The weighted M3T diagnostic remained below HGB and SSL produced zero improvement. Per the specification's downgrade criteria, do not spend the single-GPU budget on Medium/Large or external-foundation-model scaling without development evidence.

## D-009 — Economic claims downgraded

Paired MBP-10 validates available book fields, not queue position or empirical fills. Label all current economics `COUNTERFACTUAL_CROSSING; NO_HEADLINE_PNL`. The negative development result and no-trade winner remain first-class findings.

## D-010 — Freeze a negative final protocol

The absence of a development edge does not justify inspecting the lockbox informally. Freeze the best feasible simple/neural comparisons, execute them once, and prohibit post-lockbox tuning. The final evaluation count is one.

## D-011 — Final conclusion

Reject any robust-edge or complexity-justification claim. Preserve the limited predictive M3T-SUP win alongside the failed SSL/EPT/economic chain, external-data blockers, and noninferential simulator economics.

## D-012 — V1 scientific supersession

Preserve every v1 byte and hash important evidence. V1 supervised/SSL results are historical diagnostics; v1 EPT, calibration targets, and all economics are invalid for v2 claims. The spent v1 TEST/lockbox may not be represented as pristine evidence.

## D-013 — V2 development and lockbox allocation

Use ES and ICE Brent MBO for development and true counterfactual queue experiments. Seal the newly acquired XNAS/NVDA day before outcome construction as strict source/asset/instrument OOD `final_lockbox_v2`. Metadata-only coverage inspection does not authorize outcome inspection.

## D-014 — Honest sample-unit semantics

The public samples currently provide only two active development instrument-days and one sealed final instrument-day. Point transfer comparisons are feasible; conventional instrument-day inference is not. Statistical stages must use `SKIPPED_INSUFFICIENT_SAMPLE` unless additional independent units become lawfully available.

## D-015 — V2 execution labels

Only replay-derived passive-order outcomes are called `SIMULATED_L3`. Endpoint midpoint crossing, if retained, is named `ENDPOINT_CROSSING_PROXY`. The simulator assumes small orders and no endogenous impact, and reports empirical historical-order calibration error rather than perfection.

## D-016 — Freeze corrected negative v2 selection

Freeze XGBoost CUDA as the primary predictive baseline, M3T-SSL-EPT-CAL as the neural comparison, and `NO_TRADE` as the economic policy. Development XGBoost won the predictive comparison and every active controller lost after costs, so Medium/Large scaling remained stopped.

## D-017 — Outcome-blind ordinal replay identity amendment

After the one-time XNAS replay, `prediction_time` proved non-unique: one timestamp had multiplicity four. Preserve every immutable row and align by in-memory ordinal `capture_id`. Amendments 001–003 record the first pre-inference failure, the second diagnostic lookup occurrence, exact hashes, structural tests, and execution binding. No scientific choice or row changed; evaluation count remained one.

## D-018 — Final v2 conclusion

Treat `FINAL_EVALUATION_V2_WITH_OUTCOME_BLIND_STRUCTURAL_AMENDMENT` as authoritative. Preserve the frozen no-trade decision. Report final M3T/XGBoost predictive point estimates and negative active diagnostics, but do not claim population inference, provider OOD, observed fills, live PnL, or economic edge.
