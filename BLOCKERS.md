# Blockers

## B-001 — FI-2010 published reproduction (`BLOCKED_EXTERNAL_DATA`)

The authoritative Fairdata record identifies the dataset and CC BY 4.0 license, but its API reports download is not enabled. No payment, authentication bypass, or unofficial mirror is permitted. Gate 6 cannot be executed until the provider enables access or the user supplies a lawfully obtained copy.

## B-002 — Strict transfer and multi-day inference (`BLOCKED_EXTERNAL_DATA`)

The acquired Databento public samples contain ESZ5 on one trading date. They support temporal regime stress but not honest unseen-instrument, cross-source, FX↔futures transfer, or instrument×day inference. These claims must remain `NOT ESTABLISHED` unless another legal source is acquired.

Superseded by the narrower v2 status: official ICE Brent and XNAS NVDA samples now permit point-estimate cross-source/instrument experiments. However, only two active development instrument-days and one sealed final instrument-day are available, so population-level transfer inference and most dependence-aware tests remain `SKIPPED_INSUFFICIENT_SAMPLE`.

## B-003 — Official Dukascopy automated acquisition (`BLOCKED_EXTERNAL_DATA`)

The official Historical Data Export is interactive and the current review did not establish a documented, license-clear unattended download interface suitable for this pipeline. No unofficial mirror or undocumented endpoint will be used. FX transfer remains unavailable unless an official permitted export method or user-supplied lawful corpus becomes available.

## B-004 — Post-freeze final row identity (`RESOLVED_WITH_AUTHORIZED_AMENDMENT`)

The single v2 lockbox replay completed with 3,238 feature rows and 3,238 label rows, zero reconstruction errors, and immutable hashes. Final scoring initially stopped before inference because XNAS contains repeated `prediction_time` values and the frozen runner required a one-to-one timestamp merge.

The user explicitly authorized the minimal outcome-blind amendment. Amendments 001–003 pair all immutable rows by ordinal `capture_id`, propagate that identity into prediction diagnostics, and bind the exact execution source. No row or scientific choice changed. `artifacts/final_evaluation_v2.json` now exists with `evaluation_count = 1`, and repeat commands return it unchanged.

Outcome-blind structural evidence is recorded in `artifacts/final_lockbox_v2_structural_failure.json`: 3,238 rows per file, exact rowwise timestamp equality, 3,235 unique timestamps, one timestamp with multiplicity four, and no predictions or metrics generated. The amendment must use ordinal `capture_id`; it must not deduplicate or filter rows.

## Non-blocking limitations

- Paired MBP-10 validates reconstructed book fields but not historical queue position, impact, or empirical fill probabilities; headline PnL is prohibited.
- The neural funnel failed its scaling criterion, so Medium/Large, multi-seed, and external TSFM comparisons are intentionally resource-limited rather than silently omitted.
- The development TEST partition has been evaluated once for the frozen Gate 13 comparison and must not be used for further tuning.
