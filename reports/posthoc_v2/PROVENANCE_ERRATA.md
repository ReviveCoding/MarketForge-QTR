# Post-hoc Provenance Errata

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Observed chronology inconsistency

The immutable supporting artifact `artifacts/final_evaluation_v2_idempotency.json` records `verified_at_utc = 2026-09-27T18:42:00Z`, while amendment 002 and 003 embed later seal times (`18:45:00Z` and `18:50:00Z`). The final evaluation embeds `18:40:12.756441+00:00`. This makes the supporting timestamp inconsistent with the full transcript/amendment narrative if read as a total ordering.

## Controlling provenance evidence

Execution transcript ordering, immutable SHA-256 identities, filesystem mtimes, the amendment chain, and the final result identity are the meaningful evidence. The idempotency artifact still binds identical before/after final hashes and `evaluation_count = 1`. Embedded timestamps and mtimes do not form one internally consistent clock sequence.

## Disposition

No authoritative file was rewritten, and the inconsistency does not alter data, predictions, metrics, economics, or scientific results. Examination did not establish a benign clock-skew or manual-recording explanation, so none is asserted. This is a post-hoc provenance erratum only.
