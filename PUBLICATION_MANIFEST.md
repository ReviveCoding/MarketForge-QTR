# Public Publication Manifest

This manifest defines the public boundary for MarketForge-QTR v2.0.0.

## Published

- Original source code under `src/marketforge/`.
- Unit, leakage, simulator, accounting, and semantics tests under `tests/`.
- Windows-native PowerShell entry point under `scripts/`.
- Versioned instrument metadata under `config/`.
- Research design, provenance, assumptions, and limitations under `docs/` and `licenses/`.
- A curated set of immutable authoritative v2 reports under `reports/final_v2/`.
- Non-authoritative post-hoc revision-4 reports, aggregate tables, and figures under
  `reports/posthoc_v2/`.
- Public packaging, CI, contribution, citation, release, and security metadata.

## Intentionally excluded

- All `data/`: raw licensed/provider files, canonical books, features, labels, splits, and the
  spent final lockbox.
- All `artifacts/`: checkpoints, model files, calibrators, controller state, lockbox markers,
  private manifests, runtime evidence, and local audit state.
- All `results/`: row-level predictions, latent representations, experiment stores, and local
  diagnostic outputs.
- Virtual environments, caches, local databases, logs, bootstrap state, and private agent
  prompts/transcripts.
- Generated report manifests or reports that embed unnecessary workstation-absolute paths.

## Data acquisition

Market data are not redistributed. Users must acquire data independently from official
providers under their current terms. `licenses/data_sources.yaml` records the sources and
known redistribution boundary. The MIT software license does not grant data rights.

## Model artifacts

Trained weights and frozen calibration/controller objects are intentionally omitted. They are
small enough in some cases for Git, but publishing them would blur the immutable private
evidence boundary and would not make the study reproducible without the underlying licensed
data. External model weights retain their own licenses.

## Scientific identity

- Authoritative v2 final evaluation SHA-256:
  `bd4a8cfc4b903aba4ada5530c96c7f34764a258d71650e8a5c8452afe6251c1b`
- Authoritative final report-manifest SHA-256:
  `62114f72e58c58e61ed146f9f3ead9b41296b0e34851fb1b3d2e0543ecab2156`
- Authoritative evaluation count: `1`
- Public post-hoc layer: revision `4`, non-authoritative

The selected reports are research records. Their inclusion does not re-open the lockbox,
rerun evaluation, or modify the authoritative result.
