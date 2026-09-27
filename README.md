# MarketForge-QTR

Research-grade, Windows-native framework for multi-market microstructure modeling,
Transformer pretraining, counterfactual L3 execution simulation, calibration, and
execution-aware market-making evaluation.

> [!IMPORTANT]
> MarketForge-QTR is research software, not investment advice or a trading system. The v2
> study found no profitable after-cost market-making edge. Raw market data, final-lockbox
> data, and trained weights are not distributed.

## Overview

MarketForge-QTR implements a leakage-controlled research pipeline for:

- multi-market L1/L2/L3 microstructure modeling;
- a dual-stream Multi-Market Microstructure Transformer (M3T);
- supervised learning, self-supervised pretraining (SSL), and economic post-training (EPT);
- direction, return, volatility, side-specific markout, fill, and adverse-selection targets;
- frozen calibration and fair common-controller comparisons;
- FIFO counterfactual `SIMULATED_L3` queue/fill replay;
- dimensionally explicit execution accounting; and
- OOD evaluation, robustness, failure analysis, and production-style diagnostics.

The repository preserves the full implementation and selected aggregate research reports.
Licensed/provider data, immutable lockbox files, checkpoints, calibrators, controllers, and
machine-local run state remain outside the public repository.

## Research question

Can multi-market microstructure pretraining learn transferable, calibrated representations
that improve execution-aware, inventory-constrained market making on unseen instruments and
execution conditions—and do any predictive gains survive realistic transaction costs?

## Architecture

```text
Market data
  -> canonicalization and book reconstruction
  -> grouped leakage-safe windows
  -> M3T event/state representation
  -> supervised or SSL initialization
  -> economic post-training
  -> frozen calibration
  -> quote controller
  -> risk gate
  -> SIMULATED_L3 execution replay
  -> predictive and after-cost evaluation
```

M3T combines a causal event stream with an ordered book/state stream, source and instrument
embeddings, and modality masks. The corrected v2 ladder contains `M3T-SUP`, `M3T-SSL`,
`M3T-SUP-EPT`, `M3T-SSL-EPT`, and separately calibrated SUP/SSL EPT branches.

## Data domains

The completed v2 study used official Databento public samples:

- CME Globex ES MBO/L3 for development execution-aware modeling;
- ICE Futures Europe Brent L2 for compatible representation/forecasting analysis; and
- Nasdaq TotalView-ITCH NVDA MBO/L3 as the one-time unseen final lockbox.

The valid primary statistical units are only two development instrument-days and one final
instrument-day. FX acquisition was not established under the unattended public-data
constraints and was not used to manufacture multi-market evidence.

Data are not redistributed. Users must obtain compatible files from the official providers,
accept the applicable terms, and place them under their own `MARKETFORGE_DATA_ROOT`. See
[`licenses/data_sources.yaml`](licenses/data_sources.yaml) and
[`PUBLICATION_MANIFEST.md`](PUBLICATION_MANIFEST.md).

## Key results

The authoritative one-time v2 final evaluation reported:

| Metric | M3T | XGBoost |
| --- | ---: | ---: |
| Direction macro-F1 | **0.2564** | 0.1903 |
| Post-hoc bid fill Brier | 0.3032 | **0.2592** |
| Post-hoc ask fill Brier | 0.2792 | **0.2196** |

M3T transferred directional signal better on the unseen final domain, while XGBoost forecast
counterfactual fills better. Better directional prediction did **not** translate into
profitable market-making economics.

| Diagnostic | M3T | XGBoost | Frozen policy |
| --- | ---: | ---: | ---: |
| Quotes | 2,155 | 692 | 0 |
| `SIMULATED_L3` fills | 682 | 196 | 0 |
| Passive spread capture | $4.590 | $1.570 | $0 |
| Subsequent price movement | -$5.330 | -$1.795 | $0 |
| Gross PnL | -$0.740 | -$0.225 | $0 |
| Fees | $4.092 | $1.176 | $0 |
| Liquidation cost | $1.705 | $0.490 | $0 |
| Net PnL | **-$6.537** | **-$1.891** | **$0** |

The preselected authoritative policy remained `NO_TRADE`. These are small-order,
no-endogenous-impact historical replay diagnostics—not evidence of future investment returns.

## Post-hoc diagnostics

The isolated, non-authoritative post-hoc revision 4 found:

- XGBoost's fill-Brier advantage was mainly reliability-driven; bid also showed a smaller
  resolution advantage.
- M3T confidence was `NON_MONOTONIC` on development-fixed final confidence quartiles.
- The largest normalized structural shift was relative volatility (descriptive KS `0.6150`).
- M3T's largest observed fill weakness was ask-side in the lowest development-defined
  volatility bucket.
- One development-defined slice had positive diagnostic economics, but it remains an
  exploratory `V3_CANDIDATE`, not a selected v2 strategy.

These analyses did not change any v2 model, threshold, calibration object, controller,
selection decision, or claim. See the
[`post-hoc executive summary`](reports/posthoc_v2/POSTHOC_EXECUTIVE_SUMMARY.md).

## Scientific governance

- v1 is preserved as engineering/audit history but is superseded for economic claims.
- v2 is the authoritative corrected protocol.
- The final lockbox was isolated, opened once, and remains at `evaluation_count = 1`.
- Three documented outcome-blind structural amendments corrected evaluator mechanics without
  changing rows, targets, models, calibration, controllers, or assumptions.
- Post-hoc revision 4 is explicitly non-authoritative and exploratory.
- The authoritative final-evaluation SHA-256 is
  `bd4a8cfc4b903aba4ada5530c96c7f34764a258d71650e8a5c8452afe6251c1b`.

The public reports retain the negative/null findings and the conclusion that population-level
inference is not established.

## Repository structure

```text
src/marketforge/       Python package and pipeline stages
tests/                 Unit, leakage, accounting, simulator, and semantics tests
scripts/mf.ps1         Windows-native command wrapper
config/                Versioned instrument registry
docs/                  Research charter, assumptions, limitations, and release notes
licenses/              Data/model/dependency provenance boundaries
reports/final_v2/      Selected immutable authoritative v2 reports
reports/posthoc_v2/    Non-authoritative revision-4 diagnostic reports and aggregate tables
.github/               CI and dependency-update configuration
```

## Installation

The implemented research environment targets Python 3.12. Neural research stages require a
compatible NVIDIA GPU and a CUDA-enabled PyTorch build; ordinary unit tests and CI use CPU
PyTorch.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,research,ml]"
```

For CUDA work, install the current official PyTorch wheel appropriate for the host before
running GPU stages. Never silently substitute CPU for a GPU-required experiment.

## Quick start

```powershell
.\.venv\Scripts\python.exe -m marketforge.cli --help
.\scripts\mf.ps1 status
.\scripts\mf.ps1 smoke
```

The Python-native interface is `python -m marketforge.cli <stage>` and the PowerShell-native
wrapper is `.\scripts\mf.ps1 <stage>`. Commands that require the unpublished datasets,
checkpoints, or lockbox artifacts will not reproduce from a public clone alone. Public users
should not invoke final-lockbox commands without a separately authorized private artifact set.

## Testing

```powershell
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pytest -q
```

Artifact-bound tests skip cleanly in a public clone; they run in the preserved private
research workspace where the immutable evidence is available.

## Reproducibility

The public repository reproduces implementation-level unit tests, feature/label semantics,
book and queue logic, dimensional accounting, grouped-window isolation, and CLI imports. It
also publishes aggregate reports and their scientific conclusions.

Exact model training and final metrics require the original provider files, dataset
fingerprints, frozen checkpoints, calibration objects, controller state, and lockbox files.
Those are intentionally omitted for licensing, size, and lockbox-governance reasons. The
public reports state what is and is not established; they are not a substitute for the private
immutable evidence bundle.

## Limitations

- Only three instrument-day primary units support the central v2 evidence.
- The data are public/provider-specific rather than representative of all markets.
- `SIMULATED_L3` fills are counterfactual and must never be described as observed executions.
- Historical replay assumes a small order and no endogenous market impact.
- ICE rows are L2-only for execution targets and never enter L3 fill/queue claims.
- Population-level statistical significance across markets is not established.
- No profitable trading edge was established after costs.

See [`reports/final_v2/LIMITATIONS.md`](reports/final_v2/LIMITATIONS.md) and
[`reports/posthoc_v2/POSTHOC_LIMITATIONS.md`](reports/posthoc_v2/POSTHOC_LIMITATIONS.md).

## Future work

The highest-priority v3 experiment is to acquire multiple independent MBO instrument-days,
preregister the comparison, and cross-fit a queue-specialized hybrid M3T/XGBoost fill head.
Additional candidates are ranked in
[`reports/posthoc_v2/V3_CANDIDATES.md`](reports/posthoc_v2/V3_CANDIDATES.md).

## License

Original MarketForge-QTR source code is available under the [MIT License](LICENSE). This
license does not grant rights to third-party market data, external model weights, or external
code. Consult the provider and dependency licenses before acquiring or redistributing them.

## Citation

Please cite the project using [`CITATION.cff`](CITATION.cff). No DOI has been assigned.
