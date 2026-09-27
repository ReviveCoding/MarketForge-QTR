# MARKETFORGE-QTR FINAL MASTER EXECUTION SPECIFICATION

## Mission

You are responsible for completing MarketForge-QTR from this repository as an
end-to-end quantitative-research and ML-systems project.

Do not merely scaffold the repository.

You must implement, execute, validate, analyze, and document every feasible
stage from environment validation and public-data acquisition through final
models, execution-aware market-making experiments, ablations, statistical
analysis, failure analysis, final lockbox evaluation, and final technical
reports.

Work autonomously through milestones.

Do not stop after writing a plan.
Do not stop after implementing code.
Do not stop after smoke tests when later stages are feasible.
Do not fabricate results.
Do not claim experiments that did not run.
Do not silently replace GPU workloads with CPU workloads.

The project should remain useful even if the proposed model loses to a strong
baseline. Rigorous negative results are valid research outcomes.

Read and obey `AGENTS.md` and
`artifacts/system/BOOTSTRAP_PRECHECK.json` before substantial work.

---

# 1. Project identity

Project:
MarketForge-QTR

Title:
Multi-Market Foundation Modeling, Economic Post-Training, and
Execution-Aware Market Making

Primary research question:

Can multi-market microstructure pretraining learn transferable and calibrated
market representations that improve execution-aware, inventory-constrained
market making across unseen instruments, regimes, and execution conditions?

Evidence chain:

Data
-> representation
-> self-supervised pretraining
-> economic post-training
-> calibration
-> fair-value / return / markout / volatility / fill / adverse-selection estimates
-> quote controller
-> risk gate
-> execution simulator
-> after-cost economics
-> robustness
-> statistical validation
-> deployment feasibility

Target professional capabilities:

1. Quantitative Trading & Research, FX quantitative/electronic trading
2. Quantitative Trading & Research, AI/ML foundation-model research

---

# 2. First actions and externalized state

Immediately inspect:

- complete repository tree
- Git status/history if present
- AGENTS.md
- bootstrap precheck
- any previous state/checkpoints
- disk capacity
- installed environment
- available datasets

Preserve valid existing work.

Then create/update:

- PLAN.md
- IMPLEMENT.md
- DOCUMENTATION.md
- STATUS.md
- RUN_STATE.json
- DECISIONS.md
- BLOCKERS.md

Use the long-horizon pattern:

MASTER_PROMPT.md = frozen specification
PLAN.md = milestone plan and acceptance gates
IMPLEMENT.md = operational runbook and exact validation commands
DOCUMENTATION.md = continuously maintained record of execution and results

RUN_STATE.json must use machine-readable stage states:

PENDING
RUNNING
SUCCEEDED
FAILED
BLOCKED_EXTERNAL_DATA
SKIPPED
SKIPPED_RESOURCE_LIMIT

Do not repeatedly recompute successful stages when inputs/config hashes are
unchanged.

---

# 3. Native Windows architecture

This repository runs natively on Windows under PowerShell 7.

Do not make these required:

- WSL
- Bash
- GNU Make
- Snakemake

Snakemake may be documented only as a possible future Linux/WSL alternative.

The canonical execution API must be:

`python -m marketforge.cli <command>`

and the PowerShell wrapper:

`.\scripts\mf.ps1 <command>`

Required high-level commands:

bootstrap
probe
data
audit
canonicalize
study
features
splits
smoke
baselines
pretrain
posttrain
calibrate
simulator-validation
backtest
experiments
analyze
report
full-dev
freeze-final
final-evaluation
full
status
resume

Underlying implementation must be a Python-native stage/DAG orchestrator.

It must support:

- stage dependencies
- content/config hashes
- SQLite or equivalent state database
- checkpointing
- resumability
- retry policy
- resource declarations
- GPU locks
- CPU worker limits
- I/O limits
- atomic completion markers

Do not depend on shell-specific orchestration for correctness.

---

# 4. Environment and dependency strategy

Create an isolated `.venv`.

Prefer a reliable modern Python dependency workflow supported on this machine.

`uv` may be used if available or safely installable.
Otherwise use standard `venv` + pip.

Do not rely on globally installed research libraries.

Pin reproducible dependency versions after compatibility is established.

Install CUDA-compatible PyTorch based on the detected Windows/NVIDIA
environment using current official PyTorch guidance.

Before any expensive deep model:

assert and record:

- torch version
- torch.cuda.is_available()
- torch.version.cuda
- CUDA device count
- GPU names
- VRAM
- BF16 support
- a real tensor operation on CUDA
- successful backward pass on CUDA

Write:

`artifacts/system/cuda_runtime.json`

If CUDA verification fails:

1. diagnose the environment,
2. repair it if feasible,
3. request narrow Auto-review escalation if sandbox/device access is the cause,
4. never start hours-long CPU deep-learning training as a fallback.

---

# 5. GPU/resource policy

Use all available NVIDIA GPUs appropriately.

## Single GPU

All GPU-capable neural models should actually execute on CUDA.

Only one GPU-heavy training/HPO process runs at a time unless a measured
benchmark proves concurrent execution is better and VRAM-safe.

Use CPU parallelism for:

- download/decompression
- Polars/PyArrow preprocessing
- DuckDB queries
- classical statistical models
- block bootstrap
- figure generation
- report generation

Use GPU XGBoost where beneficial, scheduled around neural training.

## Multiple GPUs

Benchmark/select:

- DDP / torchrun for large selected training
- one independent experiment per GPU for screening/HPO

Prefer DistributedDataParallel over DataParallel.

Implement resource-aware scheduling and per-GPU allocation.

## Training optimization

Benchmark/use as appropriate:

- BF16 where supported
- FP16 AMP otherwise
- gradient accumulation
- activation checkpointing
- pinned memory
- persistent DataLoader workers
- prefetching
- nonblocking transfers
- torch.compile when measurably helpful

Record throughput, peak VRAM, wall time, and latency.

---

# 6. Disk and data budget

Read current disk capacity from bootstrap precheck.

Never consume the drive blindly.

Maintain a safety reserve of at least:

max(25 GiB, min(40 GiB, 5 percent of total drive capacity))

unless the user explicitly changes it.

The repository may remain on C:, but large market data must support a
configurable data root via MARKETFORGE_DATA_ROOT.

Default:
  <repo>\data

If another local drive with substantially more free space becomes available,
the pipeline may place raw/canonical market data there while keeping code,
configs, manifests, results, and reports inside the repository.

Never move data outside the repository automatically without recording the
resolved path in the data manifest.

For the current machine, do not consume the remaining system drive close to
exhaustion. Before every material download, recompute the storage budget.

Implement progressive data acquisition:

1. tiny/smoke dataset
2. public-core dataset
3. research-extended dataset only after the core pipeline succeeds
4. institutional/paid data never acquired automatically

Before every large download estimate:

- expected size where knowable
- free space
- projected remaining free space

Resume partial downloads safely where permitted.

---

# 7. Desktop study and references

Before novel modeling, perform a structured desktop study using current,
authoritative sources.

Maintain:

- docs/research_charter.md
- docs/desktop_study.md
- docs/literature_matrix.csv
- docs/baseline_matrix.csv
- docs/data_source_matrix.csv
- docs/source_registry.csv
- docs/references.bib or equivalent
- docs/hypotheses.md
- docs/assumptions_registry.csv
- docs/limitations.md

Study at minimum:

Microstructure:
- mid-price
- weighted mid-price
- microprice
- order-flow imbalance
- depth/queue concepts
- adverse selection
- markout

Market making:
- fixed quoting
- inventory skew
- Avellaneda-Stoikov
- GLFT-style policies

ML/DL:
- linear/logistic
- MLP
- XGBoost/LightGBM
- LSTM
- TCN
- generic supervised Transformer

LOB models:
- DeepLOB
- TLOB
- LOBERT or current equivalent if reproducible

Foundation models:
- Chronos family
- TimesFM family
- Kronos
- relevant current financial/time-series foundation models

For every external model record:

paper
official code
exact version/commit
license
weights license
input format
task
known limitations
local reproducibility status

Verify current information on the web rather than relying on stale assumptions.

---

# 8. Pre-registered hypotheses

Freeze hypotheses and primary comparisons before final testing.

H1 Representation:
dual-stream microstructure representation improves transfer over handcrafted
and single-stream variants.

H2 Pretraining:
M3T self-supervised pretraining improves data efficiency and OOD performance
relative to identical supervised architecture.

H3 Economic post-training:
return + markout + volatility + fill + adverse-selection post-training improves
decision-relevant predictions relative to generic SSL.

H4 Calibration:
better calibration creates incremental quoting value even when raw prediction
accuracy changes little.

H5 Transfer:
multi-market learning provides useful unseen instrument or regime transfer.

H6 Economic mechanism:
prediction improvements matter only when they translate through
markout/adverse-selection into after-cost quoting economics.

H7 Scaling:
validation-loss scaling and economic-value scaling may have different optima.

H8 Robustness:
any claimed edge must survive realistic latency, costs, queue uncertainty,
inventory constraints, and data corruption.

Hash/freeze the preregistration artifact.

---

# 9. Data sources

Build license-aware adapters.

## Public core

### FI-2010

Role:
- public LOB benchmark
- DeepLOB/TLOB reproduction
- architecture sanity

Do not use its short history as the sole evidence for regime generalization.

### Dukascopy historical FX

Role:
- long-history multi-pair FX L1
- self-supervised training
- cross-pair transfer
- temporal/regime analysis

Candidate pairs where available:

EURUSD
GBPUSD
USDJPY
USDCHF
AUDUSD
USDCAD
NZDUSD
EURJPY

Optional:
EURGBP
GBPJPY
AUDJPY
EURCHF

Verify current official acquisition method and licensing before downloading.

Do not bypass provider restrictions.

### Databento official public CME samples

Use currently available public samples where legally accessible:

MBP-10 / L2
MBO / L3

Role:
- book structure
- exchange timestamps
- L2/L3 reconstruction
- queue modeling
- fill simulation validation
- latency analysis
- execution-aware market making

## Optional robustness sources

TrueFX
HistData
Nasdaq ITCH public/sample data
other legally available sources

## Optional institutional extension

Full Databento
CME DataMine
EBS

Never purchase or require paid data automatically.

If blocked:

implement the adapter,
record exact acquisition instructions,
mark experiments BLOCKED_EXTERNAL_DATA,
continue every unaffected stage.

---

# 10. Licensing and provenance

Create:

licenses/data_sources.yaml
licenses/model_weights.yaml
licenses/third_party_code.yaml

For every source record:

official URL
license URL
retrieval method
authentication requirement
payment requirement
redistribution rights
raw Git rights
derived-output rights
notes

Raw licensed market data must not be committed.

Every file receives manifest metadata:

source
instrument
venue
contract
date range
size
SHA256
download timestamp
parser version
row count
timestamp bounds
license class
quality status

Build dataset fingerprints from:

raw hashes
source
date range
parser
schema
preprocessing config

Every experiment must reference exact dataset fingerprints.

---

# 11. Data architecture

Keep raw files immutable.

Layers:

data/raw
data/canonical
data/features
data/labels
data/splits
data/manifests

Canonical storage:

partitioned Parquet

Use:

Polars for high-performance ETL
PyArrow for storage/interchange
DuckDB for analytical queries
NumPy/PyTorch tensors or efficient binary shards for training

Do not make CSV the primary analytical format.

---

# 12. Canonical schema

At minimum:

identity:
- source
- venue
- asset_class
- instrument
- raw_contract
- instrument_id

time:
- ts_event_ns
- ts_recv_ns where available
- trading_date
- session_id
- sequence

event:
- event_type
- action
- side
- order_id where available

values:
- price
- size

LOB:
- bid_px_01..10
- ask_px_01..10
- bid_sz_01..10
- ask_sz_01..10
- bid_ct_01..10
- ask_ct_01..10

contract/session metadata:
- tick_size
- expiration
- days_to_expiry
- roll_rank
- is_roll_window

quality:
- missing_mask
- source_flags
- qa_flags

Preserve event and receive timestamps separately.

Do not call provider-specific spot FX L1 a complete centralized FX order book.

---

# 13. CME/futures semantics

For CME-like event data:

- preserve individual contract identity
- handle rollover explicitly
- do not create raw-model artifacts from unadjusted continuous roll jumps
- account for exchange-specific event semantics
- validate MBO reconstruction against MBP views where possible
- document implied/order-deletion mechanics that affect reconstruction

Store exchange-local session information while keeping canonical timestamps UTC.

---

# 14. Raw QA

Automate:

file integrity
decompression
checksum
row counts
timestamp parsing
duplicate events
sequence gaps
timestamp ordering assumptions
negative/zero prices
invalid sizes
tick-size violations
crossed/locked markets
extreme spreads
missing dates
unexpected long gaps
session classification
contract-roll anomalies
book inconsistencies

Classify:

FATAL
ERROR
WARN
INFO

Do not automatically remove valid unusual market states such as halts without
investigation.

Produce reproducible QA reports.

---

# 15. Desktop market analysis

Before deep learning, perform:

coverage analysis
events per day
activity by instrument
spread distributions
spread by session/hour
depth distributions
depth slope
order/event intensity
interarrival times
return distributions
autocorrelation
volatility
imbalance
OFI
microprice deviation

Study relationships:

E[future return | imbalance]
E[future return | OFI]
E[markout | imbalance]
E[markout | spread]
E[markout | volatility]
P(price move | OFI)
P(fill | queue) where available

Use these analyses to establish strong simple baselines and understand whether
complexity is justified.

---

# 16. Features

Past-only classical feature store:

mid-price
spread
relative spread
microprice
imbalance
OFI
top-1/top-5/top-10 depth
depth slope
depth convexity
order counts
signed volume
return lags
realized volatility
EWMA
robust volatility
event rate
trade rate
cancel/modify rates
interarrival features
session position

Normalization must use train/past information only.

Implement a leakage unit test where future rows are mutated and historical
features must remain unchanged.

---

# 17. Labels

Support event-time and clock-time targets where valid.

Returns:
log(mid_future / mid_now)

Direction:
down / neutral / up
with thresholds chosen only from development data.

Markout:
side_sign * (future_mid - execution_price)

Distinguish:

counterfactual_quote_markout
simulated_execution_markout

Volatility:
future realized volatility plus distributional/quantile targets.

Adverse selection:
markout-based economically meaningful target.

Fill metadata:

OBSERVED
SIMULATED_L3
SIMULATED_L2
UNAVAILABLE

Never label L1 FX fills as observed.

---

# 18. Leakage-safe evaluation

Each sample stores:

information_start
prediction_time
label_end

Implement:

purging
embargo
strict chronological splitting
no overlapping label leakage

Data partitions:

TRAIN
MODEL_VALIDATION
STRATEGY_VALIDATION
TEST
FINAL_LOCKBOX

Final lockbox data must not be included even as unlabeled SSL data.

Distinguish:

strict zero-shot:
target unseen during pretraining and downstream training

unsupervised-seen:
target seen unlabeled during pretraining, not downstream labeled training

few-shot:
limited target labeled adaptation

Do not mix these result classes.

Mandatory leakage tests:

scaler train-only
future-mutation feature invariance
no feature future access
no split overlap
no label-interval overlap
embargo
held-out instrument absence
strict OOD pretraining absence
final-lockbox inaccessibility during development

---

# 19. Multi-source balancing

Prevent large FX datasets from overwhelming L2/L3 or less-liquid instruments.

Compare:

proportional sampling
uniform source sampling
temperature-balanced sampling

Use a form such as:

P(source=i) proportional to N_i^alpha

with alpha below one for balancing.

Record realized batch/source/instrument distributions.

---

# 20. Baselines

Implement genuinely strong baselines.

Naive:
- persistence
- mid-price
- majority/random where meaningful
- historical volatility

Quantitative:
- weighted midpoint
- microprice
- linear/ridge
- logistic
- AR/VAR where appropriate
- EWMA
- GARCH

Tabular:
- LightGBM
- XGBoost

Use CUDA XGBoost where practical.

Simple neural:
- MLP
- MLPLOB-like baseline

Sequential:
- LSTM
- TCN
- supervised Transformer

LOB:
- DeepLOB
- TLOB

Foundation-model comparison where feasible:
- Chronos
- TimesFM
- Kronos
- LOBERT/current comparable model

Separate:

common aggregated-representation track
full microstructure/event track

Never intentionally handicap an external baseline.

---

# 21. Baseline reproduction gate

Before headline comparison with published models:

- reproduce official benchmark protocol where feasible
- use official repository or faithful implementation
- pin version/commit
- preserve license
- compare reproduced metrics against credible published range
- document deviations

If material reproduction fails:

BASELINE_VALIDATION_FAILED

Do not use that comparison as a headline claim until corrected.

---

# 22. Proposed model

Implement:

MarketForge-M3T
Multi-Market Microstructure Transformer

Dual-stream architecture.

## Event stream

event type
action
side
relative price in ticks
normalized size
delta time
venue
instrument
source

## State stream

spread
imbalance
microprice-minus-mid
depth structure
order counts
depth shape
volatility
event intensity
session state

Use:

source embeddings
instrument embeddings
modality masks
explicit missing masks

Support L1/L2/L3 without pretending missing modalities exist.

---

# 23. M3T architecture ablations

Event encoder:
categorical embeddings + continuous projection

State encoder:
preserve ordered price levels

Fusion comparisons:
- concatenation
- gated fusion
- cross-attention

Backbone:
strictly causal Transformer

Test causal-mask correctness.

Model scales:

Tiny
Small
Medium
Large-if-feasible

Tiny:
CI/smoke

Small:
architecture screening

Medium:
main results

Large:
scaling only if local GPU resources support it

If infeasible:
SKIPPED_RESOURCE_LIMIT

Do not force a huge model merely for parameter count.

---

# 24. Training ladder

## Stage 0: M3T-SUP

Supervised sanity model first.

Use it to establish:

pipeline correctness
architecture correctness
competitive baseline

## Stage 1: SSL

Objectives:

masked-event modeling
next-event prediction
future-state reconstruction
contrastive state consistency

Combined:

L_SSL =
lambda_mask * L_mask
+ lambda_next * L_next
+ lambda_state * L_state
+ lambda_contrast * L_contrast

Ablate objectives.

Avoid false-negative contrastive samples from nearly identical market states.

## Stage 2: Economic post-training

Targets:

future-return distribution
multi-horizon markout
future volatility
fill probability where meaningful
adverse-selection probability

L_EPT =
alpha * return
+ beta * markout
+ gamma * volatility
+ delta * fill
+ eta * adverse_selection

Use target availability masks.

## Stage 3: calibration

Compare where justified:

temperature scaling
isotonic calibration
quantile calibration
conformal diagnostics

Calibration uses validation data only.

Final ladder:

M3T-SUP
M3T-SSL
M3T-EPT
M3T-EPT-CAL

---

# 25. Model diagnostics

Perform:

feature-group occlusion
event-stream ablation
state-stream ablation
depth sensitivity
source embedding ablation
instrument embedding ablation
modality-mask ablation
linear probes
representation clustering by instrument/regime

For tree models:
SHAP or permutation importance where practical.

Do not equate attention visualization with causal explanation.

---

# 26. Common predictor API

Trading-capable predictors expose a common interface:

expected_return
expected_markout
volatility
fill_probability when available
adverse_selection_probability
prediction_uncertainty
model_health
timestamp

This lets the same quote controller compare predictors fairly.

---

# 27. Trading-policy baselines

Implement:

Q0 no trade
Q1 fixed symmetric quote
Q2 microprice + inventory skew
Q3 Avellaneda-Stoikov
Q4 GLFT-style inventory-aware policy
Q5 XGBoost-conditioned controller
Q6 M3T-SUP controller
Q7 M3T-SSL controller
Q8 M3T-EPT controller
Q9 M3T-EPT-CAL controller

RL is optional secondary research only after the interpretable primary path is
validated.

---

# 28. Interpretable quote controller

Use a common parametric family such as:

reservation_price =
mid
+ w_mu * expected_move
- lambda_q * inventory
- lambda_a * adverse_selection_penalty

half_spread =
base_spread
+ w_sigma * volatility
+ w_a * adverse_selection_probability
- w_f * fill_probability
+ w_q * abs(inventory)

bid = reservation_price - half_spread
ask = reservation_price + half_spread

Controller parameters are tuned only on STRATEGY_VALIDATION.

Use the same controller family across XGBoost and M3T predictor comparisons.

---

# 29. Risk controls

Add a production-style risk gate.

Controls:

max absolute inventory
max order size
max notional
price collar
max quote rate
max cancel rate
daily loss limit
drawdown limit
stale feed threshold
latency threshold
model-health threshold

Kill switch triggers:

NaN/invalid predictions
stale feed
sequence corruption
latency breach
inventory breach
PnL breach
invalid quote
critical data-quality failure

On kill:
cancel outstanding simulated quotes
block new quotes
record exact trigger and state

---

# 30. Execution simulation

Use hftbacktest if it installs and validates correctly on this Windows
environment.

It is not mandatory if a compatibility problem remains.

If hftbacktest is unsuitable, implement a narrowly scoped validated internal
replay adapter rather than switching OS merely to preserve the dependency.

Support as feasible:

feed latency
order latency
response latency
queue position
partial fills
fees
slippage
inventory
cancellation

Historical replay does not model endogenous market impact.

Primary market-making claims use small-order assumptions.

Impact is handled as a sensitivity/stress parameter, not claimed as historical
ground truth.

---

# 31. Simulator validation gate

Before headline PnL:

SV1 MBO book reconstruction
SV2 reconstructed top levels vs observed MBP
SV3 L3 exact queue validation where supported
SV4 L2 queue approximation vs L3 truth
SV5 fill calibration
SV6 latency ordering

Invariants:

no fill before effective order arrival
position conservation
cash conservation
valid queue state
no invalid post-fill cancellation
book consistency

If simulator validation materially fails:
do not issue headline trading claims.

---

# 32. Accounting

Track:

inventory q_t
cash C_t
NAV_t = C_t + q_t * mid_t

Separately record:

realized PnL
unrealized PnL
spread capture
fees
slippage
hedge cost
turnover
inventory

Primary session experiment:

forced end-of-session flatten with explicit liquidation cost.

Secondary:

carry allowed where scientifically meaningful.

Do not hide open inventory by merely marking it at mid.

---

# 33. Metrics

## Predictive

Macro-F1
MCC
balanced accuracy
MAE
RMSE
Spearman/rank IC where appropriate
NLL
Brier
CRPS where applicable
pinball loss
ECE/reliability
quantile coverage

## Economic

net after-cost PnL
PnL per turnover
daily PnL
spread capture
multi-horizon markout
negative markout
fill rate
inventory variance
95/99 percent inventory exposure
maximum drawdown
turnover
cancel rate
hedge frequency
hedge cost
quote participation

## Systems

training events/tokens per second
GPU utilization
peak VRAM
wall time
checkpoint size
restart correctness
model inference p50/p95/p99
full pipeline decision p50/p95/p99

Full decision latency includes:

features
host/device transfer
model
controller
risk gate

---

# 34. Experiment matrix

Use a staged screening/confirmation/final process.

## A: Data and representation

normalization
event-time vs clock-time
L1/L2/L3 harmonization
depth 1/5/10
handcrafted/token/dual-stream
event-only/state-only
fusion mechanisms
source embedding
instrument embedding
modality masking
sampling policy

## B: Modeling and pretraining

linear/XGB/MLP
LSTM/TCN
DeepLOB/TLOB
supervised Transformer/M3T-SUP
M3T-SUP vs M3T-SSL
individual SSL objectives
combined SSL
SSL vs EPT
EPT vs calibrated EPT
single-market vs multi-market
linear probe/frozen/full fine-tune
model scaling
data scaling
external TSFM common-input comparisons

## C: Generalization and robustness

temporal OOS
strict unseen instrument
unsupervised-seen instrument
few-shot adaptation
strict regime OOD
liquidity regime
source OOD
FX to futures
futures to FX
data efficiency
missing events
timestamp jitter
delayed features
size corruption
depth dropout
feed gap
stale quote
spread shock
volatility shock

## D: Trading

no trade
fixed quote
microprice
AS
GLFT
XGB
M3T-SUP
M3T-SSL
M3T-EPT
M3T-EPT-CAL
latency
fees
slippage
queue assumptions
inventory penalty
quote width
hedging policy
order size
liquidation
impact stress
breakeven transaction cost

## E: Systems

FP32 vs BF16/FP16
compile on/off
DataLoader workers
pin memory
prefetch
batch size
gradient accumulation
activation checkpointing
single vs multi GPU where available
model-size latency
CPU/GPU inference where meaningful
end-to-end latency
compute/economic Pareto frontier

---

# 35. Experimental budget

Do not brute-force the entire matrix at full scale.

Tier 0:
smoke, tiny data, tiny model

Tier 1:
screening, small data/model, one seed

Tier 2:
confirmation, development data, medium model, three seeds

Tier 3:
frozen final candidates, full feasible data, three to five seeds when practical

Tier 4:
final lockbox

Terminate obviously poor configurations early, but preserve the result and
reason in the registry.

---

# 36. Hyperparameter search

Use Optuna or equivalent.

Single GPU:
one GPU-heavy trial at a time.

Multiple GPUs:
trial per GPU or DDP as appropriate.

Separate:

model HPO
controller calibration
strategy parameter search

Never use TEST or FINAL_LOCKBOX for tuning.

Record every attempted configuration, including failures.

---

# 37. Statistical analysis

Do not treat millions of market events as IID sample size.

Primary inference unit:

instrument x trading day
or dependence-preserving trading-day blocks

Use:

paired effects
block/stationary bootstrap
confidence intervals
effect sizes
day win rate
instrument win rate
regime win rate

Limit headline comparisons in advance.

Primary:

P1 M3T-SSL vs M3T-SUP
P2 M3T-EPT-CAL vs M3T-SUP
P3 common-controller M3T-EPT-CAL vs common-controller XGBoost
P4 M3T-EPT-CAL controller vs AS/GLFT

Apply multiple-comparison correction to secondary hypothesis families where
appropriate.

---

# 38. Backtest selection-bias diagnostics

Where sample size/structure permits, implement:

White Reality Check or suitable implementation
SPA-style diagnostic
Probability of Backtest Overfitting / CSCV
Deflated Sharpe Ratio diagnostic

Do not present these as magical proof.

Preserve total search/trial counts so the diagnostics are interpretable.

---

# 39. Mechanism analysis

If MarketForge wins, explicitly test whether the evidence chain is consistent
with:

pretraining
-> better transfer/calibration
-> better markout/adverse-selection estimates
-> fewer adverse fills / better quote selectivity
-> improved spread capture
-> improved net PnL

Do not claim causality beyond the experimental evidence.

---

# 40. Failure analysis

Mandatory.

For all instrument-day cases where MarketForge loses to strong baselines,
analyze:

spread
volatility
depth/liquidity
event intensity
latency
inventory
confidence
calibration error
regime
source
roll/session effects

Produce explicit sections:

Where MarketForge wins
Where it ties
Where it fails
Why complexity is or is not justified

Negative results must remain in the final report.

---

# 41. Pareto analysis

Produce at minimum:

PnL vs inventory risk
economic value vs p95 end-to-end latency
prediction quality vs compute
economic value vs model size

Do not collapse everything into one arbitrary score.

---

# 42. Rolling shadow-monitoring study

On chronological held-out data, log sequentially:

input state
model version
predictions
proposed quotes
risk-gate outcome
simulated execution
markout
PnL
latency

Monitor:

prediction error
calibration drift
markout degradation
PnL degradation
feature drift
representation drift when practical
latency drift
feed quality

States:

GREEN
WATCH
REVIEW
DISABLE

---

# 43. Tests

Use pytest and appropriate static/lint tools.

Mandatory:

unit tests:
features
labels
PnL
controller
risk gate
calibration

integration:
raw slice -> canonical -> features -> labels -> model -> strategy -> simulator

leakage:
all leakage invariants

simulator:
book reconstruction
queue/fill timing
cash/position conservation
latency ordering

regression:
small fixed fixture with expected outputs

GPU tests:
CUDA tensor
CUDA model
AMP where supported
checkpoint resume
device assertions preventing silent CPU fallback

---

# 44. Smoke pipeline

Implement:

`.\scripts\mf.ps1 smoke`

It must use a tiny dataset and run the full logical chain:

data
QA
canonicalization
features
labels
split
tiny model
prediction
controller
risk gate
small backtest
aggregation
report

Full expensive experiments cannot start until smoke passes.

---

# 45. Pipeline commands

PowerShell wrapper and Python CLI must support at least:

`.\scripts\mf.ps1 bootstrap`
`.\scripts\mf.ps1 probe`
`.\scripts\mf.ps1 data`
`.\scripts\mf.ps1 audit`
`.\scripts\mf.ps1 canonicalize`
`.\scripts\mf.ps1 study`
`.\scripts\mf.ps1 features`
`.\scripts\mf.ps1 splits`
`.\scripts\mf.ps1 smoke`
`.\scripts\mf.ps1 baselines`
`.\scripts\mf.ps1 pretrain`
`.\scripts\mf.ps1 posttrain`
`.\scripts\mf.ps1 calibrate`
`.\scripts\mf.ps1 simulator-validation`
`.\scripts\mf.ps1 backtest`
`.\scripts\mf.ps1 experiments`
`.\scripts\mf.ps1 analyze`
`.\scripts\mf.ps1 report`
`.\scripts\mf.ps1 full-dev`
`.\scripts\mf.ps1 freeze-final`
`.\scripts\mf.ps1 final-evaluation`
`.\scripts\mf.ps1 full`
`.\scripts\mf.ps1 status`
`.\scripts\mf.ps1 resume`

Equivalent Python CLI must exist.

Commands must be idempotent and resumable.

---

# 46. Final lockbox

Development commands must not open final lockbox data.

Before `final-evaluation`, require a frozen manifest containing:

Git SHA or working-tree fingerprint
dataset fingerprints
model config
training config
checkpoint IDs
hyperparameters
controller parameters
strategy parameters
primary experiment list

Write:

`artifacts/final_freeze_manifest.json`

If freeze conditions are not met, final evaluation must refuse to run.

Run the frozen primary final evaluation once.

Do not tune after seeing final-lockbox performance.

---

# 47. Acceptance gates

Gate 0 environment
Gate 1 CUDA
Gate 2 licenses/data
Gate 3 raw QA
Gate 4 canonicalization
Gate 5 leakage
Gate 6 baseline reproduction
Gate 7 supervised M3T
Gate 8 SSL
Gate 9 economic post-training
Gate 10 calibration
Gate 11 simulator validation
Gate 12 risk gate
Gate 13 development economics
Gate 14 transfer
Gate 15 robustness
Gate 16 systems
Gate 17 statistical inference
Gate 18 selection-bias diagnostics
Gate 19 monitoring
Gate 20 final freeze
Gate 21 final evaluation
Gate 22 report/audit

Update STATUS.md and RUN_STATE.json after every gate.

---

# 48. Stop / downgrade criteria

Do not blindly scale broken research.

If M3T-SUP loses badly to simple models:
diagnose before SSL scaling.

If SSL does not improve relevant development endpoints:
perform failure analysis before large scaling.

If predictions improve but economics do not:
investigate calibration, horizon, and execution friction.

If profits exist only at zero latency:
label deployment-infeasible.

If profits exist only under optimistic queue assumptions:
reject robust-edge claim.

If XGBoost matches M3T after costs:
report that complexity is not economically justified.

If Large lowers loss but not economics:
report economic scaling saturation.

These are valid outcomes.

---

# 49. Result warehouse

Use MLflow or another local structured experiment store plus durable
Parquet/JSON artifacts.

Every run records:

run ID
Git/working-tree fingerprint
dataset fingerprint
config hash
seed
hardware
CUDA
model
parameters
precision
wall time
metrics
artifact paths
status

Create:

`results/warehouse/results.parquet`

Use DuckDB for post-analysis.

Do not manually copy final metrics from terminal output.

---

# 50. Figures and tables

Generate programmatically.

Figures should include where supported:

data coverage
spread/depth/activity
microstructure relationships
training curves
pretraining scaling
economic scaling
data scaling
transfer matrix
regime degradation
calibration
markouts
strategy PnL
PnL vs latency
PnL vs costs
inventory distribution
PnL vs inventory risk
economic value vs p95 latency
ablations
failure analysis
monitoring
GPU throughput / VRAM / latency

Tables:

dataset summary
data QA
baseline reproduction
predictive
calibration
transfer
regime robustness
economic
risk
ablation
scaling
systems
selection-bias diagnostics
statistical inference
failure breakdown

CSV + Markdown are mandatory.

LaTeX/HTML optional.

---

# 51. Final reports

Produce at least:

reports/final/MARKETFORGE_QTR_TECHNICAL_REPORT.md
reports/final/EXECUTIVE_SUMMARY.md
reports/final/DATASET_CARD.md
reports/final/DATA_QUALITY_REPORT.md
reports/final/MODEL_CARD.md
reports/final/TRAINING_REPORT.md
reports/final/CALIBRATION_REPORT.md
reports/final/SIMULATOR_VALIDATION_REPORT.md
reports/final/STRATEGY_SPECIFICATION.md
reports/final/RISK_CONTROL_SPECIFICATION.md
reports/final/ABLATION_REPORT.md
reports/final/ROBUSTNESS_REPORT.md
reports/final/BACKTEST_OVERFIT_REPORT.md
reports/final/SYSTEM_BENCHMARK_REPORT.md
reports/final/MONITORING_REPORT.md
reports/final/FAILURE_ANALYSIS.md
reports/final/LIMITATIONS.md
reports/final/FUTURE_WORK.md
reports/final/RESUME_EVIDENCE.md
reports/final/RESULTS_MANIFEST.json

Markdown is the source of truth.

---

# 52. Resume evidence

Generate only from actual measured artifacts.

Fields may include:

from-scratch training completed
events/tokens processed
instruments
markets/sources
parameter count
GPU
training throughput
cross-instrument improvement
OOD improvement
calibration improvement
markout improvement
after-cost economic improvement
inventory risk effect
p95 E2E latency
confidence interval
baseline comparison

Missing evidence must say:

NOT ESTABLISHED

Never invent numbers.

---

# 53. Limitations that must be explicit

At minimum:

spot FX is decentralized
public FX observations are source/provider-specific
L1 lacks exact queue/fills
FI-2010 is short and normalized
public CME samples may have limited history
historical replay lacks endogenous market impact
simulated execution is counterfactual
local RTX compute is not frontier-cluster scale
public-data market making is not JPM proprietary client-flow market making
backtests do not imply future investment returns

---

# 54. Network and external dependency failures

If sandboxed network fails:

do not silently abandon dependencies,
do not switch the entire Codex session to Full Access,
do not disable security controls.

Instead:

1. verify host network,
2. request the narrowest Auto-review escalation for the exact command,
3. keep writes within the repository/cache directories where possible,
4. record the escalation and outcome,
5. continue.

If an external dataset requires payment, unavailable credentials, or manual
license acceptance:

do not bypass it,
mark the source BLOCKED_EXTERNAL_DATA,
implement the adapter/instructions,
continue with public-core data.

---

# 55. Reboot/interruption recovery

All long operations must survive interruption.

Checkpoints include where relevant:

model state
optimizer
scheduler
AMP scaler
epoch
global step
RNG state
sampler state
config hash
dataset hash

Downloads should be resumable when legal/provider-supported.

The pipeline state database is the source of truth.

On rerun/resume:

inspect existing state,
validate artifact hashes,
continue from the last valid stage,
do not automatically restart completed multi-hour work.

Do not leave orphaned detached model-training processes when Codex exits.

---

# 56. Execution order

Use this default order unless evidence forces a documented adjustment:

1 repository inspection
2 environment / CUDA
3 desktop study
4 dependency environment
5 data source/license verification
6 public-core acquisition
7 raw QA
8 canonical schema/parser
9 EDA/desktop market study
10 features/labels
11 leakage-safe split freeze
12 smoke pipeline
13 simple baselines
14 published baseline reproduction
15 M3T-SUP
16 SSL
17 economic post-training
18 calibration
19 simulator validation
20 controllers/risk gate
21 development backtests
22 model/representation ablations
23 transfer
24 robustness
25 systems benchmarks
26 statistical inference
27 selection-bias diagnostics
28 failure analysis
29 rolling monitoring
30 freeze selected final protocol
31 final lockbox
32 aggregation/tables/figures
33 final technical reports
34 full test/audit
35 reproducibility instructions

---

# 57. Completion definition

Do not declare completion because code exists.

Completion means all feasible items below have evidence:

repository architecture complete
reproducible Python environment
CUDA verified
public-core data path working
QA generated
leakage suite passing
baseline suite executed
baseline reproduction documented
M3T ladder implemented
feasible training actually executed
simulator validated to available-data level
economic experiments actually executed where valid
ablation/transfer/robustness completed through resource-aware funnel
systems performance measured
statistical analysis produced
negative/failure results documented
final-lockbox governance honored
reports generated
resume claims audited against artifacts
README documents exact commands
tests pass or remaining failures are explicitly documented

At the end update STATUS.md with:

PROJECT STATUS
COMPLETED GATES
FAILED/BLOCKED GATES
DATASETS ACTUALLY USED
MODELS ACTUALLY TRAINED
GPU USAGE VERIFIED
EXPERIMENTS ACTUALLY RUN
KEY VERIFIED RESULTS
KEY NULL/NEGATIVE RESULTS
RESOURCE-LIMITED ITEMS
EXTERNAL-DATA BLOCKERS
FINAL REPORT PATH
RESUME EVIDENCE PATH
EXACT REPRODUCE COMMAND
EXACT RESUME COMMAND

Proceed with implementation and execution, not just planning.

---

# 58. Windows-native multiprocessing and CUDA worker rules

This project runs natively on Windows.

Python multiprocessing uses Windows spawn semantics.

Therefore:

- all multiprocessing process entry points must be import-safe
- executable process-launch code must be protected by
  if __name__ == "__main__":
- call multiprocessing.freeze_support() where appropriate
- do not depend on Linux fork semantics
- DataLoader worker functions, collate functions, datasets, and process targets
  must be top-level/pickleable objects
- avoid lambdas and local closures in objects sent to worker processes
- explicitly seed DataLoader workers
- use persistent workers only after correctness is verified
- test worker_count = 0 first, then benchmark safe parallel worker counts
- never recursively spawn workers from workers
- CPU preprocessing concurrency and GPU training concurrency must use an
  explicit resource scheduler

The current machine has one NVIDIA RTX GPU unless the runtime probe reports
otherwise.

For one GPU:

- use CUDA parallelism inside the model
- run only one GPU-heavy model training/HPO process at a time
- use BF16 if runtime validation confirms reliable support
- otherwise use FP16 AMP where appropriate
- use gradient accumulation and activation checkpointing when VRAM requires it
- run CPU ETL/statistics/reporting concurrently only when it does not starve
  GPU data loading
- GPU-capable XGBoost must share the same single-GPU resource lock
- never silently execute deep-learning training on CPU

Because the detected GPU has finite VRAM, model scale must be selected from
measured available memory rather than from an aspirational parameter count.

The public-core research study must be completable on the available local GPU.
Larger experiments may be marked SKIPPED_RESOURCE_LIMIT with evidence.

Keep Windows paths reasonably short. Avoid unnecessarily deep generated
directory/file names that risk Windows path-length problems.

---

# 59. Network-bound operations under Auto-review

The bootstrap network probe may fail inside workspace-write even when the host
network works.

For pip/uv installs, official public-data downloads, model-weight downloads,
and official repository clones:

1. first attempt the operation under the normal workspace sandbox
2. if blocked specifically by sandbox network policy, request the narrowest
   command-specific Auto-review escalation
3. do not switch the entire Codex session to Full Access
4. do not bypass authentication, licensing, or payment restrictions
5. record blocked external sources in BLOCKERS.md and continue unaffected work

Live Codex web search is separate from shell/process network access.
