# PRE_FINAL_AUDIT_V2

Generated: 2026-09-27T10:28:12.682085+00:00

Result: **PASS WITH EXPLICIT EXTERNAL/SAMPLE LIMITATIONS**. The v2 final lockbox remained sealed while this audit was produced.

## Required scientific checks

- Bid/ask markouts are side-consistent and are never averaged into the cancelling v1 target: PASS.
- ES fill labels are counterfactual `SIMULATED_L3`; BRN L3 targets are masked after failed post-clear reconstruction: PASS.
- Adverse selection and post-fill markout exist only after a simulated fill: PASS.
- Price, tick, multiplier, and USD accounting are explicit; the unit suite proves 1 ES tick = 0.25 points = $12.50/contract: PASS.
- FIFO books are keyed independently and tests deliberately interleave instruments: PASS.
- Windows cannot cross source, instrument, contract, trading date, session, reset generation, or split: PASS.
- SSL-EPT descends from SSL; SUP-EPT separately descends from SUP: PASS.
- Actual XGBoost 3.4.1 trained on `cuda:0` using the corrected split/features: PASS.
- Available AS/GLFT strategies are honestly named `AS_LIKE_HEURISTIC` and `GLFT_LIKE_HEURISTIC`, with original literature cited: PASS.
- The frozen unseen-market policy maps out-of-vocabulary source/instrument IDs to the mean fitted embedding, without checkpoint tuning: PASS.

## Data and estimability

- Actual scope: 1 provider (Databento), 3 dataset/venue domains, 3 instruments, and 3 instrument-days (including the still-sealed final day).
- Development contains two primary instrument-day units. Cross-market point estimates are computable; population transfer inference, bootstrap confidence intervals, White/SPA, CSCV/PBO, and deflated Sharpe are not interpretable and remain `SKIPPED_INSUFFICIENT_SAMPLE`.
- Strict venue/dataset, asset-class, instrument, and day OOD is estimable once on sealed XNAS/NVDA. Provider OOD is not estimable because every acquired market is from Databento. Development ES↔BRN zero-shot point estimates are diagnostic only.
- ICE MBO L3 economics are unavailable because an unrecovered clear prevents exact post-clear reconstruction; paired official MBP-10 remains valid for L2 representation/forecasting.
- Dukascopy unattended official acquisition and FI-2010 authoritative acquisition remain external-data blockers; no unofficial mirror was used.

## Lockbox and v1 exclusions

- `final_lockbox_v2` is newly acquired XNAS/NVDA and has not supplied labels, predictions, SSL data, tuning, model selection, or controller selection: PASS.
- The v1 TEST and spent v1 FINAL_LOCKBOX are historical diagnostics only. Every v1 invalid/superseded economic artifact is excluded from v2 selection and claims: PASS.
- Frozen selections: XGBoost is the primary predictive baseline; M3T-SSL-EPT-CAL is the neural comparison; the economic policy is NO_TRADE because every active development policy lost after registered costs.

## Evidence

- `data\manifests\features_v4_multi_market.json` — `336115d6c1680584c7762569954f0e70a0403634c1c45e8ea894fd794c82f9ac`
- `data\manifests\labels_v2_l3.json` — `715952b248727482f85c8b31193b7c230d5a5a871130aaba4cac9085b2949df5`
- `data\manifests\splits_v4_multi_market.json` — `ca9d98cf98139fdc4bfab07a17452bd8e060e8929186ea8e7dd1b6a6456c2863`
- `data\manifests\leakage_checks_v4_multi_market.json` — `d8ecbd175800cda9b7bb57bc29a2d137e2980eef04d305a0fb317fe670031eaf`
- `data\manifests\model_ladder_v2.json` — `73d4be1cc87784661e7b906621c15628dce02a4e159813092c50354189fe6d1a`
- `data\manifests\xgboost_gpu_v2.json` — `4e686d7b055d4432632820fc22ef9fbf12ac034c18b2ce1100e5431d734e3b54`
- `data\manifests\simulator_validation_v2.json` — `b0731a9ae8ade00bb198c4ec0a7ca0d5e630c9a1cd78b5038406afaa79d03e85`
- `data\manifests\development_economics_v2.json` — `9741847f929caa7e001bb60d68331478249956739bd4f3375c469a2feffd3205`
- `data\manifests\baselines_v2.json` — `19e4f7ffb9d925a6c6e7041914f304a301818bdb41a2e2a0dc41b1406fff236c`
- `data\manifests\transfer_v2.json` — `2b089af9b356ac14e81a86dad1bc4aa1dcddb92eee8b30495525c2404a98b5f8`
- `data\manifests\ablation_v2.json` — `a6537c3e170d9afc544657135b883cc162ded6e59c741588e6965bc477b34869`
- `data\manifests\robustness_v2.json` — `9e6aca4b45f6c7cba88d452ff23cb391891bbb3adab7ea57cee0bf014878b0fc`
- `data\manifests\systems_benchmark_v2.json` — `c47972ca80765e9d4c0098923a39e1d0d815147c154005996c986918b5ca60fd`
- `data\manifests\development_test_v2.json` — `0ef6c0094d45d61313748d0f0a23bb18ca961eac2c05e53a84fa66f2f6cbe191`
- `data\manifests\monitoring_v2.json` — `96e040288a521e0428b8c7337d377c0fb8670bfca4b49da319dd9e504aff727d`
- `data\manifests\statistical_inference_v2.json` — `7bf4ed6ea541a2d33759d9fb6fd5c397a07ce28f0cc0a374640ec64f9d442bac`
- `data\manifests\selection_bias_v2.json` — `259f9b1c5e0dce4f1bcd3557cb84a04c8b2c1b913b4dbb2cb419be7bbd14105f`
- `data\manifests\failure_analysis_v2.json` — `e0949643575013d85c6bed52750ffb4d6d7a3886ee2d58f37304fa2d5622aec0`
- Acquisition reserve satisfied: `True`.
- ES reconstruction available-data gate: `True`.
- Split fingerprint: `2f4976ad8d538647733a32b6706ef7bc3a7cf1681c19f4dc8b815688f64cc55c`.
