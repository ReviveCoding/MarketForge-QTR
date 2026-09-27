# Economic Attribution

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Frozen policy

`NO_TRADE` remains the authoritative frozen policy with zero trading PnL. Active results below are diagnostics and cannot become a selected strategy.

## Attribution

| model | quotes | fills | gross_spread_price_edge_usd | spread_capture_usd | subsequent_markout_price_movement_usd | adverse_selection_component_usd | fees_usd | liquidation_cost_usd | slippage_usd | hedge_cost_usd | realized_pnl_usd | net_pnl_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3T | 2155 | 682 | -0.74 | 4.59 | -5.33 | -6.195 | 4.092 | 1.705 | 0 | 0 | -6.537 | -6.537 |
| XGBOOST | 692 | 196 | -0.225 | 1.57 | -1.795 | -2.065 | 1.176 | 0.49 | 0 | 0 | -1.891 | -1.891 |

## Authoritative reconciliation

| model | row_type | quotes | fills | gross_price_markout_pnl_usd | fees_usd | liquidation_cost_usd | slippage_usd | hedge_cost_usd | net_pnl_usd | reconciliation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3T | AUTHORITATIVE | 2155 | 682 | -0.74 | 4.092 | 1.705 | 0 | 0 | -6.537 | REFERENCE |
| M3T | RECONSTRUCTED_POSTHOC | 2155 | 682 | -0.74 | 4.092 | 1.705 | 0 | 0 | -6.537 | MATCH |
| XGBOOST | AUTHORITATIVE | 692 | 196 | -0.225 | 1.176 | 0.49 | 0 | 0 | -1.891 | REFERENCE |
| XGBOOST | RECONSTRUCTED_POSTHOC | 692 | 196 | -0.225 | 1.176 | 0.49 | 0 | 0 | -1.891 | MATCH |
| NO_TRADE | AUTHORITATIVE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | REFERENCE |

## Accounting identity

Passive spread capture is fill-to-contemporaneous-mid. Subsequent markout/price movement is contemporaneous-mid-to-future-mid. Their sum equals total gross fill-to-future-mid PnL; spread capture is diagnostic and is never added twice. Net PnL = total gross PnL - fees - liquidation - slippage - hedge cost.

## Expected versus realized edge

| model | side | development_edge_bucket | support | predicted_edge_mean_ticks | realized_edge_mean_ticks | signed_bias_ticks | mae_ticks | spearman_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3T | ask | Q1 | 450 | -2.7312 | 0 | -2.7312 | 2.7312 | NA |
| M3T | ask | Q10 | 18 | 0.89634 | 0.51667 | 0.37967 | 1.3509 | -0.34073 |
| M3T | ask | Q2 | 490 | -0.86403 | 0 | -0.86403 | 0.86403 | NA |
| M3T | ask | Q3 | 530 | -0.50548 | 0 | -0.50548 | 0.50548 | NA |
| M3T | ask | Q4 | 508 | -0.30703 | 0 | -0.30703 | 0.30703 | NA |
| M3T | ask | Q5 | 445 | -0.16234 | 0 | -0.16234 | 0.16234 | NA |
| M3T | ask | Q6 | 408 | -0.05018 | 0 | -0.05018 | 0.05018 | NA |
| M3T | ask | Q7 | 314 | 0.00043195 | -0.13822 | 0.13865 | 0.30146 | -0.12842 |
| M3T | ask | Q8 | 32 | 0.029137 | -0.06875 | 0.097887 | 0.097887 | 0.28911 |
| M3T | ask | Q9 | 12 | 0.12667 | 0.079167 | 0.047499 | 0.7989 | 0.37634 |
| M3T | bid | Q1 | 240 | -4.9237 | 0 | -4.9237 | 4.9237 | NA |
| M3T | bid | Q10 | 672 | 0.639 | -0.30491 | 0.94392 | 0.96317 | 0.085051 |
