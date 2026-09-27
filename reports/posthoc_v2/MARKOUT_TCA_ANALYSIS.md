# Multi-horizon Markout and TCA Analysis

Analysis layer: `MARKETFORGE_QTR_V2_POSTHOC_DIAGNOSTICS` — **NON-AUTHORITATIVE / POST-HOC ONLY**.

The authoritative v2 result, frozen NO_TRADE policy, evaluation count, models, calibrators, controller, and claims are unchanged.

## Availability

Existing frozen labels support side-consistent 100 ms and 1 s quote markouts and 100 ms post-fill markouts. 10/25/50/250/500 ms and 5 s, plus event-count horizons, are unavailable without replay and were not invented.

## Selected curves

| partition | model | side | markout_type | horizon | slice_dimension | slice_value | support | mean | median | q10 | q90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | spread_bucket | Q1 | 120 | 0.0049167 | 0.005 | 0.005 | 0.005 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | spread_bucket | Q2 | 245 | 0.11965 | 0.125 | 0.005 | 0.375 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | spread_bucket | Q3 | 19 | 0.19079 | 0.125 | 0.125 | 0.375 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | realized_vol_50_bucket | Q1 | 96 | 0.13542 | 0.125 | 0.125 | 0.375 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | realized_vol_50_bucket | Q2 | 96 | 0.12891 | 0.125 | 0.125 | 0.25 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | realized_vol_50_bucket | Q3 | 96 | 0.07974 | 0.015 | 0.005 | 0.125 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | realized_vol_50_bucket | Q4 | 96 | 0.0052083 | 0.005 | 0.005 | 0.01 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | queue_ahead_bucket | Q1 | 62 | 0.13911 | 0.125 | -0.125 | 0.375 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | queue_ahead_bucket | Q2 | 62 | 0.14113 | 0.125 | 0.125 | 0.375 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | queue_ahead_bucket | Q3 | 60 | 0.12083 | 0.125 | 0.125 | 0.125 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | queue_ahead_bucket | Q4 | 59 | 0.13771 | 0.125 | 0.125 | 0.125 |
| DEVELOPMENT | M3T | bid | QUOTE_MARKOUT_100MS | 100ms | m3t_confidence_bucket | Q1 | 96 | 0.14714 | 0.125 | 0.0625 | 0.375 |

## Semantics

Quote-time forward markouts are explicitly `QUOTE_MARKOUT_100MS` or `QUOTE_MARKOUT_1000MS`; they are not fills. Only fill-conditioned replay outcomes are `SIMULATED_L3_POST_FILL_MARKOUT_100MS`. Simulated fills are never described as observed, and the small-order/no-endogenous-impact limitation remains.
