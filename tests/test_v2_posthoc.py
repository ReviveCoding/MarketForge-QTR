from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from marketforge.data import sha256_file
from marketforge.specs import load_registry
from marketforge.state import ROOT
from marketforge.v2_posthoc import (
    _adverse_error_observations,
    _available_mask,
    _bin_edges,
    _binary_calibration,
    _multiclass_metrics,
    _normalized_structural_features,
    registered_round_trip_cost_ticks,
)
from marketforge.v2_posthoc_governance import create_posthoc_charter, protected_inventory


def test_binary_calibration_support_and_decomposition_are_explicit() -> None:
    target = np.asarray([0, 0, 1, 1, 1, 0])
    probability = np.asarray([0.1, 0.2, 0.7, 0.8, 0.9, 0.4])
    metrics, curve = _binary_calibration(target, probability, bins=3)
    assert metrics["support"] == 6
    assert metrics["positive"] == 3
    assert metrics["negative"] == 3
    assert 0 <= metrics["ece"] <= 1
    assert {"brier_reliability", "brier_resolution", "brier_uncertainty"} <= metrics.keys()
    assert metrics["brier_decomposition_within_expected_binning_residual"] is True
    assert (
        metrics["brier_decomposition_absolute_residual"]
        <= metrics["brier_decomposition_expected_binning_tolerance"]
    )
    assert curve["support"].sum() == 6


def test_multiclass_metrics_use_all_three_direction_classes() -> None:
    target = np.asarray([0, 1, 2])
    probability = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    metrics = _multiclass_metrics(target, probability)
    assert metrics["support"] == 3
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["multiclass_brier"] > 0


def test_quantile_edges_cover_out_of_development_range() -> None:
    edges = _bin_edges(np.arange(100), 4)
    assert np.isneginf(edges[0])
    assert np.isposinf(edges[-1])
    assert np.all(np.diff(edges) > 0)


def test_posthoc_charter_binds_current_protected_inventory() -> None:
    if not (ROOT / "artifacts/final_evaluation_v2.json").is_file():
        pytest.skip("non-public authoritative v2 artifacts are not installed")
    charter = json.loads(create_posthoc_charter().read_text(encoding="utf-8"))
    assert charter["authoritative_v2_unchanged"] is True
    assert charter["posthoc_only"] is True
    assert charter["authoritative_evaluation_count"] == 1
    assert charter["protected_inventory"] == protected_inventory()


def test_l2_only_rows_cannot_enter_l3_execution_metrics() -> None:
    frame = pd.DataFrame(
        {
            "l3_targets_available": [False, True],
            "bid_fill": [1, 1],
            "bid_adverse_available": [True, True],
        }
    )
    assert _available_mask(frame, "bid", "post_fill_markout").tolist() == [False, True]
    assert _available_mask(frame, "bid", "adverse").tolist() == [False, True]


def test_unavailable_adverse_targets_are_excluded_not_zero_filled() -> None:
    frame = pd.DataFrame(
        {
            "l3_targets_available": [True, True],
            "bid_adverse_available": [True, False],
            "ask_adverse_available": [False, False],
            "bid_fill_conditional_adverse": [1.0, np.nan],
            "ask_fill_conditional_adverse": [np.nan, np.nan],
            "m3t_bid_adverse_probability": [0.0, 0.0],
            "m3t_ask_adverse_probability": [0.0, 0.0],
        }
    )
    observed = _adverse_error_observations(frame)
    assert len(observed) == 1
    assert observed.iloc[0]["side"] == "bid"
    assert observed.iloc[0]["adverse_absolute_error"] == pytest.approx(1.0)


def test_registry_drives_generic_economic_cost() -> None:
    registry = load_registry()
    assumptions = {
        "liquidation_cost_ticks": 0.25,
        "slippage_usd": 0.0,
        "hedge_cost_usd": 0.0,
    }
    es = registry["databento:GLBX.MDP3:ES"]
    nvda = registry["databento:XNAS.ITCH:NVDA"]
    assert registered_round_trip_cost_ticks(es, assumptions) == pytest.approx(0.45)
    assert registered_round_trip_cost_ticks(nvda, assumptions) == pytest.approx(0.85)


def test_normalized_depth_and_queue_features_are_scale_invariant() -> None:
    base = pd.DataFrame(
        {
            "source_key": ["s", "s"],
            "instrument": ["i", "i"],
            "spread": [1.0, 2.0],
            "tick_size": [1.0, 1.0],
            "mid": [100.0, 100.0],
            "depth_bid_1": [10.0, 20.0],
            "depth_ask_1": [10.0, 20.0],
            "depth_bid_5": [50.0, 100.0],
            "depth_ask_5": [50.0, 100.0],
            "realized_vol_50": [0.001, 0.002],
            "event_rate": [2.0, 4.0],
            "interarrival_ns": [10.0, 20.0],
            "l3_targets_available": [True, True],
            "bid_queue_ahead_at_entry": [5.0, 10.0],
            "ask_queue_ahead_at_entry": [5.0, 10.0],
            "bid_sz_01": [10.0, 20.0],
            "ask_sz_01": [10.0, 20.0],
        }
    )
    scaled = base.copy()
    scale_columns = [column for column in base if "depth" in column or "queue" in column]
    scale_columns += ["bid_sz_01", "ask_sz_01"]
    scaled[scale_columns] *= 100.0
    normalized_base = _normalized_structural_features(base)
    normalized_scaled = _normalized_structural_features(scaled)
    pd.testing.assert_frame_equal(normalized_base, normalized_scaled)


def test_revision_four_outputs_preserve_semantics_and_authority() -> None:
    manifest_path = ROOT / "reports/posthoc_v2/POSTHOC_RESULTS_MANIFEST.json"
    if not manifest_path.is_file():
        pytest.skip("post-hoc output has not been generated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("diagnostic_revision") != 4:
        pytest.skip("revision 4 output has not been generated")

    markouts = pd.read_csv(ROOT / "reports/posthoc_v2/tables/markout_curves.csv")
    assert set(markouts["markout_type"]) == {
        "QUOTE_MARKOUT_100MS",
        "QUOTE_MARKOUT_1000MS",
        "SIMULATED_L3_POST_FILL_MARKOUT_100MS",
    }
    assert (
        not markouts.loc[markouts["markout_type"].str.startswith("QUOTE_"), "markout_type"]
        .str.contains("SIMULATED_L3")
        .any()
    )

    queue = pd.read_csv(ROOT / "reports/posthoc_v2/tables/queue_fill_calibration.csv")
    assert queue["l3_valid_only"].all()
    assert not queue["instruments"].str.contains("BRN", regex=False).any()

    reconciliation = pd.read_csv(
        ROOT / "reports/posthoc_v2/tables/authoritative_economic_reconciliation.csv"
    )
    reconstructed = reconciliation[reconciliation["row_type"] == "RECONSTRUCTED_POSTHOC"]
    assert (reconstructed["reconciliation_status"] == "MATCH").all()
    assert dict(
        zip(reconstructed["model"], reconstructed["net_pnl_usd"], strict=True)
    ) == pytest.approx({"M3T": -6.537000000000003, "XGBOOST": -1.891000000000223})

    monitoring = pd.read_csv(ROOT / "reports/posthoc_v2/tables/monitoring_scorecard.csv")
    execution = monitoring[
        monitoring["metric"].isin(
            [
                "fill_rate",
                "fill_calibration_gap",
                "queue_ahead",
                "post_fill_markout",
                "time_to_fill_ns",
                "adverse_selection",
            ]
        )
    ]
    assert (execution["development_support"] > 0).all()
    assert (execution["final_support"] > 0).all()

    executive = (ROOT / "reports/posthoc_v2/POSTHOC_EXECUTIVE_SUMMARY.md").read_text(
        encoding="utf-8"
    )
    assert "Largest observed M3T advantage among predefined post-hoc slices" in executive
    assert "EXPLORATORY_POSTHOC_ONLY" in executive

    charter_path = ROOT / "artifacts/posthoc_v2/POSTHOC_CHARTER.json"
    if not charter_path.is_file():
        pytest.skip("private post-hoc governance artifacts are not distributed")
    charter = json.loads(charter_path.read_text(encoding="utf-8"))
    assert protected_inventory() == charter["protected_inventory"]
    assert sha256_file(ROOT / "artifacts/final_evaluation_v2.json") == (
        "bd4a8cfc4b903aba4ada5530c96c7f34764a258d71650e8a5c8452afe6251c1b"
    )


def test_posthoc_accounting_identity_is_exact() -> None:
    path = ROOT / "results/posthoc_v2/economic_attribution_detail.parquet"
    if not path.is_file():
        pytest.skip("post-hoc output has not been generated")
    detail = pd.read_parquet(path)
    if "accounting_identity_error_usd" not in detail:
        pytest.skip("revision 4 output has not been generated")
    assert detail["accounting_identity_error_usd"].max() <= 1e-9
