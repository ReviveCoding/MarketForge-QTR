from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import torch

from marketforge.grouped_windows import GROUP_COLUMNS, GroupedWindowDataset, session_position
from marketforge.model_v2 import MarketForgeM3TV2
from marketforge.specs import load_registry
from marketforge.training import BOOK_COLUMNS, EVENT_CONT, GLOBAL_STATE
from marketforge.v2_accounting import futures_round_trip
from marketforge.v2_book import (
    MBOEvent,
    MultiInstrumentBooks,
    SimulatedPassiveOrder,
    finalize_simulated_l3,
)
from marketforge.v2_finalization import _active_diagnostic, _validate_ordinal_alignment
from marketforge.v2_labels import post_fill_markout, side_specific_quote_markouts


def _event(
    instrument: int,
    action: str,
    order_id: int,
    side: str = "B",
    price: float = 100.0,
    size: int = 5,
    ts: int = 1,
) -> MBOEvent:
    return MBOEvent(
        "databento",
        instrument,
        f"C{instrument}",
        "2026-01-02",
        "S1",
        ts,
        ts,
        action,
        side,
        order_id,
        price,
        size,
    )


def test_side_markouts_retain_future_price_information() -> None:
    first = side_specific_quote_markouts(101.0, 99.0, 103.0)
    second = side_specific_quote_markouts(102.0, 99.0, 103.0)
    assert first.bid_quote_markout == 2.0 and first.ask_quote_markout == 2.0
    assert second.bid_quote_markout == 3.0 and second.ask_quote_markout == 1.0
    assert first != second
    assert post_fill_markout("B", 102.0, 100.0) == 2.0
    assert post_fill_markout("A", 98.0, 100.0) == 2.0


def test_interleaved_books_are_isolated() -> None:
    books = MultiInstrumentBooks()
    books.apply(_event(1, "A", 11, price=100.0))
    books.apply(_event(2, "A", 21, price=200.0))
    books.apply(_event(1, "C", 11, price=100.0, size=2, ts=2))
    assert books.get(_event(1, "N", 0).key).orders[11].size == 3
    assert books.get(_event(2, "N", 0).key).orders[21].size == 5
    books.apply(_event(1, "R", 0, side="N", price=None, size=0, ts=3))
    assert not books.get(_event(1, "N", 0).key).orders
    assert books.get(_event(2, "N", 0).key).orders[21].size == 5


def test_simulated_l3_only_actual_ahead_removals_reduce_queue() -> None:
    books = MultiInstrumentBooks()
    first = _event(1, "A", 11, size=2)
    second = _event(1, "A", 12, size=3, ts=2)
    books.apply(first)
    book = books.apply(second)
    order = SimulatedPassiveOrder(first.key, "B", 100.0, 2, 2, 3, 20)
    unrelated = _event(1, "C", 99, size=10, ts=3)
    order.observe(unrelated, book)
    assert sum(order.queue_ahead.values()) == 5
    fill_ahead = _event(1, "F", 11, size=2, ts=4)
    order.observe(fill_ahead, book)
    assert sum(order.queue_ahead.values()) == 3 and order.filled_quantity == 0
    cancel_ahead = _event(1, "C", 12, size=3, ts=5)
    order.observe(cancel_ahead, book)
    assert not order.queue_ahead
    hypothetical_fill = _event(1, "F", 13, size=2, ts=6)
    order.observe(hypothetical_fill, book)
    label = finalize_simulated_l3(order)
    assert label.label_type == "SIMULATED_L3"
    assert label.filled and label.fill_fraction == 1.0 and label.time_to_first_fill_ns == 3


def _window_frame() -> pd.DataFrame:
    rows = 8
    payload: dict[str, object] = {
        "source": ["a"] * 4 + ["b"] * 4,
        "instrument_id": [1] * 4 + [2] * 4,
        "contract": ["X"] * 4 + ["Y"] * 4,
        "trading_date": ["2026-01-02"] * rows,
        "session_id": ["s"] * rows,
        "reset_generation": [0] * rows,
        "event_action_id": [1] * rows,
        "event_side_id": [1] * rows,
        "direction_10": [0] * rows,
    }
    for name in EVENT_CONT + GLOBAL_STATE + BOOK_COLUMNS:
        payload[name] = np.zeros(rows)
    return pd.DataFrame(payload)


def test_grouped_windows_never_cross_instrument() -> None:
    dataset = GroupedWindowDataset(_window_frame(), sequence_length=3)
    assert len(dataset) == 4
    assert all(dataset.group_for_item(index)[1] in {1, 2} for index in range(len(dataset)))
    assert dataset.group_for_item(1)[1] == 1
    assert dataset.group_for_item(2)[1] == 2
    for index in range(len(dataset)):
        _, _, _, _, _, _, target_index = dataset[index]
        group = tuple(dataset.frame.loc[target_index.item(), GROUP_COLUMNS])
        assert group == dataset.group_for_item(index)


def test_grouped_windows_never_cross_split() -> None:
    frame = pd.concat([_window_frame().iloc[:4], _window_frame().iloc[:4]], ignore_index=True)
    frame["source"] = "a"
    frame["instrument_id"] = 1
    frame["contract"] = "X"
    frame["split"] = ["TRAIN"] * 4 + ["MODEL_VALIDATION"] * 4
    dataset = GroupedWindowDataset(frame, sequence_length=3)
    assert len(dataset) == 4
    assert {dataset.group_for_item(index)[-1] for index in range(len(dataset))} == {
        "TRAIN",
        "MODEL_VALIDATION",
    }


def test_session_position_and_es_dimensions() -> None:
    positions = session_position(
        pd.Series(["2026-01-02T23:00:00Z"]), time(17), time(16), "America/Chicago"
    )
    assert 0 <= positions[0] <= 1
    es = load_registry()["databento:GLBX.MDP3:ES"]
    assert es.ticks_to_price(1) == 0.25
    assert es.ticks_to_usd(1) == 12.50
    assert es.price_move_to_usd(0.25) == 12.50
    result = futures_round_trip(es, 6000.0, 6000.25, 1, 6000.125, 1.25)
    assert result.price_pnl_usd == 12.50
    assert result.fees_usd == 2.50
    assert result.realized_pnl_usd == 10.0
    with_liquidation = futures_round_trip(
        es, 6000.0, 6000.25, 1, 6000.125, 1.25, liquidation_cost_usd=3.0
    )
    assert with_liquidation.liquidation_cost_usd == 3.0
    assert with_liquidation.realized_pnl_usd == 7.0


def test_v2_model_has_side_specific_heads() -> None:
    model = MarketForgeM3TV2(len(GLOBAL_STATE), d_model=16, layers=1, heads=4).eval()
    output = model(
        torch.ones(2, 4, dtype=torch.long),
        torch.ones(2, 4, dtype=torch.long),
        torch.zeros(2, 4, 3),
        torch.zeros(2, 4, len(GLOBAL_STATE)),
        torch.zeros(2, 4, 10, 6),
    )
    assert output["bid_quote_markout"].shape == (2,)
    assert output["ask_quote_markout"].shape == (2,)
    assert "expected_markout" not in output


def test_v2_model_maps_unseen_market_ids_to_seen_embedding_mean() -> None:
    model = MarketForgeM3TV2(
        len(GLOBAL_STATE), d_model=16, layers=1, heads=4, num_sources=2, num_instruments=2
    ).eval()
    source_ids = torch.tensor([2, -1])
    instrument_ids = torch.tensor([99, -1])
    source = model._embedding_with_unseen_mean(model.source_embedding, source_ids)
    instrument = model._embedding_with_unseen_mean(model.instrument_embedding, instrument_ids)
    assert torch.allclose(source[0], model.source_embedding.weight.mean(dim=0))
    assert torch.allclose(source[0], source[1])
    assert torch.allclose(instrument[0], model.instrument_embedding.weight.mean(dim=0))
    assert torch.allclose(instrument[0], instrument[1])


def test_final_ordinal_alignment_preserves_repeated_exchange_timestamp_rows() -> None:
    times = np.array([100, 200, 200, 200, 200, 300], dtype=np.int64)
    result = _validate_ordinal_alignment(times, times.copy(), expected_rows=6)
    assert result["feature_rows"] == 6
    assert result["label_rows"] == 6
    assert result["unique_prediction_times"] == 3
    assert result["duplicate_timestamp_values"] == 1
    assert result["maximum_timestamp_multiplicity"] == 4
    assert result["duplicate_rows_beyond_first"] == 3
    assert result["rowwise_prediction_time_equal"] is True
    assert result["capture_id_unique"] is True
    assert result["capture_id_strictly_monotone"] is True


def test_final_diagnostic_uses_capture_id_when_timestamps_repeat() -> None:
    frame = pd.DataFrame(
        {
            "capture_id": [0, 1],
            "prediction_time": [200, 200],
            "bid_fill_fraction": [0.0, 0.0],
            "ask_fill_fraction": [0.0, 0.0],
            "bid_fill_conditional_markout_100ms": [np.nan, np.nan],
            "ask_fill_conditional_markout_100ms": [np.nan, np.nan],
        }
    )
    predictions = pd.DataFrame(
        {
            "capture_id": [0, 1],
            "prediction_time": [200, 200],
            "bid_fill_probability": [1.0, 1.0],
            "ask_fill_probability": [1.0, 1.0],
            "bid_markout_ticks": [100.0, 100.0],
            "ask_markout_ticks": [100.0, 100.0],
            "bid_adverse_probability": [0.0, 0.0],
            "ask_adverse_probability": [0.0, 0.0],
        }
    )
    result = _active_diagnostic(frame, predictions)
    assert result["quotes"] == 4
    assert result["fills"] == 0
