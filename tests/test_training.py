import pandas as pd

from marketforge.training import WindowDataset


def test_window_dataset_uses_target_at_window_end() -> None:
    columns = {
        "event_action_id": [1] * 4,
        "event_side_id": [1] * 4,
        "direction_10": [-1, 0, 1, -1],
        "future_return_10": [0.0] * 4,
        "counterfactual_bid_markout_10": [0.0] * 4,
        "counterfactual_ask_markout_10": [0.0] * 4,
        "future_volatility_10": [0.0] * 4,
    }
    from marketforge.training import BOOK_COLUMNS, EVENT_CONT, GLOBAL_STATE

    for name in EVENT_CONT + GLOBAL_STATE + BOOK_COLUMNS:
        columns[name] = [0.0] * 4
    dataset = WindowDataset(pd.DataFrame(columns), sequence_length=3)
    assert len(dataset) == 2
    assert dataset[0][5].item() == 2
