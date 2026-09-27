from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .training import BOOK_COLUMNS, EVENT_CONT, GLOBAL_STATE

GROUP_COLUMNS = [
    "source",
    "instrument_id",
    "contract",
    "trading_date",
    "session_id",
    "reset_generation",
]
SPLIT_COLUMN = "split"


def session_position(
    timestamp: pd.Series, session_start: time, session_end: time, timezone: str
) -> np.ndarray:
    localized = pd.to_datetime(timestamp, utc=True).dt.tz_convert(timezone)
    minutes = localized.dt.hour * 60 + localized.dt.minute + localized.dt.second / 60
    start = session_start.hour * 60 + session_start.minute
    end = session_end.hour * 60 + session_end.minute
    duration = (end - start) % (24 * 60)
    elapsed = (minutes - start) % (24 * 60)
    result = elapsed / duration
    return np.where(elapsed <= duration, result, np.nan)


class GroupedWindowDataset(Dataset[tuple[torch.Tensor, ...]]):
    """V2 windows that cannot cross source/instrument/contract/day/session/reset boundaries."""

    def __init__(self, frame: pd.DataFrame, sequence_length: int = 32) -> None:
        missing = set(GROUP_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"missing grouping columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.boundary_columns = GROUP_COLUMNS + (
            [SPLIT_COLUMN] if SPLIT_COLUMN in self.frame.columns else []
        )
        self.sequence_length = sequence_length
        self.action = self.frame["event_action_id"].to_numpy(np.int64)
        self.side = self.frame["event_side_id"].to_numpy(np.int64)
        self.event = np.nan_to_num(self.frame[EVENT_CONT].to_numpy(np.float32))
        self.state = np.nan_to_num(self.frame[GLOBAL_STATE].to_numpy(np.float32))
        self.book = np.nan_to_num(self.frame[BOOK_COLUMNS].to_numpy(np.float32)).reshape(-1, 10, 6)
        self.direction = self.frame["direction_10"].to_numpy(np.int64) + 1
        self.valid_ends: list[int] = []
        for positions in self.frame.groupby(
            self.boundary_columns, sort=False, dropna=False
        ).indices.values():
            ordered = np.sort(positions)
            if len(ordered) < sequence_length:
                continue
            breaks = np.flatnonzero(np.diff(ordered) != 1)
            for segment in np.split(ordered, breaks + 1):
                self.valid_ends.extend(int(index) for index in segment[sequence_length - 1 :])

    def __len__(self) -> int:
        return len(self.valid_ends)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        end = self.valid_ends[index] + 1
        start = end - self.sequence_length
        return (
            torch.from_numpy(self.action[start:end]),
            torch.from_numpy(self.side[start:end]),
            torch.from_numpy(self.event[start:end]),
            torch.from_numpy(self.state[start:end]),
            torch.from_numpy(self.book[start:end]),
            torch.tensor(self.direction[end - 1]),
            torch.tensor(end - 1),
        )

    def group_for_item(self, index: int) -> tuple[object, ...]:
        return tuple(self.frame.loc[self.valid_ends[index], self.boundary_columns])
