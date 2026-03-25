from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceDataset:
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def _iter_year_groups(df: pd.DataFrame, year_col: str = "year") -> Iterable[pd.DataFrame]:
    for _, g in df.groupby(year_col, sort=True):
        g = g.sort_values("date")
        yield g


def make_next_day_supervised(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    horizon: int = 1,
    year_col: str = "year",
) -> SequenceDataset:
    """
    Creates (X, y) for next-step prediction with season boundaries (grouped by year).

    - X: shape (N, seq_len, n_features)
    - y: shape (N, 1)
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for g in _iter_year_groups(df, year_col=year_col):
        values_x = g[feature_cols].to_numpy(dtype=np.float32)
        values_y = g[target_col].to_numpy(dtype=np.float32)

        max_start = len(g) - seq_len - horizon + 1
        for start in range(max_start):
            end = start + seq_len
            xs.append(values_x[start:end])
            ys.append(np.array([values_y[end + horizon - 1]], dtype=np.float32))

    if len(xs) == 0:
        raise ValueError("No sequences created (check seq_len/horizon vs season lengths).")

    return SequenceDataset(x=np.stack(xs, axis=0), y=np.stack(ys, axis=0), feature_names=feature_cols)


def make_season_to_one_supervised(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    year_col: str = "year",
    window: str = "last",
) -> SequenceDataset:
    """
    Creates (X, y) for end-of-season target prediction (sequence-to-one).

    For each year/season:
      - target is the *last* value of `target_col` in that year (e.g., end-of-season CWAD)
      - multiple samples are created using sliding windows of length `seq_len`

    `window` controls which windows are kept:
      - "all": all sliding windows
      - "last": only the last window per season (one sample per year)
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for g in _iter_year_groups(df, year_col=year_col):
        g = g.sort_values("date")
        values_x = g[feature_cols].to_numpy(dtype=np.float32)
        y_final = float(g[target_col].iloc[-1])

        if len(g) < seq_len:
            continue

        if window == "last":
            xs.append(values_x[-seq_len:])
            ys.append(np.array([y_final], dtype=np.float32))
            continue

        if window != "all":
            raise ValueError("window must be 'all' or 'last'")

        for end in range(seq_len, len(g) + 1):
            xs.append(values_x[end - seq_len : end])
            ys.append(np.array([y_final], dtype=np.float32))

    if len(xs) == 0:
        raise ValueError("No sequences created (check seq_len vs season lengths).")

    return SequenceDataset(x=np.stack(xs, axis=0), y=np.stack(ys, axis=0), feature_names=feature_cols)

