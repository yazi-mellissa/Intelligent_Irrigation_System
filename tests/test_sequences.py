from __future__ import annotations

import pandas as pd

from irrigation_ai.data.sequences import make_next_day_supervised


def test_make_next_day_supervised_respects_year_boundaries() -> None:
    # Two seasons/years with 5 days each.
    rows = []
    for year in (2000, 2001):
        for d in range(5):
            rows.append(
                {
                    "date": pd.Timestamp(year=year, month=4, day=15 + d),
                    "year": year,
                    "T2M_MAX": float(d),
                    "T2M_MIN": float(d),
                    "WS2M": float(d),
                    "RH2M": float(d),
                    "ALLSKY_SFC_SW_DWN": float(d),
                    "ETo": float(d),
                    "IRRC": float(d),
                    "SWTD": float(d),
                }
            )
    df = pd.DataFrame(rows)

    ds = make_next_day_supervised(
        df,
        feature_cols=[
            "T2M_MAX",
            "T2M_MIN",
            "WS2M",
            "RH2M",
            "ALLSKY_SFC_SW_DWN",
            "ETo",
            "IRRC",
            "SWTD",
        ],
        target_col="SWTD",
        seq_len=3,
        horizon=1,
        year_col="year",
    )

    # Per year: length=5 -> windows=5 - seq_len - horizon + 1 = 2
    assert ds.x.shape[0] == 4
    assert ds.x.shape[1] == 3
    assert ds.y.shape == (4, 1)

