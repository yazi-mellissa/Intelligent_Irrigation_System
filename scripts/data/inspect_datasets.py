from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    date_col: str | None = "date"
    read_csv_kwargs: dict | None = None


def _load_csv(info: DatasetInfo) -> pd.DataFrame:
    read_csv_kwargs = info.read_csv_kwargs or {}
    df = pd.read_csv(info.path, **read_csv_kwargs)
    if info.date_col and info.date_col in df.columns:
        df[info.date_col] = pd.to_datetime(df[info.date_col], errors="coerce")
    return df


def _print_df_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(f"rows={len(df)} cols={len(df.columns)}")
    print("columns:", ", ".join(df.columns))
    na_counts = df.isna().sum()
    if int(na_counts.sum()) > 0:
        print("missing values (top 10):")
        print(na_counts.sort_values(ascending=False).head(10).to_string())
    if "date" in df.columns:
        dt = df["date"].dropna()
        if len(dt) > 0:
            print(f"date range: {dt.min().date()} -> {dt.max().date()}")
            years = sorted(dt.dt.year.unique())
            preview = years[:5] + (["..."] if len(years) > 10 else []) + years[-5:]
            print("years:", preview)


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    datasets = {
        "POWER (raw)": DatasetInfo(
            path=root
            / "Data"
            / "POWER_Point_Daily_20000101_20241212_033d36N_006d88E_LST.csv",
            date_col=None,
            read_csv_kwargs={"skiprows": 25},
        ),
        "Data1 (output_data1.csv)": DatasetInfo(path=root / "Data" / "output_data1.csv"),
        "Data2 (output_data2.csv)": DatasetInfo(path=root / "Data" / "output_data2.csv"),
    }

    for name, info in datasets.items():
        df = _load_csv(info)
        _print_df_summary(name, df)

    # Small integrity check for Data2 duplicate column
    df2 = _load_csv(datasets["Data2 (output_data2.csv)"])
    if "HIAD" in df2.columns and "HIAD.1" in df2.columns:
        equal = (df2["HIAD"] == df2["HIAD.1"]).all()
        print(f"\nData2 check: HIAD == HIAD.1 -> {bool(equal)}")


if __name__ == "__main__":
    main()
