from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    data_dir: Path

    @property
    def data1(self) -> Path:
        return self.data_dir / "output_data1.csv"

    @property
    def data2(self) -> Path:
        return self.data_dir / "output_data2.csv"


def load_data1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["year"] = df["date"].dt.year
    return df


def load_data2(path: Path, drop_duplicate_hiad: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["year"] = df["date"].dt.year

    if drop_duplicate_hiad and "HIAD.1" in df.columns:
        df = df.drop(columns=["HIAD.1"])

    return df

