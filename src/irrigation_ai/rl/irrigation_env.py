from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Lstm1Bundle:
    model: object
    scaler: object
    feature_cols: list[str]


@dataclass(frozen=True)
class Lstm2Bundle:
    model: object
    feature_scaler: object
    target_scaler: object
    feature_cols: list[str]


def load_lstm1_bundle(run_dir: Path) -> Lstm1Bundle:
    import tensorflow as tf

    model = tf.keras.models.load_model(run_dir / "model.keras")
    payload = joblib.load(run_dir / "scaler.joblib")
    return Lstm1Bundle(model=model, scaler=payload["scaler"], feature_cols=list(payload["feature_cols"]))


def load_lstm2_bundle(run_dir: Path) -> Lstm2Bundle:
    import tensorflow as tf

    model = tf.keras.models.load_model(run_dir / "model.keras")
    payload = joblib.load(run_dir / "scalers.joblib")
    return Lstm2Bundle(
        model=model,
        feature_scaler=payload["feature_scaler"],
        target_scaler=payload["target_scaler"],
        feature_cols=list(payload["feature_cols"]),
    )


def _inverse_scaled_column(scaler, col_index: int, scaled: np.ndarray) -> np.ndarray:
    return scaled * scaler.scale_[col_index] + scaler.mean_[col_index]


class IrrigationEnv:
    """
    Season-based irrigation environment (report-inspired).

    Dynamics:
    - Next-day SWTD is predicted by LSTM1, conditioned on climate + irrigation action + current SWTD.
    - End-of-season yield/biomass is predicted by LSTM2 from the final `seq_len_yield` days.
    - Reward is only given at the end of a season (as described in the report pseudo-algorithm):
        r = p_y * yield - p_w * water
    """

    def __init__(
        self,
        df_data2: pd.DataFrame,
        *,
        lstm1: Lstm1Bundle,
        lstm2: Lstm2Bundle,
        actions_mm: list[float],
        seq_len_swtd: int = 7,
        seq_len_yield: int = 110,
        p_y: float = 1.0,
        p_w: float = 0.1,
        seed: int = 1,
    ):
        self.df = df_data2.copy()
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df = self.df.dropna(subset=["date"]).sort_values("date")
        if "year" not in self.df.columns:
            self.df["year"] = self.df["date"].dt.year

        self.lstm1 = lstm1
        self.lstm2 = lstm2
        self.actions_mm = [float(a) for a in actions_mm]
        self.seq_len_swtd = int(seq_len_swtd)
        self.seq_len_yield = int(seq_len_yield)
        self.p_y = float(p_y)
        self.p_w = float(p_w)
        self.rng = np.random.default_rng(int(seed))

        self._years = sorted(self.df["year"].unique().tolist())
        self._episode_df: pd.DataFrame | None = None
        self._t = 0
        self._swtd_hist: list[float] = []
        self._irrig_hist: list[float] = []

        # State uses a compact set of variables (exogenous + internal).
        self._state_weather_cols = [
            "T2M_MAX",
            "T2M_MIN",
            "WS2M",
            "RH2M",
            "ALLSKY_SFC_SW_DWN",
            "ETo",
            "PREC",
            "T2M",
        ]

    @property
    def n_actions(self) -> int:
        return len(self.actions_mm)

    @property
    def state_dim(self) -> int:
        # weather (8) + swtd + cum_irrig + day_frac
        return len(self._state_weather_cols) + 3

    def reset(self, year: int | None = None) -> np.ndarray:
        if year is None:
            year = int(self.rng.choice(self._years))
        season = self.df[self.df["year"] == year].copy()
        if len(season) < max(self.seq_len_swtd + 1, self.seq_len_yield):
            raise ValueError(f"Year {year} has too few rows for the configured sequence lengths.")

        self._episode_df = season.reset_index(drop=True)
        self._t = 0
        self._swtd_hist = [float(self._episode_df.loc[0, "SWTD"])]
        self._irrig_hist = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        assert self._episode_df is not None
        row = self._episode_df.loc[self._t]

        weather = np.array([float(row[c]) for c in self._state_weather_cols], dtype=np.float32)
        swtd = np.array([float(self._swtd_hist[self._t])], dtype=np.float32)
        cum_irrig = np.array([float(sum(self._irrig_hist))], dtype=np.float32)
        day_frac = np.array([float(self._t) / float(len(self._episode_df) - 1)], dtype=np.float32)
        return np.concatenate([weather, swtd, cum_irrig, day_frac], axis=0)

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, dict]:
        assert self._episode_df is not None
        action_index = int(action_index)
        if action_index < 0 or action_index >= self.n_actions:
            raise ValueError("Invalid action index.")

        season_len = len(self._episode_df)
        done = self._t >= season_len - 1
        if done:
            raise RuntimeError("Call reset() before stepping a finished episode.")

        irrigation_mm = float(self.actions_mm[action_index])
        self._irrig_hist.append(irrigation_mm)

        # Predict next-day SWTD using LSTM1
        next_swtd = self._predict_next_swtd()
        self._swtd_hist.append(float(next_swtd))

        self._t += 1

        terminal = self._t >= season_len - 1
        reward = 0.0
        info: dict = {"year": int(self._episode_df.loc[0, "year"]), "t": int(self._t)}

        if terminal:
            y = float(self._predict_end_of_season_yield())
            w = float(sum(self._irrig_hist))
            reward = self.p_y * y - self.p_w * w
            info.update({"yield": y, "water": w})

        return self._get_state(), float(reward), bool(terminal), info

    def _predict_next_swtd(self) -> float:
        assert self._episode_df is not None

        feat_cols = self.lstm1.feature_cols
        if "SWTD" not in feat_cols:
            raise ValueError("LSTM1 bundle must include SWTD in feature_cols.")

        # Build last `seq_len_swtd` timesteps ending at current t
        t = self._t
        start = max(0, t - self.seq_len_swtd + 1)
        idxs = list(range(start, t + 1))
        if len(idxs) < self.seq_len_swtd:
            idxs = [0] * (self.seq_len_swtd - len(idxs)) + idxs

        # For each timestep i, use climate from df, irrigation from history (or 0), SWTD from history.
        x_rows: list[list[float]] = []
        for i in idxs:
            row = self._episode_df.loc[i]
            # irrigation used on day i: if i == t it's the latest action, else from history
            irrig = self._irrig_hist[i] if i < len(self._irrig_hist) else 0.0
            swtd = self._swtd_hist[i]

            row_values: dict[str, float] = {
                "T2M_MAX": float(row.get("T2M_MAX", np.nan)),
                "T2M_MIN": float(row.get("T2M_MIN", np.nan)),
                "WS2M": float(row.get("WS2M", np.nan)),
                "RH2M": float(row.get("RH2M", np.nan)),
                "ALLSKY_SFC_SW_DWN": float(row.get("ALLSKY_SFC_SW_DWN", np.nan)),
                "ETo": float(row.get("ETo", np.nan)),
                "IRRC": float(irrig),
                "SWTD": float(swtd),
            }

            x_rows.append([row_values[c] for c in feat_cols])

        x = np.array(x_rows, dtype=np.float32)
        x_scaled = self.lstm1.scaler.transform(x)
        x_scaled = x_scaled.reshape(1, self.seq_len_swtd, -1)

        pred_scaled = np.asarray(self.lstm1.model.predict(x_scaled, verbose=0)).reshape(-1)[0]
        swtd_idx = feat_cols.index("SWTD")
        pred = _inverse_scaled_column(self.lstm1.scaler, swtd_idx, np.array([pred_scaled], dtype=np.float32))[0]
        return float(pred)

    def _predict_end_of_season_yield(self) -> float:
        assert self._episode_df is not None

        feat_cols = self.lstm2.feature_cols
        season_len = len(self._episode_df)

        # Use the last `seq_len_yield` days of the season.
        start = season_len - self.seq_len_yield
        if start < 0:
            raise ValueError("Season shorter than seq_len_yield.")

        x_rows: list[list[float]] = []
        for i in range(start, season_len):
            row = self._episode_df.loc[i]
            irrig = self._irrig_hist[i] if i < len(self._irrig_hist) else 0.0
            swtd = self._swtd_hist[i]

            row_values: dict[str, float] = {
                "T2M": float(row.get("T2M", np.nan)),
                "PREC": float(row.get("PREC", np.nan)),
                "T2M_MAX": float(row.get("T2M_MAX", np.nan)),
                "T2M_MIN": float(row.get("T2M_MIN", np.nan)),
                "WS2M": float(row.get("WS2M", np.nan)),
                "RH2M": float(row.get("RH2M", np.nan)),
                "ALLSKY_SFC_SW_DWN": float(row.get("ALLSKY_SFC_SW_DWN", np.nan)),
                "ETo": float(row.get("ETo", np.nan)),
                "IRRC": float(irrig),
                "SWTD": float(swtd),
            }

            x_rows.append([row_values[c] for c in feat_cols])

        x = np.array(x_rows, dtype=np.float32)
        x_scaled = self.lstm2.feature_scaler.transform(x).reshape(1, self.seq_len_yield, -1)
        pred_scaled = np.asarray(self.lstm2.model.predict(x_scaled, verbose=0)).reshape(-1)[0]
        pred = self.lstm2.target_scaler.inverse_transform(np.array([[pred_scaled]], dtype=np.float32))[0, 0]
        return float(pred)

