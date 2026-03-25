from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from irrigation_ai.data.load import load_data1  # noqa: E402
from irrigation_ai.data.sequences import make_next_day_supervised  # noqa: E402
from irrigation_ai.models.lstm_swtd import LstmSwtdConfig, build_lstm_swtd  # noqa: E402
from irrigation_ai.utils.artifacts import make_run_dir, save_json  # noqa: E402
from irrigation_ai.utils.seed import set_global_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM1 (next-day SWTD prediction).")
    p.add_argument("--data-path", type=Path, default=Path("Data/output_data1.csv"))
    p.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))

    p.add_argument("--seq-len", type=int, default=7)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--test-years", type=str, default="2022,2023")

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--hidden-units", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--decay", type=float, default=1e-5)
    return p.parse_args()


def _years_from_csv(s: str) -> list[int]:
    years = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        years.append(int(part))
    return years


def _inverse_swtd(scaler: StandardScaler, scaled: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    idx = feature_cols.index("SWTD")
    return scaled * scaler.scale_[idx] + scaler.mean_[idx]


def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed, deterministic=True)

    import tensorflow as tf

    test_years = set(_years_from_csv(args.test_years))

    df = load_data1(args.data_path)
    df = df.dropna()

    feature_cols = [
        "T2M_MAX",
        "T2M_MIN",
        "WS2M",
        "RH2M",
        "ALLSKY_SFC_SW_DWN",
        "ETo",
        "IRRC",
        "SWTD",  # include SWTD history as an input feature
    ]
    target_col = "SWTD"

    df = df[["date", "year", *feature_cols]].copy()

    train_df = df[~df["year"].isin(test_years)].copy()
    test_df = df[df["year"].isin(test_years)].copy()

    # Fit a scaler on the training set for all feature columns (including SWTD).
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

    def scale_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[feature_cols] = scaler.transform(out[feature_cols].to_numpy(dtype=np.float32))
        return out

    train_scaled = scale_frame(train_df)
    test_scaled = scale_frame(test_df)

    train_ds = make_next_day_supervised(
        train_scaled,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=args.seq_len,
        horizon=args.horizon,
        year_col="year",
    )
    test_ds = make_next_day_supervised(
        test_scaled,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=args.seq_len,
        horizon=args.horizon,
        year_col="year",
    )

    run = make_run_dir(args.artifacts_root, "lstm1_swtd")
    ckpt_path = run.checkpoints_dir / "best.keras"

    config = LstmSwtdConfig(
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        learning_rate=args.lr,
        decay=args.decay,
    )
    model = build_lstm_swtd(n_features=len(feature_cols), config=config)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(run.tensorboard_dir)),
    ]

    history = model.fit(
        train_ds.x,
        train_ds.y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.val_split,
        verbose=2,
        shuffle=True,
        callbacks=callbacks,
    )

    model.save(run.path("model.keras"))
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, run.path("scaler.joblib"))

    # Evaluation on test years (in original units)
    y_pred_scaled = model.predict(test_ds.x, verbose=0).reshape(-1)
    y_true_scaled = test_ds.y.reshape(-1)
    y_pred = _inverse_swtd(scaler, y_pred_scaled, feature_cols)
    y_true = _inverse_swtd(scaler, y_true_scaled, feature_cols)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    save_json(
        run.path("config.json"),
        {
            "data_path": str(args.data_path),
            "feature_cols": feature_cols,
            "target_col": target_col,
            "seq_len": args.seq_len,
            "horizon": args.horizon,
            "test_years": sorted(test_years),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "val_split": args.val_split,
            "patience": args.patience,
            "seed": args.seed,
            "model": json.loads(json.dumps(config.__dict__)),
        },
    )
    save_json(run.path("metrics.json"), metrics)

    pd.DataFrame(history.history).to_csv(run.path("history.csv"), index=False)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(run.path("predictions.csv"), index=False)

    print("\nSaved run to:", run.run_dir)
    print("Best checkpoint:", ckpt_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()

