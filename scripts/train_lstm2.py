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

from irrigation_ai.data.load import load_data2  # noqa: E402
from irrigation_ai.data.sequences import make_season_to_one_supervised  # noqa: E402
from irrigation_ai.models.lstm_yield import LstmYieldConfig, build_lstm_yield  # noqa: E402
from irrigation_ai.utils.artifacts import make_run_dir, save_json  # noqa: E402
from irrigation_ai.utils.seed import set_global_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM2 (end-of-season yield/biomass prediction via CWAD).")
    p.add_argument("--data-path", type=Path, default=Path("Data/output_data2.csv"))
    p.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))

    p.add_argument("--seq-len", type=int, default=110)
    p.add_argument("--window", type=str, choices=["all", "last"], default="all")
    p.add_argument("--test-years", type=str, default="2022,2023")

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--hidden-units", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--dense-units", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
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


def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed, deterministic=True)

    import tensorflow as tf

    test_years = set(_years_from_csv(args.test_years))

    df = load_data2(args.data_path, drop_duplicate_hiad=True)
    df = df.dropna()

    # Features aligned with the report (climate + SWTD + irrigation history).
    feature_cols = [
        "T2M",
        "PREC",
        "T2M_MAX",
        "T2M_MIN",
        "WS2M",
        "RH2M",
        "ALLSKY_SFC_SW_DWN",
        "ETo",
        "IRRC",
        "SWTD",
    ]
    target_col = "CWAD"  # proxy for end-of-season yield/biomass in DSSAT outputs

    df = df[["date", "year", *feature_cols, target_col]].copy()

    train_df = df[~df["year"].isin(test_years)].copy()
    test_df = df[df["year"].isin(test_years)].copy()

    feature_scaler = StandardScaler()
    feature_scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

    def scale_features(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[feature_cols] = feature_scaler.transform(out[feature_cols].to_numpy(dtype=np.float32))
        return out

    train_scaled = scale_features(train_df)
    test_scaled = scale_features(test_df)

    train_ds = make_season_to_one_supervised(
        train_scaled,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=args.seq_len,
        year_col="year",
        window=args.window,
    )
    test_ds = make_season_to_one_supervised(
        test_scaled,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=args.seq_len,
        year_col="year",
        window=args.window,
    )

    target_scaler = StandardScaler()
    target_scaler.fit(train_ds.y)
    y_train = target_scaler.transform(train_ds.y)
    y_test = target_scaler.transform(test_ds.y)

    run = make_run_dir(args.artifacts_root, "lstm2_yield")
    ckpt_path = run.checkpoints_dir / "best.keras"

    config = LstmYieldConfig(
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        dense_units=args.dense_units,
        learning_rate=args.lr,
        decay=args.decay,
    )
    model = build_lstm_yield(seq_len=args.seq_len, n_features=len(feature_cols), config=config)

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
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.val_split,
        verbose=2,
        shuffle=True,
        callbacks=callbacks,
    )

    model.save(run.path("model.keras"))
    joblib.dump(
        {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_cols": feature_cols,
        },
        run.path("scalers.joblib"),
    )

    # Evaluation (in original units)
    y_pred_scaled = model.predict(test_ds.x, verbose=0)
    y_pred = target_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = test_ds.y.reshape(-1)

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
            "window": args.window,
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

