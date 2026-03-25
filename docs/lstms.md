# LSTM1 & LSTM2 (how they work + how to run)

This repo originally contains Colab notebooks:

- `Colabs/LSTM1.ipynb` (SWTD next-day prediction)
- `Colabs/LSTM2.ipynb` (yield/biomass-related prediction using Data2)

To make training reproducible outside Colab, the training logic is implemented as scripts.

## LSTM1 — next-day soil water (`SWTD`)

**Goal**: predict `SWTD(t+1)` from the previous `seq_len` days of climate + irrigation + soil state.

**Training script**: `scripts/train_lstm1.py`

**Default features** (from `Data/output_data1.csv`):

- `T2M_MAX`, `T2M_MIN`, `WS2M`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `ETo`, `IRRC`, `SWTD`

**Split strategy** (default): test years = **2022, 2023** (as described in the report section for LSTM1).

**Architecture** (matches the report/notebook style):

- BiLSTM(512) → Dropout(0.3) → BiLSTM(512) → Dense(1)

**Train command**:

```bash
python scripts/train_lstm1.py --seq-len 7 --epochs 500 --batch-size 128 --test-years 2022,2023
```

**Where results go**:

- `artifacts/lstm1_swtd/<UTC_TIMESTAMP>/`
  - `model.keras` (final model)
  - `checkpoints/best.keras` (best `val_loss`)
  - `scaler.joblib` (StandardScaler for features)
  - `metrics.json`, `history.csv`, `predictions.csv`
  - `tensorboard/` (TensorBoard logs)

## LSTM2 — end-of-season yield/biomass (proxy via `CWAD`)

**Goal** (report-aligned): predict an end-of-season target from season-long sequences of climate + irrigation + soil water.

In the DSSAT outputs stored in `Data/output_data2.csv`, we use **`CWAD`** as a yield/biomass proxy (daily crop weight), and the target for a season is the **last `CWAD` value of that season**.

**Training script**: `scripts/train_lstm2.py`

**Default features** (from `Data/output_data2.csv`):

- `T2M`, `PREC`, `T2M_MAX`, `T2M_MIN`, `WS2M`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `ETo`, `IRRC`, `SWTD`

**Split strategy** (default): test years = **2022, 2023**.

**Architecture**:

- BiLSTM(128) → Dropout(0.2) → BiLSTM(128) → Dropout(0.2) → Dense(128,tanh) → Dense(1)

**Train command**:

```bash
python scripts/train_lstm2.py --seq-len 110 --window all --epochs 150 --batch-size 64 --test-years 2022,2023
```

**Where results go**:

- `artifacts/lstm2_yield/<UTC_TIMESTAMP>/`
  - `model.keras`
  - `checkpoints/best.keras`
  - `scalers.joblib` (feature + target scalers)
  - `metrics.json`, `history.csv`, `predictions.csv`
  - `tensorboard/`

## Requirements / environment

- `requirements.txt` contains the Python dependencies used by these scripts.
- For GPU training, install a compatible CUDA build of TensorFlow; otherwise CPU works (slower).

