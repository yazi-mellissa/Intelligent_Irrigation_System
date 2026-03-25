# UPSaclay AI Master — application form notes (fill & copy/paste)

This file is a **template** to answer the Google Form in the prompt. Replace placeholders like `<...>` after you run training locally and (optionally) publish the repo.

## A. Code & Repositories

**Public repo URL**: `https://github.com/yazi-mellissa/Intelligent_Irrigation_System.git`

**Core model code paths** (this repo):

- LSTM1 training: `scripts/train_lstm1.py`
- LSTM2 training: `scripts/train_lstm2.py`
- DQN training: `scripts/train_dqn.py`
- DQN environment: `src/irrigation_ai/rl/irrigation_env.py`

**Deployment-oriented detail (M2)**:

- FastAPI route: `api/main.py` (`POST /recommend`)
- Dockerfile: `Dockerfile`

**Username**: `<YOUR_GITHUB_USERNAME>`

**Three commit SHAs (authored by me in this repo)**:

- Data provenance/docs: `741226f` (docs(data): document datasets and provenance)
- LSTM1/LSTM2 scripts: `820f515` (feat(models): add reproducible LSTM1/LSTM2 training scripts)
- DQN agent + env: `aa18e53` (feat(rl): add DQN agent and irrigation environment)


**My role (≤50 words)**:

<I implemented the full ML pipeline described in the report: dataset provenance documentation, reproducible LSTM training scripts, and the DQN irrigation agent/environment. I also organized artifacts/checkpoints/logs for reproducibility and created CLI commands + documentation to train/evaluate the models.>

### 10–20 line code snippet (training or preprocessing)

Paste e.g. from `scripts/train_lstm1.py` (callbacks + `model.fit`) or from `src/irrigation_ai/data/sequences.py`.

Example (from `scripts/train_lstm1.py`):

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path), monitor="val_loss", save_best_only=True
    ),
    tf.keras.callbacks.TensorBoard(log_dir=str(run.tensorboard_dir)),
]
history = model.fit(
    train_ds.x, train_ds.y,
    batch_size=args.batch_size, epochs=args.epochs,
    validation_split=args.val_split, shuffle=True,
    callbacks=callbacks,
)
```

**Which lines are mine? Why written that way? (≤60 words)**:

<All lines. EarlyStopping prevents overfitting and restores the best validation weights. ModelCheckpoint saves the best model for reproducibility. TensorBoard writes logs for monitoring. The training call uses `validation_split` to match the report’s early stopping description while keeping a simple CLI-driven workflow.>

### Exact command(s) used to run training + environment

LSTM1:

```bash
python scripts/train_lstm1.py --seq-len 7 --epochs 500 --batch-size 128 --test-years 2022,2023
```

LSTM2:

```bash
python scripts/train_lstm2.py --seq-len 110 --window all --epochs 150 --batch-size 64 --test-years 2022,2023
```

DQN (after training LSTMs, replace `<RUN_TIMESTAMP>`):

```bash
python scripts/train_dqn.py ^
  --lstm1-run artifacts/lstm1_swtd/<RUN_TIMESTAMP> ^
  --lstm2-run artifacts/lstm2_yield/<RUN_TIMESTAMP> ^
  --episodes 500
```

**Environment**:

- Python: `<e.g. 3.11>`
- Install: `pip install -r requirements.txt`
- Requirements file path: `requirements.txt`
- CUDA (if used): `<e.g. CUDA 12.x + cuDNN ...>` (otherwise “CPU only”)

## B. Data & Reproducibility

**Most recent dataset** (≤80 words):

NASA POWER daily point climate data (file: `Data/POWER_Point_Daily_20000101_20241212_033d36N_006d88E_LST.csv`, 8949 days × 19 vars) for ~33.3614°N, 6.8753°E. We generate two derived datasets via DSSAT CROPGRO-Tomato simulations: Data1 (`Data/output_data1.csv`, 3049 rows × 12 cols) and Data2 (`Data/output_data2.csv`, 3001 rows × 22 cols). Split by year: train years excluding 2022–2023, test = 2022–2023. Cleaning: drop NA + remove duplicate `HIAD.1` (identical to `HIAD`).

Note: after selecting hyperparameters with a held-out test (e.g. 2022–2023), you can retrain a final model on **all years** using `--test-years ""`.

**Reproducibility / seeds** (≤60 words):

Seeds are set in Python `random`, NumPy, and TensorFlow (`tf.keras.utils.set_random_seed`). Scripts also set `PYTHONHASHSEED` and enable best-effort TF deterministic ops. Remaining nondeterminism can come from GPU kernels/cuDNN and parallelism during training.

## C. Modeling Decisions

**Task, model family, alternatives, key hyperparameter** (≤100 words):

Task is time-series regression + sequential decision-making: (1) predict next-day soil water (`SWTD`) with a BiLSTM (LSTM1), (2) predict end-of-season yield/biomass proxy (`CWAD`) with a BiLSTM (LSTM2), and (3) choose irrigation actions with DQN. Chosen because LSTMs model temporal dependencies better than linear regression and RandomForest, and DQN directly optimizes a long-horizon objective unlike threshold rules. Most impactful hyperparameter: sequence length (`seq_len`), searched around 4–110 (chosen 7 for LSTM1, 110 for LSTM2).

**Supervision signal** (≤60 words):

Supervised learning for LSTMs using DSSAT-simulated targets (`SWTD`, `CWAD`). Reinforcement learning for DQN with a terminal economic reward computed from LSTM2-predicted yield and cumulative irrigation water cost.

## D. Evaluation & Error Analysis

**Primary metric + trade-off** (≤60 words):

Optimized validation MSE/RMSE for regression. Trade-off: minimizing RMSE can favor “average” predictions and underreact to rare extremes (very dry/wet days), which matters for irrigation safety margins.

**Concrete failure mode** (≤100 words):

<After you run training: describe one year/day where SWTD is systematically under/over-predicted (e.g., sharp changes after irrigation). Mention why (distribution shift, insufficient features, sequence length) and one attempted fix (different seq_len, adding SWTD history, regularization, re-scaling).>

**Final validation log + checkpoint filename** (≤60 words):

Run `python scripts/print_artifact_summary.py <PATH_TO_RUN_DIR>` and paste the printed “metrics” + checkpoint file (e.g. `checkpoints/best.keras`). Add a brief note about overfitting signs (train loss decreasing while val loss increases).

## E. Compute & Systems

Hardware (≤50 words):

<CPU/GPU model, RAM/VRAM>. Longest run: <time>. Monitoring: TensorBoard logs under `artifacts/**/tensorboard/`.

Profiling (≤80 words):

<Optional: mention TF profiler / bottleneck and one improvement.>

## F. MLOps & Engineering Hygiene

Experiment tracking (≤60 words):

Experiments are tracked by artifact directories under `artifacts/<experiment>/<UTC_TIMESTAMP>/` storing `config.json`, `metrics.json`, `history.csv`, `predictions.csv` and TensorBoard logs.

Testing (≤60 words):

See `tests/test_sequences.py` (checks that sequence generation respects season/year boundaries and shapes).

## G. Teamwork & Contribution

PR / MR (≤60 words):

<Add after publishing: link to PR, title, reviewer comment, what you addressed.>

## H. Responsible & Legal AI

Bias / limitation (≤80 words):

This dataset is partly **simulated** (DSSAT), and climate inputs come from a single geographic point, so the learned policies may not generalize to other soils/regions/cultivars. Mitigation: evaluate per-year performance, hold out recent years (2022–2023), and document assumptions; when moving to real deployments, re-train with local sensor/field data.

Licensing (≤50 words):

<Add after publishing: choose a repo license (MIT/Apache-2.0) and verify external dataset/tool licenses (NASA POWER data usage terms, DSSATTools).>

## I. Math & Understanding

Loss function (≤60 words):

For LSTM regressors: mean squared error, `L(θ) = (1/N) Σ_i (ŷ_i - y_i)^2`. Regularization: dropout in the BiLSTM layers.

Early stopping (≤40 words):

Patience = 20 epochs, selection criterion = minimum validation loss (`val_loss`), with best weights restored.
