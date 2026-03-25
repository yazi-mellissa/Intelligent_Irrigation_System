# DQN agent (irrigation optimization)

The report describes a Deep Q-Network (DQN) agent that chooses daily irrigation volumes to maximize an economic reward:

**Reward (end of season)**:

- `r = p_y * y - p_w * w`
  - `y`: predicted end-of-season yield/biomass (proxy via `CWAD`)
  - `w`: cumulative irrigation water used during the season (sum of daily irrigation volumes)
  - `p_y`: tomato price coefficient
  - `p_w`: water cost coefficient

In this implementation the reward is **0 at intermediate timesteps** and is given **only at the terminal step**, matching the report pseudo-algorithm.

## Prerequisite: train LSTM1 & LSTM2

The DQN environment uses:

- LSTM1 to predict next-day soil water (`SWTD`)
- LSTM2 to predict end-of-season yield/biomass (from the last `seq_len_yield` days)

Train them first and note the artifact run directories:

```bash
python scripts/train_lstm1.py
python scripts/train_lstm2.py
```

Each script writes to:

- `artifacts/lstm1_swtd/<UTC_TIMESTAMP>/`
- `artifacts/lstm2_yield/<UTC_TIMESTAMP>/`

## Train DQN

```bash
python scripts/train_dqn.py ^
  --lstm1-run artifacts/lstm1_swtd/<RUN_TIMESTAMP> ^
  --lstm2-run artifacts/lstm2_yield/<RUN_TIMESTAMP> ^
  --episodes 500 ^
  --actions-mm 0,5,10,15,20,25,30,35,40,45,50,60 ^
  --p-y 1.0 --p-w 0.1
```

## Outputs

The DQN training run is stored under:

- `artifacts/dqn/<UTC_TIMESTAMP>/`
  - `q_network.keras`, `q_target.keras`
  - `training_log.csv` (episode returns, epsilon, eval metrics, …)
  - `config.json`

