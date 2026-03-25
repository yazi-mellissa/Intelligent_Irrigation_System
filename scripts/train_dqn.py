from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from irrigation_ai.data.load import load_data2  # noqa: E402
from irrigation_ai.rl.dqn_agent import DqnAgent, DqnConfig  # noqa: E402
from irrigation_ai.rl.irrigation_env import IrrigationEnv, load_lstm1_bundle, load_lstm2_bundle  # noqa: E402
from irrigation_ai.rl.replay_buffer import ReplayBuffer  # noqa: E402
from irrigation_ai.utils.artifacts import make_run_dir, save_json  # noqa: E402
from irrigation_ai.utils.seed import set_global_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN irrigation agent (report-inspired).")

    p.add_argument("--data2-path", type=Path, default=Path("Data/output_data2.csv"))
    p.add_argument("--lstm1-run", type=Path, required=True, help="Path to artifacts/lstm1_swtd/<run>/")
    p.add_argument("--lstm2-run", type=Path, required=True, help="Path to artifacts/lstm2_yield/<run>/")
    p.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))

    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--warmup", type=int, default=2_000, help="Transitions before training starts.")
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-update-steps", type=int, default=500)

    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-steps", type=int, default=10_000)

    p.add_argument("--seq-len-swtd", type=int, default=7)
    p.add_argument("--seq-len-yield", type=int, default=110)

    p.add_argument("--actions-mm", type=str, default="0,5,10,15,20,25,30,35,40,45,50,60")
    p.add_argument("--p-y", type=float, default=1.0, help="Tomato price coefficient in reward.")
    p.add_argument("--p-w", type=float, default=0.1, help="Water cost coefficient in reward.")

    p.add_argument("--test-years", type=str, default="2022,2023")
    p.add_argument("--eval-every", type=int, default=25)

    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def _parse_int_list(csv: str) -> list[int]:
    out: list[int] = []
    for part in csv.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_float_list(csv: str) -> list[float]:
    out: list[float] = []
    for part in csv.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _evaluate(agent: DqnAgent, env: IrrigationEnv, years: list[int]) -> dict:
    returns: list[float] = []
    waters: list[float] = []
    yields: list[float] = []

    for y in years:
        s = env.reset(year=y)
        done = False
        ep_return = 0.0
        info_last: dict = {}
        while not done:
            a = agent.act(s, greedy=True)
            s, r, done, info = env.step(a)
            ep_return += r
            info_last = info
        returns.append(ep_return)
        if "water" in info_last:
            waters.append(float(info_last["water"]))
        if "yield" in info_last:
            yields.append(float(info_last["yield"]))

    return {
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "avg_water": float(np.mean(waters)) if waters else 0.0,
        "avg_yield": float(np.mean(yields)) if yields else 0.0,
        "n_eval": int(len(returns)),
    }


def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed, deterministic=True)

    # TensorFlow is required for DQN + loading the LSTM models.
    import tensorflow as tf  # noqa: F401

    df2 = load_data2(args.data2_path, drop_duplicate_hiad=True).dropna()

    lstm1 = load_lstm1_bundle(args.lstm1_run)
    lstm2 = load_lstm2_bundle(args.lstm2_run)
    actions_mm = _parse_float_list(args.actions_mm)

    env = IrrigationEnv(
        df2,
        lstm1=lstm1,
        lstm2=lstm2,
        actions_mm=actions_mm,
        seq_len_swtd=args.seq_len_swtd,
        seq_len_yield=args.seq_len_yield,
        p_y=args.p_y,
        p_w=args.p_w,
        seed=args.seed,
    )

    test_years = set(_parse_int_list(args.test_years))
    all_years = sorted(df2["year"].unique().tolist())
    train_years = [y for y in all_years if y not in test_years]
    eval_years = [y for y in all_years if y in test_years]

    agent = DqnAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        config=DqnConfig(
            gamma=args.gamma,
            lr=args.lr,
            batch_size=args.batch_size,
            target_update_steps=args.target_update_steps,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
        ),
        seed=args.seed,
    )

    buffer = ReplayBuffer(capacity=args.buffer_size, state_dim=env.state_dim)
    rng = np.random.default_rng(args.seed)

    run = make_run_dir(args.artifacts_root, "dqn")
    save_json(
        run.path("config.json"),
        {
            "data2_path": str(args.data2_path),
            "lstm1_run": str(args.lstm1_run),
            "lstm2_run": str(args.lstm2_run),
            "actions_mm": actions_mm,
            "seq_len_swtd": args.seq_len_swtd,
            "seq_len_yield": args.seq_len_yield,
            "reward": {"p_y": args.p_y, "p_w": args.p_w},
            "train_years": train_years,
            "eval_years": eval_years,
            "dqn": {
                "gamma": args.gamma,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "target_update_steps": args.target_update_steps,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "epsilon_decay_steps": args.epsilon_decay_steps,
            },
            "seed": args.seed,
        },
    )

    logs: list[dict] = []

    for ep in range(1, args.episodes + 1):
        year = int(rng.choice(train_years))
        state = env.reset(year=year)
        done = False
        ep_return = 0.0
        ep_losses: list[float] = []
        info_last: dict = {}

        while not done:
            action = agent.act(state, greedy=False)
            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_return += reward
            info_last = info

            if len(buffer) >= args.warmup:
                batch = buffer.sample(args.batch_size, rng=rng)
                loss = agent.train_on_batch(batch)
                ep_losses.append(loss)

        row = {
            "episode": ep,
            "year": year,
            "return": float(ep_return),
            "avg_loss": float(np.mean(ep_losses)) if ep_losses else float("nan"),
            "epsilon": float(agent.epsilon()),
            "yield": float(info_last.get("yield", float("nan"))),
            "water": float(info_last.get("water", float("nan"))),
            "buffer_size": int(len(buffer)),
        }

        if args.eval_every > 0 and (ep % args.eval_every == 0) and eval_years:
            eval_metrics = _evaluate(agent, env, eval_years)
            row.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            print(f"ep={ep} train_return={row['return']:.3f} eval_avg_return={eval_metrics['avg_return']:.3f}")
        else:
            print(f"ep={ep} train_return={row['return']:.3f}")

        logs.append(row)

    pd.DataFrame(logs).to_csv(run.path("training_log.csv"), index=False)
    agent.q.save(run.path("q_network.keras"))
    agent.q_target.save(run.path("q_target.keras"))

    print("\nSaved run to:", run.run_dir)


if __name__ == "__main__":
    main()

