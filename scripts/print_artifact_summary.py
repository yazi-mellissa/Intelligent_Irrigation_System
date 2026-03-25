from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print a short summary of an artifacts run directory.")
    p.add_argument("run_dir", type=Path, help="e.g. artifacts/lstm1_swtd/<UTC_TIMESTAMP>")
    return p.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = _parse_args()
    run = args.run_dir

    config = _read_json(run / "config.json")
    metrics = _read_json(run / "metrics.json")
    ckpt = run / "checkpoints" / "best.keras"

    print("run_dir:", run)
    if config:
        print("experiment:", run.parent.name)
        print("data_path:", config.get("data_path"))
        print("seq_len:", config.get("seq_len"))
        print("test_years:", config.get("test_years"))
    if metrics:
        print("metrics:", metrics)
    print("best_checkpoint_exists:", ckpt.exists())
    print("best_checkpoint_path:", ckpt)


if __name__ == "__main__":
    main()

