from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)


def make_run_dir(artifacts_root: Path, experiment: str) -> RunArtifacts:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_root / experiment / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    return RunArtifacts(run_dir=run_dir)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

