from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Best-effort reproducibility across Python/NumPy/TensorFlow.

    Notes:
    - Exact determinism can still be affected by GPU kernels, cuDNN, parallelism, etc.
    - TensorFlow import is optional; we only touch it if installed.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

    try:
        import tensorflow as tf  # noqa: F401

        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                # Older TF builds may not expose this; keep best-effort behavior.
                pass
    except Exception:
        # TensorFlow not installed (e.g. data-only usage).
        return

