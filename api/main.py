from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_ACTIONS_MM = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]


class RecommendRequest(BaseModel):
    state: list[float] = Field(..., description="State vector used by the DQN Q-network.")
    actions_mm: Optional[list[float]] = Field(
        None, description="Optional action mapping (index -> irrigation mm/day)."
    )


class RecommendResponse(BaseModel):
    action_index: int
    irrigation_mm: float
    q_values: list[float]


@lru_cache(maxsize=1)
def _load_q_network():
    import tensorflow as tf

    model_path = os.environ.get("Q_NETWORK_PATH")
    if not model_path:
        raise RuntimeError("Q_NETWORK_PATH env var is not set (path to q_network.keras).")
    return tf.keras.models.load_model(model_path)


app = FastAPI(title="Intelligent Irrigation API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        qnet = _load_q_network()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    actions = req.actions_mm or DEFAULT_ACTIONS_MM
    if not actions:
        raise HTTPException(status_code=400, detail="actions_mm must not be empty")

    state = np.asarray(req.state, dtype=np.float32).reshape(1, -1)
    q_values = np.asarray(qnet(state, training=False)).reshape(-1)
    action_index = int(np.argmax(q_values))
    if action_index >= len(actions):
        raise HTTPException(
            status_code=400,
            detail=f"Q-network output has {len(q_values)} actions but actions_mm has {len(actions)} entries.",
        )

    return RecommendResponse(
        action_index=action_index,
        irrigation_mm=float(actions[action_index]),
        q_values=q_values.tolist(),
    )

