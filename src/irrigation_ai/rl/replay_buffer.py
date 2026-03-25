from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)

        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity,), dtype=np.int32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)

        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self._pos
        self._states[i] = state
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._next_states[i] = next_state
        self._dones[i] = 1.0 if done else 0.0

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        idx = rng.integers(0, self._size, size=int(batch_size), endpoint=False)
        return ReplayBatch(
            states=self._states[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_states=self._next_states[idx],
            dones=self._dones[idx],
        )

