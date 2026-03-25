from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DqnConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    target_update_steps: int = 500

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000


def build_q_network(state_dim: int, n_actions: int, hidden: int = 128):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(state_dim,), name="state")
    x = tf.keras.layers.Dense(hidden, activation="relu")(inputs)
    x = tf.keras.layers.Dense(hidden, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_actions, name="q_values")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="q_network")
    return model


class DqnAgent:
    def __init__(self, *, state_dim: int, n_actions: int, config: DqnConfig, seed: int = 1):
        import tensorflow as tf

        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.config = config
        self.rng = np.random.default_rng(int(seed))

        self.q = build_q_network(self.state_dim, self.n_actions)
        self.q_target = build_q_network(self.state_dim, self.n_actions)
        self.q_target.set_weights(self.q.get_weights())

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.config.lr)
        self._train_steps = 0

    def epsilon(self) -> float:
        t = min(self._train_steps, self.config.epsilon_decay_steps)
        frac = t / float(self.config.epsilon_decay_steps)
        return float(self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start))

    def act(self, state: np.ndarray, *, greedy: bool = False) -> int:
        if (not greedy) and (self.rng.random() < self.epsilon()):
            return int(self.rng.integers(0, self.n_actions))
        q_vals = self.q(np.expand_dims(state.astype(np.float32), axis=0), training=False).numpy()[0]
        return int(np.argmax(q_vals))

    def train_on_batch(self, batch) -> float:
        import tensorflow as tf

        states = tf.convert_to_tensor(batch.states, dtype=tf.float32)
        actions = tf.convert_to_tensor(batch.actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(batch.rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(batch.next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(batch.dones, dtype=tf.float32)

        gamma = tf.constant(self.config.gamma, dtype=tf.float32)

        with tf.GradientTape() as tape:
            q_values = self.q(states, training=True)  # (B, A)
            action_q = tf.gather(q_values, actions, axis=1, batch_dims=1)  # (B,)

            next_q = self.q_target(next_states, training=False)
            max_next_q = tf.reduce_max(next_q, axis=1)
            target = rewards + (1.0 - dones) * gamma * max_next_q

            loss = tf.reduce_mean(tf.square(target - action_q))

        grads = tape.gradient(loss, self.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables, strict=True))

        self._train_steps += 1
        if self._train_steps % self.config.target_update_steps == 0:
            self.q_target.set_weights(self.q.get_weights())

        return float(loss.numpy())

