from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LstmYieldConfig:
    hidden_units: int = 128
    dropout: float = 0.2
    dense_units: int = 128
    learning_rate: float = 1e-3
    decay: float = 1e-5


def build_lstm_yield(*, seq_len: int, n_features: int, config: LstmYieldConfig):
    """
    Bidirectional LSTM for season-level yield/biomass prediction (proxy via CWAD).

    This follows the report idea: input is a season window of climate + SWTD + irrigation;
    output is the end-of-season yield/biomass.
    """
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(seq_len, n_features), name="inputs")
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(config.hidden_units, activation="tanh", return_sequences=True),
        name="blstm1",
    )(inputs)
    x = tf.keras.layers.Dropout(config.dropout, name="dropout1")(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(config.hidden_units, activation="tanh", return_sequences=False),
        name="blstm2",
    )(x)
    x = tf.keras.layers.Dropout(config.dropout, name="dropout2")(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="tanh", name="dense")(x)
    outputs = tf.keras.layers.Dense(1, name="yield")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_yield")
    opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, decay=config.decay)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model

