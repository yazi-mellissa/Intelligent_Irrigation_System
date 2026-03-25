from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LstmSwtdConfig:
    hidden_units: int = 512
    dropout: float = 0.3
    learning_rate: float = 1e-4
    decay: float = 1e-5


def build_lstm_swtd(*, n_features: int, config: LstmSwtdConfig):
    """
    Bidirectional LSTM for next-day SWTD prediction (as in the report notebooks).
    """
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(None, n_features), name="inputs")
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            config.hidden_units, activation="tanh", return_sequences=True, name="lstm1"
        ),
        name="blstm1",
    )(inputs)
    x = tf.keras.layers.Dropout(config.dropout, name="dropout1")(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            config.hidden_units, activation="tanh", return_sequences=False, name="lstm2"
        ),
        name="blstm2",
    )(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="swtd_next")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_swtd")

    opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, decay=config.decay)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model

