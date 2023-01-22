""" NN time series load forecast model """

import sys

import tensorflow as tf  # type: ignore
from custom_types import LoadForecastOptions

from model.callbacks import (
    best_val_loss_checkpoint,
    early_stopping,
    reduce_lr_on_plateau,
)


def run_model(
    opts: LoadForecastOptions,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
) -> None:
    """run the load forecast model
    Args:
      opts: LoadForecastOptions object for this run
      train_dataset: training data w/ labels (windows + horizons)
      test_dataset: test data w/ labels (windows + horizons)
    Raises:
      SystemExit if no valid model type is specified
    """

    tf.keras.backend.clear_session()

    if opts["model"] == "cnn":
        model = cnn_model(opts)
    elif opts["model"] == "lstm":
        model = lstm_model(opts)
    else:
        raise sys.exit(
            """
                Invalid options.
                Must specify cnn or lstm model type in config options.
                see configuration.py
                Exiting now.
                """
        )

    model.compile(
        loss=opts["loss"], optimizer=tf.keras.optimizers.Adam(), metrics=opts["metrics"]
    )

    model.fit(
        train_dataset,
        epochs=opts["epochs"],
        validation_data=test_dataset,
        verbose=1,
        callbacks=[
            best_val_loss_checkpoint(f"{opts['model']}{opts['zone']}"),
            early_stopping(opts["es_patience"]),
            reduce_lr_on_plateau(opts["lr_patience"]),
        ],
    )


def cnn_model(
    opts: LoadForecastOptions,
) -> tf.keras.Sequential:
    """creates a Convolutional 1D forecast model
    Args:
      opts: LoadForecastOptions object for this run
      train_dataset: training data w/ labels (windows + horizons)
      test_dataset: test data w/ labels (windows + horizons)
    Returns:
      Conv1D model built using the Sequential API
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((opts["window_opts"]["window"], 1)),
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=5,
                strides=1,
                padding="causal",
                activation="relu",
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(opts["window_opts"]["horizon"]),
        ]
    )

    model.summary()

    return model


def lstm_model(
    opts: LoadForecastOptions,
) -> tf.keras.Sequential:
    """creates an LSTM forecast model
    Args:
      opts: LoadForecastOptions object for this run
      train_dataset: training data w/ labels (windows + horizons)
      test_dataset: test data w/ labels (windows + horizons)
    Returns:
      LSTM model built using the Sequential API
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((opts["window_opts"]["window"], 1)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(opts["window_opts"]["horizon"]),
        ]
    )

    model.summary()

    return model
