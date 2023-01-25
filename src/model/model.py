""" NN time series load forecast model """

import os
import sys

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore

from config import MODEL_OUT_PATH
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
    scaler,
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
            best_val_loss_checkpoint(f"{opts['model']}{opts['zone']}.hdf5"),
            early_stopping(opts["es_patience"]),
            reduce_lr_on_plateau(opts["lr_patience"]),
        ],
    )

    # predict.  soon move this to its own fcn
    model.load_weights(
        os.path.join(MODEL_OUT_PATH, f"{opts['model']}{opts['zone']}.hdf5")
    )
    pred = model.predict(tf.expand_dims(list(test_dataset)[0][0][0], axis=0))

    # plot prediction.  soon move this to its own fcn
    plt.plot(np.squeeze(scaler.inverse_transform(pred)), label="predicted")
    plt.plot(
        np.squeeze(
            scaler.inverse_transform(
                tf.expand_dims(list(test_dataset)[0][1][0], axis=0)
            )
        ),
        label="actual",
    )
    plt.legend(loc="upper left")
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.show()


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
            tf.keras.layers.Input(
                (opts["window_opts"]["window"], 1 + len(opts["additional_features"])),
                name="input",
            ),
            tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=5,
                strides=1,
                padding="causal",
                activation="relu",
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=5,
                strides=1,
                padding="causal",
                activation="relu",
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(name="cnn_flatten"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(opts["window_opts"]["horizon"], name="output"),
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
            tf.keras.layers.Input(
                (opts["window_opts"]["window"], 1 + len(opts["additional_features"])),
                name="input",
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(16, return_sequences=True, name="lstm_bidir1")
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(opts["window_opts"]["horizon"], name="output"),
        ]
    )

    model.summary()

    return model
