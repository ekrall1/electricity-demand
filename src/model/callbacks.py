""" modeling callbacks """

import os

import tensorflow as tf  # type: ignore


def best_val_loss_checkpoint(
    model_name: str, path: str = "out"
) -> tf.keras.callbacks.ModelCheckpoint:
    """callback for best val loss
    Args:
      model_name: string model name
      path: string save path
    Returns
      ModelCheckPoint callback
    """

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", path, model_name
        ),
        verbose=0,
        monitor="val_loss",
        save_best_only=True,
    )


def early_stopping(patience: int) -> tf.keras.callbacks.EarlyStopping:
    """callback for early stopping
    Returns
      EarlyStopping callback
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        restore_best_weights=True,
        patience=patience,
    )


def reduce_lr_on_plateau(patience: int) -> tf.keras.callbacks.ReduceLROnPlateau:
    """callback for learning rate reduction
    Returns
      ReduceLROnPlateau callback
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=patience,
        verbose=1,
    )
