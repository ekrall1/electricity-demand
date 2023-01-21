""" create windowed datasets for NN time series models """

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import tensorflow as tf  # type: ignore
from typing_extensions import NotRequired, TypedDict  # interpreter is Python 3.8


class WindowedDatasetOpts(TypedDict):
    """options for windowing tf dataset for time series NN"""

    window: int
    horizon: int
    batch_size: int
    shuffle_buffer_size: NotRequired[int]


class WindowOptionsValidationError(Exception):
    """exception raised for errors in windowing options object
    Attributes:
       message: explanation of the problem
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def validate_options(opts: WindowedDatasetOpts):
    """validate an instance of windowing options"""
    invalid_options = [
        {"option": key, "value": value}
        for key, value in opts.items()
        if isinstance(value, int) and value < 1
    ]
    if invalid_options:
        raise WindowOptionsValidationError(
            f"""invalid windowing options detected:\n
                {invalid_options}\n
                The values for each window dataset option must be >= 1.
                """
        )


@dataclass
class WindowedDataset:
    """class for unshuffled windowed dataset objects
    Attributes:
        opts:       windowing options object
        total_len:  lag window + forecast horizon (intervals)
        horizon:    forecast horizon
        batch_size: dataset batch size
    """

    opts: WindowedDatasetOpts

    def __post_init__(self):

        validate_options(self.opts)

        self.total_len = self.opts["window"] + self.opts["horizon"]
        self.horizon = self.opts["horizon"]
        self.batch_size = self.opts["batch_size"]

    def make_windows(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """return windowed series and labels, with shuffling
        Args:
            dataset:   an un-windowed Tf dataset
        Returns:
            windowed Tf dataset, without shuffling
        """

        return (
            dataset.window(self.total_len, shift=1, drop_remainder=True)
            .flat_map(lambda series: series.batch(self.total_len))
            .map(lambda win: (win[: -self.horizon], win[-self.horizon :]))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )


@dataclass
class ShuffledWindowedDataset:
    """class for shuffled windowed dataset objects
    Attributes:
        opts:       windowing options object
        total_len:  lag window + forecast horizon (intervals)
        horizon:    forecast horizon
        batch_size: dataset batch size
        shuffle_buffer: buffer size for shuffling
    """

    opts: WindowedDatasetOpts

    def __post_init__(self):

        validate_options(self.opts)

        self.total_len = self.opts["window"] + self.opts["horizon"]
        self.horizon = self.opts["horizon"]
        self.batch_size = self.opts["batch_size"]
        self.shuffle_buffer = self.opts["shuffle_buffer_size"]

    def make_windows(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """return windowed series and labels, with shuffling
        Args:
            dataset:   an un-windowed Tf dataset
        Returns:
            windowed Tf dataset, with shuffling
        """

        return (
            dataset.window(self.total_len, shift=1, drop_remainder=True)
            .flat_map(lambda series: series.batch(self.total_len))
            .shuffle(self.shuffle_buffer)
            .map(lambda win: (win[: -self.horizon], win[-self.horizon :]))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )


def windowed_dataset_factory(
    opts: WindowedDatasetOpts,
) -> Union[WindowedDataset, ShuffledWindowedDataset]:
    """used for creating windowed datasets
    Args:
        opts:   windowing options object
    Returns:
        windowed dataset, shuffled or unshuffled based on opts
    """

    if "shuffle_buffer_size" in opts.keys():
        return ShuffledWindowedDataset(opts)
    return WindowedDataset(opts)
