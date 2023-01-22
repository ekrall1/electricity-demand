""" train-test split for time series forecast """

from typing import Tuple

import pandas as pd
from custom_types import LoadForecastOptions


def train_test_split(
    series: pd.Series, opts: LoadForecastOptions
) -> Tuple[pd.Series, pd.Series]:
    """split time series data into train and test sets
    future issue will add validation set
    Args:
      series:   pd.Series containing load data, datetime-indexed
      opts:     LoadForecastOptions object for this run
    Returns:
      Tuple containing train/test data split at the approp. index
    """

    test_start_idx = int(len(series) * opts["train_pct"])

    train_data = series[:test_start_idx]

    test_data = series[test_start_idx:]

    return train_data, test_data
