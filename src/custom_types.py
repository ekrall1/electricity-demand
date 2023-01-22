""" defines custom object tyepe used in the forecast program """

from typing import Any, List, Literal, Union

from pandas import Timedelta
from typing_extensions import NotRequired, TypedDict  # interpreter is Python 3.8


class DtIntervalSelection(TypedDict):
    """dict type for intervals used for start of training data"""

    year: int
    month: int
    day: int
    hour: int


class TrainTestDates(TypedDict):
    """dict type for the start/end settings for train/test data"""

    start: DtIntervalSelection
    end: DtIntervalSelection


class TimeZoneOpts(TypedDict):
    """dict type for converting to offset-aware time index"""

    timezone: str
    ambiguous: Union[
        Any, Literal["raise", "infer", "NaT"]
    ]  # bypass type-hinting the bool, for now
    nonexistent: Union[
        Timedelta, Literal["shift_forward", "shift_backward", "NaT", "raise"]
    ]


class WindowedDatasetOpts(TypedDict):
    """options for windowing tf dataset for time series NN"""

    window: int
    horizon: int
    batch_size: int
    shuffle_buffer_size: NotRequired[int]


class LoadForecastOptions(TypedDict):
    """dict type for forecast options"""

    zone: Literal["DOM", "PJME"]
    train_test_dates: TrainTestDates
    train_pct: float
    window_opts: WindowedDatasetOpts
    timezone_opts: TimeZoneOpts
    min_max_scale: bool
    model: Literal["cnn", "lstm"]
    loss: Literal["mae"]
    metrics: List[Literal["mae"]]
    epochs: int
    es_patience: int
    lr_patience: int
    additional_features: List[Literal["sin_day", "cos_day", "sin_year", "cos_year"]]


class DownloadValidation(TypedDict):
    """dict type for downloaded dataset validation parameters
    used to run some simple validation checks on the Kaggle data file before using
    """

    zip_file_info: str
