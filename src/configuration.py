""" Misc. configuration settings for the forecast program """
import os

from custom_types import DownloadValidation, LoadForecastOptions

# forecast options

FORECAST_OPTIONS_OBJECT: LoadForecastOptions = {
    "zone": "DOM",
    "train_test_dates": {
        "start": {"year": 2012, "month": 1, "day": 1, "hour": 0},
        "end": {"year": 2016, "month": 12, "day": 31, "hour": 23},
    },
    "timezone_opts": {  # used to add UTC offsets to the datetimes
        "timezone": "US/Eastern",  # the tz of the data
        "ambiguous": True,  # fall back
        "nonexistent": "shift_forward",  # turn clock forward
    },
    "train_pct": 0.8,
    "min_max_scale": True,
    "window_opts": {
        "window": 24 * 30,
        "horizon": 24 * 7,
        "batch_size": 32,
        "shuffle_buffer_size": 1000,
    },
    "model": "cnn",
    "epochs": 1000,
    "loss": "huber",
    "metrics": ["mae"],
    "es_patience": 200,
    "lr_patience": 100,
    "additional_features": ["sin_day", "sin_year", "cos_day", "cos_year", "weekday"],
}


# data info
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
)

# Model weights and other output
MODEL_OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "out",
)

ZIP_FILENAME = "hourly-energy-consumption.zip"

PARQUET_ORIGINAL_FILENAME = (
    "est_hourly.paruqet"  # it is mis-spelled in the Kaggle zip archive
)

PARQUET_FILENAME = "est_hourly.parquet"

# validation of correct download
DOWNLOAD_VALIDATION_OBJECT: DownloadValidation = {
    "zip_file_info": "9464c1e4a98aa4bd9348e20629b77c16bf85a26cd9073ec2b262855a2d532b24",
}
