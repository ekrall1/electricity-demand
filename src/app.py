""" this module runs the long-term hourly load forecasting NN model """

import tensorflow as tf  # type: ignore

from configuration import FORECAST_OPTIONS_OBJECT as opts
from model.model import run_model
from preprocessing.extract_data import DataExtract
from preprocessing.scaler import scale_data
from preprocessing.train_test_splits import train_test_split
from preprocessing.windowing import windowed_dataset_factory

# extract and load data
data_extractor = DataExtract()

data_extractor.extract_data()

model_data = data_extractor.load_data_from_parquet(opts)

# scale data
(scaled_model_data, scaler) = scale_data(model_data, opts)

# split data
(train_data, test_data) = train_test_split(scaled_model_data, opts)

# preprocess windows and look-ahead horizons
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

windowing = windowed_dataset_factory(
    opts["window_opts"], features=(1 + len(opts["additional_features"]))
)

windowed_training_dataset = windowing.make_windows(train_dataset)
windowed_test_dataset = windowing.make_windows(test_dataset)

# run model
run_model(opts, windowed_training_dataset, windowed_test_dataset, scaler)
