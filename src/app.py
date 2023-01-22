""" this module runs the long-term hourly load forecasting NN model """

import tensorflow as tf  # type: ignore

from configuration import FORECAST_OPTIONS_OBJECT as opts
from model.model import run_model
from preprocessing.extract_data import DataExtract
from preprocessing.train_test_splits import train_test_split
from preprocessing.windowing import windowed_dataset_factory

# extract and load data
data_extractor = DataExtract()

data_extractor.extract_data()

load_data_series = data_extractor.load_parquet_to_df(opts)

# split data
(train_series, test_series) = train_test_split(load_data_series, opts)

# preprocess
train_dataset = tf.data.Dataset.from_tensor_slices(train_series)
test_dataset = tf.data.Dataset.from_tensor_slices(test_series)

windowing = windowed_dataset_factory(opts["window_opts"])

windowed_training_dataset = windowing.make_windows(train_dataset)
windowed_test_dataset = windowing.make_windows(test_dataset)

# run model
run_model(opts, windowed_training_dataset, windowed_test_dataset)
