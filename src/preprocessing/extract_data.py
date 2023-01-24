""" this module extracts data from the downloaded dataset """
import datetime
import hashlib
import os
import sys
from copy import deepcopy
from typing import Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytz
from dotenv.main import load_dotenv

from config import DATA_PATH, PARQUET_FILENAME, PARQUET_ORIGINAL_FILENAME, ZIP_FILENAME
from custom_types import DtIntervalSelection, LoadForecastOptions


class DataExtract:
    """contains methods to extract hourly load data from compressed format
    and load parquet to dataframe

    Attributes:
      data_path:        path to data files
      zip_filename:     name of the compressed archive
      parquet_filename: name of the final parquet file
      parquet_original_filename:    name of the parquet file in the archive
      zipfile_object:   zipfile.ZipFile object created from the archive
    """

    def __init__(self):
        self.data_path = DATA_PATH
        self.zip_filename = ZIP_FILENAME
        self.parquet_filename = PARQUET_FILENAME
        self.parquet_original_filename = PARQUET_ORIGINAL_FILENAME
        with ZipFile(self.zip_filepath, "r") as zip_file:
            self.zipfile_object = zip_file
        load_dotenv()
        self.zip_file_hash = os.environ["ZIPFILEHASH"]

    @property
    def zip_filepath(self):
        """the path to the compressed archive of data files
        Returns:
          filepath in string format
        """
        return self._path_to_file(self.zip_filename)

    @property
    def parquet_filepath(self):
        """the path to the extracted data parquet
        Returns:
          filepath in string format
        """
        return self._path_to_file(self.parquet_filename)

    def extract_data(self) -> None:
        """
        extract data from compressed archive
        """
        if self._check_for_existing_parquet_file():
            return

        zipfile_sha: str = self._get_zipfile_sha()
        self._verify_correct_data(zipfile_sha)
        # test the zipfile using built-in method
        self.zipfile_object.testzip()
        # extract
        self.zipfile_object.extract(self.parquet_original_filename, self.data_path)

        os.rename(
            self._path_to_file(self.parquet_original_filename), self.parquet_filepath
        )

    def load_data_from_parquet(
        self, opts: LoadForecastOptions
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        check that parquet has been extracted and load parquet into pandas dataframe
        Args:
          opts:     a load forecast options object specified in config file
        Returns:
          pandas object containing load and feature data
          if there is no parquet found, returns an empty series
        Note: current state only allows for one zone to be foreast at a time
        """
        if not self._check_for_existing_parquet_file():
            print(
                """warning: nothing was loaded.
                Use method `extract_data()` to create hourly load parquet"""
            )
            return pd.Series()

        df_load_data = pd.read_parquet(self.parquet_filepath)

        # localize datetime index using timezone options (make the index offset aware)
        df_load_data.index = pd.to_datetime(df_load_data.index).tz_localize(
            opts["timezone_opts"]["timezone"],
            ambiguous=opts["timezone_opts"]["ambiguous"],
            nonexistent=opts["timezone_opts"]["nonexistent"],
        )
        start = pytz.timezone(opts["timezone_opts"]["timezone"]).localize(
            self._convert_train_test_opts_to_dt(opts["train_test_dates"]["start"])
        )
        end = pytz.timezone(opts["timezone_opts"]["timezone"]).localize(
            self._convert_train_test_opts_to_dt(opts["train_test_dates"]["end"])
        )

        idx_locs = self._get_date_range_idx_locs(df_load_data.index, start, end)

        feature_df = df_load_data.iloc[idx_locs].sort_index()

        if len(opts["additional_features"]) > 0:
            feature_df = self.add_features(feature_df)

        return feature_df[[opts["zone"], *opts["additional_features"]]]

    @staticmethod
    def add_features(input_df: pd.DataFrame):
        """add features to the dataframe for multivariate model
        Args:
          inputs_df:    datetime-indexed dataframe of load data
        Returns:
          new_df:       dataframe with columns for each additional allowable feature
        """

        new_df = deepcopy(input_df)

        # get timestamps from index
        timestamps = np.array(new_df.index.map(pd.Timestamp.timestamp).to_list())

        new_df["sin_day"] = np.sin(timestamps * (2 * np.pi / 24 / 60 / 60))
        new_df["cos_day"] = np.cos(timestamps * (2 * np.pi / 24 / 60 / 60))
        new_df["sin_year"] = np.sin(timestamps * (2 * np.pi / 24 / 60 / 60 / 365.245))
        new_df["cos_year"] = np.cos(timestamps * (2 * np.pi / 24 / 60 / 60 / 365.245))
        days_of_week = new_df.index.to_series().dt.dayofweek
        new_df["weekend"] = [1 if day < 5 else 0 for day in days_of_week]
        new_df["dayofweek"] = [1 if day == 2 else 0 for day in days_of_week]
        new_df["hour"] = new_df.index.to_series().dt.hour
        new_df["dayofyear"] = new_df.index.to_series().dt.dayofyear

        return new_df

    def _get_date_range_idx_locs(
        self, dates: pd.DatetimeIndex, start: datetime.datetime, end: datetime.datetime
    ) -> pd.Index:
        return pd.Index({idx for idx, date in enumerate(dates) if start <= date <= end})

    def _convert_train_test_opts_to_dt(self, dt_interval: DtIntervalSelection):
        """"""
        return datetime.datetime(
            dt_interval["year"],
            dt_interval["month"],
            dt_interval["day"],
        ) + datetime.timedelta(hours=dt_interval["hour"])

    def _path_to_file(self, filename: str) -> str:
        """function for path to file within the data directory
        Args:
          filename:  the name of the file
        Returns:
          path to file in the data directory, in string format
        """
        return os.path.join(DATA_PATH, filename)

    def _check_for_existing_parquet_file(self) -> bool:
        """check whether parquet file already exists before extracting
        Returns:
          boolean indicator of whether the file already exists
          True -> yes it exists already
        """
        return os.path.exists(self.parquet_filepath)

    def _verify_correct_data(self, sha: str) -> None:
        """
        inspect that the correct data was downloaded for training, otherwise exit
        Args:
          sha:      representation of of the zipfile info and size
        Raises:
          SystemExit
        """
        if sha != self.zip_file_hash:
            raise sys.exit(
                """
                Unexpected data encountered.
                The hourly-energy-consumption.zip file's filesize or information have changed.
                Will not continue on to model training. Exiting now.
                """
            )

    def _get_zipfile_sha(self) -> str:
        """hash zip file's info list and size in kB
        Returns:
          unique string representation of file info and size
        """

        info = self.zipfile_object.infolist()
        size = os.path.getsize(self.zip_filepath) / 1024

        zip_info_size = f"{info}{size}".encode("utf-8")
        return hashlib.sha256(zip_info_size).hexdigest()
