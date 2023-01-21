""" this module extracts data from the downloaded dataset """

import hashlib
import os
import sys
from zipfile import ZipFile

import pandas as pd

from file_validation import download_validation_object

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
)

ZIP_FILENAME = "hourly-energy-consumption.zip"
PARQUET_ORIGINAL_FILENAME = (
    "est_hourly.paruqet"  # it is mis-spelled in the Kaggle zip archive
)
PARQUET_FILENAME = "est_hourly.parquet"


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

    @property
    def zip_filepath(self):
        """the path to the compressed archive of data files
        Returns
          filepath in string format
        """
        return self._path_to_file(self.zip_filename)

    @property
    def parquet_filepath(self):
        """the path to the extracted data parquet
        Returns
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

    def load_parquet_to_df(self) -> pd.DataFrame:
        """
        check that parquet has been extracted and load parquet into pandas dataframe
        Returns
          dataframe containing load data, if the parquet exists
          otherwise an empty dataframe
        """
        if not self._check_for_existing_parquet_file():
            print(
                """warning: nothing to load.
                Use method `extract_data()` to create hourly load parquet"""
            )
            return pd.DataFrame()
        return pd.read_parquet(self.parquet_filepath)

    def _path_to_file(self, filename: str) -> str:
        """function for path to file within the data directory
        Returns
          path to file in the data directory, in string format
        """
        return os.path.join(DATA_PATH, filename)

    def _check_for_existing_parquet_file(self) -> bool:
        """check whether parquet file already exists before extracting
        Returns
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
        if sha != download_validation_object["zip_file_info"]:
            raise sys.exit(
                """
                Unexpected data encountered.
                The hourly-energy-consumption.zip file's filesize or information have changed.
                Will not continue on to model training. Exiting now.
                """
            )

    def _get_zipfile_sha(self) -> str:
        """hash zip file's info list and size in kB
        Returns
          unique string representation of file info and size
        """
        return hashlib.sha256(
            b"{self.zipfile_object.infolist()}{os.path.getsize(self.zip_filepath)/1024}"
        ).hexdigest()
