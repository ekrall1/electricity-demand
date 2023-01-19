""" this module extracts data from the downloaded dataset """

import hashlib
import os
import sys
from zipfile import ZipFile

from file_validation import download_validation_object

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "hourly-energy-consumption.zip",
)

file_size = os.path.getsize(DATA_PATH)
with ZipFile(DATA_PATH, "r") as zf:
    file_info = zf.infolist()

# verify the zip file has not changed
ZIPFILE_HASH: str = hashlib.sha256(b"{file_info}{file_size}").hexdigest()

if ZIPFILE_HASH != download_validation_object["zip_file_info"]:
    raise sys.exit(
        """
        Unexpected data encountered.
        The hourly-energy-consumption.zip file's filesize or information have changed.
        Cannot continue on to model training.
        Exiting now.
        """
    )
