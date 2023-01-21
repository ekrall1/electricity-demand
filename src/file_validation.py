""" information to run some simple validation checks on the Kaggle data file before using """

from typing_extensions import TypedDict  # interpreter is Python 3.8


class DownloadValidation(TypedDict):
    """dict type for downloaded dataset validation parameters"""

    zip_file_info: str


download_validation_object: DownloadValidation = {
    "zip_file_info": "9464c1e4a98aa4bd9348e20629b77c16bf85a26cd9073ec2b262855a2d532b24"
}
