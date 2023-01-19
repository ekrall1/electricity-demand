""" information to run some simple validation checks on the Kaggle data file before using """

from typing_extensions import TypedDict  # interpreter is Python 3.8


class DownloadValidation(TypedDict):
    """dict type for downloaded dataset validation parameters"""

    zip_file_info: str


download_validation_object: DownloadValidation = {
    "zip_file_info": "dec601bbb17837950b27f9a470f68a9f679723fee2605ae9d57951f02a46f28b"
}
