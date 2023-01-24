""" create the .env file with variable used
to check the zip file before extracting """

import hashlib
import os
from zipfile import ZipFile

from config import DATA_PATH, ZIP_FILENAME


def get_zipfile_sha() -> None:
    """hash zip file's info list and size in kB
    Returns:
      unique string representation of file info and size
    """

    zip_filepath = os.path.join(DATA_PATH, ZIP_FILENAME)
    print(zip_filepath)

    with ZipFile(zip_filepath) as zipfile_object:
        info = zipfile_object.infolist()
        size = os.path.getsize(zip_filepath)
        combo_str = f"{info}{size/1024}".encode("utf-8")
        zipfile_hash = hashlib.sha256(combo_str).hexdigest()

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        "w",
        encoding="utf-8",
    ) as env_file:
        env_file.write(f"ZIPFILEHASH = {zipfile_hash}")
        env_file.close()


get_zipfile_sha()
