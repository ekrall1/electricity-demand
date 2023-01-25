""" create the .env file with variable used
to check the zip file before extracting """

import hashlib
import os


def get_zipfile_sha() -> None:
    """hash zip file's info list and size in kB
    Returns:
      unique string representation of file info and size
    """

    # pylint: disable=line-too-long
    file_info = """[<ZipInfo filename='AEP_hourly.csv' compress_type=deflate file_size=3395509 compress_size=624045>, <ZipInfo filename='COMED_hourly.csv' compress_type=deflate file_size=1842915 compress_size=339725>, <ZipInfo filename='DAYTON_hourly.csv' compress_type=deflate file_size=3274443 compress_size=561557>, <ZipInfo filename='DEOK_hourly.csv' compress_type=deflate file_size=1558965 compress_size=273297>, <ZipInfo filename='DOM_hourly.csv' compress_type=deflate file_size=3206580 compress_size=597197>, <ZipInfo filename='DUQ_hourly.csv' compress_type=deflate file_size=3214852 compress_size=533994>, <ZipInfo filename='EKPC_hourly.csv' compress_type=deflate file_size=1220853 compress_size=206412>, <ZipInfo filename='FE_hourly.csv' compress_type=deflate file_size=1701528 compress_size=314104>, <ZipInfo filename='NI_hourly.csv' compress_type=deflate file_size=1621599 compress_size=298389>, <ZipInfo filename='PJME_hourly.csv' compress_type=deflate file_size=4070265 compress_size=788363>, <ZipInfo filename='PJMW_hourly.csv' compress_type=deflate file_size=3866578 compress_size=699542>, <ZipInfo filename='PJM_Load_hourly.csv' compress_type=deflate file_size=921109 compress_size=177543>, <ZipInfo filename='est_hourly.paruqet' compress_type=deflate file_size=3680566 compress_size=3012008>, <ZipInfo filename='pjm_hourly_est.csv' compress_type=deflate file_size=12705938 compress_size=3551223>]"""
    file_size = "11698.7080078125"
    zipfile_hash = hashlib.sha256(f"{file_info}{file_size}".encode("utf-8")).hexdigest()

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        "w",
        encoding="utf-8",
    ) as env_file:
        env_file.write(f"ZIPFILEHASH = {zipfile_hash}")
        env_file.close()


get_zipfile_sha()
