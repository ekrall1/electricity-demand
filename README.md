Time-series forecast models for electricity demand using TensorFlow and Keras.

## Setup
- Create environment
  - using `Python 3.8.0`
  - `python -m venv venv`
  - `source venv/bin/activate` or `.\venv\Scripts\activate` (on Windows) to activate the `venv`
  - `pip install -r requirements.txt`

- Get dataset from Kaggle (need to have a Kaggle account)
  - package installation from `requirements.txt` included kaggle
  - obtain an authentication token from Kaggle (see: `https://www.kaggle.com/docs/api#getting-started-installation-&-authentication`)
  - ensure the `kaggle.json` is saved to `.kaggle` folder in the correct location (see instructions at the link above)
  - from the command line, run `kaggle datasets download robikscube/hourly-energy-consumption -p ./data`

- After getting the Kaggle dataset, create `.env` file for checking that the zipfile has not changed
  - with the `venv` activated: `python src/make_env.py`
  - verify that there is a .env file setting the ZIPFILEHASH variable

## Run the program
  - activate `venv`
  - `python src/app.py` to run
  - `src/configuration.py` contains an `LoadForecastOptions` object, which can be modified to change the run settings.  The type definition for `LoadForecastOptions` in `src/custom_types.py` specifies some limitations on allowable zones, model selections, and other parameters.

