This repository contains code for time series forecasts of electricity demand using TensorFlow and Keras.

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

- To test
  - `python src/app.py`
  - `src/configuration.py` contains an options object, which can be modified to change the run settings.  `src/custom_types.py` type definitions specify some of the limitations, such as allowable entries for zones and model types.

