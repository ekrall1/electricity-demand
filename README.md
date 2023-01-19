This repository contains code to train time series forecasting models using Tensorflow

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


