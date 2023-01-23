""" function for applying scaling transformation to model data """
from typing import Tuple, Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # type: ignore

from custom_types import LoadForecastOptions


def scale_data(
    data: Union[pd.Series, pd.DataFrame], opts: LoadForecastOptions
) -> Tuple[Union[pd.Series, pd.DataFrame], MinMaxScaler]:
    """scale the input data using the sklearn MinMaxScaler
    Args:
      data:  pd.Series or pd.DataFrame of model data
      opts:  LoadForecastOptions object
    Returs:
      data:  pandas object with load column scaled
      scaler:   MinMaxScaler object with method to invert transform
    """

    scaler = MinMaxScaler()

    data[opts["zone"]] = scaler.fit_transform(data[[opts["zone"]]])  # type: ignore
    print(data)

    return (data, scaler)
