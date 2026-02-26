import datetime
from dateutil import tz
from typing import Union

import numpy as np
import xarray as xr

def npDatetimeToDatetime(t: Union[np.datetime64, xr.DataArray]) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(
        (t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'), tz.tzutc())
  