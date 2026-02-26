from collections import OrderedDict
from typing import Union

import numpy as np
import xarray as xr

# Function destagger
# Original: WRF-Python (https://doi.org/10.5065/D6W094P1) under Apache-2.0 license
def destagger(
    data: Union[np.ndarray, xr.DataArray], staggerDim: int):
    varShape = data.shape
    nDim =  data.ndim
    staggerDimSize = varShape[staggerDim]

    fullSlice = slice(None)
    slice1 = slice(0, staggerDimSize - 1, 1)
    slice2 = slice(1, staggerDimSize, 1)

    dimRanges1= [fullSlice] * nDim
    dimRanges2 = [fullSlice] * nDim

    dimRanges1[staggerDim] = slice1
    dimRanges2[staggerDim] = slice2

    if isinstance(data, xr.DataArray):
        result = 0.5 * (data.to_numpy()[tuple(dimRanges1)] + 
                        data.to_numpy()[tuple(dimRanges2)])
    else:
        result = 0.5 * (data[tuple(dimRanges1)] + 
                        data[tuple(dimRanges2)])
        
    if isinstance(data, xr.DataArray):
        if data.name is not None:
            outname = data.name
        else:
            outname = 'data'
        outattrs = OrderedDict()
        outattrs.update(data.attrs)

        outattrs['destag_dim'] = staggerDim

        outdims = []
        outdims += data.dims
        destag_dim_name = outdims[staggerDim]
        if destag_dim_name.find('_stag') >= 0:
            new_dim_name = destag_dim_name.replace('_stag', "")
            outdims[staggerDim] = new_dim_name

        result = xr.DataArray(result, name=outname, dims=outdims, attrs=outattrs)
    
    return result