from typing import Union

import numpy as np
import metpy.calc as mpcalc
import metpy.constants as mpconst
import metpy.units as mpunits
import xarray as xr

constantT = 273.16 + 17.5
constP = 10000.0 # 100hPa
flagRidiculousMM5Test = True
RD = 287.0
G = 9.81
USSALR = 0.0065

# Function calculateSeaLevelPressure
# Original: WRF-Python (https://doi.org/10.5065/D6W094P1) under Apache-2.0 license
# See Fortran subroutine DCOMPUTESEAPRS
def calculateSeaLevelPressure(z: Union[np.ndarray, xr.DataArray], 
                              t: Union[np.ndarray, xr.DataArray], 
                              p: Union[np.ndarray, xr.DataArray], 
                              q: Union[np.ndarray, xr.DataArray]):
    '''
    Calculate the sea level pressure from geopotential height, 
    temperature, pressure and mixing ratio. The vertical dimension 
    must be z.

      Parameters
      ----------
      p : (..., P, M, N) `xarray.DataArray` or `pint.Quantity`
          pressure
      z : (..., P, M, N) `xarray.DataArray` or `pint.Quantity`
          geopotential height
      t : (..., P, M, N) `xarray.DataArray` or `pint.Quantity`
          air temperature
      q : (..., P, M, N) `xarray.DataArray` or `pint.Quantity`
          mixing ratio
    '''
    nz = p.sizes.get('z')

    pAtConstP = p.isel(z=0) - constP
    pAtConstP = pAtConstP.metpy.dequantify()
    pMask = p < pAtConstP
    count = pMask.sum(dim='z')
    level = pMask.argmax(dim='z')
    if np.any(count == 0):
        raise Exception('Cannot calculate sea level pressure.')

    kLow = np.maximum(level - 1, 0)
    kHigh = np.minimum(kLow + 1, nz - 1)
    if np.any(kLow == kHigh):
        raise Exception('Cannot calculate sea level pressure.')

    pLow = p.isel(z=kLow).metpy.dequantify()
    pHigh = p.isel(z=kHigh).metpy.dequantify()
    tvLow = t.isel(z=kLow) * (1.0 + 0.608 * q.isel(z=kLow))

    tvHigh = t.isel(z=kHigh) * (1.0 + 0.608 * q.isel(z=kHigh))

    zLow = z.isel(z=kLow).metpy.dequantify()
    zHigh = z.isel(z=kHigh).metpy.dequantify()

    tvAtConstP = tvHigh - (tvHigh - tvLow) * np.log(pAtConstP/pHigh) * np.log(pLow/pHigh)
    zAtConstP = zHigh - (zHigh - zLow) * np.log(pAtConstP/pHigh) *   np.log(pLow/pHigh)
    tvSurface = tvAtConstP * (p.isel(z=0) / pAtConstP) ** \
        (USSALR * RD / G)
    tvSeaLevel = tvAtConstP + USSALR * \
                 zAtConstP

    if flagRidiculousMM5Test:
        testMask = (tvSeaLevel >= constantT) & (tvSurface <= constantT)
        tvSeaLevel = xr.where(
            testMask, constantT, constantT - 0.005 * \
                (tvSurface.metpy.dequantify() - constantT) ** 2
        )
    
    slp = p.isel(z=0) * np.exp((2.0 * G * z.isel(z=0)) / 
              (RD * (tvSeaLevel + tvSurface)))
    
    return slp
