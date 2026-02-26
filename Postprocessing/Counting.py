from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
import xarray as xr

from Base import normalizeToMin

def densityMap(
        lon: xr.DataArray | np.ndarray, 
        lat:xr.DataArray | np.ndarray,
        gridLon: Optional[xr.DataArray | np.ndarray] = None,
        gridLat: Optional[xr.DataArray | np.ndarray] = None) -> np.ndarray:
    func = lambda x: normalizeToMin(x, 0.0, 360.0)
    if len(lon) > 0:
        lon = np.vectorize(func)(lon)
    if gridLon is None or gridLat is None:
        gridLon = np.arange(0.0, 360.0, 2.0)
        gridLat = np.arange(-90.0, 90.0, 2.0)
    gridShape = np.shape(gridLon)
    if len(gridShape) == 1:
        gridLon, gridLat = np.meshgrid(gridLon, gridLat)
        gridShape = np.shape(gridLon)

    gridLon = gridLon.flatten()
    gridLat = gridLat.flatten()
    gridCounter = np.zeros(np.shape(gridLon))
    gridPoints = np.array([gridLon, gridLat]).T
    kdTree = cKDTree(gridPoints)
    points = np.array([lon, lat]).T
    for i in points:
        dist, idx = kdTree.query(i, k=1)
        gridCounter[idx] += 1
    return gridCounter.reshape(gridShape)
    