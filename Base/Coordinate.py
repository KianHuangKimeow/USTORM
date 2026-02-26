import logging

import numpy as np
from scipy.spatial import cKDTree
import xarray as xr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def geo2XYZOnUnitSphere(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x = np.cos(lon * np.pi / 180.0) * np.cos(lat * np.pi / 180.0)
    y = np.sin(lon * np.pi / 180.0) * np.cos(lat * np.pi / 180.0)
    z = np.sin(lat * np.pi / 180.0)
    return np.array([x, y, z]).T

def normalizeToMin(x: float, min: float, modulus: float):
    diff = x - min
    return x if (0.0 <= diff and diff < modulus) else (
        diff - modulus * np.floor(diff/modulus) + min)

def getBoundary(lon: xr.DataArray | np.ndarray, 
                lat:xr.DataArray | np.ndarray) -> np.ndarray:
    shape = np.shape(lon)
    points = []
    if len(np.shape(lon)) == 1:
        for i in lon:
            points.append(np.array([i, lat[0]]))
        for j in lat:
            points.append(np.array([lon[-1], j]))
        for i in lon[::-1]:
            points.append(np.array([i, lat[-1]]))
        for j in lat[::-1]:
            points.append(np.array([lon[0], j]))
    else:
        for i in range(shape[0]):
            points.append(np.array([lon[i,0], lat[i,0]]))
        for j in range(shape[1]):
            points.append(np.array([lon[-1,j], lat[-1,j]]))
        for i in range(shape[0]-1, -1, -1):
            points.append(np.array([lon[i,-1], lat[i,-1]]))
        for j in range(shape[1]-1, -1, -1):
            points.append(np.array([lon[0,j], lat[0,j]]))

    points = np.array(points)
    return points

def getInnerBoxMask(lon: xr.DataArray | np.ndarray, 
                    lat:xr.DataArray | np.ndarray,
                    offset: float) -> np.ndarray:
    shape = np.shape(lon)
    if len(shape) == 1:
        lon, lat = np.meshgrid(lon, lat)
    shape = np.shape(lon)
    mask = np.ones(shape, dtype=int).flatten()
    boundaryPoints = getBoundary(lon, lat)
    boundaryPoints = geo2XYZOnUnitSphere(
        boundaryPoints[:,0].flatten(), boundaryPoints[:,1].flatten())
    gridPoints = geo2XYZOnUnitSphere(lon.flatten(), lat.flatten())
    gridKDTree = cKDTree(gridPoints)
    idx = gridKDTree.query_ball_point(
                    boundaryPoints, r=offset*np.pi/180.0)
    for i in idx:
        mask[i] = 0
    mask = mask.reshape(shape)
    return mask

def getBoxMask(lon: xr.DataArray | np.ndarray,
               lat: xr.DataArray | np.ndarray,
               lonMask: xr.DataArray | np.ndarray,
               latMask: xr.DataArray | np.ndarray,
               offset: float = 0.05):
    shape = np.shape(lon)
    if len(shape) == 1:
        lon, lat = np.meshgrid(lon, lat)
    shape = np.shape(lon)
    shapeMask = np.shape(lonMask)
    if len(shapeMask) == 1:
        lonMask, latMask = np.meshgrid(lonMask, latMask)
    mask = np.zeros(shape, dtype=int).flatten()

    gridPoints = geo2XYZOnUnitSphere(lon.flatten(), lat.flatten())
    gridKDTree = cKDTree(gridPoints)
    gridPointsMask = geo2XYZOnUnitSphere(lonMask.flatten(), latMask.flatten())
    idx = gridKDTree.query_ball_point(
                    gridPointsMask, r=offset*np.pi/180.0)
    for i in idx:
        mask[i] = 1
    mask = mask.reshape(shape)
    return mask