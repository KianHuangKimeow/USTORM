import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def geo2XYZOnUnitSphere(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x = np.cos(lon * np.pi / 180.0) * np.cos(lat * np.pi / 180.0)
    y = np.sin(lon * np.pi / 180.0) * np.cos(lat * np.pi / 180.0)
    z = np.sin(lat * np.pi / 180.0)
    return np.array([x, y, z]).T
