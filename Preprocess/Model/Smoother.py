from itertools import repeat
import logging
import os
from multiprocessing import Pool

import numpy as np
import scipy.ndimage
import xarray as xr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def gaussianSmoother(ds: xr.Dataset, vars: str | list, sigma: float | list):
    if not isinstance(vars, list):
        vars = [vars]
    if not isinstance(sigma, list):
        sigma = [sigma] * len(vars)
    assert(len(vars) == len(sigma))

    for v, s in zip(vars, sigma):
        da = ds[v]
        timeName = None
        if 'Time' in da.dims:
            timeName = 'Time'
        elif 'time' in da.dims:
            timeName = 'time'
        if timeName is not None:
            nTime = ds.sizes.get(timeName)
            for i in range(nTime):
                da[{timeName: i}] = scipy.ndimage.gaussian_filter(
                    da[{timeName: i}], sigma=s)
        else:
            da = scipy.ndimage.gaussian_filter(da, sigma=s)
        ds[v] = da
    
    return ds

class GaussianSmoother:
    def __init__(self, files: list, output: list = None):
        self.fileList_ = files
        if output is not None:
            assert(len(files) == len(output))
            self.outputFileList_ = output
        else:
            self.outputFileList_ = files

    def processSingle(self, input: str, output: str, vars: str | list, sigma: float | list,
                      override: bool = False):
        if not os.path.exists(output) or override:
            ds = xr.open_dataset(input)
            ds = gaussianSmoother(ds, vars, sigma)
            ds.to_netcdf(output)
        return True

    def process(self, vars: str | list, sigma: float | list, 
                override: bool = False, 
                nproc = 2):
        with Pool(processes=nproc) as pool:
            result = pool.starmap(
                self.processSingle, zip(
                    self.fileList_, self.outputFileList_,
                    repeat(vars), repeat(sigma),
                    repeat(override)
                ) 
            )

        result = np.all(result)
        if result:
            logger.warning('Succeed!')
        else:
            logger.warning('Somethings happened...')