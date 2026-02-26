from itertools import repeat
import logging
from multiprocessing import Pool
import os

import cv2 as cv
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Watershed:
    def __init__(
            self, filenames: str | list, outputs: str | list,
            nproc: int = 2):
        if not isinstance(filenames, list):
            filenames = [filenames]
        if not isinstance(outputs, list):
            outputs = [outputs]
        assert(len(filenames) == len(outputs))
        self.filenames_ = filenames
        self.outputList_ = outputs
        self.nproc_ = nproc
        self.daMask_ = None

    def processSingleTime(self,
            da: xr.DataArray, filterMin: float, filterBg: float, 
            filterMax: float, inverse: bool):
        filterMin = np.nanmin(da) if filterMin is None else filterMin
        # Mask data
        if self.daMask_ is not None:
            da = da.where(self.daMask_, filterMin)
        arrayRaw = da.to_numpy()
        # Compress data
        filterMax = np.nanmax(arrayRaw) if filterMax is None else filterMax
        filterBg = filterMin if filterBg is None else filterBg
        arrayNormal = np.where(arrayRaw>filterMin, arrayRaw, filterBg)
        arrayNormal = np.where(arrayNormal<filterMax, arrayNormal, filterMax)
        arrayNormal = arrayNormal - (0.99 * arrayNormal.min())
        arrayNormal = np.log(arrayNormal)
        arrayNormal = cv.normalize(arrayNormal, None, norm_type=cv.NORM_MINMAX)
        if inverse:
            arrayNormal = 1 - arrayNormal
        arrayNormal = (arrayNormal * 255).astype('uint8')

        arrayDummy = cv.cvtColor(arrayNormal, cv.COLOR_GRAY2RGB)
        ret, thresh = cv.threshold(arrayNormal, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        sureBg = cv.dilate(opening, kernel, iterations=3)
        distTranform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sureFg = cv.threshold(distTranform, 0.05 * distTranform.max(), 255, cv.THRESH_BINARY)
        sureFg = np.uint8(sureFg)
        unknown = cv.subtract(sureBg, sureFg)
        ret, markers = cv.connectedComponents(sureFg)
        markers = markers + 1
        markers[unknown==255] = 0
        markers = cv.watershed(arrayDummy, markers)
        labels = np.unique(markers)
        target = np.zeros(np.shape(arrayNormal))
        contours = list()
        for iLabel in labels[2:]:
            iArrayNormal = np.where(markers == iLabel, arrayRaw, 1.5e-5)
            iArrayNormal = iArrayNormal - (0.99 * iArrayNormal.min())
            iArrayNormal = np.log(iArrayNormal)
            iArrayNormal = cv.normalize(iArrayNormal, None, norm_type=cv.NORM_MINMAX)
            iArrayNormal = 1 - iArrayNormal
            iArrayNormal = (iArrayNormal * 255).astype('uint8')
            iArrayFake = cv.cvtColor(iArrayNormal, cv.COLOR_GRAY2RGB)
            ret, iThresh = cv.threshold(iArrayNormal, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            iOpening = cv.morphologyEx(iThresh, cv.MORPH_OPEN, kernel, iterations=2)
            iSureBg = cv.dilate(iOpening, kernel, iterations=3)
            iDistTranform = cv.distanceTransform(iOpening, cv.DIST_L2, 5)
            ret, iSureFg = cv.threshold(iDistTranform, 0.45 * iDistTranform.max(), 255, cv.THRESH_BINARY)
            iSureFg = np.uint8(iSureFg)
            iUnknown = cv.subtract(iSureBg, iSureFg)
            ret, iMarkers = cv.connectedComponents(iSureFg)
            iMarkers = iMarkers + 1
            iMarkers[iUnknown==255] = 0
            iMarkers = cv.watershed(iArrayFake, iMarkers)
            innerLabels = np.unique(iMarkers)
            for iInnerLabel in innerLabels[2:]:
                iTarget = np.where(iMarkers == iInnerLabel, 1, 0).astype(np.uint8)
                target = np.logical_or(target, iTarget)
                iContours, _ = cv.findContours(iTarget, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                contours.extend(iContours)

        for iContour in contours:
            for iPt in iContour:
                iLoc = iPt[0][::-1]
                target[*iLoc] = 0
        
        return target

    def processSingle(
            self, input: str, output: str, varname: str, newVarname: str,
            filterMin: float = None, filterBg: float = None, filterMax: float = None, 
            inverse: bool = False):
        ds = xr.open_dataset(input)
        da = ds[varname]
        timeName = None
        if 'Time' in da.dims:
            timeName = 'Time'
        elif 'time' in da.dims:
            timeName = 'time'
        
        if timeName is not None:
            nTime = ds.sizes.get(timeName)
            target = da.copy()
            for i in range(nTime):
                target[{timeName: i}] = self.processSingleTime(
                    da[{timeName: i}], filterMin, filterBg, filterMax, inverse)
            target = target.to_numpy()
        else:
            target = self.processSingleTime(da, inverse)
        newDs = ds.copy(data={varname: target})
        newDs = newDs.rename({varname: newVarname})
        newDs.to_netcdf(output)
    
    def process(
            self, varname: str, newVarname: str = None, 
            filterMin: float = None, filterBg: float = None, 
            filterMax: float = None, 
            inverse: bool = False, nproc: int = None, 
            maskFile: str = None, maskVar: str = None):
        if newVarname is None:
            newVarname = varname
        if nproc is None:
            nproc = self.nproc_
        if maskFile is not None and maskVar is not None:
            self.daMask_ = xr.open_dataset(maskFile)[maskVar]

        with Pool(processes=nproc) as pool:
            pool.starmap(
                self.processSingle, zip(
                    self.filenames_, self.outputList_, 
                    repeat(varname), repeat(newVarname), repeat(filterMin), 
                    repeat(filterBg), repeat(filterMax), repeat(inverse)
                    )
                )

        if self.daMask_ is not None:
            self.daMask_.close()
            