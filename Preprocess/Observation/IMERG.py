import datetime
from functools import partial
import logging
from multiprocessing import Pool
import os
import time

import numpy as np
import polars as pl
import xarray as xr
import xesmf as xe

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PreprocessIMERG:
    def __init__(self, files: list, filesMergedIR: list = None, filePerHour: int = 2):
        fileLen = len(files) // filePerHour
        if filesMergedIR is not None:
            assert(fileLen == len(filesMergedIR))
        fileList = []
        for i in range(fileLen):
            innerList = files[i*filePerHour: (i+1)*filePerHour]
            if filesMergedIR is not None:
                innerList.append(filesMergedIR[i])
            fileList.append(innerList)
        self.fileList_ = fileList
        self.filePerHour_ = filePerHour
    
    def processSingleIMERG(
            self, filename: list,
            vars: dict,
            distDir: str,
            override: bool = False,
            fileSuffix: str = None, 
            nameOnly: bool = False):
        
        f = filename[0]
        currentBasename = os.path.basename(f)
        fSplit = currentBasename.split('.')
        minOfTheDay = int(fSplit[5])
        hourOfTheDay = minOfTheDay // 60
        fDatetimeStr = fSplit[4].split('-')
        fDatetimeStr = f'{fDatetimeStr[0]}{hourOfTheDay:02d}'
        fDatetime = datetime.datetime.strptime(fDatetimeStr, "%Y%m%d%H")
        fDatetime += datetime.timedelta(hours=1)
        currentBasename = f'{fSplit[0]}.{fSplit[1]}.{fSplit[2]}.{fSplit[3]}.{fDatetime:%Y%m%d.%H}.{fSplit[6]}.nc'
        distPath = os.path.join(distDir, currentBasename)
        if nameOnly:
            return distPath
        if not os.path.exists(distPath) or override:
            ds = xr.open_mfdataset(filename, engine='h5netcdf', concat_dim='time', combine='nested', group='Grid')
            newDs = xr.Dataset()
            for var in vars.keys():
                currentVar = vars.get(var)
                varIn = currentVar.get('var')
                method = currentVar.get('method', 'mean')
                dim = currentVar.get('dim', 'time')
                transpose = currentVar.get('transpose', None)
                if method == 'mean':
                    currentDataArray = ds[varIn].mean(dim=dim)
                    currentDataArray = currentDataArray.assign_coords(time=fDatetime)
                    currentDataArray = currentDataArray.expand_dims(dim='time')
                if transpose is not None:
                    currentDataArray = currentDataArray.transpose(*transpose)
                newDs[var] = currentDataArray
                currentDataArray.close()
            if 'time' in newDs.variables.keys():
                newDs = newDs.convert_calendar('standard', dim='time', use_cftime=True)
            newDs.to_netcdf(distPath)
            newDs.close()

        return distPath
            
    def processSingleMergedIR(
            self, filename: list,
            vars: dict,
            distDir: str,
            override: bool = False,
            fileSuffix: str = None,
            nameOnly: bool = False):
        f = filename[-1]
        currentBasename = os.path.basename(f)
        distPath = os.path.join(distDir, currentBasename)
        if nameOnly:
            return distPath
        if not os.path.exists(distPath) or override:
            ds = xr.open_dataset(filename[-1])
            dsDist = xr.open_dataset(filename[0], group='Grid')
            newDs = xr.Dataset()
            for var in vars.keys():
                currentVar = vars.get(var)
                varIn = currentVar.get('var')
                method = currentVar.get('method', 'bilinear')
                timeSlice = currentVar.get('time', None)
                transpose = currentVar.get('transpose', None)
                if fileSuffix is not None:
                    regridderPath = os.path.join(distDir, f'regridder.{method}.{fileSuffix}.nc')
                else:
                    regridderPath = os.path.join(distDir, f'regridder.{method}.nc')
                if os.path.exists(regridderPath):
                    regridder = xe.Regridder(ds, dsDist, method, weights=regridderPath)
                else:
                    regridder = xe.Regridder(ds, dsDist, method)
                    regridder.to_netcdf(filename=regridderPath)
                currentDataArray = ds[varIn]
                if timeSlice is not None:
                    currentDataArray = currentDataArray.isel(time=timeSlice)
                    currentDataArray = currentDataArray.expand_dims(dim='time')
                currentDataArray = regridder(currentDataArray, keep_attrs=True)
                if transpose is not None:
                    currentDataArray = currentDataArray.transpose(*transpose)
                newDs[var] = currentDataArray
                currentDataArray.close()
            if 'time' in newDs.variables.keys():
                newDs = newDs.convert_calendar('standard', dim='time', use_cftime=True)
            newDs.to_netcdf(distPath)
            newDs.close()

        return distPath

    def processSingle(
            self, filename: str | list,
            vars: dict,
            distDir: str,
            override: bool = False,
            fileSuffix: str = None,
            nameOnly: bool = False):
        if 'Tb' in vars.keys():
            f = filename[self.filePerHour_-1:]
            return self.processSingleMergedIR(f, vars, distDir, override, fileSuffix, nameOnly)
        else:
            f = filename[:self.filePerHour_]
            return self.processSingleIMERG(f, vars, distDir, override, fileSuffix, nameOnly)
    
    def process(self, vars: dict,
                distDir: str,
                varRename: dict = {},
                varUnitChange: dict = {},
                override: bool = False,
                fileSuffix: str = None,
                nproc: int = 1, 
                nameOnly: bool = False):
        self.vars_ = vars
        timerStart = time.time()
        if nameOnly:
            self.outputList_ = []
            for i in self.fileList_:
                self.outputList_.append(
                    self.processSingle(i, vars, distDir, override, fileSuffix, nameOnly))
        else:
            with Pool(processes=nproc) as pool:
                self.outputList_ = pool.map(
                    partial(self.processSingle, vars=vars, distDir=distDir,
                            override=override, fileSuffix=fileSuffix, nameOnly=nameOnly), 
                            self.fileList_)
        timerEnd = time.time()
        logger.warning(f'Time lasped for process step: {timerEnd - timerStart} s.')
        return self.outputList_