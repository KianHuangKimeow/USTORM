import gc
from itertools import repeat
import logging
from multiprocessing import Pool
import os
from typing import Optional
import time

import numpy as np
import metpy.calc as mpcalc
import metpy.units as mpunits
import xarray as xr
import xesmf as xe

from Preprocess.Model import (
    destagger,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PreprocessERA5:
    def __init__(
            self, mlSpFiles: list, mlTFiles: list, mlQFiles: list,
            mlUFiles: list, mlVFiles: list):
        self.mlSpFileList_ = mlSpFiles
        self.mlTFileList_ = mlTFiles
        self.mlQFileList_ = mlQFiles
        self.mlUFileList_ = mlUFiles
        self.mlVFileList_ = mlVFiles
        self.currentDs_ = None
        self.invariantPath_ = None
        self.invariantDs_ = None
        self.outputList_ = []

    def processSingle(
            self, mlSpFile: str, mlTFile: str, mlQFile: str,
            mlUFile: str, mlVFile: str,
            vars: dict,
            distDir: str,
            override: Optional[bool] = False,
            fileSuffix: str = ''):
        currentBasename = os.path.basename(mlSpFile)
        currentSuffix = currentBasename.split('regn320sc.')[-1]
        currentBasename = f'e5.oper.an.ml.preprocess.regn320sc.{currentSuffix}'
        distPath = os.path.join(distDir, currentBasename)
        if len(fileSuffix) > 0:
            distPath = distPath.rsplit('.', 1)
            distPath = f'{distPath[0]}{fileSuffix}.{distPath[1]}'
        if not os.path.exists(distPath) or override:
            dsMlSp = xr.open_dataset(mlSpFile)
            dsMlT = xr.open_dataset(mlTFile)
            dsMlQ = xr.open_dataset(mlQFile)
            dsMlU = xr.open_dataset(mlUFile)
            dsMlV = xr.open_dataset(mlVFile)
            newDs = xr.Dataset()
            for var in vars.keys():
                currentVar = vars.get(var)
                zCoord = currentVar.get('zCoord', None)
                varIn = currentVar.get('var')
                level = np.array(currentVar.get('level', []))
                method = currentVar.get('method', 'log')
                suffix = currentVar.get('suffix', None)
                if suffix is None:
                    suffix = [''] * level.size
                if 'derivative' not in currentVar.keys():
                    if varIn == 'T':
                        currentDataArray = dsMlT[varIn]
                    elif varIn == 'Q':
                        currentDataArray = dsMlQ[varIn]
                    elif varIn == 'U':
                        currentDataArray = dsMlU[varIn]
                    elif varIn == 'V':
                        currentDataArray = dsMlV[varIn]
                else:
                    varDeriv = currentVar.get('derivative')
                    if varDeriv == 'potential_temperature':
                        daSp = dsMlSp['SP']
                        daT = dsMlT['T']
                        daAHalf = dsMlT['a_half']
                        daBHalf = dsMlT['b_half']
                        daPHalf = daAHalf + daBHalf * daSp
                        daPHalf = daPHalf.transpose('time', ...)
                        daPModel = destagger(daPHalf, staggerDim=1).rename({'half_level': 'level'}) * mpunits.units('Pa')
                        currentDataArray = mpcalc.potential_temperature(
                            daPModel, daT
                        )
                        daPModel.close()
                        daPHalf.close()
                        daBHalf.close()
                        daAHalf.close()
                        daT.close()
                        daSp.close()
                    elif varDeriv == 'relative_humidity_from_specific_humidity':
                        daSp = dsMlSp['SP']
                        daQ = dsMlQ['Q']
                        daT = dsMlT['T']
                        daAHalf = dsMlQ['a_half']
                        daBHalf = dsMlQ['b_half']
                        daPHalf = daAHalf + daBHalf * daSp
                        daPHalf = daPHalf.transpose('time', ...)
                        daPModel = destagger(daPHalf, staggerDim=1).rename({'half_level': 'level'}) * mpunits.units('Pa')
                        currentDataArray = mpcalc.relative_humidity_from_specific_humidity(
                            daPModel, daT, daQ, phase='auto'
                        ) * mpunits.units('100%')
                        daPModel.close()
                        daPHalf.close()
                        daBHalf.close()
                        daAHalf.close()
                        daT.close()
                        daQ.close()
                        daSp.close()
                    elif varDeriv == 'ivt':
                        daSp = dsMlSp['SP']
                        daQ = dsMlQ['Q']
                        daU = dsMlU['U']
                        daV = dsMlV['V']
                        daAHalf = dsMlQ['a_half']
                        daBHalf = dsMlQ['b_half']
                        daPHalf = daAHalf + daBHalf * daSp
                        daPHalf = daPHalf.transpose('time', ...)
                        daPModel = destagger(daPHalf, staggerDim=1).rename({'half_level': 'level'}) * mpunits.units('Pa')
                        daPModel = daQ.copy(data=daPModel.to_numpy())
                        pTop = currentVar.get('pTop', 50000)
                        if 'time' in daPModel.sizes.keys():
                            daU = daU.stack(profiles=('time', 'latitude', 'longitude'))
                            daV = daV.stack(profiles=('time', 'latitude', 'longitude'))
                            daPModel = daPModel.stack(profiles=('time', 'latitude', 'longitude'))
                            daQ = daQ.stack(profiles=('time', 'latitude', 'longitude'))
                        else:
                            daU = daU.stack(profiles=('latitude', 'longitude'))
                            daV = daV.stack(profiles=('latitude', 'longitude'))
                            daPModel = daPModel.stack(profiles=('latitude', 'longitude'))
                            daQ = daQ.stack(profiles=('latitude', 'longitude'))
                        daU = daU.transpose('profiles', ...)
                        daV = daV.transpose('profiles', ...)
                        daPModel = daPModel.transpose('profiles', ...)
                        daQ = daQ.transpose('profiles', ...)
                        nLev = daQ.sizes.get('level')
                        def ivt(dataU, dataV, dataP, dataQ):
                            ivtU = 0
                            ivtV = 0
                            ivt = 0
                            for i in range(nLev - 1, 1, -1):
                                if dataP[i] > pTop:
                                    ivtU = ivtU + 1.0 / 9.8 * (
                                        0.25 * (dataQ[i] + dataQ[i-1]) * 
                                        (dataU[i] + dataU[i-1]) * 
                                        (dataP[i] - dataP[i-1]))
                                    ivtV = ivtV + 1.0 / 9.8 * (
                                        0.25 * (dataQ[i] + dataQ[i-1]) * 
                                        (dataV[i] + dataV[i-1]) * 
                                        (dataP[i] - dataP[i-1]))
                                else:
                                    break
                            ivt = np.sqrt(np.power(ivtU, 2) + np.power(ivtV, 2))
                            return ivt
                        currentDataArray = xr.apply_ufunc(
                            ivt, daU, daV, daPModel, daQ, 
                            input_core_dims=[['level'], ['level'], ['level'], ['level']],
                            vectorize=True)
                        currentDataArray = currentDataArray.unstack('profiles')
                        daPModel.close()
                        daPHalf.close()
                        daBHalf.close()
                        daAHalf.close()
                        daV.close()
                        daU.close()
                        daQ.close()
                        daSp.close()
                    elif varDeriv == 'ivt_u':
                        daSp = dsMlSp['SP']
                        daQ = dsMlQ['Q']
                        daU = dsMlU['U']
                        daAHalf = dsMlQ['a_half']
                        daBHalf = dsMlQ['b_half']
                        daPHalf = daAHalf + daBHalf * daSp
                        daPHalf = daPHalf.transpose('time', ...)
                        daPModel = destagger(daPHalf, staggerDim=1).rename({'half_level': 'level'}) * mpunits.units('Pa')
                        daPModel = daQ.copy(data=daPModel.to_numpy())
                        pTop = currentVar.get('pTop', 50000)
                        if 'time' in daPModel.sizes.keys():
                            daU = daU.stack(profiles=('time', 'latitude', 'longitude'))
                            daPModel = daPModel.stack(profiles=('time', 'latitude', 'longitude'))
                            daQ = daQ.stack(profiles=('time', 'latitude', 'longitude'))
                        else:
                            daU = daU.stack(profiles=('latitude', 'longitude'))
                            daPModel = daPModel.stack(profiles=('latitude', 'longitude'))
                            daQ = daQ.stack(profiles=('latitude', 'longitude'))
                        daU = daU.transpose('profiles', ...)
                        daPModel = daPModel.transpose('profiles', ...)
                        daQ = daQ.transpose('profiles', ...)
                        nLev = daQ.sizes.get('level')
                        def ivtU(dataU, dataP, dataQ):
                            ivtU = 0
                            for i in range(nLev - 1, 1, -1):
                                if dataP[i] > pTop:
                                    ivtU = ivtU + 1.0 / 9.8 * (
                                        0.25 * (dataQ[i] + dataQ[i-1]) * 
                                        (dataU[i] + dataU[i-1]) * 
                                        (dataP[i] - dataP[i-1]))
                                else:
                                    break
                            return ivtU
                        currentDataArray = xr.apply_ufunc(
                            ivtU, daU, daPModel, daQ, 
                            input_core_dims=[['level'], ['level'], ['level']],
                            vectorize=True)
                        currentDataArray = currentDataArray.unstack('profiles')
                        daPModel.close()
                        daPHalf.close()
                        daBHalf.close()
                        daAHalf.close()
                        daU.close()
                        daQ.close()
                        daSp.close()
                    elif varDeriv == 'ivt_v':
                        daSp = dsMlSp['SP']
                        daQ = dsMlQ['Q']
                        daV = dsMlV['V']
                        daAHalf = dsMlQ['a_half']
                        daBHalf = dsMlQ['b_half']
                        daPHalf = daAHalf + daBHalf * daSp
                        daPHalf = daPHalf.transpose('time', ...)
                        daPModel = destagger(daPHalf, staggerDim=1).rename({'half_level': 'level'}) * mpunits.units('Pa')
                        daPModel = daQ.copy(data=daPModel.to_numpy())
                        pTop = currentVar.get('pTop', 50000)
                        if 'time' in daPModel.sizes.keys():
                            daV = daV.stack(profiles=('time', 'latitude', 'longitude'))
                            daPModel = daPModel.stack(profiles=('time', 'latitude', 'longitude'))
                            daQ = daQ.stack(profiles=('time', 'latitude', 'longitude'))
                        else:
                            daV = daV.stack(profiles=('latitude', 'longitude'))
                            daPModel = daPModel.stack(profiles=('latitude', 'longitude'))
                            daQ = daQ.stack(profiles=('latitude', 'longitude'))
                        daV = daV.transpose('profiles', ...)
                        daPModel = daPModel.transpose('profiles', ...)
                        daQ = daQ.transpose('profiles', ...)
                        nLev = daQ.sizes.get('level')
                        def ivtV(dataV, dataP, dataQ):
                            ivtV = 0
                            for i in range(nLev - 1, 1, -1):
                                if dataP[i] > pTop:
                                    ivtV = ivtV + 1.0 / 9.8 * (
                                        0.25 * (dataQ[i] + dataQ[i-1]) * 
                                        (dataV[i] + dataV[i-1]) * 
                                        (dataP[i] - dataP[i-1]))
                                else:
                                    break
                            return ivtV
                        currentDataArray = xr.apply_ufunc(
                            ivtV, daV, daPModel, daQ, 
                            input_core_dims=[['level'], ['level'], ['level']],
                            vectorize=True)
                        currentDataArray = currentDataArray.unstack('profiles')
                        daPModel.close()
                        daPHalf.close()
                        daBHalf.close()
                        daAHalf.close()
                        daV.close()
                        daU.close()
                        daQ.close()
                        daSp.close()
                    else:
                        logger.error(f'Unsupported derivative variable {varDeriv}.')
                        raise Exception(f'Unsupported derivative variable {varDeriv}.')
                    
                if level.size > 0:
                    zCoord = 'level'
                    currentDataArray = currentDataArray.isel(level=level)

                if level.size > 0:
                    for ilev in range(level.size):
                        iSuffix = suffix[ilev]
                        iVarOut = f'{var}{iSuffix}'
                        newDA = currentDataArray.isel({zCoord: ilev}).astype(dtype='float32')
                        newDs[iVarOut] = newDA.drop(zCoord)
                else:
                    newDs[var] = currentDataArray.astype(dtype='float32')

                currentDataArray.close()

            if 'Time' in newDs.variables.keys():
                newDs = newDs.convert_calendar('standard', dim='Time', use_cftime=True)
                newDs = newDs.rename({'Time': 'time'})
            newDs.to_netcdf(distPath)
            newDs.close()
            dsMlSp.close()
            dsMlT.close()
            dsMlU.close()
            dsMlV.close()
        else:
            logger.warning(f'{currentBasename} exists. Skip.')
            
        gc.collect()
        return distPath

    def process(self, vars: dict,
                distDir: str,
                override: Optional[bool] = False,
                fileSuffix: str = '',
                nproc: Optional[int] = 1):
        timerStart = time.time()

        with Pool(processes=nproc) as pool:
            self.outputList_ = pool.starmap(
                self.processSingle, zip(
                    self.mlSpFileList_, self.mlTFileList_, self.mlQFileList_, 
                        self.mlUFileList_, self.mlVFileList_, 
                        repeat(vars), repeat(distDir), 
                        repeat(override), repeat(fileSuffix),
                    )
                )

        timerEnd = time.time()
        logger.warning(f'Time lasped for process step: {timerEnd - timerStart} s.')
        return self.outputList_

    def getOutputList(self):
        return self.outputList_
    
    def regridSingle(self, distDir: str, fromFilename: str | xr.Dataset, 
                     override: bool = False, fileSuffix: str = '',
                     varRename: dict = {}):
        if isinstance(fromFilename, str):
            currentBasename = os.path.basename(fromFilename)
        else:
            currentBasename = os.path.basename(fromFilename.encoding.get('source'))
        distPath = os.path.join(distDir, currentBasename)
        if len(fileSuffix) > 0:
            distPath = distPath.rsplit('.', 1)
            distPath = f'{distPath[0]}.{fileSuffix}.{distPath[1]}'
        if not os.path.exists(distPath) or override:
            regridder = self.regridder_
            if isinstance(fromFilename, str):
                ds = xr.open_dataset(fromFilename)
            else:
                ds = fromFilename
            if varRename:
                ds = ds.rename(varRename)
            newDs = regridder(ds, keep_attrs=True)
            if 'time' in newDs.variables.keys():
                newDs = newDs.convert_calendar('standard', dim='time', use_cftime=True)
            if varRename:
                for iDim, iNewDim in varRename.items():
                    if iDim in ['longitude', 'latitude']:
                        newDs = newDs.rename({iDim: iNewDim})
            newDs.to_netcdf(distPath)
            newDs.close()
            ds.close()
        return distPath
    
    def regrid(self, distDir: str, fromFiles: list, toFiles: list | str, method: str, 
               override: bool = False, fileSuffix: str = '', varRename: dict = {}, 
               regridderSuffix: str = '', nproc: int = 1):
        timerStart = time.time()
        if isinstance(toFiles, list):
            self.regridder_ = self.loadEsmfRegridder(
                distDir, fromFiles[0], toFiles[0], method, regridderSuffix, varRename)
        else:
            self.regridder_ = self.loadEsmfRegridder(
                distDir, fromFiles[0], toFiles, method, regridderSuffix, varRename)

        with Pool(processes=nproc) as pool:
            self.outputList_ = pool.starmap(
                self.regridSingle, zip(
                        repeat(distDir), fromFiles, repeat(override), 
                        repeat(fileSuffix), repeat(varRename)
                    )
                )
        timerEnd = time.time()
        logger.warning(f'Time lasped for process step: {timerEnd - timerStart} s.')
        return self.outputList_
    
    def loadEsmfRegridder(self, distDir: str, fromFilename: str | xr.Dataset, toFilename: str, 
                          method: str, fileSuffix: str = '', varRename: dict = {}):
        if len(fileSuffix) == 0:
            regridderPath = os.path.join(distDir, f'regridder.{method}.nc')
        else:
            regridderPath = os.path.join(distDir, f'regridder.{method}.{fileSuffix}.nc')
        if isinstance(fromFilename, str):
            ds = xr.open_dataset(fromFilename)
        else:
            ds = fromFilename
        if varRename:
            ds = ds.rename(varRename)
        dsDist = xr.open_dataset(toFilename)
        if os.path.exists(regridderPath):
            regridder = xe.Regridder(ds, dsDist, method, weights=regridderPath)
        else:
            regridder = xe.Regridder(ds, dsDist, method)
            regridder.to_netcdf(filename=regridderPath)
        ds.close()
        dsDist.close()
        return regridder