import gc
from functools import partial
import logging
from multiprocessing import Pool
import os
from typing import Optional
import time
import warnings

import numpy as np
import metpy.calc as mpcalc
import metpy.units as mpunits
import xarray as xr
import xgcm
import xwrf

from Preprocess.Model import (
    calculateSeaLevelPressure, 
    olr2BrightnessTemperature,
    rttovBrightnessTemperature,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PreprocessWRF:
    def __init__(self, files: list, wrf2dFiles: list = None):
        self.fileList_ = files
        self.wrf2dList_ = wrf2dFiles
        self.currentDs_ = None
        self.invariantPath_ = None
        self.invariantDs_ = None
        self.mapFactorX_ = None
        self.mapFactorY_ = None
        self.outputList_ = []

    def setInvariantPath(self, filename: str):
        self.invariantPath_ = filename

    def findWrf2dFilename(self, suffix: str):
        if self.wrf2dList_:
            for i in self.wrf2dList_:
                if i.endswith(suffix):
                    return i
        return None
    
    def processSingle(
            self, filename: str,
            vars: dict,
            distDir: str,
            varRename: Optional[dict] = {},
            varUnitChange: Optional[dict] = {},
            override: Optional[bool] = False,
            fileSuffix: str = ''):
        currentBasename = os.path.basename(filename)
        distPath = os.path.join(distDir, currentBasename)
        currentTimeStr = currentBasename.split('d01_')[-1]
        wrf2dDistPath = self.findWrf2dFilename(currentTimeStr)
        wrf2dDistPath = os.path.join(distDir, wrf2dDistPath) if wrf2dDistPath is not None else None
        if len(fileSuffix) > 0:
            distPath = distPath.rsplit('.', 1)
            distPath = f'{distPath[0]}{fileSuffix}.{distPath[1]}'
        if not os.path.exists(distPath) or override:
            ds = xr.open_dataset(filename)
            if varRename:
                ds = ds.rename(varRename)
            if varUnitChange:
                for var, val in varUnitChange.items():
                    ds[var].attrs['units'] = val
            ds = ds.xwrf.postprocess()
            newDs = xr.Dataset()
            destaggered = ds.xwrf.destagger()
            
            if wrf2dDistPath is not None:
                ds2d = xr.open_dataset(wrf2dDistPath)
                ds2d = ds2d.xwrf.postprocess()
                destaggered2d = ds2d.xwrf.destagger()
            
            grid = xgcm.Grid(destaggered, periodic=False)
            for var in vars.keys():
                currentVar = vars.get(var)
                zCoord = currentVar.get('zCoord', None)
                varIn = currentVar.get('var')
                level = np.array(currentVar.get('level', []))
                method = currentVar.get('method', 'log')
                suffix = currentVar.get('suffix', None)
                if suffix is None:
                    suffix = [''] * level.size
                zDataArray = destaggered[zCoord] if zCoord is not None else None
                if 'derivative' not in currentVar.keys():
                    currentDataArray = destaggered[varIn]
                else:
                    varDeriv = currentVar.get('derivative')
                    if varDeriv == 'geopotential':
                        currentDataArray = mpcalc.height_to_geopotential(
                            destaggered[varIn])
                    elif varDeriv == 'vo':
                        varInU = varIn[0]
                        varInV = varIn[1]
                        currentU = destaggered[varInU]
                        currentV = destaggered[varInV]
                        currentDataArray = mpcalc.vorticity(
                            currentU, currentV, 
                            parallel_scale=self.mapFactorX_,
                            meridional_scale=self.mapFactorY_)
                        
                    elif varDeriv == 'relative_humidity_from_mixing_ratio':
                        varInP = varIn[0]
                        varInT = varIn[1]
                        varInQ = varIn[2]
                        unitT = destaggered[varInT].units
                        if unitT == 'deg K':
                            currentDataArray = mpcalc.relative_humidity_from_mixing_ratio(
                                destaggered[varInP], destaggered[varInT]*mpunits.units('K'), 
                                destaggered[varInQ], phase='auto'
                            ) * mpunits.units('100%')
                        else:
                            currentDataArray = mpcalc.relative_humidity_from_mixing_ratio(
                                destaggered[varInP], destaggered[varInT], 
                                destaggered[varInQ], phase='auto'
                            ) * mpunits.units('100%')
                        
                    elif varDeriv == 'frontogenesis_petterssen':
                        varInT = varIn[0]
                        varInP = varIn[1]
                        varInU = varIn[2]
                        varInV = varIn[3]
                        currentDataArray = mpcalc.frontogenesis(
                            mpcalc.potential_temperature(
                                destaggered[varInP], destaggered[varInT]),
                            destaggered[varInU], destaggered[varInV],
                            # dx=destaggered.metpy.grid_deltas['dx'],
                            # dy=destaggered.metpy.grid_deltas['dy'],
                            parallel_scale=self.mapFactorX_,
                            meridional_scale=self.mapFactorY_
                        )
                    elif varDeriv == 'potential_temperature':
                        varInT = varIn[0]
                        varInP = varIn[1]
                        unitT = destaggered[varInT].units
                        if unitT == 'deg K':
                            currentDataArray = mpcalc.potential_temperature(
                                destaggered[varInP], destaggered[varInT]*mpunits.units('K')
                            )
                        else:
                            currentDataArray = mpcalc.potential_temperature(
                                destaggered[varInP], destaggered[varInT]
                            )
                    elif varDeriv == 'equivalent_potential_temperature':
                        varInT = varIn[0]
                        varInP = varIn[1]
                        currentDataArray = mpcalc.equivalent_potential_temperature(
                            destaggered[varInP], destaggered[varInT],
                            mpcalc.dewpoint_from_specific_humidity(
                                destaggered[varInP], destaggered[varInT],
                                mpcalc.specific_humidity_from_mixing_ratio(
                                    destaggered[varInQ]
                                )
                            )
                        )
                    elif varDeriv == 'theta_gradient_mag':
                        varInT = varIn[0]
                        varInP = varIn[1]
                        currentDataArray = destaggered[varInT].copy()
                        gradX, gradY = mpcalc.geospatial_gradient(
                            mpcalc.potential_temperature(
                                destaggered[varInP], destaggered[varInT]),
                            parallel_scale=self.mapFactorX_,
                            meridional_scale=self.mapFactorY_
                        )
                        currentDataArray[:] = np.sqrt(gradX**2 + gradY**2)
                    elif varDeriv == 'theta_e_gradient_mag':
                        varInT = varIn[0]
                        varInP = varIn[1]
                        varInQ = varIn[2]
                        currentDataArray = destaggered[varInT].copy()
                        gradX, gradY = mpcalc.geospatial_gradient(
                            mpcalc.equivalent_potential_temperature(
                                destaggered[varInP], destaggered[varInT],
                                mpcalc.dewpoint_from_specific_humidity(
                                    destaggered[varInP], destaggered[varInT],
                                    mpcalc.specific_humidity_from_mixing_ratio(
                                        destaggered[varInQ]
                                    )
                                )),
                            parallel_scale=self.mapFactorX_,
                            meridional_scale=self.mapFactorY_
                        )
                        currentDataArray[:] = np.sqrt(gradX**2 + gradY**2)
                    elif varDeriv == 'slp':
                        varInZ = varIn[0]
                        varInT = varIn[1]
                        varInP = varIn[2]
                        varInMixingRatio = varIn[3]
                        currentDataArray = destaggered[varInP].isel(z=0).copy()
                        currentDataArray[:] = calculateSeaLevelPressure(
                            destaggered[varInZ], 
                            destaggered[varInT], 
                            destaggered[varInP], 
                            destaggered[varInMixingRatio]
                        )
                    elif varDeriv == 'tb_from_olr':
                        if wrf2dDistPath is not None:
                            currentDataArray = olr2BrightnessTemperature(destaggered2d[varIn])
                        else:
                            currentDataArray = olr2BrightnessTemperature(destaggered[varIn])
                    elif varDeriv == 'tb_from_rttov':
                        varTerrianHeight = currentVar.get('terrianHeight')
                        varSurfaceType = currentVar.get('surfaceType')
                        varLon = currentVar.get('lon')
                        varLat = currentVar.get('lat')
                        channel = currentVar.get('channel')
                        dataTerrianHeight = None
                        dataSurfaceType = None
                        if wrf2dDistPath:
                            if varTerrianHeight in destaggered2d.variables.keys():
                                dataTerrianHeight = destaggered2d[varTerrianHeight]
                            if varSurfaceType in destaggered2d.variables.keys():
                                dataSurfaceType = destaggered2d[varSurfaceType]
                        else:
                            if varTerrianHeight in destaggered.variables.keys():
                                dataTerrianHeight = destaggered[varTerrianHeight]
                            if varSurfaceType in destaggered.variables.keys():
                                dataSurfaceType = destaggered[varSurfaceType]
                        if dataTerrianHeight is None:
                            dataTerrianHeight = self.invariantDs_[varTerrianHeight]
                        if dataSurfaceType is None:
                            dataSurfaceType = self.invariantDs_[varSurfaceType]
                        dataLon = destaggered[varLon]
                        dataLat = destaggered[varLat]

                        varInT = varIn[0]
                        varInP = varIn[1]
                        varInQVapor = varIn[2]
                        varInQIce = varIn[3]
                        varInQSnow = varIn[4]
                        varInQCloud = varIn[5]
                        varInCloudFrac = varIn[6]
                        varInTSkin = varIn[7]
                        varInSnowCover = varIn[8]
                        varInT2 = varIn[9]
                        varInQ2 = varIn[10]
                        varInU10 = varIn[11]
                        varInV10 = varIn[12]
                        varInAlbedo = varIn[13]
                        varInEmissivity = varIn[14]
                        if len(varIn) == 16:
                            varW = varIn[15]
                        else:
                            varW = None
                        
                        dataT = destaggered[varInT]
                        dataP = destaggered[varInP]
                        dataQVapor = destaggered[varInQVapor]
                        dataQIce = destaggered[varInQIce]
                        dataQSnow = destaggered[varInQSnow]
                        dataQCloud = destaggered[varInQCloud]
                        dataCloudFrac = destaggered[varInCloudFrac]
                        dataW = destaggered[varW] if varW is not None else None
                        if wrf2dDistPath is not None:
                            dataTSkin = destaggered2d[varInTSkin]
                            dataSnowCover = destaggered2d[varInSnowCover]
                            dataT2 = destaggered2d[varInT2]
                            dataQ2 = destaggered2d[varInQ2]
                            dataU10 = destaggered2d[varInU10]
                            dataV10 = destaggered2d[varInV10]
                            dataAlbedo = destaggered2d[varInAlbedo]
                            dataEmissivity = destaggered2d[varInEmissivity]
                        else:
                            dataTSkin = destaggered[varInTSkin]
                            dataSnowCover = destaggered[varInSnowCover]
                            dataT2 = destaggered[varInT2]
                            dataQ2 = destaggered[varInQ2]
                            dataU10 = destaggered[varInU10]
                            dataV10 = destaggered[varInV10]
                            dataAlbedo = destaggered[varInAlbedo]
                            dataEmissivity = destaggered[varInEmissivity]
                        currentDataArray = rttovBrightnessTemperature(
                            dataLon, dataLat, dataTerrianHeight, dataSurfaceType,
                            dataT, dataP, dataQVapor, dataQIce, dataQSnow, dataQCloud,
                            dataCloudFrac, dataTSkin, dataSnowCover, 
                            dataT2, dataQ2, dataU10, dataV10, dataAlbedo, dataEmissivity,
                            irChannel=channel, w=dataW)
                    elif varDeriv == 'ivt':
                        varInU = varIn[0]
                        varInV = varIn[1]
                        varInP = varIn[2]
                        varInMixingRatio = varIn[3]
                        pTop = currentVar.get('pTop', 50000)
                        dataQ = mpcalc.specific_humidity_from_mixing_ratio(
                            destaggered[varInMixingRatio]
                        )
                        dataU = destaggered[varInU]
                        dataV = destaggered[varInV]
                        dataP = destaggered[varInP]
                        nLev = destaggered.sizes.get('z')
                        if 'Time' in dataP.sizes.keys():
                            dataU = dataU.stack(profiles=('Time', 'y', 'x'))
                            dataV = dataV.stack(profiles=('Time', 'y', 'x'))
                            dataP = dataP.stack(profiles=('Time', 'y', 'x'))
                            dataQ = dataQ.stack(profiles=('Time', 'y', 'x'))
                        else:
                            dataU = dataU.stack(profiles=('y', 'x'))
                            dataV = dataV.stack(profiles=('y', 'x'))
                            dataP = dataP.stack(profiles=('y', 'x'))
                            dataQ = dataQ.stack(profiles=('y', 'x'))
                        dataU = dataU.transpose('profiles', ...)
                        dataV = dataV.transpose('profiles', ...)
                        dataP = dataP.transpose('profiles', ...)
                        dataQ = dataQ.transpose('profiles', ...)
                        
                        def ivt(dataU, dataV, dataP, dataQ):
                            ivtU = 0
                            ivtV = 0
                            ivt = 0
                            for i in range(nLev - 1):
                                if dataP[i] > pTop:
                                    ivtU = ivtU + 1.0 / 9.8 * (
                                        0.25 * (dataQ[i] + dataQ[i+1]) * 
                                        (dataU[i] + dataU[i+1]) * 
                                        (dataP[i] - dataP[i+1]))
                                    ivtV = ivtV + 1.0 / 9.8 * (
                                        0.25 * (dataQ[i] + dataQ[i+1]) * 
                                        (dataV[i] + dataV[i+1]) * 
                                        (dataP[i] - dataP[i+1]))
                                else:
                                    break
                            ivt = np.sqrt(np.power(ivtU, 2) + np.power(ivtV, 2))
                            return ivt
                        currentDataArray = xr.apply_ufunc(
                            ivt, dataU, dataV, dataP, dataQ, 
                            input_core_dims=[['z'], ['z'], ['z'], ['z']],
                            vectorize=True)
                        currentDataArray = currentDataArray.unstack('profiles')
                        dataU.close()
                        dataV.close()
                        dataP.close()
                        dataQ.close()
                    elif varDeriv == 'ivt_u':
                        varInU = varIn[0]
                        varInV = varIn[1]
                        varInP = varIn[2]
                        varInMixingRatio = varIn[3]
                        pTop = currentVar.get('pTop', 50000)
                        dataQ = mpcalc.specific_humidity_from_mixing_ratio(
                            destaggered[varInMixingRatio]
                        )
                        dataU = destaggered[varInU]
                        dataP = destaggered[varInP]
                        nLev = destaggered.sizes.get('z')
                        if 'Time' in dataP.sizes.keys():
                            dataU = dataU.stack(profiles=('Time', 'y', 'x'))
                            dataP = dataP.stack(profiles=('Time', 'y', 'x'))
                            dataQ = dataQ.stack(profiles=('Time', 'y', 'x'))
                        else:
                            dataU = dataU.stack(profiles=('y', 'x'))
                            dataP = dataP.stack(profiles=('y', 'x'))
                            dataQ = dataQ.stack(profiles=('y', 'x'))
                        dataU = dataU.transpose('profiles', ...)
                        dataP = dataP.transpose('profiles', ...)
                        dataQ = dataQ.transpose('profiles', ...)
                        
                        def ivtU(dataU, dataP, dataQ):
                            ivtU = 0
                            for i in range(nLev - 1):
                                if dataP[i] > pTop:
                                    ivtU = ivtU + 1.0 / 9.8 * (
                                        0.25 * (dataQ[i] + dataQ[i+1]) * 
                                        (dataU[i] + dataU[i+1]) * 
                                        (dataP[i] - dataP[i+1]))
                                else:
                                    break
                            return ivtU
                        currentDataArray = xr.apply_ufunc(
                            ivtU, dataU, dataP, dataQ, 
                            input_core_dims=[['z'], ['z'], ['z']],
                            vectorize=True)
                        currentDataArray = currentDataArray.unstack('profiles')
                        dataU.close()
                        dataP.close()
                        dataQ.close()
                    elif varDeriv == 'ivt_v':
                        varInU = varIn[0]
                        varInV = varIn[1]
                        varInP = varIn[2]
                        varInMixingRatio = varIn[3]
                        pTop = currentVar.get('pTop', 50000)
                        dataQ = mpcalc.specific_humidity_from_mixing_ratio(
                            destaggered[varInMixingRatio]
                        )
                        dataV = destaggered[varInV]
                        dataP = destaggered[varInP]
                        nLev = destaggered.sizes.get('z')
                        if 'Time' in dataP.sizes.keys():
                            dataV = dataV.stack(profiles=('Time', 'y', 'x'))
                            dataP = dataP.stack(profiles=('Time', 'y', 'x'))
                            dataQ = dataQ.stack(profiles=('Time', 'y', 'x'))
                        else:
                            dataV = dataV.stack(profiles=('y', 'x'))
                            dataP = dataP.stack(profiles=('y', 'x'))
                            dataQ = dataQ.stack(profiles=('y', 'x'))
                        dataV = dataV.transpose('profiles', ...)
                        dataP = dataP.transpose('profiles', ...)
                        dataQ = dataQ.transpose('profiles', ...)
                        
                        def ivtV(dataV, dataP, dataQ):
                            ivtV = 0
                            for i in range(nLev - 1):
                                if dataP[i] > pTop:
                                    ivtV = ivtV + 1.0 / 9.8 * (
                                        0.25 * (dataQ[i] + dataQ[i+1]) * 
                                        (dataV[i] + dataV[i+1]) * 
                                        (dataP[i] - dataP[i+1]))
                                else:
                                    break
                            return ivtV
                        currentDataArray = xr.apply_ufunc(
                            ivtV, dataV, dataP, dataQ, 
                            input_core_dims=[['z'], ['z'], ['z']],
                            vectorize=True)
                        currentDataArray = currentDataArray.unstack('profiles')
                        dataV.close()
                        dataP.close()
                        dataQ.close()
                    else:
                        logger.error(f'Unsupported derivative variable {varDeriv}.')
                        raise Exception(f'Unsupported derivative variable {varDeriv}.')
                    
                if level.size > 0:
                    if zCoord is not None:
                        with warnings.catch_warnings(action='ignore', category=FutureWarning):
                            currentDataArray = grid.transform(
                                currentDataArray.metpy.dequantify(), 'Z',
                                level, target_data=zDataArray, method=method) 
                        currentDataArray = currentDataArray.compute()
                    else:
                        zCoord = 'z'
                        currentDataArray = currentDataArray.isel(z=level)

                if level.size > 0:
                    for ilev in range(level.size):
                        iSuffix = suffix[ilev]
                        iVarOut = f'{var}{iSuffix}'
                        newDA = currentDataArray.isel({zCoord: ilev}).astype(dtype='float32')
                        newDs[iVarOut] = newDA.drop(zCoord)
                elif varDeriv == 'tb_from_rttov':
                    channel = currentVar.get('channel')
                    if isinstance(channel, list):
                        if len(channel) > 1:
                            for ich in range(len(channel)):
                                iSuffix = suffix[ich]
                                iVarOut = f'{var}{iSuffix}'
                                newDA = currentDataArray.isel({'z': ich}).astype(dtype='float32')
                                newDs[iVarOut] = newDA
                        else:
                            iSuffix = suffix[0]
                            iVarOut = f'{var}{iSuffix}'
                            newDA = currentDataArray.astype(dtype='float32')
                            newDs[iVarOut] = newDA
                    else:
                        newDs[var] = currentDataArray.astype(dtype='float32')
                else:
                    newDs[var] = currentDataArray.astype(dtype='float32')

                currentDataArray.close()

            if 'Time' in newDs.variables.keys():
                newDs = newDs.convert_calendar('standard', dim='Time', use_cftime=True)
                newDs = newDs.rename({'Time': 'time'})
            newDs.to_netcdf(distPath)
            newDs.close()
            destaggered.close()
            if wrf2dDistPath is not None:
                destaggered2d.close()
                ds2d.close()
            ds.close()
        else:
            logger.warning(f'{currentBasename} exists. Skip.')
        
        gc.collect()
        return distPath

    def process(self, vars: dict,
                distDir: str,
                requireMapFactor: Optional[bool] = True,
                varRename: Optional[dict] = {},
                varUnitChange: Optional[dict] = {},
                override: Optional[bool] = False,
                fileSuffix: str = '',
                nproc: Optional[int] = 1):
        '''
        Args:
            vars (dict): A dictionary contains all information of the variables
                to be processed. zCoord is the name of the vertical coordinate 
                used as reference. var is the variable name in the WRF files.
                levels is a list contains all levels to be interpolated to.
                Derivative variables are also supported, such as relative 
                vorticity.
                For example,
                dict(
                    Z = dict(
                        zCoord = 'P',
                        var = 'HGT',
                        level = [925, 850, 700, 500, 300],
                        method = 'log',
                    ),
                    Vo = dict(
                        derivative = 'vo',
                        zCoord = 'P',
                        var = ['U', 'V'],
                        level = [500],
                        mapFacX = 'MAPFAC_MX',
                        mapFacY = 'MAPFAC_MY',
                        method = 'log',
                    )
                )

        '''
        self.vars_ = vars
        needMapFactor = False
        needTerrianHeight = False
        needSurfaceType = False
        needMesh = False
        varNeedMapFactor = ['vo', 'div']
        varOutList = vars.keys()
        varMapFacX = None
        varMapFacY = None
        timerStart = time.time()

        for i in varOutList:
            currentVar = vars.get(i)
            if 'derivative' in currentVar.keys():
                if currentVar.get('derivative') in varNeedMapFactor:
                    needMapFactor = True
                    varMapFacX = currentVar.get('mapFacX')
                    varMapFacY = currentVar.get('mapFacY')
            if 'terrianHeight' in currentVar.keys():
                needTerrianHeight = True
            if 'surfaceType' in currentVar.keys():
                needSurfaceType = True

        if self.invariantPath_ and (
            needMapFactor or needTerrianHeight or needMesh or needSurfaceType):
            self.invariantDs_ = xr.open_dataset(self.invariantPath_).xwrf.postprocess()
            if (varMapFacX in self.invariantDs_.variables.keys()):
                self.mapFactorX_ = self.invariantDs_[varMapFacX].isel(Time=0)
                self.mapFactorY_ = self.invariantDs_[varMapFacY].isel(Time=0)
        else:
            if requireMapFactor:
                logger.error('Cannot find map factor.')
                raise Exception('Cannot find map factor.')

        with Pool(processes=nproc) as pool:
            self.outputList_ = pool.map(
                partial(self.processSingle, vars=vars, distDir=distDir, 
                        varRename=varRename, varUnitChange=varUnitChange,
                        override=override, fileSuffix=fileSuffix), self.fileList_)

        timerEnd = time.time()
        logger.warning(f'Time lasped for process step: {timerEnd - timerStart} s.')
        return self.outputList_

    def getOutputList(self):
        return self.outputList_