import datetime
import logging
import os

import geopandas
import numpy as np
import polars as pl
import xarray as xr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

varMap = dict(
    core_id='coreId',
    core_lon='coreLon',
    core_lat='coreLat',
    storm_lon='stormLon',
    storm_lat='stormLat',
    core_area='coreArea',
    core_rain_rate_mean='coreRainRateMean',
    core_rain_rate_max='coreRainRateMax',
    core_top_ht='coreTopHeight',
    core_ter_ht='coreTerrianHeight',
    core_rain_pix_all='coreRainPixAll',
    core_rain_pix_stra='coreRainPixStra',
    core_rain_pix_conv='coreRainPixConv',
)

class PreprocessGPMStormModeTabular:
    def __init__( self, 
            files: list, regionCodeMap: dict = None, 
            geoDf: geopandas.GeoDataFrame = None):
        self.fileList_ = files
        self.regionCodeMap_ = regionCodeMap
        self.geoDf_ = geoDf
        self.df_ = None

    def getVarMap(self):
        return varMap

    def process(self, filename: str, vars: dict, override: bool = False):
        if os.path.exists(filename) and not override:
            if os.path.splitext(filename)[1] == 'parquet':
                self.df_ = pl.read_parquet(filename)
            else:
                self.df_ = pl.read_csv(filename)
            varsNew = [varMap[var] for var in vars]
            varsExisted = self.df_.columns
            if set(varsNew) <= set(varsExisted):
                return self.df_
            
        ds = xr.open_mfdataset(self.fileList_, concat_dim='case', combine='nested')
        orbit = ds['orbit'].to_numpy().astype('U6')
        nCase = len(orbit)
        coreId = ds['core_id'].to_numpy().astype(int)
        stormId = np.zeros(np.shape(coreId), dtype=int)
        coreLon = ds['core_lon'].to_numpy()
        coreLat = ds['core_lat'].to_numpy()
        stormLon = ds['storm_lon'].to_numpy()
        stormLat = ds['storm_lat'].to_numpy()
        coreData = ds['date'].to_numpy()
        coreTime = ds['time'].to_numpy()
        coreDatetime = coreData + coreTime
        coreDatetime = np.array([datetime.datetime.strptime(i.astype('U14'), '%Y%m%d%H%M%S') for i in coreDatetime], dtype=np.datetime64)
        
        currentStormId = -1
        lastStormLon = -1
        lastStormLat = -1
        for i in range(nCase):
            if (stormLon[i] != lastStormLon) and (stormLat[i] != lastStormLat):
                currentStormId += 1
                lastStormLon = stormLon[i]
                lastStormLat = stormLat[i]
            stormId[i] = currentStormId

        if self.df_ is None:
            self.df_ = pl.DataFrame()
            self.df_ = self.df_.with_columns(
                orbit=orbit,
                coreId=coreId,
                stormId=stormId,
                coreDatetime=coreDatetime,
                coreLon=coreLon,
                coreLat=coreLat,
                stormLon=stormLon,
                stormLat=stormLat,
            )
            if self.regionCodeMap_ and self.geoDf_ is not None:
                corePoints = geopandas.GeoSeries.from_xy(coreLon, coreLat, crs='EPSG:4326')
                stormPoints = geopandas.GeoSeries.from_xy(stormLon, stormLat, crs='EPSG:4326')
                coreRegion = np.ones(np.shape(coreLon), dtype=int) * -1
                stormRegion = np.ones(np.shape(coreLon), dtype=int) * -1
                for region, code in self.regionCodeMap_.items():
                    pointsContain = np.array(corePoints.apply(
                        lambda x: self.geoDf_[self.geoDf_['RegionName']==region]
                        .geometry.contains_properly(x)).values).flatten()
                    coreRegion[pointsContain] = code

                    pointsContain = np.array(stormPoints.apply(
                        lambda x: self.geoDf_[self.geoDf_['RegionName']==region]
                        .geometry.contains_properly(x)).values).flatten()
                    stormRegion[pointsContain] = code

                self.df_ = self.df_.with_columns(
                    coreRegion=coreRegion,
                    stormRegion=stormRegion,
                )

        for var in vars:
            self.df_ = self.df_.with_columns(
                pl.Series(varMap[var], ds[var].to_numpy())
            )

        if os.path.splitext(filename)[1] == 'parquet':
            self.df_.write_parquet(filename)
        else:
            self.df_.write_csv(filename)

        return self.df_
        