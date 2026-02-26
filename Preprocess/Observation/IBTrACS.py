import datetime
import os

import numpy as np
import polars as pl
import xarray as xr

class PreprocessIBTrACS:
    def __init__(self, file):
        self.file_ = file
        self.df_ = None

    def process(self, filename: str, 
                beginTime: datetime.datetime,
                endTime: datetime.datetime,
                override: bool = False):
        if os.path.exists(filename) and not override:
            if os.path.splitext(filename)[1] == 'parquet':
                self.df_ = pl.read_parquet(filename, try_parse_dates=True)
            else:
                self.df_ = pl.read_csv(filename, try_parse_dates=True)

        else:
            ds = xr.open_dataset(self.file_)

            nStorm = ds.dims.get('storm')
            nDateTime = ds.dims.get('date_time')

            schema = {
                'Sid': str,
                'Lon': pl.Float32,
                'Lat': pl.Float32,
                'Label': str,
                'UTC': datetime.datetime,
                # 'IFlag': str,
                'Basin': str,
                'Nature': str,
                'USAStatus': str,
            }
            self.df_ = pl.DataFrame({
                'Sid': [],
                'Lon': [],
                'Lat': [],
                'Label': [],
                'UTC': [],
                # 'IFlag': [],
                'Basin': [],
                'Nature': [],
                'USAStatus': []}, schema=schema)
            
            for i in range(nStorm):
                nObs = ds['numobs'].isel(storm=i).to_numpy().astype(int)
                if nObs >= nDateTime:
                    print()
                iSid = ds['sid'].isel(storm=i).to_numpy().astype('U13')
                iSid = np.repeat(iSid, nObs)
                iLon = ds['lon'].isel(storm=i, date_time=slice(0, nObs)).to_numpy()
                iLat = ds['lat'].isel(storm=i, date_time=slice(0, nObs)).to_numpy()
                iTime = ds['time'].isel(storm=i, date_time=slice(0, nObs)).to_numpy().astype(
                    'datetime64[s]').astype('datetime64[ms]')
                iLabel = ds['nature'].isel(storm=i, date_time=slice(0, nObs)).to_numpy().astype('U2')
                iBasin = ds['basin'].isel(storm=i, date_time=slice(0, nObs)).to_numpy().astype('U2')
                iNature = ds['nature'].isel(storm=i, date_time=slice(0, nObs)).to_numpy().astype('U2')
                iUSAStatus = ds['usa_status'].isel(storm=i, date_time=slice(0, nObs)).to_numpy().astype('U2')
                iDf = pl.DataFrame({
                    'Sid': iSid,
                    'Lon': iLon,
                    'Lat': iLat,
                    'Label': iLabel,
                    'UTC': iTime,
                    # 'IFlag': [],
                    'Basin': iBasin,
                    'Nature': iNature,
                    'USAStatus': iUSAStatus}, schema=schema)
                self.df_ = pl.concat([self.df_, iDf])
            
        if beginTime is not None:
            self.df_ = self.df_.filter(pl.col('UTC') >= beginTime)
        if endTime is not None:
            self.df_ = self.df_.filter(pl.col('UTC') <= endTime)

        if not os.path.exists(filename) or override:
          if os.path.splitext(filename)[1] == 'parquet':
              self.df_.write_parquet(filename)
          else:
              self.df_.write_csv(filename)

        return self.df_