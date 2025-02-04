'''
Details of SyCLoPS algorithms can be found at 
https://doi.org/10.1029/2024JD041287
'''
import logging
import os
import math
from multiprocessing import Pool
import time
from typing import Optional

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from scipy import stats
import xarray as xr

from Base import geo2XYZOnUnitSphere

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SyCLoPSClassifier():
    def __init__(self,
                 stichNodesOutput: str, blobStatsOutput: str,
                 z0File: str, nproc: int = 4, timeInterval: int = 3):
        self.trackFile_ = stichNodesOutput
        self.blobStatsFile_ = blobStatsOutput
        self.z0File_ = z0File
        self.nproc_ = nproc
        self.timeInterval_ = timeInterval
        self.tracks_ = None
        self.nNode_ = -1
        self.z0_ = None
        self.blobStats_ = None
        self.trackStartLocs_ = None
        self.loadData()

    def loadData(self) -> None:
        self.loadStichNodesOutput()
        self.z0_ = xr.load_dataset(self.z0File_)
        self.loadBlobStatsOutput()

    def loadStichNodesOutput(self) -> None:
        '''
        TID: Track ID begining from 0
        '''
        tid = []
        ntrack = -1
        trackRecords = []
        with open(self.trackFile_) as f:
            trackRecords = f.readlines()
        nline = len(trackRecords)
        for i in range(nline):
            if 'start' in trackRecords[i]:
                ntrack += 1
            else:
                tid.append(ntrack)
        tid = np.array(tid)
        self.tracks_ = pl.read_csv(
            self.trackFile_, separator='\t', has_header=False, comment_prefix='start')
        self.tracks_.columns = [
            'TID', 'NodeX', 'NodeY', 'Lon', 'Lat',
            # Mean sea level pressure
            'MSLP',
            # Maximum 10m wind speed within 2.0 GCD
            'WS10',
            # Greatest positive closed contour delta of MSLP over a 2.0 GCD 
            'MSLPCC20',
            # Greatest positive closed contour delta of MSLP over a 5.5 GCD 
            'MSLPCC55',
            # Average environmental deep-layer (200 hPa - 850 hPa) wind shear 
            # over a 10.0 GCD
            'DeepShear',
            # Greatest decline of the upper-level (300 - 500 hPa) geopotential 
            # thickness within a 6.5 GCD of the maximum thickness node within 
            # a 1.0 GCD of the current node
            'UppThkCC',
            # Greatest decline of the mid-level (500 - 700 hPa) geopotential 
            # thickness within a 3.5 GCD of the maximum thickness node within 
            # a 1.0 GCD of the current node
            'MidThkCC',
            # Greatest decline of the lower-level (700 - 925 hPa) geopotential 
            # thickness within a 3.5 GCD of the maximum thickness node within 
            # a 1.0 GCD of the current node
            'LowThkCC',
            # Greatest increase of the 500 hPa geopotential within a 3.5 GCD 
            # of the minimum geopotential node within a 1.0 GCD of the current node
            'Z500CC',
            # Avarage relative vorticity over a 2.5 GCD
            'Vo500Avg',
            # Maximum 100 hPa relative humidity within a 2.5 GCD
            'RH100Max',
            # Avarage 850 hPa relative humidity within a 2.5 GCD
            'RH850Avg',
            # 850 hPa air temperature at the node
            'T850',
            # 850 hPa geopotential at the node
            'Z850',
            # Surface geopotential at the node
            'Z0',
            # Difference between the weighted area mean of positive and negative 
            # values of 850 hPa eastward wind over a 5.5 GCD 
            'U850Diff',
            # Maximun poleward 200 hPa wind speed within a 1.0 GCD
            'WS200PMax',
            'Year',
            'Month',
            'Day',
            'Hour'
        ]
        self.tracks_ = self.tracks_.with_columns(TID=tid)

        self.tracks_ = self.tracks_.with_columns(
            pl.datetime('Year', 'Month', 'Day', 'Hour').alias('UTC'))

        newColums = [
            'TID', 'Lat', 'Lon', 'UTC',
            'MSLP', 'WS10', 'MSLPCC20', 'MSLPCC55', 'DeepShear',
            'UppThkCC', 'MidThkCC', 'LowThkCC', 'Z500CC', 'Vo500Avg',
            'RH100Max', 'RH850Avg', 'T850', 'Z850', 'Z0', 'U850Diff', 'WS200PMax']
        self.tracks_ = self.tracks_[newColums]
        self.tracks_ = self.tracks_.with_columns(Area=0.0)
        self.tracks_ = self.tracks_.with_columns(IKE=0.0)
        self.tracks_ = self.tracks_.with_columns(TrackSpread=0.0)
        self.tracks_ = self.tracks_.with_columns(TrackCorrelation=0.0)
        self.tracks_ = self.tracks_.with_columns(Z0Max=0.0)
        self.tracks_ = self.tracks_.with_columns(InlandRatio=0.0)
        self.tracks_ = self.tracks_.with_row_index()
        self.nNode_ = tid.size

    def loadBlobStatsOutput(self) -> None:
        self.blobStats_ = pl.read_csv(
            self.blobStatsFile_, separator='\t', has_header=False)
        self.blobStats_ = self.blobStats_.drop(
            'column_2')
        self.blobStats_.columns = [
            'BID', 'Time', 'CenLon', 'CenLat',
            'MinLat', 'MaxLat', 'MinLon', 'MaxLon', 'Area',
            'IKE'
        ]
        self.blobStats_ = self.blobStats_.with_columns(
            pl.col('Time').str.to_datetime(
                format='%Y-%m-%d %H:%M:%S').alias('UTC'))
        self.blobStats_ = self.blobStats_.drop('Time')
        self.blobStats_ = self.blobStats_.with_columns(PairedNode=-1)

    def blobPairing(self, k):
        nodePair = []
        currentTrack = self.tracks_.filter(pl.col('UTC') == k)
        lonTrack = currentTrack['Lon']
        latTrack = currentTrack['Lat']
        trackPoints = geo2XYZOnUnitSphere(lonTrack, latTrack)
        currentBlob = self.blobStats_.filter(pl.col('UTC') == k)
        lonBlob = currentBlob['CenLon']
        latBlob = currentBlob['CenLat']
        blobPoints = geo2XYZOnUnitSphere(lonBlob, latBlob)
        trackTree = cKDTree(trackPoints)
        for i in range(len(blobPoints)):
            idx = trackTree.query_ball_point(blobPoints[i], r=5*np.pi/180.0)
            if idx:
                finalIdx = idx[currentTrack[idx, 'MSLP'].arg_min()]
                nodeIdx = currentTrack[finalIdx, 'index']
                nodePair.append(
                    (currentBlob[i, 'BID'], nodeIdx))
            else:
                if (currentBlob[i, 'MaxLon'] - currentBlob[i, 'MinLon'] > 180.0):
                    idxApprox = currentTrack.with_row_index(name='indexA').filter(
                        ((pl.col('Lon') >= 350.0) & (pl.col('Lon') <= 360.0)) |
                        ((pl.col('Lon') >= 0.0) & (pl.col('Lon') <= 10.0)) &
                        (pl.col('Lat') >= currentBlob[i, 'MinLat']) &
                        (pl.col('Lat') <= currentBlob[i, 'MaxLat']))['indexA'].to_list()
                else:
                    idxApprox = currentTrack.with_row_index(name='indexA').filter(
                        (pl.col('Lon') >= currentBlob[i, 'MinLon']) & 
                        (pl.col('Lon') <= currentBlob[i, 'MaxLon']) &
                        (pl.col('Lat') >= currentBlob[i, 'MinLat']) &
                        (pl.col('Lat') <= currentBlob[i, 'MaxLat']))['indexA'].to_list()
                if len(idxApprox) > 0:
                    finalIdx = idxApprox[currentTrack[idxApprox, 'MSLP'].arg_min()]
                    nodeIdx = currentTrack[finalIdx, 'index']
                    nodePair.append(
                        (currentBlob[i, 'BID'], nodeIdx))
        return nodePair

    def lowTerrainRatio(self, k):
        lonTrack = self.tracks_['Lon'][k]
        latTrack = self.tracks_['Lat'][k]
        trackPoints = geo2XYZOnUnitSphere(lonTrack, latTrack)
        idx = self.z0KDTree_.query_ball_point(trackPoints, r=5.0*np.pi/180.0)
        z0 = self.z0_['Z0'].to_numpy().flatten()[idx]
        ratio = len(np.where(z0 < 7000)[0]) / len(z0)
        return ratio

    def maxTerrianNearby(self, k):
        lonTrack = self.tracks_['Lon'][k]
        latTrack = self.tracks_['Lat'][k]
        trackPoints = geo2XYZOnUnitSphere(lonTrack, latTrack)
        idx = self.z0KDTree_.query_ball_point(trackPoints, r=1.0*np.pi/180.0)
        z0Max = self.z0_['Z0'].to_numpy().flatten()[idx].max()
        return z0Max

    def inlandRatio(self, tid):
        z0Max = self.tracks_.filter(pl.col('TID') == tid)['Z0Max'].to_numpy()
        ratio = len(np.where(z0Max > 150.0)[0]) / len(z0Max)
        return ratio

    def trackSpread(self, tid):
        lonTrack = self.tracks_['Lon'][self.trackStartLocs_[
            tid]:self.trackEndLocs_[tid]+1]
        latTrack = self.tracks_['Lat'][self.trackStartLocs_[
            tid]:self.trackEndLocs_[tid]+1]
        trackPoints = geo2XYZOnUnitSphere(lonTrack, latTrack)
        trackKDTree = cKDTree(trackPoints)
        spread = np.std(trackKDTree.query(trackPoints[0],
                                          k=self.trackEndLocs_[tid]-self.trackStartLocs_[tid]+1)[0]
                        ) / np.pi * 180.0
        return spread

    def trackCorrelation(self, tid):
        lonTrack = self.tracks_['Lon'][self.trackStartLocs_[
            tid]:self.trackEndLocs_[tid]+1]
        latTrack = self.tracks_['Lat'][self.trackStartLocs_[
            tid]:self.trackEndLocs_[tid]+1]
        corr = np.abs(stats.pearsonr(lonTrack, latTrack).statistic)
        if math.isnan(corr):
            corr = 0
        return corr

    def preprocess(self, filename: Optional[str] = None, override: bool = False):
        timerStart = time.time()
        if os.path.exists(filename) and not override:
            self.tracks_ = pl.read_parquet(filename)
            self.nNode_ = self.tracks_.height
        else:
            self.trackStartLocs_ = np.unique(
                self.tracks_['TID'], return_index=1)[1]
            self.trackEndLocs_ = self.tracks_.height - np.unique(
                self.tracks_.with_columns(pl.col('TID').reverse())['TID'],
                return_index=1)[1] - 1
            z0Lon2D, z0Lat2D = np.meshgrid(
                self.z0_['longitude'], self.z0_['latitude'])
            z0Points = geo2XYZOnUnitSphere(
                z0Lon2D.flatten(), z0Lat2D.flatten())
            self.z0KDTree_ = cKDTree(z0Points)
            datetimes = np.unique(self.tracks_['UTC'])
            pool = Pool(processes=self.nproc_)

            # Pair nodes with corresponding blobs
            pairGroup = pool.map(self.blobPairing, datetimes)
            pairGroupBlodId = [inner[0] for outer in pairGroup for inner in outer]
            pairGroupNodeMatched = np.array(
                [inner[1] for outer in pairGroup for inner in outer])
            self.blobStats_ = self.blobStats_.update(self.blobStats_.filter(
                pl.col('BID').is_in(pairGroupBlodId)
            ).with_columns(PairedNode=pairGroupNodeMatched), on='BID')
            lowTerrainRatio = pool.map(
                self.lowTerrainRatio, range(self.nNode_))

            blobGroups = self.blobStats_.filter(
                pl.col('PairedNode') > -1).group_by(
                    'PairedNode', maintain_order=True)

            blobGroupAreaSum = blobGroups.agg(pl.col('Area').sum()).sort('PairedNode')
            blobGroupIkeSum = blobGroups.agg(pl.col('IKE').sum()).sort('PairedNode')
            
            self.tracks_ = self.tracks_.update(self.tracks_.filter(
                pl.col('index').is_in(blobGroupAreaSum['PairedNode'])
            ).with_columns(Area=blobGroupAreaSum['Area'] * 1e-6), on='index')

            self.tracks_ = self.tracks_.update(self.tracks_.filter(
                pl.col('index').is_in(blobGroupAreaSum['PairedNode'])
            ).with_columns(IKE=blobGroupIkeSum['IKE'] * 1e-12), on='index')

            # Magnify LPS area if the nearby terrian is high (925 hPa wind speed may 
            # be weak)
            nodeIdxAjustArea = np.where((np.array(lowTerrainRatio) >= 0.3) & (
                np.array(lowTerrainRatio) <= 0.7))[0]
            self.tracks_ = self.tracks_.with_columns(
                Area = pl.when(pl.col('index').is_in(nodeIdxAjustArea)).then(
                    pl.col('Area') * 2.0
                ).otherwise(pl.col('Area')))
            
            trackIds = np.unique(self.tracks_['TID'])
            trackSpread = pool.map(self.trackSpread, trackIds)
            trackCorrelation = pool.map(self.trackCorrelation, trackIds)
            z0MaxPerNode = np.array(pool.map(self.maxTerrianNearby, range(self.nNode_)))
            self.tracks_ = self.tracks_.with_columns(
                Z0Max = z0MaxPerNode)
            
            inlandRatio = pool.map(self.inlandRatio, trackIds)
            for i in trackIds:
                self.tracks_ = self.tracks_.with_columns(
                    TrackSpread = pl.when(pl.col('TID') == i).then(
                        trackSpread[i]).otherwise(pl.col('TrackSpread')),
                    TrackCorrelation = pl.when(pl.col('TID') == i).then(
                        trackCorrelation[i]).otherwise(pl.col('TrackCorrelation')),
                    InlandRatio = pl.when(pl.col('TID') == i).then(
                        inlandRatio[i]).otherwise(pl.col('InlandRatio')),
                )
            pool.close()

            if filename:
                self.tracks_.write_parquet(filename)

            timerEnd = time.time()
            logger.warning(f'Time lasped for preprocess step: {timerEnd - timerStart} s.')

    def classify(self, filename: Optional[str] = None, full: Optional[bool] = False, 
                 resultFormat: Optional[str] = 'parquet'):
        timerStart = time.time()
        # Step 1: Low pressure system categorization
        self.tracks_ = self.tracks_.with_columns(
            MSLPCCRatio = pl.col('MSLPCC20') / pl.col('MSLPCC55')
        )
        labelFull = np.array([''] * self.nNode_, dtype=object)
        labelShort = np.array([''] * self.nNode_, dtype=object)

        # High-altitude branch
        condHighAltitude = self.tracks_['Z850'] < self.tracks_['Z0']
        condHighThermalLow = (self.tracks_['MidThkCC'] < 0) | (
            self.tracks_['UppThkCC'] < 0)
        idxHighAltitudeThermal = self.tracks_.filter(
            condHighAltitude & condHighThermalLow)['index']
        idxHighAltitude = self.tracks_.filter(
            condHighAltitude & ~condHighThermalLow)['index']
        labelFull[idxHighAltitudeThermal] = 'High-altitude Thermal Low'
        labelShort[idxHighAltitudeThermal] = 'HATHL'
        labelFull[idxHighAltitude] = 'High-altitude Low'
        labelShort[idxHighAltitude] = 'HAL'

        # Dryness branch
        condDry = self.tracks_['RH850Avg'] <= 60.0
        condCyclonic = ((self.tracks_['Vo500Avg'] > 0.0) &
                        (self.tracks_['Lat'] >= 0.0)) | (
            (self.tracks_['Vo500Avg'] < 0.0) &
            (self.tracks_['Lat'] < 0.0))
        condLowThermalLow = self.tracks_['LowThkCC'] < 0
        idxDryDisturbance = self.tracks_.filter(
            ~condHighAltitude & condDry & ~condLowThermalLow)['index']
        idxDeepOrogThermalLow = self.tracks_.filter(
            ~condHighAltitude & condDry & condLowThermalLow & condCyclonic)['index']
        idxThermalLow = self.tracks_.filter(
            ~condHighAltitude & condDry & condLowThermalLow & ~condCyclonic)['index']
        labelFull[idxDryDisturbance] = 'Dry Disturbance'
        labelShort[idxDryDisturbance] = 'DSD'
        labelFull[idxDeepOrogThermalLow] = 'Deep (Orographic) Thermal Low'
        labelShort[idxDeepOrogThermalLow] = 'DOTHL'
        labelFull[idxThermalLow] = 'Thermal Low'
        labelShort[idxThermalLow] = 'THL'

        # Tropical branch
        if self.timeInterval_ == 3:
            condTropical = (
                (self.tracks_['RH100Max'] > 20.0) &
                (self.tracks_['DeepShear'] < 18.0) &
                (self.tracks_['T850'] > 280.0))
            condTC = (
                (self.tracks_['UppThkCC'] < -107.8) &
                (self.tracks_['LowThkCC'] < 0.0) &
                (self.tracks_['MSLPCC20'] > 215.0))
        elif self.timeInterval_ == 6:
            condTropical = (
                (self.tracks_['RH100Max'] > 20.0) &
                (self.tracks_['DeepShear'] < 13.0) &
                (self.tracks_['T850'] > 280.0))
            condTC = (
                (self.tracks_['UppThkCC'] < -147.0) &
                (self.tracks_['LowThkCC'] < 0.0) &
                (self.tracks_['MSLPCC20'] > 225.0))
        condTD = (
            (self.tracks_['UppThkCC'] < 0.0) &
            (self.tracks_['MSLPCC55'] > 160.0))
        condMosoon = (
            (self.tracks_['RH850Avg'] > 85.0) &
            (self.tracks_['U850Diff'] > 0.0))
        idxTropicalDisturbance = self.tracks_.filter(
            ~condHighAltitude & ~condDry & condTropical & ~condCyclonic)['index']
        idxTC = self.tracks_.filter(
            ~condHighAltitude & ~condDry & condTropical & condCyclonic & condTC
            )['index']
        idxTDMosoon = self.tracks_.filter(
            ~condHighAltitude & ~condDry & condTropical & condCyclonic & ~condTC &
            condTD & condMosoon)['index']
        idxTD = self.tracks_.filter(
            ~condHighAltitude & ~condDry & condTropical & condCyclonic & ~condTC &
            condTD & ~condMosoon)['index']
        idxMosoonLow = self.tracks_.filter(
            ~condHighAltitude & ~condDry & condTropical & condCyclonic & ~condTC &
            ~condTD & condMosoon)['index']
        idxTropicalLow = self.tracks_.filter(
            ~condHighAltitude & ~condDry & condTropical & condCyclonic & ~condTC &
            ~condTD & ~condMosoon)['index']
        # Include mosoon low and mosoon TD
        idxMosoonProperty = self.tracks_.filter(
            ~condHighAltitude & ~condDry & condTropical & condCyclonic & ~condTC &
            condMosoon)['index']
        labelFull[idxTropicalDisturbance] = 'Tropical Disturbance'
        labelShort[idxTropicalDisturbance] = 'DST'
        labelFull[idxTC] = 'Tropical Cyclone'
        labelShort[idxTC] = 'TC'
        labelFull[idxTDMosoon] = (
            'Tropical Depression (Mosoon Depression)')
        labelShort[idxTDMosoon] = 'TD(MD)'
        labelFull[idxTD] = 'Tropical Depression'
        labelShort[idxTD] = 'TD'
        labelFull[idxMosoonLow] = 'Tropical Low (Mosoon Low)'
        labelShort[idxMosoonLow] = 'TLO(ML)'
        labelFull[idxTropicalLow] = 'Tropical Low'
        labelShort[idxTropicalLow] = 'TLO'
        # Extratropical branch
        idxExtDisturbance = self.tracks_.filter(
            ~condHighAltitude & ~condDry & ~condTropical & ~condCyclonic)['index']
        labelFull[idxExtDisturbance] = 'Extratropical Disturbance'
        labelShort[idxExtDisturbance] = 'DSE'
        if self.timeInterval_ == 3:
            condSubtropical = (
                (self.tracks_['LowThkCC'] < 0.0) &
                (self.tracks_['Z500CC'] > 0.0) &
                (self.tracks_['WS200PMax'] > 30.0))
        elif self.timeInterval_ == 6:
            condSubtropical = (
                (self.tracks_['LowThkCC'] < 0.0) &
                (self.tracks_['Z500CC'] > 0.0) &
                (self.tracks_['WS200PMax'] > 35.0))
        condTropicalLike = (
            (self.tracks_['LowThkCC'] < 0) &
            (self.tracks_['MidThkCC'] < 0) &
            (
              ((self.tracks_['Area'] > 0) &
               (self.tracks_['Area'] <= 5.5e5) &
               (self.tracks_['MSLPCC20'] > 190)
              ) | ((self.tracks_['MSLPCC20'] > 420) &
               (self.tracks_['MSLPCCRatio'] > 0.5)
              )
            ))
        condTropicalLikeCyclone = (
            ~condHighAltitude & ~condDry & ~condTropical & condCyclonic &
            condTropicalLike)
        idxTropicalLikeCyclone = self.tracks_.filter(
            condTropicalLikeCyclone)['index']

        if self.timeInterval_ == 3:
            condSubtropicalTropicalLikeCyclone = (
                condTropicalLikeCyclone &
                (self.tracks_['WS200PMax'] >= 25.0))
            condPolarLow = (
                condTropicalLikeCyclone &
                (self.tracks_['WS200PMax'] < 25.0))
        elif self.timeInterval_ == 6:
            condSubtropicalTropicalLikeCyclone = (
                condTropicalLikeCyclone &
                (self.tracks_['WS200PMax'] >= 30.0))
            condPolarLow = (
                condTropicalLikeCyclone &
                (self.tracks_['WS200PMax'] < 30.0))
        idxSubtropicalTropicalLikeCyclone = self.tracks_.filter(
            condSubtropicalTropicalLikeCyclone)['index']
        idxPolarLow = self.tracks_.filter(condPolarLow)['index']

        condSubtropicalCyclone = (
            ~condHighAltitude & ~condDry & ~condTropical & condCyclonic &
            ~condTropicalLike & condSubtropical)
        idxSubtropicalCyclone = self.tracks_.filter(
            condSubtropicalCyclone)['index']
        condExtCyclone = (
            ~condHighAltitude & ~condDry & ~condTropical & condCyclonic &
            ~condTropicalLike & ~condSubtropical)
        idxExtCyclone = self.tracks_.filter(condExtCyclone)['index']

        labelFull[idxSubtropicalTropicalLikeCyclone] = (
            'Subtropical Tropical-like Cyclone '
            '(Subtropical Storm)')
        labelShort[idxSubtropicalTropicalLikeCyclone] = 'SS(STLC)'
        labelFull[idxPolarLow] = (
            'Polar Low (Polar Tropical-like Cyclone)')
        labelShort[idxPolarLow] = 'PL(PTLC)'
        labelFull[idxSubtropicalCyclone] = 'Subtropical Cyclone'
        labelShort[idxSubtropicalCyclone] = 'SC'
        labelFull[idxExtCyclone] = 'Extratropical Cyclone'
        labelShort[idxExtCyclone] = 'EX'

        self.tracks_ = self.tracks_.with_columns(
            FullLabel = labelFull,
            ShortLabel = labelShort
        )

        # Step 2: Track information, including transition
        idxTropical = self.tracks_.filter(condTropical)['index']
        condTrans = (
            condTropical &
            ((self.tracks_['RH100Max'] < 55.0) |
             (self.tracks_['DeepShear'] > 10.0)) &
            (np.absolute(self.tracks_['Lat']) > 15.0))
        idxTrans = self.tracks_.filter(condTrans)['index']
        flagTrans = np.array([0] * self.nNode_, dtype=np.int64)
        flagTropical = np.array([0] * self.nNode_, dtype=np.int64)
        flagTrans[idxTrans] = 1
        flagTropical[idxTropical] = 1
        self.tracks_ = self.tracks_.with_columns(
            TransitionZone = flagTrans,
            TropicalFlag = flagTropical
        )

        if self.timeInterval_ == 3:
            tidTCTrack = self.tracks_.filter(
                pl.col('index').is_in(idxTC)).group_by(
                    'TID', maintain_order=True).count().filter(
                    pl.col('count') >= 8)['TID']
            tidMosoonTrack = self.tracks_.filter(
                pl.col('index').is_in(idxMosoonProperty)).group_by(
                    'TID', maintain_order=True
                    ).count().filter(pl.col('count') >= 10)['TID']
            tidTropicalLikeCycloneTrack = self.tracks_.filter(
                pl.col('index').is_in(idxTropicalLikeCyclone)).group_by(
                    'TID', maintain_order=True).count().filter(
                        pl.col('count') >= 2)['TID']
        elif self.timeInterval_ == 6:
            tidTCTrack = self.tracks_.filter(
                pl.col('index').is_in(idxTC)).group_by(
                    'TID', maintain_order=True).count().filter(
                    pl.col('count') >= 4)['TID']
            tidMosoonTrack = self.tracks_.filter(
                pl.col('index').is_in(idxMosoonProperty)).group_by(
                    'TID', maintain_order=True).count().filter(
                        pl.col('count') >= 5)['TID']
            tidTropicalLikeCycloneTrack = self.tracks_.filter(
                pl.col('index').is_in(idxTropicalLikeCyclone)).group_by(
                    'TID', maintain_order=True).count().filter(
                        pl.col('count') >= 1)['TID']

        tidSubtropicalTropicalLikeCycloneTrack = np.intersect1d(
            tidTropicalLikeCycloneTrack,
            self.tracks_.filter(
                pl.col('index').is_in(idxSubtropicalTropicalLikeCyclone)).unique(
                    pl.col('TID'))['TID']
            )
        tidPolarLowTrack = np.intersect1d(
            tidTropicalLikeCycloneTrack,
            self.tracks_.filter(
                pl.col('index').is_in(idxPolarLow)).unique(
                    pl.col('TID'))['TID']
            )

        idxTCTrack = self.tracks_.filter(
                pl.col('TID').is_in(tidTCTrack))['index']
        idxMosoonTrack = self.tracks_.filter(
                pl.col('TID').is_in(tidMosoonTrack))['index']
        idxSubtropicalTropicalLikeCycloneTrack = self.tracks_.filter(
                pl.col('TID').is_in(tidSubtropicalTropicalLikeCycloneTrack))['index']
        idxPolarLowTrack = self.tracks_.filter(
                pl.col('TID').is_in(tidPolarLowTrack))['index']
        condQuasiStationary = (
            (self.tracks_['TrackCorrelation'] < 0.55) &
            (self.tracks_['TrackSpread'] < 3) &
            (self.tracks_['InlandRatio'] > 0.65))
        idxQuasiStationary = self.tracks_.filter(condQuasiStationary)['index']
        flagExtTransition = np.zeros(len(tidTCTrack), dtype=np.int64) - 1
        tracksTC = self.tracks_.filter(
            pl.col('TID').is_in(tidTCTrack))
        for i, j in enumerate(tidTCTrack):
            trackCurrent = tracksTC.filter(pl.col('TID') == j)
            idxLastTC = trackCurrent.filter(
                pl.col('ShortLabel') == 'TC')['index'][-1]
            trackRemain = trackCurrent.filter(
                (pl.col('index') > idxLastTC) & 
                (pl.col('TropicalFlag') == 0))
            if trackRemain.height > 0 or (
                (trackCurrent['index'][-1] - idxLastTC == 1) and 
                (len(trackRemain['index']) == 1)):
                firstExt = trackRemain['index'][0]
                flagExtTransition[i] = firstExt
        flagExtTransition = np.delete(
            flagExtTransition, np.argwhere(flagExtTransition == -1))

        flagTropicalTransition = np.zeros(len(tidTCTrack), dtype=np.int64) - 1
        for i, j in enumerate(tidTCTrack):
            trackCurrent = tracksTC.filter(pl.col('TID') == j)
            if (trackCurrent['TropicalFlag'][0] == 0) or (
                    trackCurrent['TransitionZone'][0] > 0):
                idxFirstTC = trackCurrent.filter(
                    pl.col('ShortLabel') == 'TC')['index'][0]
                flagTropicalTransition[i] = idxFirstTC
        flagTropicalTransition = np.delete(
            flagTropicalTransition, np.argwhere(flagTropicalTransition == -1))

        labelTrack = np.array(['Track'] * self.nNode_, dtype=object)
        labelTrack[idxTCTrack] += '_TC'
        labelTrack[idxMosoonTrack] += '_MS'
        labelTrack[idxSubtropicalTropicalLikeCycloneTrack] += '_SS(STLC)'
        labelTrack[idxPolarLowTrack] += '_PL(PTLC)'
        labelTrack[flagExtTransition] += '_EXT'
        labelTrack[flagTropicalTransition] += '_TT'
        labelTrack[idxQuasiStationary] += '_QS'
        self.tracks_ = self.tracks_.with_columns(
            TrackInfo = labelTrack
        )
        if filename:
            self.save(filename, full, resultFormat)
        timerEnd = time.time()

        logger.warning(f'Time lasped for classify step: {timerEnd - timerStart} s.')

    def save(self, filename: str, full: Optional[bool] = False, 
             resultFormat: Optional[str] = 'parquet'):
        if not full:
            newColums = [
                'index', 'TID', 'Lat', 'Lon', 'UTC',
                'MSLP', 'WS10', 'FullLabel', 'ShortLabel', 
                'TropicalFlag', 'TransitionZone', 'TrackInfo',
                'Area', 'IKE']
            self.tracks_ = self.tracks_[newColums]
        if resultFormat == 'parquet':
            self.tracks_.write_parquet(f'{filename}.parquet')
        elif resultFormat == 'csv':
            self.tracks_.write_csv(f'{filename}.csv')
        elif resultFormat == 'all':
            self.tracks_.write_parquet(f'{filename}.parquet')
            self.tracks_.write_csv(f'{filename}.csv')