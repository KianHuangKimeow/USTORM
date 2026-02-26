'''
Details of SyCLoPS algorithms can be found at 
https://doi.org/10.1029/2024JD041287
'''
from itertools import repeat
import logging
import os
import math
from multiprocessing import Pool
import time
from typing import Optional
import warnings

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from scipy import stats
import xarray as xr

from Base import geo2XYZOnUnitSphere, getBoundary

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SyCLoPSClassifier():
    def __init__(self,
                 stichNodesOutput: str, blobStatsOutput: str,
                 z0File: str,
                 lpsMaskBlobStatsOutput: str = None,
                 nproc: Optional[int] = 4, 
                 timeInterval: Optional[int] = 3,
                 flagGeopotentialHeight: Optional[bool] = False, 
                 lonName: Optional[str] = None,
                 latName: Optional[str] = None,
                 flagWS250: Optional[bool] = False,
                 rhTropicalThreshold: float = 20,
                 regional: Optional[bool] = False):
        self.trackFile_ = stichNodesOutput
        self.blobStatsFile_ = blobStatsOutput
        self.z0File_ = z0File
        self.lpsMaskBlobStatsOutput_ = lpsMaskBlobStatsOutput
        self.nproc_ = nproc
        self.timeInterval_ = timeInterval
        self.flagWS250_ = flagWS250
        self.rhTropicalThreshold_ = rhTropicalThreshold
        self.regional_ = regional
        self.tracks_ = None
        self.nNode_ = -1
        self.z0Ds_ = None
        self.z0_ = None
        self.blobStats_ = None
        self.trackStartLocs_ = None
        self.inputGeopotentialHeight_ = flagGeopotentialHeight
        self.lonName_ = lonName
        self.latName_ = latName
        self.trackPointXYZ_ = None
        self.blobPointXYZ_ = None
        self.lpsMaskBlobPointXYZ_ = None
        self.trackDatetimeGroupIdx_ = None
        self.blobDatetimeGroupIdx_ = None
        self.trackDatetime_ = None
        self.blobDatetime_ = None
        self.lpsMaskBlobDatetime_ = None
        self.boundaryKDTree_ = None
        self.loadData()

    def loadData(self) -> None:
        self.loadStichNodesOutput()
        self.z0Ds_ = xr.load_dataset(self.z0File_)
        if self.lonName_ is not None:
            self.z0Ds_ = self.z0Ds_.rename_vars(
                {self.lonName_: 'longitude'})
        if self.latName_ is not None:
            self.z0Ds_ = self.z0Ds_.rename_vars(
                {self.latName_: 'latitude'})
        if self.inputGeopotentialHeight_:
            self.z0Ds_['Z0'] *= 9.81
        self.loadBlobStatsOutput()
        if self.lpsMaskBlobStatsOutput_ is not None:
            self.loadLpsMaskBlobStatsOutput()

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
            self.trackFile_, separator='\t', has_header=False, comment_prefix='start',
            null_values='nan')
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
            # Avarage 700 hPa relative humidity within a 2.5 GCD
            'RH700Avg',
            # 850 hPa air temperature at the node
            'T850',
            # 850 hPa geopotential at the node
            'Z850',
            # 700 hPa geopotential at the node
            'Z700',
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
            'RH100Max', 'RH850Avg', 'RH700Avg', 'T850', 'Z850', 'Z700', 
            'Z0', 'U850Diff', 'WS200PMax']
        self.tracks_ = self.tracks_[newColums]
        self.tracks_ = self.tracks_.with_columns((pl.col('Lon') + 360.0).mod(360.0).alias('Lon'))
        if self.inputGeopotentialHeight_:
            self.tracks_ = self.tracks_.with_columns(
                Z0 = pl.col('Z0') * 9.81,
                Z850 = pl.col('Z850') * 9.81,
                Z700 = pl.col('Z700') * 9.81,
                UppThkCC = pl.col('UppThkCC') * 9.81,
                MidThkCC = pl.col('MidThkCC') * 9.81,
                LowThkCC = pl.col('LowThkCC') * 9.81,
                Z500CC = pl.col('Z500CC') * 9.81
            )
        self.tracks_ = self.tracks_.with_columns(Area=0.0)
        # self.tracks_ = self.tracks_.with_columns(IKE=0.0)
        self.tracks_ = self.tracks_.with_columns(TrackSpread=0.0)
        self.tracks_ = self.tracks_.with_columns(TrackCorrelation=0.0)
        self.tracks_ = self.tracks_.with_columns(Z0Max=0.0)
        self.tracks_ = self.tracks_.with_columns(InlandRatio=0.0)
        self.tracks_ = self.tracks_.with_columns(BoundaryRatio=0.0)
        self.tracks_ = self.tracks_.with_columns(Bid=[])
        self.tracks_ = self.tracks_.with_row_index()
        self.nNode_ = tid.size

    def loadBlobStatsOutput(self) -> None:
        self.blobStats_ = pl.read_csv(
            self.blobStatsFile_, separator='\t', has_header=False, null_values='nan')
        self.blobStats_ = self.blobStats_.drop(
            'column_2')
        self.blobStats_.columns = [
            'BID', 'Time', 'CenLon', 'CenLat',
            'MinLat', 'MaxLat', 'MinLon', 'MaxLon', 'Area',
            # 'IKE'
        ]
        self.blobStats_ = self.blobStats_.with_columns(
            pl.col('Time').str.to_datetime(
                format='%Y-%m-%d %H:%M:%S').alias('UTC'))
        self.blobStats_ = self.blobStats_.drop('Time')
        self.blobStats_ = self.blobStats_.with_columns(PairedNode=-1)

    def loadLpsMaskBlobStatsOutput(self) -> None:
        self.lpsMaskBlobStats_ = pl.read_csv(
            self.lpsMaskBlobStatsOutput_, separator='\t', has_header=False, null_values='nan')
        self.lpsMaskBlobStats_ = self.lpsMaskBlobStats_.drop(
            'column_2')
        self.lpsMaskBlobStats_.columns = [
            'BID', 'Time', 'CenLon', 'CenLat',
            'MinLat', 'MaxLat', 'MinLon', 'MaxLon', 'Area',
            'LPSBlobs'
        ]
        self.lpsMaskBlobStats_ = self.lpsMaskBlobStats_.with_columns(
            pl.col('Time').str.to_datetime(
                format='%Y-%m-%d %H:%M:%S').alias('UTC'))
        self.lpsMaskBlobStats_ = self.lpsMaskBlobStats_.drop('Time')
        self.lpsMaskBlobStats_ = self.lpsMaskBlobStats_.with_columns(PairedNode=-1)

    def lpsMaskBlobNodePairing(self, dt):
        track = self.tracks_.filter(UTC=dt)
        lpsMaskBlobStats = self.lpsMaskBlobStats_.filter(UTC=dt)
        trackIdx = track['index']
        trackPoints = self.trackPointXYZ_[trackIdx]
        trackTree = cKDTree(trackPoints)
        result = pl.DataFrame({
                        'BID': [], 
                        'PairedNode': []}, 
                    schema={
                        'BID': pl.Int32, 
                        'PairedNode': pl.Int32})
        for iLpsMask in lpsMaskBlobStats.rows(named=True):
            iLpsMaskIdx = iLpsMask['BID']
            iLpsMaskBlobPoint = self.lpsMaskBlobPointXYZ_[iLpsMaskIdx-1]
            idx = trackTree.query_ball_point(iLpsMaskBlobPoint, r=0.1*np.pi/180.0)
            if len(idx) == 1:
                iResult = pl.DataFrame({
                    'BID': [iLpsMaskIdx], 
                    'PairedNode': [trackIdx[idx[0]]]}, 
                    schema={
                        'BID': pl.Int32, 
                        'PairedNode': pl.Int32})
                result.extend(iResult)
            elif len(idx) > 1:
                logger.warning(f'Special situation: LPS region mask {iLpsMaskIdx} has multiple LPS nodes {trackIdx[idx]}')
                iResult = pl.DataFrame({
                    'BID': [iLpsMaskIdx], 
                    'PairedNode': [trackIdx[idx[0]]]}, 
                    schema={
                        'BID': pl.Int32, 
                        'PairedNode': pl.Int32})
                result.extend(iResult)
        self.lpsMaskBlobStats_ = self.lpsMaskBlobStats_.update(result, on='BID')
        
    def blobNodePairingNew(self):
        blobPairStrs = self.lpsMaskBlobStats_.with_columns(LPSBlobs = pl.col('LPSBlobs').str.split(';'))['LPSBlobs'].to_numpy()
        lpsMaskNodes = self.lpsMaskBlobStats_['PairedNode'].to_numpy()
        nodePair = []
        nodePairDict = dict()
        for iblobPairStr, iLpsMaskNode in zip(blobPairStrs, lpsMaskNodes):
            item = iblobPairStr[:-1]
            newItem = dict()
            for j in item:
                k, v = j.split(':')
                if int(k) > 0:
                    newItem.update({int(iLpsMaskNode): float(v)})
                    nodePairDict.update({int(k): newItem})
        for k, v in nodePairDict.items():
            iNodes = list(v.keys())
            if len(iNodes) == 1:
                nodePair.append((k, iNodes[0]))
            else:
                logger.warning(f'LPS impact area {k} has multiple LPSs {v}')
                maxArea = 0
                finalNode = -1
                for iNode, iArea in v.items():
                    if iArea > maxArea:
                        maxArea = iArea
                        finalNode = iNode
                nodePair.append((k, finalNode))
        return [nodePair]

    def blobNodePairingOld(self, dt):
        nodePair = []
        dtIdxTrack = np.argwhere(self.trackDatetime_ == dt)
        dtIdxBlob = np.argwhere(self.blobDatetime_ == dt)
        if len(dtIdxTrack) == 1 and len(dtIdxBlob) == 1:
            dtIdxTrack = dtIdxTrack[0,0]
            dtIdxBlob = dtIdxBlob[0,0]
        else:
            return nodePair
        currentTrackIdx = self.trackDatetimeGroupIdx_[dtIdxTrack]
        trackPoints = self.trackPointXYZ_[currentTrackIdx]
        currentBlobIdx = self.blobDatetimeGroupIdx_[dtIdxBlob]
        blobPoints = self.blobPointXYZ_[currentBlobIdx-1]
        trackTree = cKDTree(trackPoints)
        for i in range(len(blobPoints)):
            idx = trackTree.query_ball_point(blobPoints[i], r=5*np.pi/180.0)
            if idx:
                finalIdx = idx[np.argmin(self.trackDatetimeGroupMslp_[dtIdxTrack][idx])]
                nodeIdx = currentTrackIdx[finalIdx]
                nodePair.append(
                    (currentBlobIdx[i], nodeIdx))
            else:
                if (self.blobDatetimeGroupMaxLon_[dtIdxBlob][i] - 
                    self.blobDatetimeGroupMinLon_[dtIdxBlob][i] > 180.0):
                    idxApprox = np.where(
                        ((self.trackDatetimeGroupLon_[dtIdxTrack] >= 350.0) & 
                         (self.trackDatetimeGroupLon_[dtIdxTrack] <= 360.0)) |
                        ((self.trackDatetimeGroupLon_[dtIdxTrack] >= 0.0) & 
                         (self.trackDatetimeGroupLon_[dtIdxTrack] <= 10.0)) &
                        (self.trackDatetimeGroupLat_[dtIdxTrack] >= self.blobDatetimeGroupMinLat_[dtIdxBlob][i]) &
                        (self.trackDatetimeGroupLat_[dtIdxTrack] <= self.blobDatetimeGroupMaxLat_[dtIdxBlob][i])
                    )[0]
                else:
                    idxApprox = np.where(
                        (self.trackDatetimeGroupLon_[dtIdxTrack] >= self.blobDatetimeGroupMinLon_[dtIdxBlob][i]) & 
                        (self.trackDatetimeGroupLon_[dtIdxTrack] <= self.blobDatetimeGroupMaxLon_[dtIdxBlob][i]) &
                        (self.trackDatetimeGroupLat_[dtIdxTrack] >= self.blobDatetimeGroupMinLat_[dtIdxBlob][i]) &
                        (self.trackDatetimeGroupLat_[dtIdxTrack] <= self.blobDatetimeGroupMaxLat_[dtIdxBlob][i])
                    )[0]
                if len(idxApprox) > 0:
                    finalIdx = idxApprox[np.argmin(
                        self.trackDatetimeGroupMslp_[dtIdxTrack][idxApprox])]
                    nodeIdx = currentTrackIdx[finalIdx]
                    nodePair.append(
                        (currentBlobIdx[i], nodeIdx))
        return nodePair

    def lowTerrainRatio(self, k):
        idx = self.z0KDTree_.query_ball_point(self.trackPointXYZ_[k], r=5.0*np.pi/180.0)
        z0 = self.z0_[idx]
        ratio = len(np.where(z0 < 7000)[0]) / len(z0)
        return ratio

    def maxTerrianNearby(self, k):
        idx = self.z0KDTree_.query_ball_point(
            self.trackPointXYZ_[k], r=1.0*np.pi/180.0)
        z0Max = self.z0_[idx].max()
        return z0Max

    def inlandRatio(self, tid):
        z0Max = self.trackZ0Max_[tid]
        test = np.where(z0Max > 150.0)[0]
        ratio = len(test) / len(z0Max)
        return ratio

    def trackSpread(self, tid):
        trackPoints = self.trackPointXYZ_[self.trackStartLocs_[
            tid]:self.trackEndLocs_[tid]+1]
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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore', 
                message=('An input array is constant; '
                         'the correlation coefficient is not defined.'))
            corr = np.abs(stats.pearsonr(lonTrack, latTrack).statistic)
        if math.isnan(corr):
            corr = 0
        return corr
    
    def trackBoundaryMask(self, tid):
        trackPoints = self.trackPointXYZ_[self.trackStartLocs_[
            tid]:self.trackEndLocs_[tid]+1]
        trackBoundaryMask = np.zeros(len(trackPoints), dtype=int)
        if self.regional_:
            idx = self.boundaryKDTree_.query_ball_point(
                    trackPoints, r=2.0*np.pi/180.0)
            maskCloseToBoundary = np.array([len(i) > 0 for i in idx], dtype=bool)
            idxAwayFromBoundary = np.where(~ maskCloseToBoundary)[0]
            trackBoundaryMask[maskCloseToBoundary] = 1
            if len(idxAwayFromBoundary) > 0:
                firstIdxAwayFromBoundary = idxAwayFromBoundary[0]
                lastIdxAwayFromBoundary = idxAwayFromBoundary[-1]
                trackBoundaryMask[firstIdxAwayFromBoundary:lastIdxAwayFromBoundary] *= 2

        return trackBoundaryMask

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
            self.trackPointXYZ_ = geo2XYZOnUnitSphere(
                self.tracks_['Lon'], self.tracks_['Lat'])
            self.blobPointXYZ_ = geo2XYZOnUnitSphere(
                self.blobStats_['CenLon'], self.blobStats_['CenLat'])
            self.lpsMaskBlobPointXYZ_ = geo2XYZOnUnitSphere(
                self.lpsMaskBlobStats_['CenLon'], self.lpsMaskBlobStats_['CenLat'])
            if len(self.z0Ds_['longitude'].shape) > 1:
                z0Lon2D = self.z0Ds_['longitude'].to_numpy()
                z0Lat2D = self.z0Ds_['latitude'].to_numpy()
            else:
                z0Lon2D, z0Lat2D = np.meshgrid(
                    self.z0Ds_['longitude'], self.z0Ds_['latitude'])
            z0Points = geo2XYZOnUnitSphere(
                z0Lon2D.flatten(), z0Lat2D.flatten())
            if self.regional_:
                boundaryPoints = getBoundary(z0Lon2D, z0Lat2D)
                boundaryPoints = geo2XYZOnUnitSphere(
                    boundaryPoints[:,0].flatten(), boundaryPoints[:,1].flatten())
                self.boundaryKDTree_ = cKDTree(boundaryPoints)
            self.z0KDTree_ = cKDTree(z0Points)
            self.z0_ = self.z0Ds_['Z0'].to_numpy().flatten()
            self.z0Ds_.close()

            # Pair nodes with corresponding blobs
            trackDatetimeGroup = self.tracks_.select(
                ['index', 'UTC', 'MSLP', 'Lon', 'Lat']
            ).to_pandas()
            trackDatetimeGroup = trackDatetimeGroup.groupby(
                by=trackDatetimeGroup['UTC'])
            self.trackDatetimeGroupIdx_ = np.array(
                trackDatetimeGroup['index'].apply(np.array, dtype=np.int64))
            self.trackDatetimeGroupMslp_ = np.array(
                trackDatetimeGroup['MSLP'].apply(np.array, dtype=np.float64))
            self.trackDatetimeGroupLon_ = np.array(
                trackDatetimeGroup['Lon'].apply(np.array, dtype=np.float64))
            self.trackDatetimeGroupLat_ = np.array(
                trackDatetimeGroup['Lat'].apply(np.array, dtype=np.float64))
            blobDatetimeGroup = self.blobStats_.select(
                ['BID', 'UTC', 'MinLon', 'MaxLon', 'MinLat', 'MaxLat']
            ).to_pandas()
            blobDatetimeGroup = blobDatetimeGroup.groupby(
                by=blobDatetimeGroup['UTC'])
            self.blobDatetimeGroupIdx_ = np.array(
                blobDatetimeGroup['BID'].apply(np.array, dtype=np.int64))
            self.blobDatetimeGroupMinLon_ = np.array(
                blobDatetimeGroup['MinLon'].apply(np.array, dtype=np.float64))
            self.blobDatetimeGroupMaxLon_ = np.array(
                blobDatetimeGroup['MaxLon'].apply(np.array, dtype=np.float64))
            self.blobDatetimeGroupMinLat_ = np.array(
                blobDatetimeGroup['MinLat'].apply(np.array, dtype=np.float64))
            self.blobDatetimeGroupMaxLat_ = np.array(
                blobDatetimeGroup['MaxLat'].apply(np.array, dtype=np.float64))

            
            self.trackDatetime_ = np.unique(self.tracks_['UTC'])
            self.blobDatetime_ = np.unique(self.blobStats_['UTC'])
            self.lpsMaskBlobDatetime_ = np.unique(self.lpsMaskBlobStats_['UTC'])
            
            for i in self.trackDatetime_:
                self.lpsMaskBlobNodePairing(i)

            pairGroup = []
            if self.lpsMaskBlobStatsOutput_ is None:
                for i in self.trackDatetime_:
                    pairGroup.append(self.blobNodePairingOld(i))
            else:
                pairGroup = self.blobNodePairingNew()
            
            pairGroupBlodId = np.array([inner[0] for outer in pairGroup for inner in outer])
            pairGroupNodeMatched = np.array(
                [inner[1] for outer in pairGroup for inner in outer], dtype=int)
            self.blobStats_ = self.blobStats_.update(self.blobStats_.filter(
                pl.col('BID').is_in(pairGroupBlodId)
            ).with_columns(PairedNode=pairGroupNodeMatched), on='BID')

            trackNodeBlobPair = pl.DataFrame({'index': pairGroupNodeMatched, 'Bid': pairGroupBlodId})
            trackNodeBlobPairNodeGroup = trackNodeBlobPair.group_by('index', maintain_order=True).all()
            self.tracks_ = self.tracks_.update(trackNodeBlobPairNodeGroup, on='index')
            
            with Pool(processes=self.nproc_) as pool:
                lowTerrainRatio = pool.map(
                    self.lowTerrainRatio, range(self.nNode_))

            blobGroups = self.blobStats_.filter(
                pl.col('PairedNode') > -1).group_by(
                    'PairedNode', maintain_order=True)

            blobGroupAreaSum = blobGroups.agg(pl.col('Area').sum()).sort('PairedNode')
            # blobGroupIkeSum = blobGroups.agg(pl.col('IKE').sum()).sort('PairedNode')

            self.tracks_ = self.tracks_.update(self.tracks_.filter(
                pl.col('index').is_in(blobGroupAreaSum['PairedNode'])
            ).with_columns(Area=blobGroupAreaSum['Area'] * 1e-6), on='index')

            # self.tracks_ = self.tracks_.update(self.tracks_.filter(
            #     pl.col('index').is_in(blobGroupAreaSum['PairedNode'])
            # ).with_columns(IKE=blobGroupIkeSum['IKE'] * 1e-12), on='index')

            # Magnify LPS area if the nearby terrian is high (925 hPa wind speed may 
            # be weak)
            nodeIdxAjustArea = np.where((np.array(lowTerrainRatio) >= 0.3) & (
                np.array(lowTerrainRatio) <= 0.7))[0]
            self.tracks_ = self.tracks_.with_columns(
                Area = pl.when(pl.col('index').is_in(nodeIdxAjustArea)).then(
                    pl.col('Area') * 2.0
                ).otherwise(pl.col('Area')))
            
            trackIds = np.unique(self.tracks_['TID'])

            trackSpread = []
            trackCorrelation = []
            trackBoundaryRatio = []
            trackBoundaryMask = []
            for i in trackIds:
                trackSpread.append(self.trackSpread(i))
                trackCorrelation.append(self.trackCorrelation(i))
                boundaryMask = self.trackBoundaryMask(i)
                boundaryRatio = np.count_nonzero(boundaryMask) / len(boundaryMask)
                trackBoundaryMask.append(boundaryMask)
                trackBoundaryRatio.append(boundaryRatio)
            trackBoundaryMask = np.concatenate(trackBoundaryMask, axis=0)
            if self.regional_:
                self.tracks_ = self.tracks_.with_columns(
                    BoundaryMask = trackBoundaryMask
                )

            z0MaxPerNode = []
            for i in range(self.nNode_):
                z0MaxPerNode.append(self.maxTerrianNearby(i))
            self.tracks_ = self.tracks_.with_columns(
                Z0Max = np.array(z0MaxPerNode))

            trackTidGroup = self.tracks_.select(
                ['TID', 'Z0Max']
            ).to_pandas()
            trackTidGroup = trackTidGroup.groupby(
                by=trackTidGroup['TID'])
            self.trackZ0Max_ = np.array(trackTidGroup['Z0Max'].apply(np.array, dtype=np.float64))
            inlandRatio = []
            for i in range(len(trackIds)):
                inlandRatio.append(self.inlandRatio(i))

            for i in trackIds:
                self.tracks_ = self.tracks_.with_columns(
                    TrackSpread = pl.when(pl.col('TID') == i).then(
                        trackSpread[i]).otherwise(pl.col('TrackSpread')),
                    TrackCorrelation = pl.when(pl.col('TID') == i).then(
                        trackCorrelation[i]).otherwise(pl.col('TrackCorrelation')),
                    InlandRatio = pl.when(pl.col('TID') == i).then(
                        inlandRatio[i]).otherwise(pl.col('InlandRatio')),
                    BoundaryRatio = pl.when(pl.col('TID') == i).then(
                        trackBoundaryRatio[i]).otherwise(pl.col('BoundaryRatio')),
                )

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
        labelHA = np.array([''] * self.nNode_, dtype=object)

        # High-altitude branch
        condHighAltitude = (
            self.tracks_.with_columns(pl.col('Z850').fill_null(0.0))['Z850'] < 
            self.tracks_['Z0']
        )
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
        if not self.flagWS250_:
            condTropical = (
                (self.tracks_['RH100Max'] > self.rhTropicalThreshold_) &
                (self.tracks_['DeepShear'] < 18.0) &
                (self.tracks_.with_columns(pl.col('T850').fill_null(0.0))['T850'] > 280.0))
            condTC = (
                (self.tracks_['UppThkCC'] < -107.8) &
                (self.tracks_['LowThkCC'] < 0.0) &
                (self.tracks_['MSLPCC20'] > 215.0))
        else:
            condTropical = (
                (self.tracks_['RH100Max'] > self.rhTropicalThreshold_) &
                (self.tracks_['DeepShear'] < 13.0) &
                (self.tracks_.with_columns(pl.col('T850').fill_null(0.0))['T850'] > 280.0))
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
        if not self.flagWS250_:
            condSubtropical = (
                (self.tracks_['LowThkCC'] < 0.0) &
                (self.tracks_['Z500CC'] > 0.0) &
                (self.tracks_['WS200PMax'] > 30.0))
        else:
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

        if not self.flagWS250_:
            condSubtropicalTropicalLikeCyclone = (
                condTropicalLikeCyclone &
                (self.tracks_['WS200PMax'] >= 25.0))
            condPolarLow = (
                condTropicalLikeCyclone &
                (self.tracks_['WS200PMax'] < 25.0))
        else:
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

        # Further classification for high-altitude (HA) branch
        condExtremeHA = (
            self.tracks_.with_columns(pl.col('Z700').fill_null(0.0))['Z700'] < 
            self.tracks_['Z0']
        )
        condDryHA = self.tracks_['RH700Avg'] <= 60.0
            
        if not self.flagWS250_:
            condTropicalHA = (
                (self.tracks_['RH100Max'] > self.rhTropicalThreshold_) &
                (self.tracks_.with_columns(pl.col('DeepShear').fill_null(
                    999.0))['DeepShear'] < 18.0) 
            )
            condSubtropicalHA = (
                (self.tracks_['MidThkCC'] < 0.0) &
                (self.tracks_['Z500CC'] > 0.0) &
                (self.tracks_['WS200PMax'] > 30.0))
        else:
            condTropicalHA = (
                (self.tracks_['RH100Max'] > self.rhTropicalThreshold_) &
                (self.tracks_.with_columns(pl.col('DeepShear').fill_null(
                    999.0))['DeepShear'] < 13.0) 
            )
            condSubtropicalHA = (
                (self.tracks_['MidThkCC'] < 0.0) &
                (self.tracks_['Z500CC'] > 0.0) &
                (self.tracks_['WS200PMax'] > 35.0))
        idxExtremeHA = self.tracks_.filter(
            condExtremeHA)['index']
        idxDryHA = self.tracks_.filter(
            condHighAltitude & ~condExtremeHA & condDryHA)['index']
        idxTropicalHA = self.tracks_.filter(
            condHighAltitude & ~condExtremeHA & ~condDryHA & condTropicalHA)['index']
        idxSubtropicalHA = self.tracks_.filter(
            condHighAltitude & ~condExtremeHA & ~condDryHA & ~condTropicalHA &
            condSubtropicalHA)['index']
        idxExtHA = self.tracks_.filter(
            ~condExtremeHA & condHighAltitude & ~condDryHA & ~condTropicalHA &
            ~condSubtropicalHA & condCyclonic)['index']
        idxExtDisturbanceHA = self.tracks_.filter(
            condHighAltitude & ~condExtremeHA & ~condDryHA & ~condTropicalHA &
            ~condSubtropicalHA & ~condCyclonic)['index']
        
        labelHA[idxExtremeHA] = 'Extreme'
        labelHA[idxDryHA] = 'Dry'
        labelHA[idxTropicalHA] = 'Tropical'
        labelHA[idxSubtropicalHA] = 'Subtropical'
        labelHA[idxExtDisturbanceHA] = 'DSE'
        labelHA[idxExtHA] = 'EX'

        self.tracks_ = self.tracks_.with_columns(
            FullLabel = labelFull,
            ShortLabel = labelShort,
            HALabel = labelHA,
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

        convertRate = 3.0 / self.timeInterval_
        if not self.flagWS250_:
            tidTCTrack = self.tracks_.filter(
                pl.col('index').is_in(idxTC)).group_by(
                    'TID', maintain_order=True).count().filter(
                    pl.col('count') >= 8 * convertRate)['TID']
        else:
            tidTCTrack = self.tracks_.filter(
                pl.col('index').is_in(idxTC)).group_by(
                    'TID', maintain_order=True).count().filter(
                    pl.col('count') >= 6 * convertRate)['TID']

        tidMosoonTrack = self.tracks_.filter(
            pl.col('index').is_in(idxMosoonProperty)).group_by(
                'TID', maintain_order=True
                ).count().filter(pl.col('count') >= 10 * convertRate)['TID']
        tidTropicalLikeCycloneTrack = self.tracks_.filter(
            pl.col('index').is_in(idxTropicalLikeCyclone)).group_by(
                'TID', maintain_order=True).count().filter(
                    pl.col('count') >= 2 * convertRate)['TID']

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

        idxBoundaryTrack = self.tracks_.filter(
                    pl.col('BoundaryRatio') >= 0.8)['index']
        idxCloseBoundaryTrack = self.tracks_.filter(
                    (pl.col('BoundaryRatio') < 0.8) & 
                    (pl.col('BoundaryRatio') > 0.4))['index']
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
        labelTrack[idxBoundaryTrack] += '_BDY'
        labelTrack[idxCloseBoundaryTrack] += '_CBDY'
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
                'MSLP', 'WS10', 'FullLabel', 'ShortLabel', 'HALabel',
                'TropicalFlag', 'TransitionZone', 'TrackInfo',
                'Area', 'Bid']
            if self.regional_:
                newColums.append('BoundaryMask')
            self.tracks_ = self.tracks_[newColums]
        if resultFormat == 'parquet':
            self.tracks_.write_parquet(f'{filename}.parquet')
        elif resultFormat == 'csv':
            self.tracks_.write_csv(f'{filename}.csv')
        elif resultFormat == 'excel':
            self.tracks_.write_excel(f'{filename}.xlsx')
        elif resultFormat == 'all':
            self.tracks_.write_parquet(f'{filename}.parquet')
            self.tracks_.write_excel(f'{filename}.xlsx')
            try:
                self.tracks_.write_csv(f'{filename}.csv')
            except:
                pass
        self.lpsMaskBlobStats_.write_excel(f'{filename}_LPSMask.xlsx')