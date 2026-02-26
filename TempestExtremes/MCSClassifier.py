import itertools
import logging
import os
import time
from typing import Optional

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
import xarray as xr

from Base import geo2XYZOnUnitSphere, getBoundary

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MCSClassifier():
    def __init__(self,
                 blobStatsOutput: str,
                 syCloPSOutput: str,
                 z0File: str,
                 classifierType: str = 'MCS',
                 frontProcessedOutput: str = None,
                 mcsProcessedOutput: str = None,
                 lonName: Optional[str] = None,
                 latName: Optional[str] = None,
                 regional: bool = False,
                 nproc: Optional[int] = 4, 
                 timeInterval: Optional[int] = 3,
                 rhTropicalThreshold: float = 20):
        self.currentBlobStatsFile_ = blobStatsOutput
        self.lpsTrackFile_ = syCloPSOutput
        self.frontTrackFile_ = frontProcessedOutput
        self.mcsProcessedFile_ = mcsProcessedOutput
        self.nproc_ = nproc
        self.timeInterval_ = timeInterval
        self.rhTropicalThreshold_ = rhTropicalThreshold
        self.z0File_ = z0File
        self.type_ = classifierType.lower()
        self.regional_ = regional
        self.lonName_ = lonName
        self.latName_ = latName
        self.currentTracks_ = None
        self.lpsTracks_ = None
        self.frontTracks_ = None
        self.mcsTracks_ = None
        self.nNode_ = -1
        self.blobStats_ = None
        self.blobPairs_ = []
        self.boundaryKDTree_ = None
        self.withFront_ = False
        if frontProcessedOutput is not None:
            self.withFront_ = True
        self.loadData()

    def loadData(self) -> None:
        if self.lpsTrackFile_.endswith('.xlsx'):
            self.lpsTracks_ = pl.read_excel(self.lpsTrackFile_)
        elif self.lpsTrackFile_.endswith('.parquet'):
            self.lpsTracks_ = pl.read_parquet(self.lpsTrackFile_)

        if self.regional_:
            self.z0Ds_ = xr.load_dataset(self.z0File_)
            if self.lonName_ is not None:
                self.z0Ds_ = self.z0Ds_.rename_vars(
                    {self.lonName_: 'longitude'})
            if self.latName_ is not None:
                self.z0Ds_ = self.z0Ds_.rename_vars(
                    {self.latName_: 'latitude'})
                
        self.loadBlobStatsOutput()
        if self.withFront_:
            self.loadFrontProcessedInfo()

    def loadBlobStatsOutput(self) -> None:
        self.currentTracks_ = pl.read_csv(
            self.currentBlobStatsFile_, separator='\t', has_header=False, null_values='nan')
        self.currentTracks_ = self.currentTracks_.drop(
            'column_2')
        if self.type_ == 'mcs':
            if self.withFront_:
                self.currentTracks_.columns = [
                    'McsTid', 'Time', 'CenLon', 'CenLat',
                    'MinLat', 'MaxLat', 'MinLon', 'MaxLon', 'Area',
                    'Vo850Sum', 'Vo850Avg', 'LPSBlobs', 'FrontalBlobs',
                ]
            else:
                self.currentTracks_.columns = [
                    'McsTid', 'Time', 'CenLon', 'CenLat',
                    'MinLat', 'MaxLat', 'MinLon', 'MaxLon', 'Area',
                    'Vo850Sum', 'Vo850Avg', 'LPSBlobs'
                ]
        elif self.type_ == 'front':
            self.currentTracks_.columns = [
                'FrontTid', 'Time', 'CenLon', 'CenLat',
                'MinLat', 'MaxLat', 'MinLon', 'MaxLon', 'Area',
                'RH100Sum', 'RH100Avg', 'LPSBlobs'
            ]
        self.currentTracks_ = self.currentTracks_.with_columns(
            pl.col('Time').str.to_datetime(
                format='%Y-%m-%d %H:%M:%S').alias('UTC'))
        self.currentTracks_ = self.currentTracks_.drop('Time')
        self.currentTracks_ = self.currentTracks_.with_row_index()

    def loadFrontProcessedInfo(self):
        self.frontTracks_ = pl.read_parquet(self.frontTrackFile_)

    def loadMcsProcessedInfo(self):
        self.mcsTracks_ = pl.read_parquet(self.mcsProcessedFile_)

    def pairLpsInfo(self):
        currentDatetime = self.currentTracks_['UTC']
        currentDatetime = np.unique(currentDatetime.to_numpy()).tolist()
        self.lpsTracks_ = self.lpsTracks_.filter(pl.col('UTC').is_in(currentDatetime))
        result = pl.DataFrame({
            'index': [], 'LPSIndex': [], 'LPSOverlap': [], 'LPSTID': [], 
            'LPSLabel': [], 'HALabel': []}, 
            schema={
                'index': pl.Int32, 
                'LPSIndex': pl.List(pl.Int32), 
                'LPSOverlap': pl.List(pl.Float32),
                'LPSTID': pl.List(pl.Int32),
                'LPSLabel': pl.List(pl.String),
                'HALabel': pl.List(pl.String)})
        for iTime in currentDatetime:
            iLpsDf = self.lpsTracks_.filter(pl.col('UTC') == iTime)
            iMcsDf = self.currentTracks_.filter(pl.col('UTC') == iTime)
            for itemMcs in iMcsDf.rows(named=True):
                mcsPairStrs = itemMcs['LPSBlobs'].split(';')[:-1]
                mcsPair = dict()
                for i in mcsPairStrs:
                    k, v = i.split(':')
                    if int(k) > 0:
                        mcsPair.update({int(k): float(v)})
                matchLps = []
                matchOverlap = []
                matchTid = []
                matchLabel = []
                matchHALabel = []
                for itemLps in iLpsDf.rows(named=True):
                    idxLps = itemLps['index']
                    tidLps = itemLps['TID']
                    labelLps = itemLps['ShortLabel']
                    labelHA = itemLps['HALabel']
                    idxBlob = itemLps['Bid']
                    blobIntersect = np.intersect1d(list(mcsPair.keys()), idxBlob)
                    if len(blobIntersect) > 0:
                        totalOverlap = 0
                        for j in blobIntersect:
                            totalOverlap += mcsPair[j]
                        matchLps.append(idxLps)
                        matchOverlap.append(totalOverlap)
                        matchTid.append(tidLps)
                        matchLabel.append(labelLps)
                        matchHALabel.append(labelHA)
                if matchLps:
                    iResult = pl.DataFrame({
                        'index': [itemMcs['index']],
                        'LPSIndex': [matchLps],
                        'LPSOverlap': [matchOverlap],
                        'LPSTID': [matchTid],
                        'LPSLabel': [matchLabel],
                        'HALabel': [matchHALabel]},
                        schema={
                            'index': pl.Int32, 
                            'LPSIndex': pl.List(pl.Int32), 
                            'LPSOverlap': pl.List(pl.Float32),
                            'LPSTID': pl.List(pl.Int32),
                            'LPSLabel': pl.List(pl.String),
                            'HALabel': pl.List(pl.String)})
                    result.extend(iResult)
        return result
    
    def pairFrontInfo(self):
        timerStart = time.time()
        mcsDatetime = self.currentTracks_['UTC']
        mcsDatetime = np.unique(mcsDatetime.to_numpy()).tolist()
        self.frontTracks_ = self.frontTracks_.filter(pl.col('UTC').is_in(mcsDatetime))
        result = pl.DataFrame({
            'index': [], 'FrontTid': [], 'FrontOverlap': [], 'FrontRH100Avg': [],
            'FrontLPSIndex': [], 'FrontLPSTID': [], 'FrontLPSLabel': [], 
            'FrontHALabel': []}, 
            schema={
                'index': pl.Int32, 
                'FrontTid': pl.List(pl.Int32),
                'FrontOverlap': pl.List(pl.Float32),
                'FrontRH100Avg': pl.List(pl.Float32),
                'FrontLPSIndex': pl.List(pl.List(pl.Int32)), 
                'FrontLPSTID': pl.List(pl.List(pl.Int32)), 
                'FrontLPSLabel': pl.List(pl.List(pl.String)),
                'FrontHALabel': pl.List(pl.List(pl.String))})
        for iTime in mcsDatetime:
            iFrontDf = self.frontTracks_.filter(pl.col('UTC') == iTime)
            iMcsDf = self.currentTracks_.filter(pl.col('UTC') == iTime)
            for itemMcs in iMcsDf.rows(named=True):
                mcsPairStrs = itemMcs['FrontalBlobs'].split(';')[:-1]
                mcsPair = dict()
                for i in mcsPairStrs:
                    k, v = i.split(':')
                    if int(k) > 0:
                        mcsPair.update({int(k): float(v)})
                matchFront = []
                matchFrontRH = []
                matchOverlap = []
                matchTid = []
                matchLabel = []
                matchFrontLpsIdx = []
                matchHALabel = []
                for itemFront in iFrontDf.rows(named=True):
                    idxFrontalBlob = itemFront['FrontTid']
                    frontRH = itemFront['RH100Avg']
                    idxLpsList = itemFront['LPSIndex']
                    tidLpsList = itemFront['LPSTID']
                    labelLpsList = itemFront['LPSLabel']
                    labelHaList = itemFront['HALabel']
                    blobIntersect = np.intersect1d(list(mcsPair.keys()), idxFrontalBlob)
                    if len(blobIntersect) > 0:
                        totalOverlap = 0
                        for j in blobIntersect:
                            totalOverlap += mcsPair[j]
                        matchFront.append(idxFrontalBlob)
                        matchFrontRH.append(frontRH)
                        matchOverlap.append(totalOverlap)
                        matchFrontLpsIdx.append(idxLpsList)
                        matchTid.append(tidLpsList)
                        matchLabel.append(labelLpsList)
                        matchHALabel.append(labelHaList)
                if matchFront:
                    iResult = pl.DataFrame({
                        'index': [itemMcs['index']],
                        'FrontTid': [matchFront],
                        'FrontOverlap': [matchOverlap],
                        'FrontRH100Avg': [matchFrontRH],
                        'FrontLPSIndex': [matchFrontLpsIdx],
                        'FrontLPSTID': [matchTid],
                        'FrontLPSLabel': [matchLabel],
                        'FrontHALabel': [matchHALabel]},
                        schema={
                            'index': pl.Int32, 
                            'FrontTid': pl.List(pl.Int32),
                            'FrontOverlap': pl.List(pl.Float32),
                            'FrontRH100Avg': pl.List(pl.Float32),
                            'FrontLPSIndex': pl.List(pl.List(pl.Int32)), 
                            'FrontLPSTID': pl.List(pl.List(pl.Int32)), 
                            'FrontLPSLabel': pl.List(pl.List(pl.String)),
                            'FrontHALabel': pl.List(pl.List(pl.String))})
                    result.extend(iResult)
        timerEnd = time.time()
        logger.warning(f'Time lasped for mergeInfoMcs step: {timerEnd - timerStart} s.')
        return result
    
    def mergeInfoMcs(self):
        timerStart = time.time()
        result = pl.DataFrame({
            'index': [], 'FinalType': [], 'FinalLPSLabel': [], 'AuxLabel': [],
            'HybridFlag': []}, 
            schema={
                'index': pl.Int32, 
                'FinalType': pl.String,
                'FinalLPSLabel': pl.List(pl.String),
                'AuxLabel': pl.List(pl.String),
                'HybridFlag': pl.Int32})
        for iMcs in self.currentTracks_.rows(named=True):
            frontIdx = iMcs['FrontTid']
            finalType = ''
            auxLabel = []
            finalLabel = []
            flagHybrid = 0
            if len(frontIdx) > 0:
                lpsIndex = iMcs['LPSIndex']
                lpsLabels = iMcs['LPSLabel']
                haLabels = iMcs['HALabel']
                frontLpsIndex = iMcs['FrontLPSIndex']
                frontLpsLabel = iMcs['FrontLPSLabel']
                frontRH = iMcs['FrontRH100Avg']
                frontHALabel = iMcs['FrontHALabel']
                finalType = 'Front'
                iFrontLpsList = []
                iFrontLpsLabelList = []
                iFrontRHList = []
                iFrontHALabelList = []
                for iCount, iFront in enumerate(frontIdx):
                    if len(frontLpsIndex[iCount]) > 0:
                        iFrontLpsList.extend(frontLpsIndex[iCount])
                        iFrontLpsList = list(set(iFrontLpsList))
                        iFrontLpsLabelList.extend(frontLpsLabel[iCount])
                        iFrontLpsLabelList = list(set(iFrontLpsLabelList))
                        iFrontRHList.extend(frontRH)
                        iFrontRHList = list(set(iFrontRHList))
                        iFrontHALabelList.extend(frontHALabel[iCount])
                        iFrontHALabelList = list(set(iFrontHALabelList))
                if len(lpsIndex) > 0:
                    finalType = 'Direct'
                    if len(lpsLabels) > 1:
                        mixLabels = []
                        for iLpsLabel, iHaLabel in zip(lpsLabels, haLabels):
                            iMixLabel = f'{iLpsLabel}_{iHaLabel}'
                            mixLabels.append(iMixLabel)
                        mixLabels = sorted(set(mixLabels))
                        lpsLabels = []
                        haLabels = []
                        for iMixLabel in mixLabels:
                            iLpsLabel, iHaLabel = iMixLabel.split('_')
                            lpsLabels.append(iLpsLabel)
                            haLabels.append(iHaLabel)
                        flagHybrid = 1
                    finalLabel = lpsLabels
                    auxLabel = haLabels
                elif len(iFrontLpsList) > 0:
                    finalType = 'Indirect'
                    lpsLabels = []
                    haLabels = []
                    if len(iFrontLpsList) == 1:
                        lpsLabels = iFrontLpsLabelList
                        haLabels = iFrontHALabelList
                    elif len(iFrontLpsList) > 1:
                        # In some cases of a sticky front area, a distance detection 
                        # is needed.
                        lpsPairs = list(itertools.combinations(iFrontLpsList, 2))
                        if len(lpsPairs) > 2:
                            mcsIdx = iMcs['index']
                            logger.warning(f'Multiple LPSs: MCS index {mcsIdx}')
                        for iPair in lpsPairs:
                            iLpsIdx1 = iPair[0]
                            iLpsIdx2 = iPair[1]
                            iLps1 = self.lpsTracks_.filter(index=iLpsIdx1)
                            iLps2 = self.lpsTracks_.filter(index=iLpsIdx2)
                            iLpsLon1 = iLps1.item(0,'Lon')
                            iLpsLat1 = iLps1.item(0,'Lat')
                            iLpsLon2 = iLps2.item(0,'Lon')
                            iLpsLat2 = iLps2.item(0,'Lat')
                            iMcsLon = iMcs['CenLon']
                            iMcsLat = iMcs['CenLat']
                            disLps = np.sqrt((iLpsLon1 - iLpsLon2)**2 + (iLpsLat1 - iLpsLat2)**2)
                            dis1 = np.sqrt((iLpsLon1 - iMcsLon)**2 + (iLpsLat1 - iMcsLat)**2)
                            dis2 = np.sqrt((iLpsLon2 - iMcsLon)**2 + (iLpsLat2 - iMcsLat)**2)
                            if dis1 < disLps and dis2 < disLps:
                                lpsLabels.append(iLps1.item(0,'ShortLabel'))
                                lpsLabels.append(iLps2.item(0,'ShortLabel'))
                                haLabels.append(iLps1.item(0,'HALabel'))
                                haLabels.append(iLps2.item(0,'HALabel'))

                            elif dis1 < dis2 and dis1 < disLps:
                                lpsLabels.append(iLps1.item(0,'ShortLabel'))
                                haLabels.append(iLps1.item(0,'HALabel'))
                            elif dis2 <= dis1 and dis2 < disLps:
                                lpsLabels.append(iLps2.item(0,'ShortLabel'))
                                haLabels.append(iLps2.item(0,'HALabel'))
                            else:
                                mcsIdx = iMcs['index']
                                logger.warning(f'Special situation: MCS index {mcsIdx}')
                                if dis1 < dis2:
                                    lpsLabels.append(iLps1.item(0,'ShortLabel'))
                                    haLabels.append(iLps1.item(0,'HALabel'))
                                else:
                                    lpsLabels.append(iLps2.item(0,'ShortLabel'))
                                    haLabels.append(iLps2.item(0,'HALabel'))
                        if len(lpsLabels) > 1:
                            flagHybrid = 1
                    
                    mixLabels = []
                    for iLpsLabel, iHaLabel in zip(lpsLabels, haLabels):
                        iMixLabel = f'{iLpsLabel}_{iHaLabel}'
                        mixLabels.append(iMixLabel)
                    mixLabels = sorted(set(mixLabels))
                    lpsLabels = []
                    haLabels = []
                    for iMixLabel in mixLabels:
                        iLpsLabel, iHaLabel = iMixLabel.split('_')
                        lpsLabels.append(iLpsLabel)
                        haLabels.append(iHaLabel)
                    finalLabel = lpsLabels
                    auxLabel = haLabels
                else:
                    finalLabel = ['Front']
                    auxLabel = ['']
                    for iFrontRH in frontRH:
                        if iFrontRH >= self.rhTropicalThreshold_:
                            auxLabel = ['MS']

                iResult = pl.DataFrame({
                    'index': [iMcs['index']], 
                    'FinalType': [finalType], 
                    'FinalLPSLabel': [finalLabel],
                    'AuxLabel': [auxLabel],
                    'HybridFlag': [flagHybrid]}, 
                    schema={
                        'index': pl.Int32, 
                        'FinalType': pl.String,
                        'FinalLPSLabel': pl.List(pl.String),
                        'AuxLabel': pl.List(pl.String),
                        'HybridFlag': pl.Int32})
                result.extend(iResult)
            else:
                if len(iMcs['LPSIndex']) > 0:
                    lpsLabels = iMcs['LPSLabel']
                    haLabels = iMcs['HALabel']
                    finalType = 'Direct'
                    if len(lpsLabels) > 1:
                        mixLabels = []
                        for iLpsLabel, iHaLabel in zip(lpsLabels, haLabels):
                            iMixLabel = f'{iLpsLabel}_{iHaLabel}'
                            mixLabels.append(iMixLabel)
                        mixLabels = sorted(set(mixLabels))
                        lpsLabels = []
                        haLabels = []
                        for iMixLabel in mixLabels:
                            iLpsLabel, iHaLabel = iMixLabel.split('_')
                            lpsLabels.append(iLpsLabel)
                            haLabels.append(iHaLabel)
                        flagHybrid = 1
                    
                    finalLabel = lpsLabels
                    auxLabel = haLabels

                    iResult = pl.DataFrame({
                        'index': [iMcs['index']], 
                        'FinalType': [finalType],
                        'HybridFlag': [flagHybrid],
                        'AuxLabel': [auxLabel],
                        'FinalLPSLabel': [finalLabel]}, 
                    schema={
                        'index': pl.Int32, 
                        'FinalType': pl.String,
                        'FinalLPSLabel': pl.List(pl.String),
                        'AuxLabel': pl.List(pl.String),
                        'HybridFlag': pl.Int32})
                    result.extend(iResult)
        
        self.currentTracks_ = self.currentTracks_.with_columns(
                    pl.lit('').alias('FinalType'),
                    pl.lit(0).alias('HybridFlag'),
                    FinalLPSLabel=[],
                    AuxLabel=[])
        self.currentTracks_ = self.currentTracks_.update(result, on='index')
        timerEnd = time.time()
        logger.warning(f'Time lasped for pairFrontInfo step: {timerEnd - timerStart} s.')
    
    def pairMcsInfo(self):
        timerStart = time.time()
        precipDatetime = self.currentTracks_['UTC']
        precipDatetime = np.unique(precipDatetime.to_numpy()).tolist()
        self.mcsTracks_ = self.mcsTracks_.filter(pl.col('UTC').is_in(precipDatetime))
        result = pl.DataFrame({
            'index': [], 'McsTid': [], 'McsOverlap': [], 'HybridFlag': [], 
            'LPSTID': [], 'FinalType': [], 'FinalLPSLabel': [], 
            'AuxLabel': []}, 
            schema={
                'index': pl.Int32, 
                'McsTid': pl.List(pl.Int32),
                'McsOverlap': pl.List(pl.Float32),
                'HybridFlag': pl.Int32,
                'LPSTID': pl.List(pl.Int32),
                'FinalType': pl.String,
                'FinalLPSLabel': pl.List(pl.String), 
                'AuxLabel': pl.List(pl.String)})
        for iTime in precipDatetime:
            iMcsDf = self.mcsTracks_.filter(pl.col('UTC') == iTime)
            iPrecipDf = self.currentTracks_.filter(pl.col('UTC') == iTime)
            for iPrecipDf in iPrecipDf.rows(named=True):
                precipPairStrs = iPrecipDf['MCSBlobs'].split(';')[:-1]
                precipPair = dict()
                for i in precipPairStrs:
                    k, v = i.split(':')
                    if int(k) > 0:
                        precipPair.update({int(k): float(v)})
                matchMcs = []
                matchOverlap = []
                matchFinalLps = []
                matchFinalType = []
                matchLabel = []
                matchHybridFlag = 0
                matchAuxLabel = []
                for itemMcs in iMcsDf.rows(named=True):
                    idxMcsBlob = itemMcs['McsTid']
                    flagHybrid = itemMcs['HybridFlag']
                    mcsFrontalLpsTidList = itemMcs['FrontLPSTID']
                    if len(mcsFrontalLpsTidList) > 0:
                        newMcsFrontalLpsTidList = []
                        for i in mcsFrontalLpsTidList:
                            newMcsFrontalLpsTidList.extend(i)
                        mcsFrontalLpsTidList = newMcsFrontalLpsTidList
                    mcsLpsTidList = itemMcs['LPSTID']
                    finalLps = np.intersect1d(mcsFrontalLpsTidList, mcsLpsTidList).tolist()
                    finalType = itemMcs['FinalType']
                    labelLpsList = itemMcs['FinalLPSLabel']
                    labelAuxList = itemMcs['AuxLabel']
                    blobIntersect = np.intersect1d(list(precipPair.keys()), idxMcsBlob)
                    if len(blobIntersect) > 0:
                        totalOverlap = 0
                        for j in blobIntersect:
                            totalOverlap += precipPair[j]
                        matchMcs.append(idxMcsBlob)
                        matchOverlap.append(totalOverlap)
                        matchHybridFlag = np.logical_or(matchHybridFlag, flagHybrid).astype(int)
                        matchFinalLps.extend(finalLps)
                        matchFinalType.append(finalType)
                        matchLabel.extend(labelLpsList)
                        matchAuxLabel.extend(labelAuxList)
                if matchMcs:
                    matchFinalLps = list(set(matchFinalLps))
                    matchFinalType = list(set(matchFinalType))
                    matchFinalType = [i for i in matchFinalType if len(i) > 0]
                    matchLabel = list(set(matchLabel))
                    matchLabel = [i for i in matchLabel if len(i) > 0]
                    if len(matchFinalType) > 1:
                        logger.warning(f'Precipitation blob {iPrecipDf['index']} has multiple matched types: {matchFinalType}.')
                        if 'Direct' in matchFinalType:
                            matchFinalType = 'Direct'
                        elif 'Indirect' in matchFinalType:
                            matchFinalType = 'Indirect'
                        elif 'Front' in matchFinalType:
                            matchFinalType = 'Front'
                        else:
                            matchFinalType = matchFinalType[0]
                    elif len(matchFinalType) == 1:
                        matchFinalType = matchFinalType[0]
                    else:
                        matchFinalType = 'MCS'
                    iResult = pl.DataFrame({
                        'index': [iPrecipDf['index']],
                        'McsTid': [matchMcs],
                        'McsOverlap': [matchOverlap],
                        'HybridFlag': [matchHybridFlag],
                        'LPSTID': [matchFinalLps], 
                        'FinalType': [matchFinalType], 
                        'FinalLPSLabel': [matchLabel],
                        'AuxLabel': [matchAuxLabel]},
                        schema={
                            'index': pl.Int32, 
                            'McsTid': pl.List(pl.Int32),
                            'McsOverlap': pl.List(pl.Float32),
                            'HybridFlag': pl.Int32,
                            'LPSTID': pl.List(pl.Int32),
                            'FinalType': pl.String,
                            'FinalLPSLabel': pl.List(pl.String), 
                            'AuxLabel': pl.List(pl.String)})
                    result.extend(iResult)
        timerEnd = time.time()
        logger.warning(f'Time lasped for pairMcsInfo step: {timerEnd - timerStart} s.')
        return result

    def preprocess(self, filename: Optional[str] = None, override: bool = False):
        if not os.path.exists(filename) or override:
            if self.regional_:
                if len(self.z0Ds_['longitude'].shape) > 1:
                    z0Lon2D = self.z0Ds_['longitude'].to_numpy()
                    z0Lat2D = self.z0Ds_['latitude'].to_numpy()
                else:
                    z0Lon2D, z0Lat2D = np.meshgrid(
                        self.z0Ds_['longitude'], self.z0Ds_['latitude'])
                boundaryPoints = getBoundary(z0Lon2D, z0Lat2D)
                boundaryPoints = geo2XYZOnUnitSphere(
                    boundaryPoints[:,0].flatten(), boundaryPoints[:,1].flatten())
                self.boundaryKDTree_ = cKDTree(boundaryPoints)
                trackPoints = geo2XYZOnUnitSphere(
                    self.currentTracks_['CenLon'], self.currentTracks_['CenLat'])
                idx = self.boundaryKDTree_.query_ball_point(
                        trackPoints, r=2.0*np.pi/180.0)
                maskCloseToBoundary = np.array([len(i) > 0 for i in idx], dtype=bool)
                trackBoundaryMask = np.zeros(len(trackPoints), dtype=int)
                trackBoundaryMask[maskCloseToBoundary] = 1
                self.currentTracks_ = self.currentTracks_.with_columns(
                    BoundaryMask = trackBoundaryMask
                )

            pairDf = self.pairLpsInfo()
            self.currentTracks_ = self.currentTracks_.with_columns(
                LPSIndex=[], LPSOverlap=[], LPSTID=[], LPSLabel=[], HALabel=[])
            self.currentTracks_ = self.currentTracks_.update(pairDf, on='index')
            if self.type_ == 'mcs':
                if self.withFront_:
                    pairDf = self.pairFrontInfo()
                    self.currentTracks_ = self.currentTracks_.with_columns(
                        FrontTid=[], FrontOverlap=[], FrontRH100Avg=[], 
                        FrontLPSIndex=[], FrontLPSTID=[], FrontLPSLabel=[], 
                        FrontHALabel=[])
                    self.currentTracks_ = self.currentTracks_.update(pairDf, on='index')
                    self.mergeInfoMcs()

            filename, suffix = filename.rsplit('.', 1)
            self.currentTracks_.write_parquet(f'{filename}.parquet')
            self.currentTracks_.write_excel(f'{filename}.xlsx')
        else:
            if filename.endswith('parquet'):
                self.currentTracks_ = pl.read_parquet(filename)
            elif filename.endswith('xlsx'):
                self.currentTracks_ = pl.read_excel(filename)