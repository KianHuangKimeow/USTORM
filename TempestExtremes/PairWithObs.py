import datetime
import logging
import warnings

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from scipy import stats

from Base import geo2XYZOnUnitSphere

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def trackPairWithObs(
        trackDf: pl.DataFrame, obsDf: pl.DataFrame,
        tidName: str = 'TID', sidName: str = 'Sid', 
        beginTime: datetime.datetime = None, 
        endTime: datetime.datetime = None,
        appendRecords: str | list = None, 
        renameRecords: str | list = None):
    if appendRecords is not None:
        if not isinstance(appendRecords, list):
            appendRecords = [appendRecords]
        if renameRecords is not None:
            if not isinstance(renameRecords, list):
                renameRecords = [renameRecords]
            assert(len(appendRecords) == len(renameRecords))

    if beginTime is not None:
        trackDf = trackDf.filter(pl.col('UTC') >= beginTime)
        obsDf = obsDf.filter(pl.col('UTC') >= beginTime)
    if endTime is not None:
        trackDf = trackDf.filter(pl.col('UTC') <= endTime)
        obsDf = obsDf.filter(pl.col('UTC') <= endTime)

    obsSid = np.unique(obsDf[sidName])
    trackTid = np.unique(trackDf[tidName])
    obsSidGroup = obsDf.group_by(sidName, maintain_order=True)
    trackTidGroup = trackDf.group_by(tidName, maintain_order=True)

    # Compare the datetime of tracks and observations
    obsGroupTime1 = obsSidGroup.first()['UTC'].to_numpy()
    obsGroupTime2 = obsSidGroup.last()['UTC'].to_numpy()

    trackGroupTime1 = trackTidGroup.first()['UTC'].to_numpy()
    trackGroupTime2 = trackTidGroup.last()['UTC'].to_numpy()

    timePair = dict()
    for iSid, obsT1, obsT2 in zip(obsSid, obsGroupTime1, obsGroupTime2):
        timePair[iSid] = []
        for iTid, trackT1, trackT2 in zip(trackTid, trackGroupTime1, trackGroupTime2):
            start = obsT1 if (obsT1 - trackT1) > 0 else trackT1
            end = obsT2 if (obsT2 - trackT2) < 0 else trackT2
            if start <= end:
                timePair[iSid].append(iTid)

    finalMatch = dict()
    for iSid, iTidList in timePair.items():
        iObs = obsDf.filter(pl.col(sidName) == iSid)
        obsTime = iObs['UTC'].to_numpy()
        disList = []
        corrList = []
        nMatchList = []
        for iTid in iTidList:
            iTrack = trackDf.filter(pl.col(tidName) == iTid)
            trackTime = iTrack['UTC'].to_numpy()
            timeIntersect = np.intersect1d(obsTime, trackTime).tolist()
            obsIntersect = iObs.filter(pl.col('UTC').is_in(timeIntersect))
            trackIntersect = iTrack.filter(pl.col('UTC').is_in(timeIntersect))
            obsLon = obsIntersect['Lon'].to_numpy()
            obsLat = obsIntersect['Lat'].to_numpy()
            obsPoints = geo2XYZOnUnitSphere(obsLon, obsLat)
            trackLon = trackIntersect['Lon'].to_numpy()
            trackLat = trackIntersect['Lat'].to_numpy()
            trackPoints = geo2XYZOnUnitSphere(trackLon, trackLat)
            dis = np.sqrt(np.sum(np.power(obsPoints - trackPoints, 2), axis=1))
            nMatch = np.sum(dis < (2.0 * np.pi / 180.0))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore', 
                    message=('An input array is constant; '
                            'the correlation coefficient is not defined.'))
                corr = np.abs(stats.pearsonr(trackPoints, obsPoints, axis=1).statistic)
            disList.append(np.mean(dis))
            nMatchList.append(nMatch)
            corrList.append(corr)
        iCandidate = np.argwhere(np.array(nMatchList) > 0)
        if len(iCandidate) == 1:
            finalMatch.update({iTidList[iCandidate[0][0]]: iSid})
        elif len(iCandidate) > 1:
            iCandidate = np.argmax(np.bincount([
                np.argmax(np.array(nMatchList))[0], np.argmax(np.array(corrList))[0],
                np.argmin(np.array(disList))[0]]))[0]
            if len(iCandidate) == 1:
                finalMatch.update({iTidList[iCandidate[0]]: iSid})
            else:
                raise Exception('Cannot find the candidate.')

    trackDf = trackDf.with_columns(
        ObsSid = pl.lit(None),
        ObsLon = pl.lit(None),
        ObsLat = pl.lit(None),
    )
    if appendRecords is not None:
        newRecords = appendRecords if renameRecords is None else renameRecords
        for i in newRecords:
            trackDf = trackDf.with_columns(
                new = pl.lit(None),
            ).rename({'new': i})
    for iTid, iSid in finalMatch.items():
        iTrack = trackDf.filter(pl.col(tidName) == iTid)
        trackTime = iTrack['UTC'].to_numpy()
        iObs = obsDf.filter(pl.col(sidName) == iSid)
        obsTime = iObs['UTC'].to_numpy()
        timeIntersect = np.intersect1d(obsTime, trackTime).tolist()
        obsIntersect = iObs.filter(pl.col('UTC').is_in(timeIntersect))
        trackIntersect = iTrack.filter(pl.col('UTC').is_in(timeIntersect))
        
        appendDf = trackIntersect.with_columns(
            ObsSid = obsIntersect['Sid'],
            ObsLon = obsIntersect['Lon'],
            ObsLat = obsIntersect['Lat']
        )
        if appendRecords is not None:
            for i, j in zip(appendRecords, newRecords):
                appendDf = appendDf.with_columns(
                    obsIntersect[i]
                )
                if i!=j:
                    appendDf = appendDf.rename({i: j})
        trackDf = trackDf.update(appendDf, on='index')
    return trackDf