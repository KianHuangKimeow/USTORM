import datetime
import logging
import matplotlib.dates as mdates
import os
import sys

import numpy as np
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, os.path.abspath('.'))

from Base import npDatetimeToDatetime
from Postprocessing import densityMap
from Visualization import Plot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

labelColorMap = {
    'HATHL': 'gold', 'HAL': 'beige',
    'THL': 'yellow', 'DOTHL': 'tan',
    'DST': 'silver', 'DSD': 'khaki', 'DSE': 'wheat',
    'TD': 'blue', 'TLO': 'skyblue', 'TD(MD)': 'aqua', 'TLO(ML)': 'cadetblue',
    'TC': 'orangered',
    'EX': 'lime', 'PL(PTLC)': 'hotpink', 
    'SC': 'olive', 'SS(STLC)': 'mediumorchid',
    'Others': 'grey', 'Front': 'silver'
}

combinedLabelGroup = {
    'HighAltitude': ['HATHL', 'HAL'],
    'Dry': ['DSD', 'DOTHL', 'THL'],
    'Tropical': ['DST', 'TC', 'TD(MD)', 'TD', 'TLO(ML)', 'TLO'],
    'Extratropical': ['DSE', 'SS(STLC)', 'PL(PTLC)', 'SC', 'EX'],
    'Front': ['Front'],
    'MonsoonTrough': ['MonsoonTrough'],
}

highAltitudeLabelGroup = {
    'Dry': ['Dry'],
    'HighAltitude': ['Extreme'],
    'Tropical': ['Tropical'],
    'Extratropical': ['DSE', 'EX', 'Subtropical'],
}

combinedLabelColorMap = {
    'Dry': 'gold',
    'HighAltitude': 'brown',
    'Tropical': 'orangered',
    'Extratropical': 'lime',
    'Front': 'silver',
    'MonsoonTrough': 'wheat',
    'Others': 'grey',
    'All': 'skyblue',
}

regionCodeMap = {
    'Northeast': 1,
    'Southeast': 2, 'Midwest': 3, 'Northern Great Plains': 4, 
    'Southern Plains': 5, 'Southwest': 6, 'Northwest': 7
}

def findGroup(lps: str):
    for i, j in combinedLabelGroup.items():
        if lps in j:
            return i
        
def findGroupHA(aux: str):
    for i, j in highAltitudeLabelGroup.items():
        if aux in j:
            return i

def calculateNYear(beginTime, endTime):
    nYear = 1
    dYear = endTime.year - beginTime.year
    if (dYear >= 1):
        nYear = dYear - 1
        nDayOfYearHead = (datetime.date(beginTime.year, 12, 31) - beginTime.date()).days + 1
        nDayOfYearTail = (endTime.date() - datetime.date(endTime.year, 1, 1)).days + 1
        nYearHead = nDayOfYearHead / datetime.date(beginTime.year, 12, 31).timetuple().tm_yday
        nYearTail= nDayOfYearTail / datetime.date(endTime.year, 12, 31).timetuple().tm_yday
        nYear = nYear + nYearHead + nYearTail
        logger.warning(f'The amount of years: {nYear}')
    if nYear <= 1:
        nYear = 1
        logger.warning(f'Because nYear < 1, the program will not calculate annual average.')
    return nYear

def plotMCSFrequency(
        df: pl.DataFrame, figDir: str,
        beginTime: datetime.datetime = None,
        endTime: datetime.datetime = None,
        flagGroup: bool = True,
        flagHACat: bool = True,
        flagMS: bool = True,
        timeGroup: str = 'month',
        title: str = None,
        panelConfig: dict = None, plot: Plot = None,
        figsize: tuple = (30, 9), yMax: int = 0,
        regions: list = None):
    pathLabelGroup = ''
    pathLabelHA = ''
    pathLabelMS = ''
    pathRegion = ''
    if flagGroup:
        pathLabelGroup = '_Group'
    if flagHACat:
        pathLabelHA = '_HACat'
    if flagMS:
        pathLabelMS = '_MS'

    figPath = os.path.join(
        workDir, f'MCSFrequency_Series_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}{pathLabelMS}_{pathRegion}{timeGroup.title()}.pdf')
    dfPath = os.path.join(
        figDir, f'MCSFrequency_Series_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}{pathLabelMS}_{pathRegion}{timeGroup.title()}.csv')
    
    dfStat = pl.DataFrame(
        {'time': [], 'type': [], 'amount':[], 'percentage': []},
        schema={
            'time': pl.Datetime,
            'type': pl.String,
            'amount': pl.Int64,
            'percentage': pl.Float64})
    
    if plot is None:
        plot = Plot(figsize=figsize, dpi=640, fontFamily='Arial')
        plot.setPlot()
    if panelConfig is not None:
        nRow = panelConfig.get('nRow')
        nCol = panelConfig.get('nCol')
        currentPanel = panelConfig.get('currentPanel')
        plot.switchAx(
            (nRow, nCol), (currentPanel // nCol, np.mod(currentPanel, nCol)))
        
    if beginTime is not None:
        df = df.filter(pl.col('UTC') >= beginTime)
    if endTime is not None:
        df = df.filter(pl.col('UTC') <= endTime)

    # Remove MCSs close to the boundary.
    df = df.filter(~(pl.col('BoundaryMask') == 1))
    lpsLabelUniqueList = []
    lpsLabelColorList = []
    if flagMS:
        combinedLabelGroup.update({'MonsoonTrough': ['MonsoonTrough']})
    if flagGroup:
        for i in combinedLabelGroup.keys():
            lpsLabelUniqueList.append(i)
            lpsLabelColorList.append(combinedLabelColorMap[i])
    else:
        lpsLabelUniqueArray = df.unique(subset='FinalLPSLabel')['FinalLPSLabel'].to_numpy()
        for i in lpsLabelUniqueArray:
            if len(i) > 1:
                logger.warning(f'Hybrid MCS {i}')
                # Exception(f'Hybrid MCS {i}')
            elif len(i) == 1:
                lpsLabelUniqueList.append(i[0])
                lpsLabelColorList.append(labelColorMap[i[0]])

    if timeGroup in ['month', 'mon']:
        dfGroupMonth = df.group_by(pl.col('UTC').dt.month(), maintain_order=True)
    elif timeGroup in ['none', 'all']:
        dfGroupMonth = df.group_by(pl.col('UTC').is_not_null(), maintain_order=True)
    countGroupMonth = dfGroupMonth.len()
    monthList = countGroupMonth['UTC'].to_numpy()
    monthList = np.sort(monthList)
    monthAllList = []
    monthHybridList = []
    monthHybridPercentage = []
    width = 20
    iMonth = 0
    yMax = yMax
    others = []
    otherPercentage = []
    x = []
    for currentMon in monthList:
        for i in dfGroupMonth:
            iMon = np.unique(i[1]['UTC'].dt.month())[0]
            if currentMon == iMon:
                others.append(0)
                otherPercentage.append(0)
                nYear = len(np.unique(i[1]['UTC'].dt.year()))
                dfGroupLabel = i[1].group_by('FinalLPSLabel','AuxLabel')
                dfGroupLabelDirect = i[1].filter(FinalType='Direct').group_by(
                    'FinalLPSLabel','AuxLabel')
                dfGroupLabelIndirect = i[1].filter(FinalType='Indirect').group_by(
                    'FinalLPSLabel','AuxLabel')
                labelCount = dfGroupLabel.len()
                labelCountDirect = dfGroupLabelDirect.len()
                labelCountIndirect = dfGroupLabelIndirect.len()
                monthAll = np.round(labelCount['len'].to_numpy().sum() / nYear).astype(np.int64)
                yMap = dict()
                yMapDirect = dict()
                yMapIndirect = dict()
                yMapPercentage = dict()
                yMapPercentageDirect = dict()
                yMapPercentageIndirect = dict()
                hybridCount = 0
                for j in lpsLabelUniqueList:
                    yMap[j] = 0
                    yMapDirect[j] = 0
                    yMapIndirect[j] = 0
                    yMapPercentage[j] = 0
                    yMapPercentageDirect[j] = 0
                    yMapPercentageIndirect[j] = 0
                #  Check whether there are any hybrid type MCSs
                for j in labelCount.rows():
                    jRowLabel = ''
                    jRowLabelList = j[0]
                    jRowAuxLabelList = j[1]
                    jRowCount = np.round(j[2]  / nYear).astype(np.int64)
                    try:
                        jRowCountDirect = np.round(labelCountDirect.filter(FinalLPSLabel=jRowLabelList).item(0, 'len')  / nYear).astype(np.int64)
                    except:
                        jRowCountDirect = 0
                    try:
                        jRowCountIndirect = np.round(labelCountIndirect.filter(FinalLPSLabel=jRowLabelList).item(0, 'len')  / nYear).astype(np.int64)
                    except:
                        jRowCountIndirect = 0
                    if len(jRowLabelList) == 0:
                        others[-1] = jRowCount
                        otherPercentage[-1] = int(jRowCount / monthAll * 100)
                    elif len(jRowLabelList) == 1:
                        jRowLabel = jRowLabelList[0]
                        jAuxLabel = jRowAuxLabelList[0]
                        if flagGroup:
                            groupLabel = findGroup(jRowLabel)
                            if flagHACat and (groupLabel == 'HighAltitude'):
                                groupLabel = findGroupHA(jAuxLabel)
                        else:
                            groupLabel = jRowLabel
                        if flagMS and groupLabel == 'Front' and jAuxLabel == 'MS':
                            groupLabel = 'MonsoonTrough'
                        yMap[groupLabel] += jRowCount
                        yMapDirect[groupLabel] += jRowCountDirect
                        yMapIndirect[groupLabel] += jRowCountIndirect
                        
                        if yMap[groupLabel] > yMax:
                            yMax = yMap[groupLabel]
                    else:
                        hybridCount += jRowCount

                for j in lpsLabelUniqueList:
                    if flagGroup:
                        yMapPercentage[j] = f'{np.round(yMap[j] / monthAll * 100).astype(int)}\n%'
                        yMapPercentageDirect[j] = f'{np.round(yMapDirect[j] / monthAll * 100).astype(int)}\n%'
                        yMapPercentageIndirect[j] = f'{np.round(yMapIndirect[j] / monthAll * 100).astype(int)}\n%'
                    else:
                        yMapPercentage[j] = f'{np.round(yMap[j] / monthAll * 100).astype(int)}\n%'
                        yMapPercentageDirect[j] = f'{np.round(yMapDirect[j] / monthAll * 100).astype(int)}\n%'
                        yMapPercentageIndirect[j] = f'{np.round(yMapIndirect[j] / monthAll * 100).astype(int)}\n%'
                    if timeGroup in ['month', 'mon']:
                        iTime = datetime.date(year=1970, month=iMonth+1, day=1)
                    elif timeGroup in ['none', 'all']:
                        iTime = datetime.datetime(year=1970, month=1, day=1)
                    dfStat = dfStat.extend(
                        pl.DataFrame(
                            {'time': [iTime], 'type': [j], 'amount': [yMap[j]], 'percentage': [yMap[j] / monthAll]},
                            schema={
                                'time': pl.Datetime,
                                'type': pl.String,
                                'amount': pl.Int64,
                                'percentage': pl.Float64}))
                    
                monthAllList.append(monthAll)
                monthHybridList.append(hybridCount)
                monthHybridPercentage.append(int(hybridCount / monthAll * 100))
                
                finalLabelList = [i for i in lpsLabelUniqueList]
                if flagGroup:
                    for i in range(len(lpsLabelUniqueList)):
                        iLabel = lpsLabelUniqueList[i]
                        iLabel = 'Extreme High Altitude' if iLabel == 'HighAltitude' else iLabel
                        iLabel = 'Monsoon Trough' if iLabel == 'MonsoonTrough' else iLabel
                        finalLabelList[i] = iLabel
                
                plot.bar(
                    list(yMap.values()), offset=width*len(lpsLabelUniqueList)*iMonth + width*iMonth, width=width, 
                    color=lpsLabelColorList, label=finalLabelList, yLimit=(0,yMax*(1+0.05)), showValue=True,
                    valueConfig=dict(value=list(yMapPercentage.values()), format='', yloc=8, 
                                     fontsize=8), 
                    zorder=10)
                plot.bar(
                    list(yMapIndirect.values()), offset=width*len(lpsLabelUniqueList)*iMonth + width*iMonth, width=width, 
                    color=lpsLabelColorList, label=finalLabelList, yLimit=(0,yMax*(1+0.05)), showValue=False,
                    valueConfig=dict(value=list(yMapPercentageIndirect.values()), format='', yLoc=1.5), 
                    hatch='///', 
                    edgecolor='black', zorder=100)
                x.append(width*(len(lpsLabelUniqueList)*(iMonth+0.5)+0.5) + width*iMonth)
                iMonth += 1

    dfStat = dfStat.sort(by=['time', 'percentage'], descending=[False, True])
    dfStat.write_csv(dfPath)

    xLabels = []
    for i, j, k, l, n, m in zip(monthList, monthAllList, others, otherPercentage, monthHybridList, monthHybridPercentage):
        if timeGroup in ['month', 'mon']:
            iLabel = f'{datetime.date(year=2000, month=i, day=1).strftime('%b')} ({j})\n'
        elif timeGroup in ['none', 'all']:
            iLabel = 'All\n'
        iLabel = iLabel + f'O: {k} ({l}%)'
        if n > 0:
            iLabel = iLabel + f'\nH: {n} ({m}%) '
        xLabels.append(iLabel)

    plot.currentAx_.set_xlim(min(x) - width * len(lpsLabelUniqueList) / 2, max(x) + width * len(lpsLabelUniqueList) / 2)
    plot.currentAx_.set_xticks(x, xLabels)
    plot.currentAx_.set_xlabel('Month')
    plot.currentAx_.set_ylabel('Count')
    plot.legend(loc='upper left')
        
    if title:
        if panelConfig is not None:
            titleSize = 14
            if len(title) > 50:
                titleSize = 11
            plot.title(title, fontsize=titleSize, location=[0, 1.085])
        else:
            plot.title(title)

    if panelConfig is None:
        plot.save(figPath)
        plot.clear()
    else:
        if panelConfig.get('drawNumber', False):
            plot.panelNumber(currentPanel, location=[0.5, -0.1], fontsize=10)
        return plot

def plotMCSTimeSeries(
        filename: str, figDir: str,
        beginTime: datetime.datetime = None,
        endTime: datetime.datetime = None,
        flagGroup: bool = True,
        flagHACat: bool = True,
        timeGroup: str = 'month',
        title: str = None,
        xlim: tuple = None,
        panelConfig: dict = None, plot: Plot = None,
        figsize: tuple = (20, 4), yMax: int = 0, 
        yMin2: int = 0, yMax2: int = 0,
        flagPlot: bool = False,
        windowSize: int = 1,
        rollingMode: str = 'mean',
        kernal: list | np.ndarray = np.array([1, 4, 6, 4, 1]),
        regions: list = None
    ):
    pathLabelGroup = ''
    pathLabelHA = ''
    pathRegion = ''
    pathPlotType = ''
    if flagGroup:
        pathLabelGroup = '_Group'
    if flagHACat:
        pathLabelHA = '_HACat'
    if flagMS:
        pathLabelMS = '_MS'
    if flagPlot:
        pathPlotType = '_Plot'
    figPath = os.path.join(
        figDir, f'MCSTime_Series{pathPlotType}_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}{pathLabelMS}_{pathRegion}{timeGroup.title()}.pdf')
    dfPath = os.path.join(figDir, f'MCSTime_Series_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}{pathLabelMS}_{pathRegion}{timeGroup.title()}.csv')

    dfStat = pl.DataFrame(
        {'time': [], 'type': [], 'amount':[], 'percentage': []},
        schema={
            'time': pl.Datetime,
            'type': pl.String,
            'amount': pl.Int64,
            'percentage': pl.Float64})
    if plot is None:
        plot = Plot(figsize=figsize, dpi=640, fontFamily='Arial')
        plot.setPlot()
    if panelConfig is not None:
        nRow = panelConfig.get('nRow', 1)
        nCol = panelConfig.get('nCol', 1)
        currentPanel = panelConfig.get('currentPanel')
        if nRow * nCol > 1:
            plot.switchAx(
                (nRow, nCol), (currentPanel // nCol, np.mod(currentPanel, nCol)))
    
    if filename.endswith('.csv'):
        df = pl.read_csv(filename, null_values='nan', try_parse_dates=True)
    else:
        df = pl.read_parquet(filename)

    if beginTime is not None:
        df = df.filter(pl.col('UTC') >= beginTime)
    if endTime is not None:
        df = df.filter(pl.col('UTC') <= endTime)

    df = df.sort(['UTC'])
    # Remove MCSs close to the boundary.
    df = df.filter(~(pl.col('BoundaryMask') == 1))
    lpsLabelUniqueList = []
    lpsLabelColorList = []
    if flagMS:
        combinedLabelGroup.update({'MonsoonTrough': ['MonsoonTrough']})
    if flagGroup:
        for i in combinedLabelGroup.keys():
            lpsLabelUniqueList.append(i)
            lpsLabelColorList.append(combinedLabelColorMap[i])
    else:
        lpsLabelUniqueArray = df.unique(subset='FinalLPSLabel')['FinalLPSLabel'].to_numpy()
        for i in lpsLabelUniqueArray:
            if len(i) > 1:
                logger.info(f'Hybrid MCS {i}')
            elif len(i) == 1:
                lpsLabelUniqueList.append(i[0])
                lpsLabelColorList.append(labelColorMap[i[0]])

    if timeGroup in ['month', 'mon']:
        dfGroupMonth = df.group_by_dynamic(
            'UTC', every='1mo', closed='left', label='left')
    elif timeGroup in ['year']:
        dfGroupMonth = df.group_by_dynamic('UTC', every='1y', closed='left', label='left')
    elif timeGroup in ['wyear']:
        firstTimeStep = df.head(1)['UTC'][0]
        lastTimeStep = df.tail(1)['UTC'][0]
        firstWyear = firstTimeStep.year
        lastWyear = lastTimeStep.year
        if firstTimeStep.month < 10:
            df = df.filter(pl.col('UTC') >= datetime.datetime(firstWyear, 10, 1))
        elif firstTimeStep.month > 10:
            df = df.filter(pl.col('UTC') >= datetime.datetime(firstWyear+1, 10, 1))
        if lastTimeStep.month < 9:
            df = df.filter(pl.col('UTC') < datetime.datetime(lastWyear-1, 10, 1))
        elif lastTimeStep.month > 9:
            df = df.filter(pl.col('UTC') < datetime.datetime(lastWyear, 10, 1))
        offsetWyear = datetime.datetime(firstWyear, 10, 1) - datetime.datetime(firstWyear, 1, 1)
        dfGroupMonth = df.group_by_dynamic('UTC', every='1y', closed='left', label='left', offset=offsetWyear)
    elif timeGroup in ['none', 'all']:
        dfGroupMonth = df.group_by(pl.col('UTC').is_not_null(), maintain_order=True)

    monthAllList = []
    monthHybridList = []
    monthHybridPercentage = []
    yMap = dict()
    yMapDirect = dict()
    yMapIndirect = dict()
    yMapPercentage = dict()
    yMapPercentageDirect = dict()
    yMapPercentageIndirect = dict()
    others = []
    otherPercentage = []
    for j in lpsLabelUniqueList:
        yMap[j] = []
        yMapDirect[j] = []
        yMapIndirect[j] = []
        yMapPercentage[j] = []
        yMapPercentageDirect[j] = []
        yMapPercentageIndirect[j] = []
    x = []
    for i in dfGroupMonth:
        iMon = npDatetimeToDatetime(np.unique(i[0])[0]).replace(tzinfo=None)
        if timeGroup in ['wyear']:
            iMon = datetime.datetime(iMon.year+1, 1, 1)
        x.append(iMon)
        dfGroupLabel = i[1].group_by('FinalLPSLabel','AuxLabel')
        dfGroupLabelDirect = i[1].filter(FinalType='Direct').group_by(
            'FinalLPSLabel','AuxLabel')
        dfGroupLabelIndirect = i[1].filter(FinalType='Indirect').group_by(
            'FinalLPSLabel','AuxLabel')
        labelCount = dfGroupLabel.len()
        labelCountDirect = dfGroupLabelDirect.len()
        labelCountIndirect = dfGroupLabelIndirect.len()
        monthAll = labelCount['len'].to_numpy().sum()
        hybridCount = 0
        for j in lpsLabelUniqueList:
            yMap[j].append(0)
            yMapDirect[j].append(0)
            yMapIndirect[j].append(0)
            yMapPercentage[j].append(0)
            yMapPercentageDirect[j].append(0)
            yMapPercentageIndirect[j].append(0)
        others.append(0)
        otherPercentage.append(0)
        for j in labelCount.rows():
            jRowLabel = ''
            jRowLabelList = j[0]
            jRowAuxLabelList = j[1]
            jRowCount = j[2]
            try:
                jRowCountDirect = labelCountDirect.filter(FinalLPSLabel=jRowLabelList).item(0, 'len')
            except:
                jRowCountDirect = 0
            try:
                jRowCountIndirect = labelCountIndirect.filter(FinalLPSLabel=jRowLabelList).item(0, 'len')
            except:
                jRowCountIndirect = 0
            if len(jRowLabelList) == 0:
                others[-1] = jRowCount
                otherPercentage[-1] = int(jRowCount / monthAll * 100)
            elif len(jRowLabelList) == 1:
                jRowLabel = jRowLabelList[0]
                jAuxLabel = jRowAuxLabelList[0]
                if flagGroup:
                    groupLabel = findGroup(jRowLabel)
                    if flagHACat and (groupLabel == 'HighAltitude'):
                        groupLabel = findGroupHA(jAuxLabel)
                else:
                    groupLabel = jRowLabel
                if flagMS and groupLabel == 'Front' and jAuxLabel == 'MS':
                    groupLabel = 'MonsoonTrough'
                yMap[groupLabel][-1] += jRowCount
                yMapDirect[groupLabel][-1] += jRowCountDirect
                yMapIndirect[groupLabel][-1] += jRowCountIndirect

                if yMap[groupLabel][-1] > yMax:
                    yMax = yMap[groupLabel][-1]
            else:
                hybridCount += jRowCount

        for j in lpsLabelUniqueList:
            yMapPercentage[j][-1] = yMap[j][-1] / monthAll * 100
            yMapPercentageDirect[j][-1] = yMapDirect[j][-1] / monthAll * 100
            yMapPercentageIndirect[j][-1] = yMapIndirect[j][-1] / monthAll * 100
            dfStat = dfStat.extend(
                pl.DataFrame(
                    {'time': [iMon], 'type': [j], 'amount': [yMap[j][-1]], 'percentage': [yMapPercentage[j][-1]]},
                    schema={
                        'time': pl.Datetime,
                        'type': pl.String,
                        'amount': pl.Int64,
                        'percentage': pl.Float64}))
            
        monthAllList.append(monthAll)
        monthHybridList.append(hybridCount)
        monthHybridPercentage.append(int(hybridCount / monthAll * 100))

    yMap['Others'] = others
    yMapPercentage['Others'] = otherPercentage
    for iMon, iValue, iPercentage in zip(x, yMap['Others'], yMapPercentage['Others']):
        dfStat = dfStat.extend(
            pl.DataFrame(
                {'time': [iMon], 'type': ['Others'], 'amount': [iValue], 'percentage': [iPercentage]},
                schema={
                    'time': pl.Datetime,
                    'type': pl.String,
                    'amount': pl.Int64,
                    'percentage': pl.Float64}))
    lpsLabelUniqueList.append('Others')
    lpsLabelColorList.append(combinedLabelColorMap['Others'])
    finalLabelList = [i for i in lpsLabelUniqueList]
    if flagGroup:
        for i in range(len(lpsLabelUniqueList)):
            allTypeInGroup = combinedLabelGroup.get(lpsLabelUniqueList[i], lpsLabelUniqueList[i])
            iLabel = lpsLabelUniqueList[i]
            iLabel += '('
            for j in allTypeInGroup:
                iLabel += f'{j}, '
            iLabel = iLabel[:-2] + ')'
            finalLabelList[i] = iLabel
        
    dfStat = dfStat.sort(by=['time', 'percentage'], descending=[False, True])
    dfStat.write_csv(dfPath)

    xlabel = 'Date'
    if timeGroup in ['wyear']:
        xlabel = 'Date (Water Year)'
    plot.currentAx_.set_xlabel(xlabel)
    if flagPlot:
        for j, jColor in zip(lpsLabelUniqueList, lpsLabelColorList):
            if rollingMode == 'mean':
                currentValueList = pl.Series(
                    'Value', yMap[j], pl.Float64).rolling_mean(windowSize, center=True).to_numpy()
            elif rollingMode == 'absmax':
                currentValueList = pl.DataFrame({'Value': yMap[j]}, schema={'Value': pl.Float64})
                currentValueList = currentValueList.with_columns(
                    pl.col('Value').rolling_max(windowSize, center=True).alias('ValueMax'))
                currentValueList = currentValueList.with_columns(
                    pl.col('Value').rolling_min(windowSize, center=True).alias('ValueMin'))
                currentValueList = currentValueList.with_columns(
                    pl.when(
                        pl.col('ValueMax').abs() >= pl.col('ValueMin').abs()
                    ).then(pl.col('ValueMax')).otherwise(pl.col('ValueMin')).alias('ValueFinal'))
                currentValueList = currentValueList['ValueFinal'].to_numpy()
            elif rollingMode == 'sum':
                currentValueList = pl.Series(
                    'Value', yMap[j], pl.Float64).rolling_sum(windowSize, center=True).to_numpy()
            elif rollingMode == 'lowpass':
                currentValueList = np.array(yMap[j], dtype=np.float64)
                kernal = kernal / np.sum(kernal, dtype=np.float64)
                currentValueList = np.convolve(currentValueList, kernal, mode='same')
                kernalHalfLength = len(kernal) // 2
                currentValueList[0:kernalHalfLength] = np.nan
                currentValueList[-kernalHalfLength:] = np.nan
            jLabel = j
            jLabel = 'Extreme High Altitude' if jLabel == 'HighAltitude' else jLabel
            jLabel = 'Monsoon Trough' if jLabel == 'MonsoonTrough' else jLabel
            plot.currentAx_.plot(x, currentValueList, label=jLabel, color=jColor, linewidth=1.5)
            yMax = np.max([np.nanmax(currentValueList), yMax])

        plot.currentAx_.set_ylim(0, yMax)
        plot.axes_[0] = plot.currentAx_
        plot.currentAx_ = plot.axes_[0].twinx()
        plot.axes_[1] = plot.currentAx_
        if rollingMode == 'mean':
            currentValueList = pl.Series(
                'Value', monthAllList, pl.Float64).rolling_mean(windowSize, center=True).to_numpy()
        elif rollingMode == 'absmax':
            currentValueList = pl.DataFrame({'Value': monthAllList}, schema={'Value': pl.Float64})
            currentValueList = currentValueList.with_columns(
                pl.col('Value').rolling_max(windowSize, center=True).alias('ValueMax'))
            currentValueList = currentValueList.with_columns(
                pl.col('Value').rolling_min(windowSize, center=True).alias('ValueMin'))
            currentValueList = currentValueList.with_columns(
                pl.when(
                    pl.col('ValueMax').abs() >= pl.col('ValueMin').abs()
                ).then(pl.col('ValueMax')).otherwise(pl.col('ValueMin')).alias('ValueFinal'))
            currentValueList = currentValueList['ValueFinal'].to_numpy()
        elif rollingMode == 'sum':
            currentValueList = pl.Series(
                'Value', monthAllList, pl.Float64).rolling_sum(windowSize, center=True).to_numpy()
        elif rollingMode == 'lowpass':
            currentValueList = np.array(monthAllList, dtype=np.float64)
            kernal = kernal / np.sum(kernal, dtype=np.float64)
            currentValueList = np.convolve(currentValueList, kernal, mode='same')
            kernalHalfLength = len(kernal) // 2
            currentValueList[0:kernalHalfLength] = np.nan
            currentValueList[-kernalHalfLength:] = np.nan
        yMin2 = np.min([np.nanmin(currentValueList), yMin2])
        yMax2 = np.max([np.nanmax(currentValueList), yMax2])
        plot.plot(
                x=x, y=currentValueList, label='All', yLimit=(-yMax2, yMax2+20), 
                color='black', linestyle='dashed', linewidth=2, xAxisConfig=dict(
                    ticks=dict(
                        type='datetime',
                        format='%Y')))
        plot.currentAx_.set_ylim(yMin2, yMax2)
    else:
        plot.currentAx_.stackplot(
            x, yMap.values(), labels=lpsLabelUniqueList, colors=lpsLabelColorList)
        plot.currentAx_.set_yticks(np.arange(0, 2000, 100))
        plot.currentAx_.set_ylim(0, yMax)
    if timeGroup in ['month', 'mon']:
        plot.currentAx_.xaxis.set_major_locator(mdates.YearLocator(2, month=6))
        plot.currentAx_.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plot.currentAx_.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        plot.currentAx_.tick_params(axis='x', which='major', width=1.5, length=6)
        plot.currentAx_.tick_params(axis='x', which='minor', width=1, length=3)
    elif timeGroup in ['year', 'wyear']:
        plot.currentAx_.xaxis.set_major_locator(mdates.YearLocator(2))
        plot.currentAx_.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plot.currentAx_.tick_params(axis='x', which='major', width=1.5, length=6)

    for l in plot.currentAx_.get_xticklabels():
        l.set_rotation(45)
    if flagPlot:
        plot.legend(bbox_to_anchor=[0.97, 1], loc='upper left', axis='all')
    else:
        plot.legend(bbox_to_anchor=[1, 1], loc='upper left')
    
    if flagPlot:
        plot.axes_[0].set_ylabel('Count')
        plot.currentAx_.set_ylabel('Count (All)')
    else:
        plot.currentAx_.set_ylabel('Count')

    if xlim:
        plot.currentAx_.set_xlim(xlim)
        
    if title:
        if panelConfig is not None:
            titleSize = 14
            if len(title) > 50:
                titleSize = 11
            plot.title(title, fontsize=titleSize, location=[0, 1.085])
        else:
            plot.title(title)

    if panelConfig is None:
        plot.save(figPath)
        plot.clear()
    else:
        if panelConfig.get('drawNumber', False):
            plot.panelNumber(currentPanel, location=[0.5, -0.088], fontsize=12)
        if panelConfig.get('save', False):
            plot.save(figPath)
        return plot

def plotDensistyMap(filename: str, figPath: str, map: str = 'conus404', 
                    labelType: str | list = None,
                    flagGroup: bool = True,
                    flagHACat: bool = True,
                    flagNonMS: bool = False,
                    flagMS: bool = False,
                    finalType: str = None,
                    beginTime: datetime.datetime = None,
                    endTime: datetime.datetime = None,
                    months: list = None,
                    levels: list = None, cmapScale: str = 'linear',
                    maxDensity: int = None, over: str = None,
                    title: str = None,
                    panelConfig: dict = None, plot: Plot = None,
                    figsize: tuple = (16, 9)):
    if plot is None:
        plot = Plot(figsize=figsize, dpi=640, fontFamily='Arial')
    if panelConfig is not None:
        nRow = panelConfig.get('nRow')
        nCol = panelConfig.get('nCol')
        currentPanel = panelConfig.get('currentPanel')
        plot.switchAx(
            (nRow, nCol), (currentPanel // nCol, np.mod(currentPanel, nCol)),
            onHoldForSetMap=True)
    plot.setMap(map=map)
    plot.drawMesh(fontsize=10)
    plot.drawFeature(['coastline', 'country', 'state'])
    if filename.endswith('.csv'):
        df = pl.read_csv(filename, null_values='nan', try_parse_dates=True)
    else:
        df = pl.read_parquet(filename)

    if beginTime is not None:
        df = df.filter(pl.col('UTC') >= beginTime)
    if endTime is not None:
        df = df.filter(pl.col('UTC') <= endTime)

    nYear = 1
    if beginTime is not None and endTime is not None:
        nYear = calculateNYear(beginTime, endTime)

    # Remove MCSs close to the boundary.
    df = df.filter(~(pl.col('BoundaryMask') == 1))
    
    if labelType is not None:
        if not isinstance(labelType, list):
            labelType = [labelType]
        finalLabel = []
        auxLabel = []
        if flagGroup:
            for i in labelType:
                if i in combinedLabelGroup.keys():
                    if not (flagHACat and i == 'HighAltitude'):
                        finalLabel.extend(combinedLabelGroup[i])
                if flagHACat and i in highAltitudeLabelGroup.keys():
                    auxLabel.extend(highAltitudeLabelGroup[i])
        else:
            finalLabel = labelType
        
        if 'Others' in labelType and len(labelType) == 1:
            df = df.filter(
                pl.col('FinalLPSLabel').list.len() == 0
            )
        else:
            df = df.filter(
                (pl.col('FinalLPSLabel').list.set_intersection(finalLabel).list.len() > 0) | 
                    ((pl.col('FinalLPSLabel').list.set_intersection(['HATHL', 'HAL']).list.len() > 0) &
                    (pl.col('AuxLabel').list.set_intersection(auxLabel).list.len() > 0))
                )
    
    if finalType is not None:
        df = df.filter(pl.col('FinalType') == finalType)

    if flagNonMS:
        df = df.filter(pl.col('AuxLabel').list.set_intersection(['MS']).list.len() == 0)

    if flagMS:
        df = df.filter(pl.col('AuxLabel').list.set_intersection(['MS']).list.len() > 0)

    if months is not None:
        df = df.filter(pl.col('UTC').dt.month().is_in(months))

    lon = df['CenLon'].to_numpy()
    lat = df['CenLat'].to_numpy()
    gridLon = np.arange(0.5, 360.0, 1.0)
    gridLat = np.arange(-89.5, 90.0, 1.0)
    gridLon, gridLat = np.meshgrid(gridLon, gridLat)
    density = densityMap(lon, lat, gridLon, gridLat) / nYear
    densityMaxValue = np.round(np.max(density), decimals=1)

    over = over
    if levels is None:
        maxDensity = np.nanmax(density) if maxDensity is None else maxDensity
        if maxDensity <= 10:
            levels = np.linspace(0, maxDensity, 5*maxDensity+1) if levels is None else levels        
        elif maxDensity < 100:
            if maxDensity <= 20:
                levels = np.linspace(0, maxDensity, int(maxDensity)+1)
            elif maxDensity <= 25:
                levels = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
            elif maxDensity <= 30:
                levels = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
            elif maxDensity < 50:
                levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50]
            else:
                levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, maxDensity]
        elif maxDensity < 300:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300, 500]
        elif maxDensity < 500:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300, 500]
        else:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300, 500]
            over = 'auto'
    if cmapScale == 'log':
        levels[0] = 1
    plot.pcolormesh(
        density, gridLon, gridLat, cmap='WhiteBlueGreenYellowRed.rgb', 
        cmapConfig=dict(unit='1°$^{-2}$', over=over),
        levels=levels, cmapScale=cmapScale, cbar=True, cbarConfig=dict(
            fontsize=10, unitPos=[1.07, 1.01]))
    plot.figure_.text(
        0.75, 1.01, f'Max: {densityMaxValue:.1f}', 
        fontsize=12, horizontalalignment='left', verticalalignment='bottom',
        family='Source Sans Pro',
        transform=plot.currentAx_.transAxes)
    if title:
        if panelConfig is not None:
            titleSize = 14
            if len(title) > 50:
                titleSize = 11
            plot.title(title, fontsize=titleSize, location=[0, 1.01])
        else:
            plot.title(title)
        
    if panelConfig is None:
        plot.save(figPath)
        plot.clear()
    else:
        if panelConfig.get('drawNumber', False):
            plot.panelNumber(currentPanel, location=[0.5, -0.08], fontsize=14)
        return plot

def statsModelWithTimeSeries(
        filename: str,
        figDir: str,
        beginTime: datetime.datetime = None,
        endTime: datetime.datetime = None,
        flagGroup: bool = True,
        flagHACat: bool = True,
        timeGroup: str = 'month',
        model: str = 'linear',
        regions: list = None):
    
    pathLabelGroup = ''
    pathLabelHA = ''
    pathRegion = ''
    if flagGroup:
        pathLabelGroup = '_Group'
    if flagHACat:
        pathLabelHA = '_HACat'
    pathModel = f'_{model}'
    
    dfPath = os.path.join(figDir, f'MCSStatsModel_Series_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}{pathModel}_{pathRegion}{timeGroup.title()}.csv')
    
    dfStat = pl.DataFrame(
        {'type': [], 'slope':[], 'r2': [], 'p': [], 
         'cv': [], 'mean': [], 'std': [],
         'std_residuals':[], },
        schema={
            'type': pl.String,
            'slope': pl.Float64,
            'r2': pl.Float64,
            'p': pl.Float64,
            'cv': pl.Float64,
            'mean': pl.Float64,
            'std': pl.Float64,
            'std_residuals': pl.Float64})
    
    if filename.endswith('.csv'):
        df = pl.read_csv(filename, null_values='nan', try_parse_dates=True)
    else:
        df = pl.read_parquet(filename)

    if beginTime is not None:
        df = df.filter(pl.col('UTC') >= beginTime)
    if endTime is not None:
        df = df.filter(pl.col('UTC') <= endTime)

    df = df.sort(['UTC'])
    df = df.filter(~(pl.col('BoundaryMask') == 1))

    lpsLabelUniqueList = []
    lpsLabelColorList = []
    if flagGroup:
        for i in combinedLabelGroup.keys():
            lpsLabelUniqueList.append(i)
            lpsLabelColorList.append(combinedLabelColorMap[i])
        lpsLabelUniqueList.append('All')
        lpsLabelColorList.append(combinedLabelColorMap['All'])
    else:
        lpsLabelUniqueArray = df.unique(subset='ShortLabel')['ShortLabel'].to_numpy()
        for i in lpsLabelUniqueArray:
            lpsLabelUniqueList.append(i)
            lpsLabelColorList.append(labelColorMap[i])

    if timeGroup in ['month', 'mon']:
        dfGroupMonth = df.group_by_dynamic(
            'UTC', every='1mo', closed='left', label='left')
    elif timeGroup in ['year']:
        dfGroupMonth = df.group_by_dynamic('UTC', every='1y', closed='left', label='left')
    elif timeGroup in ['wyear']:
        firstTimeStep = df.head(1)['UTC'][0]
        lastTimeStep = df.tail(1)['UTC'][0]
        firstWyear = firstTimeStep.year
        lastWyear = lastTimeStep.year
        if firstTimeStep.month < 10:
            df = df.filter(pl.col('UTC') >= datetime.datetime(firstWyear, 10, 1))
        elif firstTimeStep.month > 10:
            df = df.filter(pl.col('UTC') >= datetime.datetime(firstWyear+1, 10, 1))
        if lastTimeStep.month < 9:
            df = df.filter(pl.col('UTC') < datetime.datetime(lastWyear-1, 10, 1))
        elif lastTimeStep.month > 9:
            df = df.filter(pl.col('UTC') < datetime.datetime(lastWyear, 10, 1))

        offsetWyear = datetime.datetime(firstWyear, 10, 1) - datetime.datetime(firstWyear, 1, 1)
        dfGroupMonth = df.group_by_dynamic('UTC', every='1y', closed='left', label='left', offset=offsetWyear)
    elif timeGroup in ['none', 'all']:
        dfGroupMonth = df.group_by(pl.col('UTC').is_not_null(), maintain_order=True)

    monthAllList = []
    hybridList = []
    yMap = dict()
    yMapDirect = dict()
    yMapIndirect = dict()
    yMapCorrelation = dict()
    yMapPValue = dict()
    others = []
    for j in lpsLabelUniqueList:
        yMap[j] = []
        yMapDirect[j] = []
        yMapIndirect[j] = []
        yMapCorrelation[j] = []
        yMapPValue[j] = []
    yMapCorrelation['Others'] = []
    yMapPValue['Others'] = []
    x = []
    for i in dfGroupMonth:
        iMon = npDatetimeToDatetime(np.unique(i[0])[0]).replace(tzinfo=None)
        if timeGroup in ['wyear']:
            iMon = datetime.datetime(iMon.year+1, 1, 1)
        dfGroupLabel = i[1].group_by('FinalLPSLabel','AuxLabel')
        dfGroupLabelDirect = i[1].filter(FinalType='Direct').group_by(
            'FinalLPSLabel','AuxLabel')
        dfGroupLabelIndirect = i[1].filter(FinalType='Indirect').group_by(
            'FinalLPSLabel','AuxLabel')
        labelCount = dfGroupLabel.len()
        labelCountDirect = dfGroupLabelDirect.len()
        labelCountIndirect = dfGroupLabelIndirect.len()
        monthAll = labelCount['len'].to_numpy().sum()
        hybridCount = 0
        for j in lpsLabelUniqueList:
            yMap[j].append(0)
            yMapDirect[j].append(0)
            yMapIndirect[j].append(0)
        others.append(0)
        for j in labelCount.rows():
            jRowLabel = ''
            jRowLabelList = j[0]
            jRowAuxLabelList = j[1]
            jRowCount = j[2]
            try:
                jRowCountDirect = labelCountDirect.filter(FinalLPSLabel=jRowLabelList).item(0, 'len')
            except:
                jRowCountDirect = 0
            try:
                jRowCountIndirect = labelCountIndirect.filter(FinalLPSLabel=jRowLabelList).item(0, 'len')
            except:
                jRowCountIndirect = 0
            if len(jRowLabelList) == 0:
                others[-1] = jRowCount
            elif len(jRowLabelList) == 1:
                jRowLabel = jRowLabelList[0]
                jAuxLabel = jRowAuxLabelList[0]
                if flagGroup:
                    groupLabel = findGroup(jRowLabel)
                    if flagHACat and (groupLabel == 'HighAltitude'):
                        groupLabel = findGroupHA(jAuxLabel)
                else:
                    groupLabel = jRowLabel
                if flagMS and groupLabel == 'Front' and jAuxLabel == 'MS':
                    groupLabel = 'MonsoonTrough'
                yMap[groupLabel][-1] += jRowCount
                yMapDirect[groupLabel][-1] += jRowCountDirect
                yMapIndirect[groupLabel][-1] += jRowCountIndirect
            else:
                hybridCount += jRowCount

        monthAllList.append(monthAll)
        hybridList.append(hybridCount)

    yMap['All'] = np.array(monthAllList, dtype=np.int64).tolist()
    yMap['Others'] = np.array(others, dtype=np.float64).tolist()
    lpsLabelUniqueList.append('Others')
    lpsLabelColorList.append(combinedLabelColorMap['Others'])

    for j in lpsLabelUniqueList:
        currentValueList = np.array(yMap[j], dtype=np.float64)
        x = np.arange(len(currentValueList))
        currentX = sm.add_constant(x)
        if model == 'linear':
            currentModel = sm.OLS(currentValueList, currentX).fit()
        elif model == 'poisson':
            currentModel = sm.GLM(currentValueList, currentX, family=sm.families.Poisson()).fit()
        slope = currentModel.params[1]
        pValueSlope = currentModel.pvalues[1]
        rSquared = currentModel.rsquared
        mseResidual = currentModel.mse_resid
        stdResidual = np.sqrt(mseResidual)
        stdRaw = np.std(currentValueList)
        meanRaw = np.mean(currentValueList)
        stdRatio = stdRaw / meanRaw

        dfStat = dfStat.extend(
            pl.DataFrame(
                {
                    'type': [j], 'slope': [slope], 'r2': [rSquared], 
                    'p': [pValueSlope], 'cv': [stdRatio], 
                    'mean': [meanRaw], 'std': [stdRaw], 
                    'std_residuals': [stdResidual]},
                schema={
                    'type': pl.String,
                    'slope': pl.Float64,
                    'r2': pl.Float64,
                    'p': pl.Float64,
                    'cv': pl.Float64,
                    'mean': pl.Float64,
                    'std': pl.Float64, 
                    'std_residuals': pl.Float64}))
        if pValueSlope < 0.05:
            print(f'Group {j}:')
            print((f'Slope {slope}, r^2 {rSquared}, p Value {pValueSlope}, '
                  f'std_residuals {stdResidual}', f'std {stdRaw}'))

    dfStat = dfStat.sort(by=['type', 'p', 'slope'], descending=[False, True, False])
    dfStat.write_csv(dfPath)

if __name__ == '__main__':
    modelType = 'conus'
    timeGroup = 'mon'
    regions = None
    if modelType == 'obs':
        modelTypePrefix = 'OBS'
    elif modelType == 'rttov':
        modelTypePrefix = 'RTTOV'
    else:
        modelTypePrefix = 'CONUS'

    if timeGroup == 'mon':
        if modelType == 'conus':
            beginTimeStrList = ['19791001_00']
            endTimeStrList = ['20220930_23']
        elif modelType == 'rttov':
            beginTimeStrList = ['20210101_00']
            endTimeStrList = ['20211231_23']
        elif modelType == 'obs':
            beginTimeStrList = ['20000601_00']
            endTimeStrList = ['20220930_23']

    if modelType == 'conus':
        mcsFile = f'Result/CONUS/Preprocess_1979100100_2022093023.parquet'
    elif modelType == 'rttov':
        mcsFile = f'Result/RTTOV/Preprocess_2020100100_2022010100.parquet'
    elif modelType == 'obs':
        mcsFile = f'Result/OBS/Preprocess_2000060101_2022093023.parquet'
    workDir = f'Result/fig/{modelTypePrefix}'
    for beginTimeStr, endTimeStr in zip(beginTimeStrList, endTimeStrList):
        beginTime = datetime.datetime.strptime(beginTimeStr, "%Y%m%d_%H")
        flagGroup = True
        flagHACat = True
        flagMS = True
        figDir = workDir
        finalType = None #'Direct'
        if finalType is not None:
            finalTypeStr = f'_{finalType}'
        else:
            finalTypeStr = ''

        preprocessDir = f'{workDir}/preprocess'

        beginTime = datetime.datetime.strptime(beginTimeStr, "%Y%m%d_%H")
        endTime = datetime.datetime.strptime(endTimeStr, "%Y%m%d_%H")
        if mcsFile is None:
            mcsFile = os.path.join(workDir, f'Preprocess_{beginTime:%Y%m%d%H}_{endTime:%Y%m%d%H}.parquet')

        if mcsFile.endswith('.xlsx'):
            mcsTrack = pl.read_excel(mcsFile)
        elif mcsFile.endswith('.parquet'):
            mcsTrack = pl.read_parquet(mcsFile)

        plot = plotMCSFrequency(
            mcsTrack, figDir, beginTime, endTime, 
            flagGroup=flagGroup, flagHACat=flagHACat, flagMS=flagMS, 
            timeGroup=timeGroup, yMax=300, figsize=(18, 6))
        
        months = None
        currentPanel = 0
        panelConfig = dict(
            nRow=3, nCol= 3, currentPanel=0, drawNumber=True,
        )
        figPath = os.path.join(workDir, f'MCSDensityMap_All_{beginTime:%Y%m}_{endTime:%Y%m}.png')
        plot = plotDensistyMap(
            mcsFile, figPath, beginTime=beginTime, endTime=endTime, maxDensity=20, over='auto',
            title='All MCSs', panelConfig=panelConfig, figsize=(16, 12),
            months=months)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)

        pathLabelGroup = '_Group'
        pathLabelHA = '_HACat'
        # All MCS-Tropical
        figPath = os.path.join(
            workDir, f'MCSDensityMap_Tropical{finalTypeStr}{pathLabelGroup}{pathLabelHA}_{beginTime:%Y%m}_{endTime:%Y%m}.png')
        plot = plotDensistyMap(
            mcsFile, figPath, labelType='Tropical', flagGroup=True, flagHACat=True,
            finalType=finalType,
            beginTime=beginTime, endTime=endTime, maxDensity=3, over='auto',
            title='MCS-Tropical', panelConfig=panelConfig, plot=plot,
            months=months)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)

        # All MCS-Extratropical
        figPath = os.path.join(
            workDir, f'MCSDensityMap_EX{finalTypeStr}{pathLabelGroup}{pathLabelHA}_{beginTime:%Y%m}_{endTime:%Y%m}.png')
        plot = plotDensistyMap(
            mcsFile, figPath, labelType='Extratropical', flagGroup=True, flagHACat=True,
            finalType=finalType,
            beginTime=beginTime, endTime=endTime, maxDensity=4, over='auto',
            title='MCS-Extratropical', panelConfig=panelConfig, plot=plot,
            months=months)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)

        # All MCS-Dry
        figPath = os.path.join(
            workDir, f'MCSDensityMap_Dry{finalTypeStr}{pathLabelGroup}{pathLabelHA}_{beginTime:%Y%m}_{endTime:%Y%m}.png')
        plot = plotDensistyMap(
            mcsFile, figPath, labelType='Dry', flagGroup=True, flagHACat=True,
            finalType=finalType,
            beginTime=beginTime, endTime=endTime, maxDensity=2, over='auto',
            title='MCS-Dry', panelConfig=panelConfig, plot=plot,
            months=months)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)

        # All MCS-Others
        figPath = os.path.join(workDir, f'MCSDensityMap_Others_{beginTime:%Y%m}_{endTime:%Y%m}.png')
        plot = plotDensistyMap(
            mcsFile, figPath, labelType='Others', flagGroup=True, flagHACat=True,
            beginTime=beginTime, endTime=endTime, maxDensity=20, over='auto',
            title='MCS-Others', panelConfig=panelConfig, plot=plot,
            months=months)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)

        # All MCS-MT
        figPath = os.path.join(
            workDir, f'MCSDensityMap_MT{finalTypeStr}_{beginTime:%Y%m}_{endTime:%Y%m}.png')
        plot = plotDensistyMap(
            mcsFile, figPath, labelType='Front', flagGroup=True, flagHACat=True, flagMS=True,
            beginTime=beginTime, endTime=endTime, maxDensity=20, over='auto',
            title='MCS-MT', panelConfig=panelConfig, plot=plot,
            months=months)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)

        # All MCS-Front
        figPath = os.path.join(
            workDir, f'MCSDensityMap_Front{finalTypeStr}_{beginTime:%Y%m}_{endTime:%Y%m}.png')
        plot = plotDensistyMap(
            mcsFile, figPath, labelType='Front', flagGroup=True, flagHACat=True, flagNonMS=True,
            beginTime=beginTime, endTime=endTime, maxDensity=5, over='auto',
            title='MCS-Front', panelConfig=panelConfig, plot=plot,
            months=months)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)

        pathMonth = ''
        if months:
            pathMonth = '_mon_'
            for iMonth in months:
                pathMonth = f'{pathMonth}{iMonth}_'
            pathMonth = pathMonth[:-1]
        figPath = os.path.join(workDir, f'MCSDensityMap{finalTypeStr}{pathLabelGroup}{pathLabelHA}_{beginTime:%Y%m%d}_{endTime:%Y%m%d}{pathMonth}.pdf')
        
        plot.save(figPath)
        plot.clear()

        xLim1 = datetime.datetime.strptime("19791001_00", "%Y%m%d_%H")
        xLim2 = datetime.datetime.strptime("20220930_23", "%Y%m%d_%H")
        yMax = 2500
        yMin2 = 3400
        yMax2 = 7000
        if modelType == 'conus':
            currentPanel = 0
        else: 
            currentPanel = 1
        panelConfig = dict(
            currentPanel=currentPanel, drawNumber=True, save=True
        )
        plot = plotMCSTimeSeries(
            mcsFile, workDir, beginTime, endTime, flagGroup=flagGroup, 
            flagHACat=flagHACat, timeGroup='wyear', yMax=yMax,
            yMin2=yMin2, yMax2=yMax2,
            figsize=(11, 6), xlim=(xLim1, xLim2), 
            flagPlot=True, panelConfig=panelConfig,
            regions=regions)

        statsModelWithTimeSeries(
            mcsFile, workDir, beginTime, endTime, flagGroup=flagGroup, 
            flagHACat=flagHACat, timeGroup='wyear', model='linear',
            regions=regions
        )