import datetime
import logging
import matplotlib.dates as mdates
import os
import sys

import numpy as np
import polars as pl
from scipy.stats import pearsonr
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
    'Others': 'grey'
}

combinedLabelGroup = {
    'HighAltitude': ['HATHL', 'HAL'],
    'Tropical': ['DST', 'TC', 'TD(MD)', 'TD', 'TLO(ML)', 'TLO'],
    'Dry': ['DSD', 'DOTHL', 'THL'],
    'Extratropical': ['DSE', 'SS(STLC)', 'PL(PTLC)', 'SC', 'EX'],
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

def plotLPSFrequency(
        filename: str, figDir: str,
        beginTime: datetime.datetime = None,
        endTime: datetime.datetime = None,
        flagGroup: bool = True,
        flagHACat: bool = True,
        timeGroup: str = 'month',
        title: str = None,
        panelConfig: dict = None, plot: Plot = None,
        figsize: tuple = (30, 9), yMax: int = 0,
        regions: list = None):
    pathLabelGroup = ''
    pathLabelHA = ''
    pathRegion = ''
    if flagGroup:
        pathLabelGroup = '_Group'
    if flagHACat:
        pathLabelHA = '_HACat'
    if regions:
        for i in regions:
            pathRegion = f'{pathRegion}_{i}'
        pathRegion = f'{pathRegion[1:]}_'
    figPath = os.path.join(figDir, f'LPSFrequency_Series_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}_{pathRegion}{timeGroup.title()}.pdf')
    dfPath = os.path.join(figDir, f'LPSFrequency_Series_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}_{pathRegion}{timeGroup.title()}.csv')

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
    
    if filename.endswith('.csv'):
        df = pl.read_csv(filename, null_values='nan', try_parse_dates=True)
    else:
        df = pl.read_parquet(filename)

    if beginTime is not None:
        df = df.filter(pl.col('UTC') >= beginTime)
    if endTime is not None:
        df = df.filter(pl.col('UTC') <= endTime)
    
    lpsLabelUniqueList = []
    lpsLabelColorList = []
    if flagGroup:
        for i in combinedLabelGroup.keys():
            lpsLabelUniqueList.append(i)
            lpsLabelColorList.append(combinedLabelColorMap[i])
    else:
        lpsLabelUniqueArray = df.unique(subset='ShortLabel')['ShortLabel'].to_numpy()
        for i in lpsLabelUniqueArray:
            lpsLabelUniqueList.append(i)
            lpsLabelColorList.append(labelColorMap[i])

    if timeGroup in ['month', 'mon']:
        dfGroupMonth = df.group_by(pl.col('UTC').dt.month(), maintain_order=True)
    elif timeGroup in ['none', 'all']:
        dfGroupMonth = df.group_by(pl.col('UTC').is_not_null(), maintain_order=True)
    countGroupMonth = dfGroupMonth.len()
    monthList = countGroupMonth['UTC'].to_numpy()
    monthList = np.sort(monthList)
    monthAllList = []
    width = 20
    iMonth = 0
    yMax = yMax
    x = []
    for currentMon in monthList:
        for i in dfGroupMonth:
            iMon = np.unique(i[1]['UTC'].dt.month())[0]
            if currentMon == iMon:
                nYear = len(np.unique(i[1]['UTC'].dt.year()))
                dfGroupLabel = i[1].group_by(['ShortLabel', 'HALabel'])
                labelCount = dfGroupLabel.len()
                monthAll = np.round(labelCount['len'].to_numpy().sum() / nYear).astype(np.int64)
                yMap = dict()
                yMapPercentage = dict()
                for j in lpsLabelUniqueList:
                    yMap[j] = 0
                    yMapPercentage[j] = 0
                for j in labelCount.rows():
                    jRowLabel = j[0]
                    jHALabel = j[1]
                    if flagGroup:
                        groupLabel = findGroup(jRowLabel)
                        if flagHACat and (groupLabel == 'HighAltitude'):
                            groupLabel = findGroupHA(jHALabel)
                    else:
                        groupLabel = jRowLabel
                    jRowCount = np.round(j[2]  / nYear).astype(np.int64)
                    yMap[groupLabel] += jRowCount

                    if yMap[groupLabel] > yMax:
                        yMax = yMap[groupLabel]

                for j in lpsLabelUniqueList:
                    yMapPercentage[j] = f'{yMap[j] / monthAll * 100:.1f}\n%'
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
                
                finalLabelList = [i for i in lpsLabelUniqueList]
                if flagGroup:
                    for i in range(len(lpsLabelUniqueList)):
                        iLabel = lpsLabelUniqueList[i]
                        iLabel = 'Extreme High Altitude' if iLabel == 'HighAltitude' else iLabel
                        finalLabelList[i] = iLabel

                plot.bar(
                    list(yMap.values()), offset=width*len(lpsLabelUniqueList)*iMonth + width*iMonth, width=width, 
                    color=lpsLabelColorList, label=finalLabelList, yLimit=(0,yMax*(1+0.05)), showValue=True, 
                    valueConfig=dict(value=list(yMapPercentage.values()), format='', yloc=12, 
                                     fontsize=8))
                x.append(width*(len(lpsLabelUniqueList)*(iMonth+0.5)+0.5) + width*iMonth)
                iMonth += 1

    dfStat = dfStat.sort(by=['time', 'percentage'], descending=[False, True])
    dfStat.write_csv(dfPath)
    
    xLabels = []
    for i, j in zip(monthList, monthAllList):
        if timeGroup in ['month', 'mon']:
            iLabel = f'{datetime.date(year=2000, month=i, day=1).strftime('%b')} ({j})'
            xLabels.append(iLabel)
        elif timeGroup in ['none', 'all']:
            xLabels.append('All')
    plot.currentAx_.set_xlim(min(x) - width * len(lpsLabelUniqueList) / 2, max(x) + width * len(lpsLabelUniqueList) / 2)
    plot.currentAx_.set_xticks(x, xLabels)
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

def plotLPSTimeSeries(
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
        regions: list = None):
    pathLabelGroup = ''
    pathLabelHA = ''
    pathRegion = ''
    if flagGroup:
        pathLabelGroup = '_Group'
    if flagHACat:
        pathLabelHA = '_HACat'
    if regions:
        for i in regions:
            pathRegion = f'{pathRegion}_{i}'
        pathRegion = f'{pathRegion[1:]}_'
    
    if flagPlot:
        figPath = os.path.join(figDir, f'LPSTime_Series_Plot_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}_{pathRegion}{timeGroup.title()}.pdf')
    else:
        figPath = os.path.join(figDir, f'LPSTime_Series_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}_{pathRegion}{timeGroup.title()}.pdf')
    
    dfPath = os.path.join(figDir, f'LPSTime_Series_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}_{pathRegion}{timeGroup.title()}.csv')

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

    df = df.sort(['UTC', 'ShortLabel', 'HALabel'])
    lpsLabelUniqueList = []
    lpsLabelColorList = []
    if flagGroup:
        for i in combinedLabelGroup.keys():
            lpsLabelUniqueList.append(i)
            lpsLabelColorList.append(combinedLabelColorMap[i])
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
        lastWyear = firstTimeStep.year
        if firstTimeStep.month < 10:
            lastWyear = lastWyear - 1
        offsetWyear = datetime.datetime(lastWyear, 10, 1) - datetime.datetime(lastWyear+1, 1, 1)
        dfGroupMonth = df.group_by_dynamic('UTC', every='1y', closed='left', label='left', offset=offsetWyear)
    elif timeGroup in ['none', 'all']:
        dfGroupMonth = df.group_by(pl.col('UTC').is_not_null(), maintain_order=True)
    monthAllList = []
    yMap = dict()
    yMapPercentage = dict()
    yMax = yMax
    for j in lpsLabelUniqueList:
        yMap[j] = []
        yMapPercentage[j] = []
    x = []
    for i in dfGroupMonth:
        iMon = npDatetimeToDatetime(np.unique(i[0])[0]).replace(tzinfo=None)
        if timeGroup in ['wyear']:
            iMon = datetime.datetime(iMon.year+1, 1, 1)
        x.append(iMon)
        dfGroupLabel = i[1].group_by(['ShortLabel', 'HALabel'])
        labelCount = dfGroupLabel.len()
        monthAll = labelCount['len'].to_numpy().sum()
        for j in lpsLabelUniqueList:
            yMap[j].append(0)
            yMapPercentage[j].append(0)
        for j in labelCount.rows():
            jRowLabel = j[0]
            jHALabel = j[1]
            if flagGroup:
                groupLabel = findGroup(jRowLabel)
                if flagHACat and (groupLabel == 'HighAltitude'):
                    groupLabel = findGroupHA(jHALabel)
            else:
                groupLabel = jRowLabel
            jRowCount = j[2]
            yMap[groupLabel][-1] += jRowCount

            if yMap[groupLabel][-1] > yMax:
                yMax = yMap[groupLabel][-1]

        for j in lpsLabelUniqueList:
            yMapPercentage[j][-1] = yMap[j][-1] / monthAll * 100
            dfStat = dfStat.extend(
                pl.DataFrame(
                    {'time': [iMon], 'type': [j], 'amount': [yMap[j][-1]], 'percentage': [yMapPercentage[j][-1]]},
                    schema={
                        'time': pl.Datetime,
                        'type': pl.String,
                        'amount': pl.Int64,
                        'percentage': pl.Float64}))

        monthAllList.append(monthAll)
        
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
        if regions is not None:
            plot.currentAx_.set_yticks(np.arange(0, 1000, 50))
        else:
            plot.currentAx_.set_yticks(np.arange(0, yMax+200, 200))
        plot.currentAx_.set_ylim(0, yMax)
        plot.currentAx_.set_ylabel('Count')
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
        yMax2 = np.max([np.nanmax(currentValueList), yMax2, yMax])
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

def plotDensistyMapLps(
        filename: str, figPath: str, map: str = 'conus404', 
        trackTypes: str | list = None, labelType: str | list = None,
        excludeTrackTypes: str | list = None,
        flagGroup: bool = False,
        flagHACat: bool = False,
        beginTime: datetime.datetime = None,
        endTime: datetime.datetime = None,
        levels: list = None, cmapScale: str = 'linear',
        maxDensity: int = None, over: str = None,
        title: str = None,
        panelConfig: dict = None, plot: Plot = None,
        figsize: tuple = (16, 9),
        regions: list = None):
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

    if trackTypes is not None:
        if not isinstance(trackTypes, list):
            trackTypes = [trackTypes]
        df = df.filter(pl.col('TrackInfo').str.contains_any(trackTypes))
    
    if excludeTrackTypes is not None:
        if not isinstance(excludeTrackTypes, list):
            excludeTrackTypes = [excludeTrackTypes]
        df = df.filter(pl.col('TrackInfo').str.contains_any(excludeTrackTypes).not_())
    
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
        if len(finalLabel) > 0:
            if len(auxLabel) > 0:
                df = df.filter(
                    pl.col('ShortLabel').str.contains_any(finalLabel) | 
                    (pl.col('ShortLabel').str.contains_any(['HATHL', 'HAL']) &
                    pl.col('HALabel').str.contains_any(auxLabel)))
            else:
                df = df.filter(pl.col('ShortLabel').str.contains_any(finalLabel))
        else:
            df = df.filter(
                pl.col('ShortLabel').str.contains_any(['HATHL', 'HAL']) &
                pl.col('HALabel').str.contains_any(auxLabel))

    lon = df['Lon'].to_numpy()
    lat = df['Lat'].to_numpy()
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
        elif maxDensity <= 200:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200]
        elif maxDensity <= 300:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300]
        elif maxDensity <= 500:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300, 400, 500]
        else:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300, 400, 500]
        if maxDensity < densityMaxValue:
            over = 'auto'
    if cmapScale == 'log':
        levels[0] = 1
    plot.pcolormesh(
        density, gridLon, gridLat, cmap='WhiteBlueGreenYellowRed.rgb', 
        cmapConfig=dict(unit='1°$^{-2}$', over=over),
        levels=levels, cmapScale=cmapScale, cbar=True, cbarConfig=dict(
            fontsize=10, unitPos=[1.07, 1.01]))
    plot.figure_.text(
        0.72, 1.01, f'Max: {densityMaxValue:.1f}', 
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

def plotDensistyMapFront(
        filename: str, figPath: str, map: str = 'conus404', 
        flagMS: bool = False,
        flagFront: bool = False,
        flagLPS: bool = False,
        thresholdMS: float = 20,
        beginTime: datetime.datetime = None,
        endTime: datetime.datetime = None,
        beginHour: int = None,
        endHour: int = None,
        levels: list = None, cmapScale: str = 'linear',
        maxDensity: int = None, title: str = None,
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
    plot.setMap(map=map, spacing=10)
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

    if beginHour is not None and endHour is not None:
        if beginHour < endHour:
            df = df.filter(
                (pl.col('UTC').dt.hour() >= beginHour) & (pl.col('UTC').dt.hour() < endHour))
        else:
            df = df.filter(
                (pl.col('UTC').dt.hour() >= beginHour) | (pl.col('UTC').dt.hour() < endHour))

    nYear = 1
    if beginTime is not None and endTime is not None:
        nYear = calculateNYear(beginTime, endTime)
        
    if flagLPS:
        df = df.filter(pl.col('LPSIndex').list.len() > 0)
    else:
        df = df.filter(pl.col('LPSIndex').list.len() == 0)
    
    if flagMS:
        df = df.filter(pl.col('RH100Avg') >= thresholdMS)
    elif flagFront:
        df = df.filter(pl.col('RH100Avg') < thresholdMS)

    lon = df['CenLon'].to_numpy()
    lat = df['CenLat'].to_numpy()
    gridLon = np.arange(0.5, 360.0, 1.0)
    gridLat = np.arange(-89.5, 90.0, 1.0)
    gridLon, gridLat = np.meshgrid(gridLon, gridLat)
    density = densityMap(lon, lat, gridLon, gridLat) / nYear
    densityMaxValue = np.round(np.max(density), decimals=1)

    over = None
    if levels is None:
        maxDensity = np.nanmax(density) if maxDensity is None else maxDensity
        if maxDensity < 10:
            maxDensity = 10
            levels = np.linspace(0, 10, 11) if levels is None else levels
        elif maxDensity < 100:
            if maxDensity <= 20:
                levels = np.linspace(0, maxDensity, int(maxDensity)+1)
            elif maxDensity < 50:
                levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50]
            else:
                levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, maxDensity]
        elif maxDensity <= 200:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200]
        elif maxDensity <= 300:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300]
        elif maxDensity <= 500:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300, 400, 500]
        else:
            levels = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300, 400, 500]
        if maxDensity < densityMaxValue:
            over = 'auto'
    if cmapScale == 'log':
        levels[0] = 1
    plot.pcolormesh(
        density, gridLon, gridLat, cmap='WhiteBlueGreenYellowRed.rgb', 
        cmapConfig=dict(unit='1°$^{-2}$', over=over),
        levels=levels, cmapScale=cmapScale, cbar=True, cbarConfig=dict(
            fontsize=10, unitPos=[1.07, 1.01]))
    plot.figure_.text(
        0.72, 1.01, f'Max: {densityMaxValue:.1f}', 
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
    if regions:
        for i in regions:
            pathRegion = f'{pathRegion}_{i}'
        pathRegion = f'{pathRegion[1:]}_'
    pathModel = f'_{model}'
    
    dfPath = os.path.join(figDir, f'LPSStatsModel_Series_{beginTime:%Y%m}_{endTime:%Y%m}{pathLabelGroup}{pathLabelHA}{pathModel}_{pathRegion}{timeGroup.title()}.csv')
    
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

    df = df.sort(['UTC', 'ShortLabel', 'HALabel'])

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
        elif lastTimeStep.month > 10:
            df = df.filter(pl.col('UTC') < datetime.datetime(lastWyear, 10, 1))

        offsetWyear = datetime.datetime(firstWyear, 10, 1) - datetime.datetime(firstWyear, 1, 1)
        
        dfGroupMonth = df.group_by_dynamic('UTC', every='1y', closed='left', label='left', offset=offsetWyear)
    elif timeGroup in ['none', 'all']:
        dfGroupMonth = df.group_by(pl.col('UTC').is_not_null(), maintain_order=True)
    monthAllList = []
    yMap = dict()
    yMapCorrelation = dict()
    yMapPValue = dict()
    for j in lpsLabelUniqueList:
        yMap[j] = []
        yMapCorrelation[j] = []
        yMapPValue[j] = []
    yMapCorrelation['All'] = []
    yMapPValue['All'] = []
    for i in dfGroupMonth:
        iMon = npDatetimeToDatetime(np.unique(i[0])[0]).replace(tzinfo=None)
        if timeGroup in ['wyear']:
            iMon = datetime.datetime(iMon.year+1, 1, 1)
        dfGroupLabel = i[1].group_by(['ShortLabel', 'HALabel'])
        labelCount = dfGroupLabel.len()
        monthAll = labelCount['len'].to_numpy().sum()
        for j in lpsLabelUniqueList:
            yMap[j].append(0)
        for j in labelCount.rows():
            jRowLabel = j[0]
            jHALabel = j[1]
            if flagGroup:
                groupLabel = findGroup(jRowLabel)
                if flagHACat and (groupLabel == 'HighAltitude'):
                    groupLabel = findGroupHA(jHALabel)
            else:
                groupLabel = jRowLabel
            jRowCount = j[2]
            yMap[groupLabel][-1] += jRowCount

        monthAllList.append(monthAll)

    yMap['All'] = np.array(monthAllList, dtype=np.int64).tolist()

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
        thresholdMS = 20
    else:
        modelTypePrefix = 'CONUS'
        thresholdMS = 17

    if timeGroup == 'mon':
        if modelType == 'conus':
            beginTimeStrList = ['19791001_00']
            endTimeStrList = ['20220930_23']
        elif modelType == 'obs':
            beginTimeStrList = ['19791001_00']
            endTimeStrList = ['20220930_23']

    if modelType == 'conus':
        resultFile = f'Result/CONUS/Result_1979100100_2022093023.parquet'
        frontFile = f'Result/CONUS/FrontPreprocess_1979100100_2022093023.parquet'
    elif modelType == 'obs':
        resultFile = f'Result/OBS/Result_1979100100_2022093023.parquet'
        frontFile = f'Result/OBS/FrontPreprocess_1979100100_2022093023.parquet'

    workDir = f'Result/fig/{modelTypePrefix}'
    for beginTimeStr, endTimeStr in zip(beginTimeStrList, endTimeStrList):
        flagGroup = True
        flagHACat = True
        beginTime = datetime.datetime.strptime(beginTimeStr, "%Y%m%d_%H")
        endTime = datetime.datetime.strptime(endTimeStr, "%Y%m%d_%H")

        currentPanel = 0
        panelConfig = dict(
            nRow=2, nCol= 3, currentPanel=0, drawNumber=True,
        )
        figPath = os.path.join(workDir, f'DensityMap_All_{beginTime:%Y%m}_{endTime:%Y%m}.pdf')
        plot = plotDensistyMapLps(
            resultFile, figPath, beginTime=beginTime, endTime=endTime, maxDensity=500, over='auto',
            title='All LPSs', panelConfig=panelConfig, figsize=(16, 8.15))
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)
        # All tropical
        pathLabelGroup = '_Group'
        pathLabelHA = '_HACat'
        figPath = os.path.join(workDir, f'DensityMap_Tropical{pathLabelGroup}{pathLabelHA}_{beginTime:%Y%m}_{endTime:%Y%m}.pdf')
        plot = plotDensistyMapLps(
            resultFile, figPath, labelType='Tropical', flagGroup=True, flagHACat=True,
            beginTime=beginTime, endTime=endTime, maxDensity=5, over='auto',
            title='Tropical LPSs', panelConfig=panelConfig,  plot=plot)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)
        # All dry
        figPath = os.path.join(workDir, f'DensityMap_Dry{pathLabelGroup}{pathLabelHA}_{beginTime:%Y%m}_{endTime:%Y%m}.pdf')
        plot = plotDensistyMapLps(
            resultFile, figPath, labelType='Dry', flagGroup=True, flagHACat=True,
            beginTime=beginTime, endTime=endTime, maxDensity=300, over='auto',
            title='Dry LPSs', panelConfig=panelConfig, plot=plot)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)
        # All extratropical
        figPath = os.path.join(workDir, f'DensityMap_Extratropical{pathLabelGroup}{pathLabelHA}_{beginTime:%Y%m}_{endTime:%Y%m}.pdf')
        plot = plotDensistyMapLps(
            resultFile, figPath, labelType='Extratropical', flagGroup=True, flagHACat=True,
            beginTime=beginTime, endTime=endTime, maxDensity=30, over='auto',
            title='Extratropical LPSs', panelConfig=panelConfig, plot=plot)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)
        # All high-altitude (over-700hPa) LPSs
        figPath = os.path.join(workDir, f'DensityMap_HighAltitude{pathLabelGroup}{pathLabelHA}_{beginTime:%Y%m}_{endTime:%Y%m}.pdf')
        plot = plotDensistyMapLps(
            resultFile, figPath, labelType='HighAltitude', flagGroup=True, flagHACat=True,
            beginTime=beginTime, endTime=endTime, maxDensity=30, over='auto',
            title='Extreme High Altitude LPSs', panelConfig=panelConfig, plot=plot)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)

        figPath = os.path.join(
            workDir, f'DensityMap{pathLabelGroup}{pathLabelHA}_{beginTime:%Y%m}_{endTime:%Y%m}.pdf')
        plot.save(figPath)
        plot.clear()

        ############################################################################
        beginHour = None
        endHour = None
        if modelType == 'conus':
            currentPanel = 0
        else:
            currentPanel = 3
        panelConfig = dict(
            nRow=2, nCol= 3, currentPanel=currentPanel, drawNumber=True,
        )
        # Monsoon Trough
        figPath = os.path.join(workDir, f'DensityMapFront_MS_{beginTime:%Y%m}_{endTime:%Y%m}.pdf')
        plot = plotDensistyMapFront(
            frontFile, figPath, flagMS=True, thresholdMS=thresholdMS,
            beginTime=beginTime, endTime=endTime, 
            beginHour=beginHour, endHour=endHour,
            title='Monsoon Troughs', panelConfig=panelConfig, figsize=(14, 10))
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)
        # Front
        figPath = os.path.join(workDir, f'DensityMapFront_Front_{beginTime:%Y%m}_{endTime:%Y%m}.pdf')
        plot = plotDensistyMapFront(
            frontFile, figPath, flagFront=True, thresholdMS=thresholdMS,
            beginTime=beginTime, endTime=endTime, 
            beginHour=beginHour, endHour=endHour,
            title='Fronts', panelConfig=panelConfig, plot=plot,
            maxDensity=200)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)

        # LPS-Related 
        figPath = os.path.join(workDir, f'DensityMapFront_Front_{beginTime:%Y%m}_{endTime:%Y%m}.pdf')
        plot = plotDensistyMapFront(
            frontFile, figPath, flagLPS=True,
            beginTime=beginTime, endTime=endTime, 
            beginHour=beginHour, endHour=endHour,
            title='LPS-Related', panelConfig=panelConfig, plot=plot,
            maxDensity=15)
        currentPanel += 1
        panelConfig.update(currentPanel=currentPanel)
        
        figPath = os.path.join(
            workDir, f'DensityMapFront_{beginTime:%Y%m}_{endTime:%Y%m}.pdf')
        if beginHour is not None and endHour is not None:
            figPath = os.path.join(
                workDir, 
                f'DensityMapFront_{beginTime:%Y%m}_{endTime:%Y%m}_{beginHour:02d}Z_To_{endHour:02d}Z.pdf')
        plot.save(figPath)
        plot.clear()

        plot = plotLPSFrequency(
            resultFile, workDir, beginTime, endTime, flagGroup=flagGroup, 
            flagHACat=flagHACat, timeGroup=timeGroup, yMax=480, figsize=(18, 6))

        xLim1 = datetime.datetime.strptime("19791001_00", "%Y%m%d_%H")
        xLim2 = datetime.datetime.strptime("20220930_23", "%Y%m%d_%H")
        yMax = 3900
        yMin2 = 6000
        yMax2 = 7700
        if modelType == 'conus':
            currentPanel = 0
        else: 
            currentPanel = 1
        panelConfig = dict(
            currentPanel=currentPanel, drawNumber=True, save=True
        )
        plot = plotLPSTimeSeries(
            resultFile, workDir, beginTime, endTime, flagGroup=flagGroup, 
            flagHACat=flagHACat, timeGroup='wyear', yMax=yMax, 
            yMin2=yMin2, yMax2=yMax2,
            figsize=(12, 6), xlim=(xLim1, xLim2), 
            flagPlot=True, panelConfig=panelConfig,
            regions=regions)

        statsModelWithTimeSeries(
            resultFile, workDir, beginTime, endTime, flagGroup=flagGroup, 
            flagHACat=flagHACat, timeGroup='wyear', model='linear',
            regions=regions
        )