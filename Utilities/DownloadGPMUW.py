import datetime
import dateutil
import os
from typing import Optional

import dateutil.relativedelta
import polars as pl

from Utilities import downloadFromWebpage

def downloadGPMUW(
        begin: datetime.datetime, end: datetime.datetime, 
        version: str, region: str, dataType: str, dataRoot: str, 
        downloadedRecordFile: str, step: Optional[int] = 1,
        override: Optional[bool] = False, cacheDir: Optional[str] = None):
    
    urlUWGPM = 'https://gpm.atmos.washington.edu/'
    downloadedRecord = None
    flagUpdateRecord = False

    schema = {
        'version': str,
        'region': str,
        'dataType': str,
        'datetime': datetime.datetime,
        'filename': str}
    if os.path.exists(downloadedRecordFile):
        downloadedRecord = pl.read_csv(
            downloadedRecordFile, try_parse_dates=True)
    else:
        downloadedRecord = pl.DataFrame({
            'version': [],
            'region': [],
            'dataType': [],
            'datetime': [],
            'filename': []}, schema=schema)
        
    if dataType[:3] in ['BSR', 'DCC', 'DWC', 'SHI', 'WCC']:
        currentDate = begin.date().replace(day=1)
        currentTime = datetime.datetime.combine(currentDate, datetime.time(hour=0, minute=0, second=0))
    else:
        currentTime = begin
    while currentTime <= end:
        searchResult = downloadedRecord.filter(
            (pl.col('version') == version) &
            (pl.col('region') == region) &
            (pl.col('dataType') == dataType) &
            (pl.col('datetime') == currentTime)
        )
        if searchResult['filename'].count() == 0:
            if dataType == 'interp_data':
                dataDirSuffix = f'{version}/{region}/{dataType}/{currentTime:%Y/%m}'
            elif dataType[:3] in ['BSR', 'DCC', 'DWC', 'SHI', 'WCC']:
                dataDirSuffix = f'{version}/{region}/{dataType[:3]}/{currentTime:%m}'
            else:
                raise Exception(f'{dataType} has not been supported yet.')
            currentUrl = f'{urlUWGPM}{dataDirSuffix}'
            distDir = os.path.join(dataRoot, dataDirSuffix)
            if dataType == 'interp_data':
                reStr = f'GPM2Ku7_uw4_{currentTime:%Y%m%d.%H}'
            elif dataType[:3] in ['BSR', 'DCC', 'DWC', 'SHI', 'WCC']:
                reStr = f'GPM2Ku7_uw4_{dataType}_{currentTime:%Y%m}'
            else:
                raise Exception(f'{dataType} has not been supported yet.')
            currentDownloaded = downloadFromWebpage(
                currentUrl, distDir, reStr, override=override, cacheDir=cacheDir)
            for i in currentDownloaded:
                downloadedRecord.extend(pl.DataFrame({
                    'version': version, 'region': region, 'dataType': dataType,
                    'datetime': currentTime, 'filename': i}))
            flagUpdateRecord = True

        if dataType[:3] in ['BSR', 'DCC', 'DWC', 'SHI', 'WCC']:
            currentDate = currentTime.date() + dateutil.relativedelta.relativedelta(months=1)
            currentTime = datetime.datetime.combine(currentDate, currentTime.time())
        else:
            currentTime += datetime.timedelta(hours=step)
    if flagUpdateRecord:
        downloadedRecord = downloadedRecord.sort(by=[
            'version', 'region', 'dataType', 'datetime'])
        downloadedRecord.write_csv(downloadedRecordFile)
    
    searchResult = downloadedRecord.filter(
        (pl.col('datetime') >= begin) & (pl.col('datetime') <= end) &
        (pl.col('dataType') == dataType))
    if dataType == 'interp_data':
        plFormat = pl.format(
            '/{}/{}/{}/{}/{}', 
            pl.col('version'), pl.col('region'), pl.col('dataType'),
            pl.col('datetime').dt.strftime('%Y/%m'), pl.col('filename')
        )
    elif dataType[:3] in ['BSR', 'DCC', 'DWC', 'SHI', 'WCC']:
        plFormat = pl.format(
            '/{}/{}/{}/{}/{}', 
            pl.col('version'), pl.col('region'), 
            pl.col('dataType').str.slice(0, length=3),
            pl.col('datetime').dt.strftime('%m'), pl.col('filename')
        )
    fileList = searchResult.select([
        (dataRoot + plFormat).alias('filePath')])['filePath'].to_list()
    
    return fileList