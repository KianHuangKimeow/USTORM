import datetime

from .Downloader import downloadWget

def downloadGesDiscMergedIR(
        distDir: str, beginTimeStr: str, endTimeStr: str, step: int = 1,
        override: bool = False):
    beginTime = datetime.datetime.strptime(beginTimeStr, "%Y%m%d_%H")
    endTime = datetime.datetime.strptime(endTimeStr, "%Y%m%d_%H")

    urlPrefix = 'https://disc2.gesdisc.eosdis.nasa.gov/data/MERGED_IR/'

    currentTime = beginTime
    while currentTime <= endTime:
        dayOfYear = currentTime.timetuple().tm_yday
        yearStr = currentTime.strftime("%Y")
        filenameStr = f'GPM_MERGIR.1/{yearStr}/{dayOfYear:03d}/' + \
          f'merg_{currentTime:%Y%m%d%H}_4km-pixel.nc4'
        
        currentUrl = urlPrefix + filenameStr
        currentDist = distDir + '/' + filenameStr
        downloadWget(currentUrl, currentDist, override)
        currentTime += datetime.timedelta(hours=step)