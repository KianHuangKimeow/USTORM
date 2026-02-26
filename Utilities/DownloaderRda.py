import calendar
import datetime

from .Downloader import download

def downloadRdaERA5(distDir: str, beginTime: datetime.datetime, 
                    endTime: datetime.datetime, override: bool = False):

    urlPrefix = 'https://data.rda.ucar.edu/d633000/'
    urlPrefixMl = 'https://data.rda.ucar.edu/d633006/'

    variableMap = {
        'e5.oper.invariant': [
            '128_129_z.ll025sc',  # Geopotential at surface (m2 s-2)
        ],
        # e5.oper.an.pl: pressure-level analysis
        'e5.oper.an.pl': [
            '128_060_pv.ll025sc',  # Potential vorticity (K m2 kg-1 s-1)
            '128_129_z.ll025sc',  # Geopotential (m2 s-2)
            '128_130_t.ll025sc',  # Temperature (K)
            '128_131_u.ll025uv',  # U component of wind (m s-1)
            '128_132_v.ll025uv',  # V component of wind (m s-1)
            '128_133_q.ll025sc',  # Specific humidity (kg kg-1)
            '128_138_vo.ll025sc',  # Vorticity (relative) (s-1)
            '128_157_r.ll025sc',  # Relative humidity (%)
        ],
        # e5.oper.an.sfc: surface (single-level) analysis
        'e5.oper.an.sfc': [
            '128_151_msl.ll025sc',  # Mean sea level pressure (Pa)
            '128_165_10u.ll025sc',  # 10 metre U wind component (m s-1)
            '128_166_10v.ll025sc',  # 10 metre U wind component (m s-1)
        ],
        # e5.oper.an.ml: model-level analysis
        'e5.oper.an.ml': [
            '0_5_0_0_0_t.regn320sc',  # Temperature (K)
            '0_5_0_1_0_q.regn320sc',  # Specific humidity (kg kg-1)
            '0_5_0_2_2_u.regn320uv',  # U component of wind (m s-1)
            '0_5_0_2_3_v.regn320uv',  # V component of wind (m s-1)
            '128_134_sp.regn320sc',  # Surface pressure (Pa)
        ],
    }
    typeList = ['e5.oper.an.pl', 'e5.oper.an.sfc', 'e5.oper.an.ml']
    sixHourlyTypeList = ['e5.oper.an.ml']
    monthlyTypeList = ['e5.oper.an.sfc']
    downloadedMap = {}
    for t in typeList:
        downloadedMap[t] = False

    for var in variableMap['e5.oper.invariant']:
        filenameStr = (
            'e5.oper.invariant/197901/e5.oper.invariant.' +
            var + '.1979010100_1979010100.nc')
        currentUrl = urlPrefix + filenameStr
        currentDist = distDir + '/' + filenameStr
        download(currentUrl, currentDist, override)
        
    currentTime = beginTime
    while currentTime <= endTime:
        yearStr = currentTime.strftime("%Y")
        monthStr = currentTime.strftime("%m")
        dateStr = currentTime.strftime("%Y%m%d")
        if currentTime.day == 1:
            for t in monthlyTypeList:
                downloadedMap[t] = False
        for t in typeList:
            if not downloadedMap[t]:
                if t in monthlyTypeList:
                    lastDay = datetime.datetime(currentTime.year, currentTime.month,
                                                calendar.monthrange(
                                                    currentTime.year, currentTime.month)[1])
                    lastDayStr = lastDay.strftime("%Y%m%d")
                    fileSuffix = yearStr + monthStr + '0100_' + lastDayStr + '23.nc'
                    for var in variableMap[t]:
                        filenameStr = t + '/' + yearStr + monthStr + \
                            '/' + t + '.' + var + '.' + fileSuffix
                        currentUrl = urlPrefix + filenameStr
                        currentDist = distDir + '/' + filenameStr
                        download(currentUrl, currentDist, override)
                elif t in sixHourlyTypeList:
                    for var in variableMap[t]:
                        innerTime = datetime.datetime.strptime(f'{currentTime:%Y%m%d}_00', '%Y%m%d_%H')
                        innerEndTime = innerTime + datetime.timedelta(hours=23)
                        while innerTime < innerEndTime:
                            innerTimeE = innerTime + datetime.timedelta(hours=5)
                            fileSuffix = dateStr + f'{innerTime:%H}_' + dateStr + f'{innerTimeE:%H}.nc'
                            filenameStr = t + '/' + yearStr + monthStr + \
                                '/' + t + '.' + var + '.' + fileSuffix
                            currentUrl = urlPrefixMl + filenameStr
                            currentDist = distDir + '/' + filenameStr
                            download(currentUrl, currentDist, override)
                            innerTime += datetime.timedelta(hours=6)
                else:
                    for var in variableMap[t]:
                        fileSuffix = dateStr + '00_' + dateStr + '23.nc'
                        filenameStr = t + '/' + yearStr + monthStr + \
                            '/' + t + '.' + var + '.' + fileSuffix
                        currentUrl = urlPrefix + filenameStr
                        currentDist = distDir + '/' + filenameStr
                        download(currentUrl, currentDist, override)
                if t in monthlyTypeList:
                    downloadedMap[t] = True
        currentTime += datetime.timedelta(days=1)


def downloadRdaCONUS404(distDir: str, beginTimeStr: str, endTimeStr: str, step: int = 3,
                     override: bool = False, month: int = None):
    beginTime = datetime.datetime.strptime(beginTimeStr, "%Y%m%d_%H")
    endTime = datetime.datetime.strptime(endTimeStr, "%Y%m%d_%H")

    urlPrefix = 'https://data.rda.ucar.edu/d559000/'
  
    typeList = ['wrf2d', 'wrf3d']

    # CONUS404 invariant
    filenameStr = 'INVARIANT/wrfconstants_usgs404.nc'
    currentUrl = urlPrefix + filenameStr
    currentDist = distDir + '/' + filenameStr
    download(currentUrl, currentDist, override)
        
    currentTime = beginTime
    while currentTime <= endTime:
        currentYear = currentTime.year
        currentMonth = currentTime.month
        currentWaterYear = currentYear + 1 if currentMonth >= 10 else currentYear
            
        yearStr = currentTime.strftime("%Y")
        monthStr = currentTime.strftime("%m")
        dateStr = currentTime.strftime("%Y-%m-%d_%H:%M:%S")
        fileSuffix = '_d01_' + dateStr + '.nc'

        for t in typeList:
            filenameStr = f'wy{currentWaterYear}/' + yearStr + monthStr + \
                '/' + t + fileSuffix
            currentUrl = urlPrefix + filenameStr
            currentDist = distDir + '/' + filenameStr
            if (month is None) or (currentMonth == month):
                download(currentUrl, currentDist, override)
        currentTime += datetime.timedelta(hours=step)

def downloadRdaIMERG(distDir: str, beginTimeStr: str, endTimeStr: str, step: int = 1,
                     override: bool = False):
    beginTime = datetime.datetime.strptime(beginTimeStr, "%Y%m%d_%H")
    endTime = datetime.datetime.strptime(endTimeStr, "%Y%m%d_%H")

    beginTime -= datetime.timedelta(hours=1)
    endTime -= datetime.timedelta(hours=1)

    urlPrefix = 'https://data.rda.ucar.edu/d731000/'

    currentTime = beginTime
    while currentTime <= endTime:
        yearStr = currentTime.strftime("%Y")
        monthStr = currentTime.strftime("%m")
        dayStr = currentTime.strftime("%d")
        minOfTheDay = currentTime.hour * 60
        filenamePrefix = f'gpm_3imerghh_v07/' + yearStr + '/' + monthStr + \
            '/' + dayStr + '/' + '3B-HHR.MS.MRG.3IMERG.' + yearStr + monthStr + \
            dayStr
        
        innerNowS = currentTime
        innerNowE = currentTime + datetime.timedelta(minutes=30) -  datetime.timedelta(seconds=1)
        filenameStrList = []
        for t in range(2):
            filenameStrList.append(
                filenamePrefix + f'-S{innerNowS:%H%M%S}-E{innerNowE:%H%M%S}.' + \
                f'{minOfTheDay+t*30:04d}.V07B.HDF5')
            innerNowS += datetime.timedelta(minutes=30)
            innerNowE += datetime.timedelta(minutes=30)

        for i in filenameStrList:
            currentUrl = urlPrefix + i
            currentDist = distDir + '/' + i
            download(currentUrl, currentDist, override)
        currentTime += datetime.timedelta(hours=step)