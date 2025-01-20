import calendar
import datetime
import sys
import os
from urllib.request import build_opener


def downloadFromRda(distDir: str, beginTimeStr: str, endTimeStr: str):
    beginTime = datetime.datetime.strptime(beginTimeStr, "%Y%m%d_%H")
    endTime = datetime.datetime.strptime(endTimeStr, "%Y%m%d_%H")

    urlPrefix = 'https://data.rda.ucar.edu/d633000/'

    variableMap = {
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
        ]
    }
    typeList = ['e5.oper.an.pl', 'e5.oper.an.sfc']
    monthlyTypeList = ['e5.oper.an.sfc']
    downloadedMap = {}
    for t in typeList:
        downloadedMap[t] = False

    opener = build_opener()

    currentTime = beginTime
    while currentTime <= endTime:
        yearStr = currentTime.strftime("%Y")
        monthStr = currentTime.strftime("%m")
        dateStr = currentTime.strftime("%Y%m%d")
        fileSuffix = dateStr + '00_' + dateStr + '23.nc'
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
                    ofile = os.path.basename(currentUrl)
                    currentDir = os.path.dirname(currentDist)
                    if not os.path.isdir(currentDir):
                        sys.stdout.write(
                            currentDir + " does not exist, making one ... ")
                        sys.stdout.flush()
                        os.makedirs(currentDir)
                        sys.stdout.write("done\n")
                    sys.stdout.write("downloading " + ofile + " ... ")
                    sys.stdout.flush()
                    infile = opener.open(currentUrl)
                    outfile = open(currentDist, "wb")
                    outfile.write(infile.read())
                    outfile.close()
                    sys.stdout.write("done\n")
                if t in monthlyTypeList:
                    downloadedMap[t] = True
        currentTime += datetime.timedelta(days=1)
