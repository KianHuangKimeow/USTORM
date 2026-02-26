import logging
import os

from .InputNameBase import InputNameBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

typeList = ['wrf2d', 'wrf3d']

class CONUS404InputName(InputNameBase):
    def __init__(self, root, step: int = 3):
        super().__init__(root, step)

    def generateInput(self, types: list = [], output: bool = False, month: int = None):
        currentTime = self.begin_
        self.inputNameList_ = ''
        if not types:
            for t in typeList:
                types.append(t)

        while currentTime <= self.end_:
            currentYear = currentTime.year
            currentMonth = currentTime.month
            currentWaterYear = currentYear + 1 if currentMonth >= 10 else currentYear
                
            yearStr = currentTime.strftime("%Y")
            monthStr = currentTime.strftime("%m")
            dateStr = currentTime.strftime("%Y-%m-%d_%H:%M:%S")
            fileSuffix = '_d01_' + dateStr + '.nc'

            for t in types:
                if t == 'INVARIANT':
                    fileSuffix = '1979010100_1979010100.nc'
                    filenameStr = 'INVARIANT/wrfconstants_usgs404.nc'
                else:
                    filenameStr = f'wy{currentWaterYear}/' + yearStr + monthStr + \
                        '/' + t + fileSuffix
                currentDist = self.root_ + '/' + filenameStr
                if (month is None) or (currentMonth == month):
                    if not os.path.exists(currentDist):
                        raise FileExistsError(
                            f'{currentDist} does not exist!')
                    self.inputNameList_ += currentDist + ';'
            self.inputNameList_ = self.inputNameList_[:-1]
            self.inputNameList_ += '\n'
            currentTime += self.step_

    def getInvariantPaths(self):
        filenameStr = 'INVARIANT/wrfconstants_usgs404.nc'
        self.invariantPaths_ = dict(
            invariant = self.root_ + '/' + filenameStr
        )
        return super().getInvariantPaths()