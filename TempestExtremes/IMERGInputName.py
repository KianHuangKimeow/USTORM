import datetime
import logging

from .InputNameBase import InputNameBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class IMERGInputName(InputNameBase):
    def __init__(self, root, step: int = 3):
        super().__init__(root, step)

    def findRoot(self, type: str):
        if isinstance(self.root_, dict):
            return self.root_[type]
        else:
            return self.root_

    def generateInput(self, types: list = [], output: bool = False, month: int = None):
        self.inputNameList_ = ''
        if not types:
            types = ['IMERG', 'MergedIR']
        for t in types:
            if t == 'IMERG':
                self.inputNameList_ += self.generateInputIMERG(output=True, month=month)
            elif t == 'MergedIR':
                self.inputNameList_ += self.generateInputMergedIR(output=True, month=month)
        if output:
            return self.inputNameList_
        
    def generateInputIMERG(self, output: bool = False, month: int = None):
        currentTime = self.begin_
        currentTime -= datetime.timedelta(hours=1)
        endTime = self.end_ - datetime.timedelta(hours=1)
        self.inputNameList_ = ''
        dataRoot = self.findRoot('IMERG')
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
                currentDist = dataRoot + '/' + i
                self.inputNameList_ += currentDist + '\n'

            currentTime += self.step_

        if output:
            return self.inputNameList_

    def generateInputMergedIR(self, output: bool = False, month: int = None):
        currentTime = self.begin_
        self.inputNameList_ = ''
        dataRoot = self.findRoot('MergedIR')
        while currentTime <= self.end_:
            dayOfYear = currentTime.timetuple().tm_yday
            yearStr = currentTime.strftime("%Y")
            filenameStr = f'GPM_MERGIR.1/{yearStr}/{dayOfYear:03d}/' + \
              f'merg_{currentTime:%Y%m%d%H}_4km-pixel.nc4'
            currentDist = dataRoot + '/' + filenameStr
            self.inputNameList_ += currentDist + '\n'

            currentTime += self.step_

        if output:
            return self.inputNameList_