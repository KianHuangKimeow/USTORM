import calendar
import datetime
import logging
import os

from .InputNameBase import InputNameBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    ],
    'e5.oper.invariant': [
        '128_129_z.ll025sc',  # Geopotential at the surface (m2 s-2)
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
typeList = ['e5.oper.an.pl', 'e5.oper.an.sfc', 'e5.oper.invariant', 'e5.oper.an.ml']
sixHourlyTypeList = ['e5.oper.an.ml']
monthlyTypeList = ['e5.oper.an.sfc']

class ERA5InputName(InputNameBase):
    def __init__(self, root, step: int = 24):
        super().__init__(root, step)

    def generateInput(self, variables: list = [], exclusiveType: list = [], 
                      output: bool = False, step: int = None, addInvariant: list = []):
        currentTime = self.begin_
        self.inputNameList_ = ''
        step = self.step_ if step is None else step
        if isinstance(step, int):
            step = datetime.timedelta(hours=step)
        if not variables:
            for varType in typeList:
                if varType not in exclusiveType:
                    for var in variableMap[varType]:
                        variables.append(var)
        while currentTime <= self.end_:
            yearStr = currentTime.strftime("%Y")
            monthStr = currentTime.strftime("%m")
            dateStr = currentTime.strftime("%Y%m%d")
            fileSuffix = dateStr + '00_' + dateStr + '23.nc'
            filenameStr = ''
            for varType in typeList:
                if varType not in exclusiveType:
                    for var in variableMap[varType]:
                        if var in variables:
                            if varType == 'e5.oper.invariant':
                                fileSuffix = '1979010100_1979010100.nc'
                                filenameStr = varType + '/197901/' + varType + '.' + var + '.' + fileSuffix
                                self.invariantPaths_[
                                    var] = self.root_ + '/' + filenameStr
                            elif varType in monthlyTypeList:
                                lastDay = datetime.datetime(currentTime.year, currentTime.month, calendar.monthrange(
                                    currentTime.year, currentTime.month)[1])
                                lastDayStr = lastDay.strftime("%Y%m%d")
                                fileSuffix = yearStr + monthStr + '0100_' + lastDayStr + '23.nc'
                                filenameStr = varType + '/' + yearStr + monthStr + \
                                    '/' + varType + '.' + var + '.' + fileSuffix
                            elif varType in sixHourlyTypeList:
                                currentSixHourTime = currentTime
                                while currentSixHourTime.timetuple().tm_hour % 6 != 0:
                                    currentSixHourTime -= datetime.timedelta(hours=1)
                                currentSixHourTimeE = currentSixHourTime + datetime.timedelta(hours=5)
                                fileSuffix =  f'{currentSixHourTime:%Y%m%d%H}_' + f'{currentSixHourTimeE:%Y%m%d%H}.nc'
                                filenameStr = varType + '/' + yearStr + monthStr + \
                                    '/' + varType + '.' + var + '.' + fileSuffix
                            else:
                                fileSuffix = dateStr + '00_' + dateStr + '23.nc'
                                filenameStr = varType + '/' + yearStr + monthStr + \
                                    '/' + varType + '.' + var + '.' + fileSuffix
                            currentDist = self.root_ + '/' + filenameStr
                            if not os.path.exists(currentDist):
                                raise FileExistsError(
                                    f'{currentDist} does not exist!')
                            self.inputNameList_ += currentDist + ';'
            for invariantVar in addInvariant:
                self.inputNameList_ += self.invariantPaths_[invariantVar] + ';'
            self.inputNameList_ = self.inputNameList_[:-1]
            self.inputNameList_ += '\n'
            currentTime += step
        if output:
            return self.inputNameList_
