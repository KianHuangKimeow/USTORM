import colorsys
import logging
import os
import re
from typing import Optional

import matplotlib.cm as cm
import matplotlib.colors as mclr
import numpy as np

from Utilities import download

logger = logging.getLogger(__name__)

LAST_COLOR = 0
CONTROL_COLOR = 1
TRANSIT_COLOR = 2

_cmapDir = os.path.join(os.path.split(__file__)[0], 'Colormap')
if os.getenv('TEF_COLORMAP'):
    _cmapDir = os.getenv('TEF_COLORMAP')


def cmap(name: str, over: Optional[str] = None, 
        under: Optional[str] = None, unit: Optional[str] = None,
        type: Optional[str] = 'auto', levels: Optional[str] = 'auto'):
    colormap = ColorMap(name, over, under, unit, type, levels)
    return colormap.process()

class ColorMap:
    def __init__(self, name: str, over: Optional[str] = None, 
            under: Optional[str] = None, unit: Optional[str] = None,
            type: Optional[str] = 'auto', levels: Optional[str] = 'auto'):
        self.config_ = dict(
            name = name, over = over, under = under, unit = unit, type = type, 
            levels = levels)
        if name.lower().endswith('.rgb'):
            self.cmapPath_ = self._findCmapFile(name)
            self.process = self.nclrgb
        elif name.lower().endswith('.cmap'):
            self.cmapPath_ = self._findCmapFile(name)
            self.process = self.cmgntLinear
        else:
            raise Exception(f'Unknown cmap type: {name}')
    
    def _findCmapFile(self, filename: str):
        filePath = ''
        if os.path.exists(filename):
            filePath = filename
        else:
            filePath = os.path.join(_cmapDir, filename)
            if not os.path.exists(filePath):
                if '/' in filename:
                    _, filename = os.path.split(filename)
                if filename.lower().endswith('rgb'):
                    resource = r'https://raw.github.com/NCAR/ncl/develop/ni/src/db/colormaps'
                elif filename.lower().endswith('cmap'):
                    resource = r'https://raw.github.com/crazyapril/mpkit/master/colormap'
                else:
                    raise Exception(f'file not found: {filePath}')
                
                url = os.path.join(resource, filename)
                url = url.replace('\\', '/')
                download(url, filePath)
        return filePath
    
    def _setCmapExtend(self, cmap: any):
        extend = 'neither'
        cmapArray = cm.ScalarMappable(cmap=cmap).to_rgba([0, 1])
        over = self.config_.get('over', None)
        if over is not None:
            if over.lower() == 'auto':
                color = (cmapArray[-1])
                cmap.set_over(color)
            else:
                cmap.set_over(self.getColor(over))
            extend = 'max'
        
        under = self.config_.get('under', None)
        if under is not None:
            if under.lower() == 'auto':
                color = (cmapArray[0])
                cmap.set_under(color)
            else:
                cmap.set_under(self.getColor(under))
            extend = 'min'

        if over is not None and under is not None:
            extend = 'both'
        
        return (cmap, extend)
    
    def cmgntLineList(self, line):
        if line[-1] == '\n':
            line = line[:-1]
        lineSplit = line.split(' ')
        if len(lineSplit) != 2:
            raise Exception(f'Invalid cmap format: line {line}')
        try:
            value = float(lineSplit[0])
        except (SyntaxError, ValueError, NameError):
            raise Exception(f'Invalid value: line {line}')
        if lineSplit[1].lower() == 'end':
            return value, CONTROL_COLOR
        elif lineSplit[1] == '~':
            return value, TRANSIT_COLOR
        else:
            return value, self.getColor(lineSplit[1])
        
    def cmgntLineLinear(self, line):
        if line[-1] == '\n':
            line = line[:-1]
        lineSplit = line.split(' ')
        if len(lineSplit) != 3:
            raise Exception(f'Invalid cmap format: line {line}')
        try:
            value = float(lineSplit[0])
        except (SyntaxError, ValueError, NameError):
            raise Exception(f'Invalid value: line {line}')
        if lineSplit[1].lower() == 'begin':
            return value, (0.0, 0.0, 0.0), self.getColor(lineSplit[2])
        elif lineSplit[2].lower() == 'end':
            return value, self.getColor(lineSplit[1]), (0.0, 0.0, 0.0)
        else:
            return value, self.getColor(
                lineSplit[1]), self.getColor(lineSplit[2])
        
    def getColor(self, rgbStr: str, mode: Optional[str] = 'rgb'):
        if rgbStr == '~':
            return LAST_COLOR
        rgbList = rgbStr.split('/')
        if len(rgbList) != 3:
            rgbList = rgbStr.split()
        if len(rgbList) != 3:
            raise Exception('Invalid cmap format.')
        
        try:
            rgbList = np.array(rgbList, dtype=np.float64)
        except:
            raise Exception('Invalid cmap value.')
            
        if np.any(rgbList > 1) and mode == 'rgb':
            rgbList = rgbList / 255.0
        if np.any((rgbList < 0) | (rgbList > 1)):
            raise('Invalid cmap value: value out of range.')
        if mode == 'hsv':
            rgbList = colorsys.hsv_to_rgb(*rgbList)
        return np.array(rgbList).tolist()
    
    def getLevels(self, level: str, cmapList):
        valMin = cmapList[0][0]
        valMax = cmapList[-1][0]
        clev = []
        step = 0
        if level == 'file':
            for i in cmapList:
                clev.append(i[0])
        else:
            if level == 'auto':
                step = 1.0
            elif level.startswith('s'):
                step = float(level[1:])
            elif level.isdigit():
                step = level
            clev = np.arange(valMin, valMax + float(step), float(step))
        return np.array(clev)
    
    def nclrgb(self):
        nColorCount = 0
        colorList = []
        fileLines = []
        with open(self.cmapPath_) as f:
            fileLines = [
                re.sub('\s+', ' ', i) for i in f.read().splitlines() 
                if not (i.startswith('#') or i.endswith('n'))]
            
        fileLines = [i.split('#', 1)[0] for i in fileLines]
        fileLines = [re.sub('\s+', ' ', i) for i in fileLines]
        fileLines = [i for i in fileLines if i != '']
        fileLines = [i for i in fileLines if 'ncolor' not in i]
        colorList = np.array([self.getColor(i) for i in fileLines])
        nColorCount = len(fileLines)
        cmap = mclr.LinearSegmentedColormap.from_list(
            name=self.config_.get('name'), 
            colors=colorList, N=nColorCount)
        cmap, extend = self._setCmapExtend(cmap)
        unit = self.config_.get('unit', None)
        return dict(cmap = cmap, unit = unit, extend = extend)

    def cmgntList(self):
        lastValue = -1e7
        valueList = []
        cmapList = []
        filePath = self.cmapPath_
        with open(filePath, 'r') as f:
            transitCount = 0
            for line in f:
                if line[0] == '*':
                    lineSplit = line[1:].split(':')
                    configName, configValue = lineSplit[0].lower(
                    ), lineSplit[1][:-1]
                    if configName == 'type':
                        if configValue.lower() == 'linear':
                            return self.cmgntLinear
                        elif configValue.lower() != 'listed':
                            raise Exception(f'Unknown cmap type: {configValue}')
                    if configName not in self.config_ or self.config_[configName] == 'auto':
                        self.config_[configName] = configValue
                else:
                    value, color = self.cmgntLineList(line)
                    if value < lastValue:
                        raise Exception(f'Not in order: line {line}')
                    valueList.append(value)
                    if color == TRANSIT_COLOR:
                        cmapList.append(0)
                        transitCount += 1
                    elif color != CONTROL_COLOR:
                        if transitCount > 0:
                            lastColor = cmapList[-transitCount-1]
                            for i in range(-transitCount, 0):
                                w = (transitCount + i + 1) / (transitCount + 1)
                                cmapList[i] = tuple(
                                    (k - j) * w + j for j,
                                    k in zip(lastColor, color)
                                )
                            transitCount = 0
                        cmapList.append(color)
                    lastValue = value
        cmap = mclr.ListedColormap(cmapList)
        unit = self.config_.get('unit', None)
        cmap, extend = self._setCmapExtend(cmap)
        norm = mclr.BoundaryNorm(valueList, cmap.N)
        return dict(
            cmap=cmap, clev=valueList, norm=norm, unit=unit, extend=extend)

    def cmgntLinear(self):
        lastColor = None
        lastValue = -1e7
        cmapList = []
        filePath = self.cmapPath_

        with open(filePath, 'r') as f:
            for line in f:
                if line[0] == '*':
                    lineSplit = line[1:].split(':')
                    configName, configValue = lineSplit[0].lower(
                    ), lineSplit[1][:-1]
                    if configName == 'type':
                        if configValue.lower() == 'listed':
                            return self.cmgntList()
                        elif configValue.lower() != 'linear':
                            raise Exception(f'Unknown cmap type: {configValue}')
                    if configName not in self.config_ or self.config_[configName] == 'auto':
                        self.config_[configName] = configValue
                else:
                    value, color1, color2 = self.cmgntLineLinear(line)
                    if value < lastValue:
                        raise Exception(f'Not in order: line {line}')
                    if color1 == LAST_COLOR:
                        color1 = lastColor
                    if color2 == LAST_COLOR:
                        color2 = color1
                    cmapList.append((value, color1, color2))
                    lastValue = value
                    lastColor = color2
        valMin = cmapList[0][0]
        valMax = cmapList[-1][0]
        span = valMax - valMin
        cmapDict = dict(
            red=[], green=[], blue=[]
        )
        for val, color1, color2 in cmapList:
            valNorm = (val - valMin) / span
            cmapDict['red'].append((valNorm, color1[0], color2[0]))
            cmapDict['green'].append((valNorm, color1[1], color2[1]))
            cmapDict['blue'].append((valNorm, color1[2], color2[2]))
        cmap = mclr.LinearSegmentedColormap('cmap', cmapDict)
        configLevel = self.config_['levels']
        clev = self.getLevels(configLevel, cmapList)
        unit = self.config_.get('unit', None)
        cmap, extend = self._setCmapExtend(cmap)
        return dict(cmap=cmap, clev=clev, unit=unit, extend=extend)