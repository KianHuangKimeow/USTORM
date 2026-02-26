import functools
import logging
import matplotlib
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import os
from typing import AnyStr, Optional, Sequence, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as cshreader
import cartopy.mpl.gridliner as cgridliner
from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr

from Base import npDatetimeToDatetime
import Visualization.ColorManager as cmgr

logger = logging.getLogger(__name__)

matplotlib.use('agg')

_shpDir = os.path.join(os.path.split(__file__)[0], 'shapefile')

def mergeDict(d1: dict, d2: dict):
    for key, val in d2.items():
        if key not in d1:
            d1[key] = val
    return d1

class Plot:
    def __init__(
            self, figsize: Optional[Sequence] = (7, 5), 
            dpi: Optional[int] = 180, aspect: Optional[Union[AnyStr, float]] = None,
            full: Optional[bool] = False, boundary: Optional[bool] = None, 
            **kwargs) -> None:
        '''
        Initialize an Plpy instance.
        '''
        self.config_ = dict()
        self.config_['figure'] = dict()
        self.config_['figure'].update(figsize = figsize)
        self.config_['figure'].update(dpi = dpi)
        self.config_['figure'].update(full = full)
        # Instantiate a matplotlib.figure.Figure object
        self.figure_ = plt.figure(figsize=figsize)
        self.axes_ = dict()
        self.axesInfo_ = dict()
        self.currentAx_ = None
        self.currentAxInfo_ = dict()
        self.currentAxIdx_ = None
        # Map configuration
        self.config_['map'] = dict()
        self.config_['map'].update(aspect = aspect)
        self.config_['map'].update(boundary = boundary)
        self.config_['map'].update(gridSpacing = kwargs.pop('gridSpacing', None))
        # Font setting
        self.config_['font'] = dict()
        self.config_['font'].update(family = 
                                    kwargs.pop('fontFamily', 'Source Sans Pro'))
        self.config_['font']['size'] = dict(
            title=14, timestamp=12, clabel=12, cbar=12, gridValue=10, annot=10,
            gridTick=10, legend=10, boxText=8
        )
        self.config_['font']['color'] = dict(
            gridValue='white', tick='black', cbar='black', title='black', 
            timestamp='black'
        )
        self.config_['font']['zorder'] = dict(
            gridValue=250, text=250, timestamp=250
        )
        # Line setting
        self.config_['line'] = dict()
        self.config_['line']['color'] = dict(
            coastline='gray', country='gray', state='gray', city='gray',
            gridLine='black', contour='black', stream='white', barbs='white',
            polygon='black'
        )
        self.config_['line']['style'] = dict(
            coastline='solid', country='solid', state='dotted', city='none',
            gridLine='dashed', contour='solid', stream='solid', barbs='solid',
            polygon='solid'
        )
        self.config_['line']['width'] = dict(
            coastline=0.5, country=0.45, state=0.45, city=0.45,
            gridLine=0.45, contour=0.6, stream=0.6, barbs=0.6,
            polygon=0.5, hatch=0
        )
        self.config_['line']['alpha'] = dict(
            coastline=1, country=1, state=1, city=1,
            gridLine=0.6, contour=1, stream=1, barbs=1,
            polygon=1
        )
        self.config_['line']['zorder'] = dict(
            coastline=100, country=100, state=100, city=100,
            gridLine=300, contour=100, stream=100, barbs=100,
            polygon=300
        )
        # Fill setting
        self.config_['fill'] = dict()
        self.config_['fill']['alpha'] = dict(
            contourf=1, pcolormesh=1, scatter=1
        )
        self.config_['fill']['zorder'] = dict(
            contourf=50, pcolormesh=50, scatter=200
        )

    def updateLineConfig(self, config: dict):
        for key, val in config.items():
            if key in self.config_['line'].keys():
                self.config_['line'][key].update(val)
            else:
                self.config_['line'].update(config)

    def setMesh(self, 
                x: Union[Sequence, np.ndarray, xr.DataArray],
                y: Union[Sequence, np.ndarray, xr.DataArray], 
                unstructured: Optional[bool] = False):
        self.currentAxInfo_['x'] = x
        self.currentAxInfo_['y'] = y
        self.currentAxInfo_['unstructured'] = unstructured

    def _findBuiltInMapConfig(self, map: str):
        map = map.lower()
        if map == 'global':
            proj = 'PlateCarree'
            mapConfig = dict(
                map_range = (-90, 90, -180, 180),
                central_longitude=0, grid_spacing=20
            )
        elif map == 'north_america':
            proj = 'LambertConformal'
            mapConfig = dict(
                map_range=(8, 72, -145, -55),
                central_longitude=-100, central_latitude=40,
                standard_parallels=(24, 56),
                grid_spacing=5
            )
        elif map == 'conus':
            proj = 'LambertConformal'
            mapConfig = dict(
                map_range=(18, 54.2, -135.8, -60),
                central_longitude=-97.9, central_latitude=39.1,
                standard_parallels=(30, 50),
                grid_spacing=5
            )
        elif map == 'conus404':
            proj = 'LambertConformal'
            mapConfig = dict(
                map_range=(20.86, 55.36, -123.45, -72.35),
                central_longitude=-97.9, central_latitude=39.1,
                standard_parallels=(30, 50),
                grid_spacing=5
            )

        return proj, mapConfig

    def setMap(self, 
               map: Optional[str] = None,
               mapConfig: Optional[dict] = None,
               proj: Optional[Union[ccrs.Projection, str]] = 'PlateCarree',
               resolution: Optional[str] = 'medium',
               spacing: Optional[Union[int, float]] = None,
               spineColor: Optional[str] = None,
               spineThickness: Optional[float] = None, **kwargs):
        _projMap = dict(
            P='PlateCarree', L='LambertConformal', M='Mercator',
            N='NorthPolarStereo', G='Geostationary'
        )
        _resMap = dict(high='10m', medium='50m', low='110m')

        if mapConfig is None:
            mapConfig = dict()
        if map is not None:
            proj, mapConfig = self._findBuiltInMapConfig(map=map)

        if 'map_range' not in mapConfig:
            mapRange = (-90, 90, -180, 180)
        else:
            mapRange = mapConfig.pop('map_range')
        
        gridSpacing = mapConfig.pop('grid_spacing', None)
        gridSpacing = gridSpacing if spacing is None else spacing

        if isinstance(proj, ccrs.Projection):
            _proj = proj
            proj = type(_proj).__name__
        else:
            proj = _projMap.get(proj.upper(), proj)
            if (proj in ['PlateCarree', 'Mercator']) and (mapRange[2] < 180 and 
                                                          180 < mapRange[3]):
                mapConfig['central_longitude'] = 180
            self.config_['map']['proj'] = proj
            _proj = getattr(ccrs, proj)(**mapConfig)

        mapConfig['gridSpacing'] = \
            self.config_['map']['gridSpacing'] if gridSpacing is None else gridSpacing
  
        if self.currentAxIdx_ is not None:
            shape, loc, rowspan, colspan = self.currentAxIdx_
            self.currentAx_ = plt.subplot2grid(
                shape, loc, rowspan=rowspan, colspan=colspan,
                fig=self.figure_, projection=_proj
            )
        else:
            self.currentAx_ = plt.axes(projection=_proj)
        
        mapConfig['resolution'] = _resMap.get(
            resolution.lower(), resolution
        )
        mapConfig['mapRange'] = mapRange
        extend = mapConfig['mapRange'][2:] + mapConfig['mapRange'][:2]
        if proj != 'NearsidePerspective':
            self.currentAx_.set_extent(extend, crs=ccrs.PlateCarree())
            width, height = self.figure_.get_size_inches()
            dLon = mapConfig['mapRange'][3] - mapConfig['mapRange'][2]
            dLat = mapConfig['mapRange'][1] - mapConfig['mapRange'][0]

            aspect = self.config_['map']['aspect']
            if aspect == 'auto':
                aspectRatio = (height * dLon) / (width * dLat)
                self.currentAx_.set_aspect(aspectRatio)
            elif aspect is not None:
                self.currentAx_.set_aspect(aspect)
        
        for i in self.currentAx_.spines.values():
            if spineColor is not None:
                i.set_edgecolor(spineColor)
            if spineThickness is not None:
                i.set_linewidth(spineThickness)

        boundary = self.config_['map']['boundary']
        if boundary is None:
            self.currentAx_.patch.set_linewidth(0)
        elif boundary == 'rect':
            self.currentAx_.patch.set_linewidth(1)
        elif boundary == 'circle':
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5, 0.5]
            unitCircle = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(unitCircle * radius + center)
            self.currentAx_.set_boundary(
                circle, transform=self.currentAx_.transAxes, linewidth=1)
        else:
            raise Exception(f'Unknown boundary value: {boundary}')
        
        if 'map' not in self.currentAxInfo_:
            self.currentAxInfo_.update(map=dict())
        self.currentAxInfo_.update(projection=_proj)
        self.currentAxInfo_['map'].update(mapConfig)

    def getMap(self, 
            map: Optional[str] = None,
            mapConfig: Optional[dict] = None,
            proj: Optional[Union[ccrs.Projection, str]] = 'PlateCarree',
            **kwargs):
        _projMap = dict(
            P='PlateCarree', L='LambertConformal', M='Mercator',
            N='NorthPolarStereo', G='Geostationary'
        )

        ret = dict()
        if mapConfig is None:
            mapConfig = dict()
        if map is not None:
            proj, mapConfig = self._findBuiltInMapConfig(map=map)

        if 'map_range' not in mapConfig:
            mapRange = (-90, 90, -180, 180)
        else:
            mapRange = mapConfig.pop('map_range')
        
        mapConfig.pop('grid_spacing', None)
        if isinstance(proj, ccrs.Projection):
            _proj = proj
            proj = type(_proj).__name__
        else:
            proj = _projMap.get(proj.upper(), proj)
            if (proj in ['PlateCarree', 'Mercator']) and (mapRange[2] < 180 and 
                                                          180 < mapRange[3]):
                mapConfig['central_longitude'] = 180
            self.config_['map']['proj'] = proj
            _proj = getattr(ccrs, proj)(**mapConfig)
        
        ret.update(proj=_proj)

        mapConfig['map_range'] = mapRange
        extend = mapConfig['map_range'][2:] + mapConfig['map_range'][:2]
        ret.update(extend=extend)
        return ret
    
    def setPlot(self, xType: str = None, yType: str = None, 
                xUnit: str = None, yUnit: str = None, 
                xMin: float = None, xMax: float = None,
                yMin: float = None, yMax: float = None,
                spineColor: str = None, spineThickness: float = None, 
                tickConfig: dict = None,
                projection: str = None):
        tickConfig = tickConfig or dict()
        if self.currentAxIdx_ is not None:
            self.axes_[self.currentAxIdx_] = self.currentAx_
        else:
            self.currentAx_ = plt.axes(projection=projection)
        
        xType = 'linear' if xType is None else xType
        yType = 'linear' if yType is None else yType
        self.currentAx_.set_xscale(xType)
        self.currentAx_.set_yscale(yType)
        if xMin is not None:
            self.currentAx_.set_xlim(left=xMin)
        if xMax is not None:
            self.currentAx_.set_xlim(right=xMax)
        if yMin is not None:
            self.currentAx_.set_ylim(bottom=yMin)
        if yMax is not None:
            self.currentAx_.set_ylim(top=yMax)
        if xUnit is not None:
            self.currentAx_.set_xlabel(xUnit)
        if yUnit is not None:
            self.currentAx_.set_ylabel(yUnit)
        self.currentAx_.tick_params(**tickConfig)
        for spine in self.currentAx_.spines.values():
            if spineColor:
                spine.set_edgecolor(spineColor)
            if spineThickness:
                spine.set_linewidth(spineThickness)
        
        if yType == 'log':
            self.currentAx_.yaxis.set_major_formatter(mticker.ScalarFormatter())
            self.currentAx_.yaxis.set_major_locator(mticker.MultipleLocator(100))
            self.currentAx_.yaxis.set_minor_formatter(mticker.NullFormatter())
        if xType == 'log':
            self.currentAx_.xaxis.set_major_formatter(mticker.ScalarFormatter())
            self.currentAx_.xaxis.set_major_locator(mticker.MultipleLocator(100))
            self.currentAx_.xaxis.set_minor_formatter(mticker.NullFormatter())
    
    def _findShpFile(self, filename: str):
        filePath = ''
        if os.path.exists(filename):
            filePath = filename
        else:
            filePath = os.path.join(_shpDir, filename)
            if not os.path.exists(filePath):
                raise Exception(f'File not found: {filePath}')
        return filePath
    
    @functools.lru_cache(maxsize=32)
    def getFeature(self, *args, **kwargs):
        return cfeature.ShapelyFeature(*args, **kwargs)
    
    def drawFeature(self,
                    name: Union[Sequence, str], linewidth: Optional[float] = None,
                    linestyle: Optional[str] = None, color: Optional[str] = None,
                    alpha: Optional[Union[int, float]] = None, 
                    resolution: Optional[str] = None, zorder: Optional[int] = None, 
                    **kwargs):
        naturalEarthNameMap = dict(
            coastline='coastline', country='admin_0_boundary_lines_land',
            state='admin_1_states_provinces_lines',
            province='admin_1_states_provinces_lines'
        )
        naturalEarthCategoryMap = dict(
            coastline='physical', admin_0_boundary_lines_land='cultural',
            admin_1_states_provinces_lines='cultural'
        )
        if isinstance(name, str):
            name = [name]
        for i in name:
            feature = None
            linewidth = self.config_['line']['width'][i] if linewidth is None else linewidth
            linestyle = self.config_['line']['style'][i] if linestyle is None else linestyle
            color = self.config_['line']['color'][i] if color is None else color
            alpha = self.config_['line']['alpha'][i] if alpha is None else alpha
            zorder = self.config_['line']['zorder'][i] if zorder is None else zorder
            shpFile = ''
            flagNaturalEarth = i in naturalEarthNameMap.keys() or \
                i in naturalEarthCategoryMap.keys()
            if (i.endswith('.shp')) or (not flagNaturalEarth):
                shpFile = i
                shpFile = self._findShpFile(shpFile)
            else:
                resolution = self.currentAxInfo_['map']['resolution'] \
                    if resolution is None else resolution
                naturalEarthName = naturalEarthNameMap.get(i, i)
                category = naturalEarthCategoryMap.get(naturalEarthName)
                shpFile = cshreader.natural_earth(
                    resolution=resolution, category=category,
                    name=naturalEarthName)
                
            feature = self.getFeature(
                cshreader.Reader(shpFile).geometries(),
                crs=ccrs.PlateCarree(), facecolor='none', 
                edgecolor=color)
            self.currentAx_.add_feature(
                feature, linewidth=linewidth, 
                linestyle=linestyle, zorder=zorder, **kwargs)
    
    def drawMesh(self, linewidth: Optional[float] = None,
                 linestyle: Optional[str] = None, color: Optional[str] = None,
                 alpha: Optional[Union[int, float]] = None, 
                 spacing: Optional[str] = None, zorder: Optional[int] = None, 
                 **kwargs):
        linewidth = self.config_['line']['width']['gridLine'] if linewidth is None else linewidth
        linestyle = self.config_['line']['style']['gridLine'] if linestyle is None else linestyle
        color = self.config_['line']['color']['gridLine'] if color is None else color
        alpha = self.config_['line']['alpha']['gridLine'] if alpha is None else alpha
        zorder = self.config_['line']['zorder']['gridLine'] if zorder is None else zorder
        labelSize = kwargs.pop('fontsize', None)
        labelSize = self.config_['font']['size']['gridValue'] if labelSize is None else labelSize
        labelFamily = self.config_['font']['family']

        gridSpacing = self.currentAxInfo_['map']['gridSpacing'] if spacing is None else spacing

        flagLine = linewidth is not None
        flagDrawLabel = self.config_['map'].get('proj') != 'NearsidePerspective'
        gl = self.currentAx_.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=flagDrawLabel,
            linewidth=linewidth, color=color, linestyle=linestyle, 
            zorder=zorder, alpha=alpha, **kwargs
        )
        if flagDrawLabel:
            gl.top_labels = self.config_['map'].get('top_labels', False)
            gl.bottom_labels = self.config_['map'].get('bottom_labels', True)
            gl.left_labels = self.config_['map'].get('left_labels', True)
            gl.right_labels = self.config_['map'].get('right_labels', False)
            gl.xformatter = cgridliner.LONGITUDE_FORMATTER
            gl.yformatter = cgridliner.LATITUDE_FORMATTER
            gl.xlabel_style = dict(
                size=labelSize, color=color, family=labelFamily
            )
            gl.ylabel_style = dict(
                size=labelSize, color=color, family=labelFamily
            )

        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, gridSpacing))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, gridSpacing))
        gl.x_inline = self.config_['map'].get('x_inline', False)
        gl.y_inline = self.config_['map'].get('y_inline', False)
        gl.rotate_labels = self.config_['map'].get('rotate_labels', False)
        if not flagLine:
            gl.xlines = False
            gl.ylines = False
        
        if self.config_['figure']['full']:
            gl.xpadding = -8
            gl.ypadding = -12
        else:
            gl.xpadding = 3
            gl.xpadding = 2

    def mapStyle(self, mapStyle: str, resolution: Optional[str] = None,
                 zorder: Optional[Union[dict, int]] = 1):
        mapStyle = mapStyle.lower()
        resolution = self.currentAxInfo_['map']['resolution'] if resolution is None else resolution
        if mapStyle not in ['jma', 'bom']:
            logger.error(f'Unknown map style: {mapStyle}')
        if mapStyle == 'jma':
            oceanColor = '#87A9D2'
            riverColor = '#87A9D2'
            lakeColor = '#87A9D2'
            landColor = '#AAAAAA'
            lineColor = dict(
                coastline='#666666', country='#666666', state='#888888',
                city='#888888', coutry='#888888', grid_line='#666666'
            )
            self.updateLineConfig(dict(color=lineColor))
        elif mapStyle == 'bom':
            oceanColor = '#E6E6FF'
            riverColor = '#E6E6FF'
            lakeColor = '#E6E6FF'
            landColor = '#E8E1C4'
            lineColor = dict(
                coastline='#D0A85E', country='#D0A85E', state='#D0A85E',
                city='#D0A85E', coutry='#D0A85E', grid_line='#D0A85E'
            )
            self.updateLineConfig(dict(color=lineColor))
        
        oceanZorder = zorder if isinstance(zorder, int) else zorder.get('ocean', 1)
        self.currentAx_.add_feature(
            cfeature.OCEAN.with_scale(resolution),
            color=oceanColor, zorder=oceanZorder
        )
        riverZorder = zorder if isinstance(zorder, int) else zorder.get('river', 1)
        self.currentAx_.add_feature(
            cfeature.RIVERS.with_scale(resolution),
            color=riverColor, zorder=riverZorder
        )
        if resolution == '10m':
            self.currentAx_.add_feature(
                cfeature.NaturalEarthFeature(
                    'physical', 'rivers_north_america', 
                    resolution, edgecolor=cfeature.COLORS['water'],
                    facecolor='never'),
                color=riverColor, zorder=riverZorder
            )
        lakeZorder = zorder if isinstance(zorder, int) else zorder.get('lake', 1)
        self.currentAx_.add_feature(
            cfeature.LAKES.with_scale(resolution),
            color=lakeColor, zorder=lakeZorder
        )
        if resolution == '10m':
            self.currentAx_.add_feature(
                cfeature.NaturalEarthFeature(
                    'physical', 'lakes_north_america', 
                    resolution, edgecolor=cfeature.COLORS['water'],
                    facecolor='never'),
                color=lakeColor, zorder=lakeZorder
            )
        landZorder = zorder if isinstance(zorder, int) else zorder.get('land', 1)
        self.currentAx_.add_feature(
            cfeature.LAND.with_scale(resolution),
            color=landColor, zorder=landZorder
        )

    def switchAx(self, shape: tuple, loc: tuple, 
                 rowspan: Optional[int] = 1, colspan: Optional[int] = 1,
                 onHoldForSetMap: Optional[bool] = False):
        if self.currentAxIdx_ is not None:
            self.axes_[self.currentAxIdx_] = self.currentAx_
        
        idx = (shape, loc, rowspan, colspan)
        if idx not in self.axes_.keys():
            if not onHoldForSetMap:
                self.currentAx_ = plt.subplot2grid(
                    shape, loc, rowspan=rowspan, colspan=colspan, fig=self.figure_
                )
        else:
            self.currentAx_ = self.axes_[idx]

        self.currentAxIdx_ = idx

    def getAxIndex(self, shape: tuple, loc: tuple,
                   rowspan: Optional[int] = 1, colspan: Optional[int] = 1):
        if self.currentAxIdx_ is not None:
            self.axes_[self.currentAxIdx_] = self.currentAx_
        
        idx = (shape, loc, rowspan, colspan)
        if idx not in self.axes_.keys():
            idx = None
        
        return idx
    
    def getAx(self):
        return self.currentAx_
    
    def _checkMesh(self,
            x: Union[Sequence, np.ndarray, xr.DataArray],
            y: Union[Sequence, np.ndarray, xr.DataArray], 
            data: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None):
        xShape = np.shape(x)
        yShape = np.shape(y)
        if xShape != yShape:
            raise Exception('The shape of x and y must be consistent.')
        
        if data is not None:
            dataShape = np.shape(data)
            if xShape != dataShape:
                raise Exception('The shape of x, y and data must be consistent.')
        
        return True
    
    def transformData(self, 
            x: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None,
            y: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None,
            data: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None, 
            checkCyclicPoint: Optional[bool] = False):
        x = self.currentAxInfo_.get('x', None) if x is None else x
        y = self.currentAxInfo_.get('y', None) if y is None else y

        x = x.to_numpy() if isinstance(x, xr.DataArray) else x
        y = y.to_numpy() if isinstance(y, xr.DataArray) else y
        data = data.to_numpy() if isinstance(x, xr.DataArray) else data

        if 'map' in self.currentAxInfo_.keys():
            if isinstance(self.currentAx_.projection, ccrs.Projection):
                _proj = self.currentAx_.projection
                projName = type(_proj).__qualname__
                xLimits = _proj.x_limits
                if projName == 'PlateCarree' and xLimits == (-180, 180) and checkCyclicPoint:
                    x = add_cyclic_point(x)
                    if x[0, -2] * x[0, -1] < 0:
                        x[:, -1] *= -1
                    y = add_cyclic_point(y)
                    data = add_cyclic_point(data)
            if x is not None and y is not None:
                result = self.currentAx_.projection.transform_points(
                    ccrs.PlateCarree(), x, y, data
                )
                x = result[..., 0]
                y = result[..., 1]
                if data is not None:
                    data = result[..., 2]
        return x, y, data
    
    def colorbar(self,
            mappable,
            unit: Optional[str] = None,
            flagGlobal: Optional[bool] = False,
            unitPos: Optional[Union[tuple, list]] = [1.05, 1.02],
            addAxes: Optional[list] = None, 
            size: Optional[str] = '2%', 
            pad: Optional[str] = '1%', **kwargs):
        fontsize = kwargs.pop('fontsize', None)
        fontColor = kwargs.pop('color', None)
        labelSize = self.config_['font']['size']['cbar'] if fontsize is None else fontsize
        fontColor = self.config_['font']['color']['cbar'] if fontColor is None else fontColor
        fontFamily = self.config_['font']['family']

        location = kwargs.pop('location', None)
        orientation = kwargs.pop('orientation', None)
        if orientation == 'horizontal':
            location = 'bottom' if location is None else location
        else:
            location = 'right' if location is None else location
            orientation = 'vertical'
        if unit is not None:
            self.currentAx_.text(
                *unitPos, unit, transform=self.currentAx_.transAxes,
                ha='right', fontsize=fontsize, family=fontFamily,
                color=fontColor)
        divider = make_axes_locatable(self.currentAx_)
        aspect = self.currentAx_.get_aspect()
        if isinstance(aspect, (int, float)):
            padValue = int(pad.strip('%')) / 100
            padValue = padValue - (1 - 1 / aspect) / 2
            pad = f'{padValue:.02%}'
        if addAxes:
            cax = plt.axes(addAxes)
        elif flagGlobal:
            cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        else:
            cax = divider.append_axes(location, size=size, pad=pad,
                                      axes_class=plt.Axes)
        cbarNorm = kwargs.pop('norm', None)
        if cbarNorm is not None:
            vmax = np.max(kwargs.get('ticks'))
            vmin = np.min(kwargs.get('ticks'))
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = None
        cb = plt.colorbar(mappable=mappable, cax=cax, orientation=orientation, **kwargs)
        cb.ax.tick_params(labelsize=labelSize, length=0, pad=2)
        cb.outline.set_linewidth(0.4)
        for i in cb.ax.xaxis.get_ticklabels():
            i.set(fontfamily=fontFamily, color=fontColor)
        for i in cb.ax.yaxis.get_ticklabels():
            i.set(fontfamily=fontFamily, color=fontColor)
        return cb
    
    def contour(self, 
            data: Union[Sequence, np.ndarray, xr.DataArray], 
            x: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None,
            y: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None,
            clabelFlag: Optional[bool] = True,
            clabelConfig: Optional[dict] = None,
            color: Optional[Sequence | str] = None,
            linewidth: float = None, linestyle: str = None,
            alpha: float = None, zorder: int = None,
            flagUnstructured: bool = None, **kwargs):
        clabelConfig = clabelConfig or dict()
        color = self.config_['line']['color']['contour'] if color is None else color
        linewidth = self.config_['line']['width']['contour'] if linewidth is None else linewidth
        linestyle = self.config_['line']['style']['contour'] if linestyle is None else linestyle
        alpha = self.config_['line']['alpha']['contour'] if alpha is None else alpha
        zorder = self.config_['line']['alpha']['contour'] if zorder is None else zorder
        flagUnstructured = self.currentAxInfo_.get('unstructured', False) if flagUnstructured is None else flagUnstructured
        
        checkCyclicPoint = False if flagUnstructured else True
        x, y, data = self.transformData(x=x, y=y, data=data, checkCyclicPoint=checkCyclicPoint)
        
        kwargs.update(colors=color, linewidth=linewidth, linestyles=linestyle,
                      alpha=alpha, zorder=zorder)
        if 'projection' in self.currentAxInfo_.keys():
            kwargs.update(transform=self.currentAx_.projection,
                          transform_first=True)
        if flagUnstructured:
            maskGood = ~(np.isnan(x) | np.isnan(y) | np.isnan(data))
            x = x[maskGood]
            y = y[maskGood]
            data = data[maskGood]
            contour = self.currentAx_.tricontour(x, y, data, **kwargs)
        else:
            contour = self.currentAx_.contour(x, y, data, **kwargs)
        
        if clabelFlag:
            if 'levels' in clabelConfig:
                levels = clabelConfig.pop('levels')
            elif 'levels' in kwargs:
                levels = clabelConfig.get('levels')
            else:
                levels = contour.levels
            fontsize = clabelConfig.pop('fontsize', None)
            fontsize = self.config_['font']['size']['clabel'] if fontsize is None else fontsize
            clabelConfig = mergeDict(
                clabelConfig, dict(
                    fmt='%d', fontsize=fontsize, zorder=zorder
                ))
            clabel = self.currentAx_.clabel(contour, levels=levels, **clabelConfig)
            if clabel is not None:
                for i in range(len(clabel)):
                    clabel[i].set(fontfamily=self.config_['font']['family'])
            return contour
    
    def contourf(self, 
            data: Union[Sequence, np.ndarray, xr.DataArray], 
            x: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None,
            y: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None,
            cmap: Optional[str | mcolors.Colormap] = None,
            cmapConfig: Optional[dict] = None, 
            cbar: Optional[bool] = False,
            cbarConfig: Optional[dict] = None,
            alpha: Optional[float] = None,
            zorder: Optional[int] = None, hatch: Optional[dict] = None,
            unstructured: Optional[bool] = None, **kwargs):
        cbarConfig = cbarConfig or dict()
        kwargs = kwargs or dict()
        alpha = self.config_['fill']['alpha']['contourf'] if alpha is None else alpha
        zorder = self.config_['fill']['zorder']['contourf'] if zorder is None else zorder
        if cmap is not None:
            if isinstance(cmap, str):
                if cmapConfig is not None:
                    cmapConfig.update(name=cmap)
                    kwargs = mergeDict(kwargs, cmgr.cmap(**cmapConfig))
                else:
                    kwargs = mergeDict(kwargs, cmgr.cmap(cmap))
            else:
                kwargs = mergeDict(kwargs, dict(cmap=cmap))
        levels = kwargs.get('levels', None)
        levels = kwargs.get('clev', kwargs.get('cmap').N) if levels is None else levels
        unit = cmapConfig.pop('unit', kwargs.pop(
            'unit', None)) if cmapConfig is not None else kwargs.pop('unit', None)
        
        flagUnstructured = self.currentAxInfo_.get('unstructured', False) if unstructured is None else unstructured
        flagCheckCyclicPoint = False if unstructured else True
        x, y, data = self.transformData(x=x, y=y, data=data, checkCyclicPoint=flagCheckCyclicPoint)
        kwargs.update(alpha=alpha, zorder=zorder, levels=levels)
        if isinstance(levels, Union[list, np.ndarray]) and cmap is not None:
            if len(levels) == kwargs.get('cmap').N:
                norm = mcolors.BoundaryNorm(levels, kwargs.get('cmap').N)
                kwargs.update(norm=norm)

        if 'projection' in self.currentAxInfo_.keys():
            kwargs.update(transform=self.currentAx_.projection,
                          transform_first=True)
        
        if flagUnstructured:
            maskGood = ~(np.isnan(x) | np.isnan(y) | np.isnan(data))
            x = x[maskGood]
            y = y[maskGood]
            data = data[maskGood]
            contourf = self.currentAx_.tricontourf(x, y, data, **kwargs) \
              if cmap is not None else None
        else:
            contourf = self.currentAx_.contourf(x, y, data, **kwargs) \
              if cmap is not None else None
            
        if hatch is not None:
            hatchKwargs = dict()
            if 'projection' in self.currentAxInfo_.keys():
                hatchKwargs.update(
                    transform=self.currentAx_.projection,
                    transform_first=True)
                hatchLevels = hatch.pop('levels')
                hatchTexture = hatch.pop('texture', '...')
                hatchColor = hatch.pop('colors', None)
                hatchAlpha = hatch.pop('alpha', 1)
                hatchLinewidth = hatch.pop('linewidth', None)
                hatchLinewidth = self.config_['line']['width']['hatch'] if hatchLinewidth is None else hatchLinewidth
                for i in range(len(hatchTexture)):
                    if 'none' == hatchTexture[i].lower():
                        hatchTexture[i] = None
                hatchKwargs.update(
                    hatches=hatchTexture, alpha=hatchAlpha, 
                    linewidth=hatchLinewidth, zorder=zorder, extend='both',
                    facecolor=None, levels=hatchLevels
                )
                if unstructured:
                    hatching = self.currentAx_.tricontourf(x, y, data, **hatchKwargs)
                else:
                    hatching = self.currentAx_.contourf(x, y, data, **hatchKwargs)
                if hatchColor:
                    hatching.set_facecolor('none')
                    hatching.set_edgecolor(hatchColor)
                    hatching.set_linewidth(hatchLinewidth)
        
        if cbar:
            if 'ticks' not in cbarConfig:
                levels = contourf.levels if not isinstance(kwargs.get(
                    'levels'), list) else levels
                step = len(levels) // 40 + 1
                if cbarConfig.get('orientation', None) == 'horizontal':
                    step = step * 2 - 1
                cbarConfig.update(ticks=levels[::step])
            if 'extend' in kwargs:
                cbarConfig.update(extend=kwargs.pop('extend'), extendfrac=0.02)
            self.colorbar(contourf, unit=unit, **cbarConfig)
        return contourf
    
    def pcolormesh(self, 
            data: Union[Sequence, np.ndarray, xr.DataArray], 
            x: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None,
            y: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None,
            cmap: Optional[str] = None,
            cmapConfig: Optional[dict] = None, 
            cmapScale: str = 'linear',
            cbar: Optional[bool] = False,
            cbarConfig: Optional[dict] = None,
            alpha: Optional[float] = None,
            zorder: Optional[int] = None, 
            hatch: Optional[dict] = None, **kwargs):
        cbarConfig = cbarConfig or dict()
        kwargs = kwargs or dict()
        alpha = self.config_['fill']['alpha']['pcolormesh'] if alpha is None else alpha
        zorder = self.config_['fill']['zorder']['pcolormesh'] if zorder is None else zorder
        if cmap is not None:
            if cmapConfig is not None:
                cmapConfig.update(name=cmap)
                kwargs = mergeDict(kwargs, cmgr.cmap(**cmapConfig))
            else:
                kwargs = mergeDict(kwargs, cmgr.cmap(cmap))
        unit = cmapConfig.pop('unit', kwargs.pop(
            'unit', None)) if cmapConfig is not None else kwargs.pop('unit', None)
        
        # x, y, data = self.transformData(x=x, y=y, data=data, checkCyclicPoint=True)
        kwargs.update(alpha=alpha, zorder=zorder)
        if 'projection' in self.currentAxInfo_.keys():
            kwargs.update(transform=ccrs.PlateCarree())
            # kwargs.update(transform=self.currentAx_.projection)
        
        levels = kwargs.pop('levels', kwargs.pop('clev', []))
        cmap = kwargs.get('cmap')

        if cmapScale == 'linear':
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        elif cmapScale == 'log':
            norm = mcolors.LogNorm(vmin=levels[0], vmax=levels[1], 
                                   clip=True)
        extend = kwargs.pop('extend', None)
        pcolormesh = self.currentAx_.pcolormesh(x, y, data, norm=norm, **kwargs) \
            if cmap is not None else None
        
        if cbar:
            if 'ticks' not in cbarConfig:
                step = len(levels) // 40 + 1
                if cbarConfig.get('orientation', None) == 'horizontal':
                    step = step * 2 - 1
                cbarConfig.update(ticks=levels[::step])
            if extend is not None:
                cbarConfig.update(extend=extend, extendfrac=0.02)
            cb = self.colorbar(pcolormesh, unit=unit, **cbarConfig)
            cb.minorticks_off()
        return pcolormesh
    
    def scatter(self, 
            data: Union[Sequence, np.ndarray, xr.DataArray], 
            x: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None,
            y: Optional[Union[Sequence, np.ndarray, xr.DataArray]] = None,
            cmap: Optional[str] = None,
            cmapConfig: Optional[dict] = None, 
            cbar: Optional[bool] = False,
            cbarConfig: Optional[dict] = None,
            alpha: Optional[float] = None,
            zorder: Optional[int] = None, 
            hatch: Optional[dict] = None, **kwargs):
        cbarConfig = cbarConfig or dict()
        kwargs = kwargs or dict()
        alpha = self.config_['fill']['alpha']['scatter'] if alpha is None else alpha
        zorder = self.config_['fill']['zorder']['scatter'] if zorder is None else zorder
        if cmap is not None:
            if cmapConfig is not None:
                cmapConfig.update(name=cmap)
                kwargs = mergeDict(kwargs, cmgr.cmap(**cmapConfig))
            else:
                kwargs = mergeDict(kwargs, cmgr.cmap(cmap))
        unit = cmapConfig.pop('unit', kwargs.pop(
            'unit', None)) if cmapConfig is not None else kwargs.pop('unit', None)
        
        x, y, data = self.transformData(x=x, y=y, data=data, checkCyclicPoint=False)
        kwargs.update(alpha=alpha, zorder=zorder)
        if 'projection' in self.currentAxInfo_.keys():
            kwargs.update(transform=self.currentAx_.projection)
        
        levels = kwargs.pop('levels', [])
        extend = kwargs.pop('extend', None)
        vmin = kwargs.get('vmin', None)
        vmax = kwargs.get('vmax', None)
        if isinstance(levels, Union[list, np.ndarray]) and cmap is not None:
            if len(levels) == kwargs.get('cmap').N:
                norm = mcolors.BoundaryNorm(levels, kwargs.get('cmap').N)
                kwargs.update(norm=norm)
            elif len(levels) > 1:
                vmax = levels[0]
                vmin = levels[-1]
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                kwargs.update(norm=norm)
        elif vmax is not None and vmin is not None:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            kwargs.update(norm=norm)

        scatter = self.currentAx_.scatter(
            x, y, c=data, edgecolor='face', **kwargs)
        
        if cbar:
            if extend is not None:
                cbarConfig.update(extend=extend, extendfrac=0.02)
            self.colorbar(scatter, unit=unit, **cbarConfig)
        return scatter
    
    def polygon(self, 
            points: Union[Sequence, np.ndarray], label: Optional[str] = None, 
            fill: Optional[bool] = False, color: Optional[str] = None, 
            linewidth: Optional[float] = None, linestyle: Optional[str] = None,
            alpha: Optional[float] = None, zorder: Optional[float] = None, 
            labelOnPolygon: Optional[bool] = False, **kwargs):
        linewidth = self.config_['line']['width']['polygon'] if linewidth is None else linewidth
        linestyle = self.config_['line']['style']['polygon'] if linestyle is None else linestyle
        color = self.config_['line']['color']['polygon'] if color is None else color
        alpha = self.config_['line']['alpha']['polygon'] if alpha is None else alpha
        zorder = self.config_['line']['zorder']['polygon'] if zorder is None else zorder
        labelSize = kwargs.pop('labelSize', None)
        labelSize = self.config_['font']['size']['annot'] if labelSize is None else labelSize
        labelWeight = kwargs.pop('labelWeight', None)
        fontFamily = self.config_['font']['family']
        transform = kwargs.get('transform', None)

        kwargs.update(color=color, linewidth=linewidth, linestyle=linestyle,
                      alpha=alpha, zorder=zorder, label=label, fill=fill)
        if transform and 'map' in self.currentAxInfo_.keys():
            kwargs.update(transform=ccrs.PlateCarree())
        patch = mpatches.Polygon(points, closed=True, **kwargs)
        polygon = self.currentAx_.add_patch(patch)

        if label is not None and labelOnPolygon:
            labelX = np.mean(points[:,0])
            labelY = np.mean(points[:,1])
            dLabelY = np.max(points[:,1]) - np.min(points[:,1])
            labelY += 0.8 * dLabelY
            self.text(labelX, labelY, label, color=color,
                fontsize=labelSize, family=fontFamily, 
                fontweight=labelWeight, horizontalalignment='center')

        return polygon
    
    def plot(self, x = None, y = None, *args, 
             transform: bool = False, add_axes = False, 
             xLimit: float = None, yLimit: float = None,
             label: str = None, format: str = '%s',
             xAxisConfig = None,
             yAxisConfig = None,
             **kwargs):
        if 'zorder' not in kwargs.keys():
            kwargs.update(zorder=self.config_['line']['zorder']['polygon'])
        if 'linestyle' not in kwargs.keys():
            kwargs.update(linestyle=self.config_['line']['style']['polygon'])
        if 'linewidth' not in kwargs.keys():
            kwargs.update(linewidth=self.config_['line']['width']['polygon'])
        if 'alpha' not in kwargs.keys():
            kwargs.update(alpha=self.config_['line']['alpha']['polygon'])

        if add_axes:
            currentAx = plt.axes(add_axes)
        else:
            currentAx = self.currentAx_

        if xLimit == 'auto':
            xLimit = (np.nanmin(x), np.nanmax(x))
        if yLimit == 'auto':
            yLimit = (np.nanmin(y), np.nanmax(y))
        x, y, _ = self.transformData(x=x, y=y, checkCyclicPoint=False)
        if 'map' in self.currentAxInfo_.keys() and transform:
            kwargs.update(transform=self.currentAx_.projection)
        # if 'map' in self.currentAxInfo_ and :
        #     kwargs.update(transform=ccrs.PlateCarree())
        # else:
        #     kwargs.update(transform=currentAx.transAxes)
        
        if format is not None and label is not None:
            try:
                fmt = '{:' + format + '}'
                label = fmt.format(label)
            except:
                try:
                    if isinstance(label, np.datetime64):
                        ts = npDatetimeToDatetime(label)
                        label = ts.strftime(format)
                except:
                    label = label
            kwargs.update(label=label)

        if xLimit is not None:
            currentAx.set_xlim(xLimit)
        if yLimit is not None:
            currentAx.set_ylim(yLimit)
        if x is not None and y is not None:
            ret = currentAx.plot(x, y, *args, **kwargs)
        else:
            ret = currentAx.plot(*args, **kwargs)

        if xAxisConfig:
            xTicksConfig = xAxisConfig.get('ticks', None)
            xLabelConfig = xAxisConfig.get('label', None)
            if xTicksConfig:
                xTicksRotation = xTicksConfig.get('rotation', None)
                xTicksFormat = xTicksConfig.get('format', format)
                xTicksType = xTicksConfig.get('type', None)
                if xTicksRotation:
                    for l in currentAx.get_xticklabels():
                        l.set_rotation(xTicksRotation)
                if xTicksFormat:
                    if xTicksType == 'datetime':
                        currentAx.xaxis.set_major_formatter(mdates.DateFormatter(xTicksFormat))
                    else:
                        currentAx.xaxis.set_major_formatter(mticker.FormatStrFormatter(xTicksFormat))
            if xLabelConfig:
                xLabel = xLabelConfig.pop('text', None)
                if xLabel:
                    currentAx.set_xlabel(xLabel, **xLabelConfig)

        if yAxisConfig:
            yTicksConfig = yAxisConfig.get('ticks', None)
            yLabelConfig = yAxisConfig.get('label', None)
            if yTicksConfig:
                yTicksRotation = yTicksConfig.get('rotation', None)
                yTicksFormat = yTicksConfig.get('format', None)
                yTicksType = yTicksConfig.get('type', None)
                if yTicksRotation:
                        for l in currentAx.get_yticklabels():
                            l.set_rotation(yTicksRotation)
                if yTicksFormat:
                    if yTicksType == 'datetime':
                        currentAx.yaxis.set_major_formatter(mdates)
                    else:
                        currentAx.yaxis.set_major_formatter(mticker.FormatStrFormatter(format))
            if yLabelConfig:
                yLabel = yLabelConfig.pop('text', None)
                if yLabel:
                    currentAx.set_xlabel(yLabel, **xLabelConfig)

        return ret
    
    def colorplot(
            self, data, x = None, y = None, 
            cmap: Optional[str] = None,
            cmapConfig: Optional[dict] = None, 
            cbar: Optional[bool] = False,
            cbarConfig: Optional[dict] = None,
            linestyle: Optional[str] = None,
            linewidth: Optional[float] = None,
            alpha: Optional[float] = None,
            zorder: Optional[int] = None, **kwargs):
        if cmap is not None:
            if cmapConfig is not None:
                cmapConfig.update(name=cmap)
                kwargs = mergeDict(kwargs, cmgr.cmap(**cmapConfig))
            else:
                kwargs = mergeDict(kwargs, cmgr.cmap(cmap))

        levels = kwargs.get('levels', kwargs.get('clev', kwargs.get('cmap').N))
        unit = cmapConfig.pop('unit', kwargs.pop(
            'unit', None
        )) if cmapConfig is not None else kwargs.pop('unit', None)

        kwargs.update(linestyle=linestyle, linewidth=linewidth, alpha=alpha,
                      zorder=zorder)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        vmin = np.min(levels) if vmin is None else vmin
        vmax = np.max(levels) if vmax is None else vmax
        norm = plt.Normalize(vmin, vmax)
        extend = kwargs.pop('extend', None)
        if 'projection' in self.currentAxInfo_.keys():
            kwargs.update(transform=self.currentAx_.projection)
        x, y, data = self.transformData(x=x, y=y, data=data, checkCyclicPoint=False)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments=segments, norm=norm, **kwargs)
        lc.set_array(data)
        colorplot = self.currentAx_.add_collection(lc)
        if cbar:
            if 'ticks' not in cbarConfig:
                step = len(levels) // 40 + 1
                cbarConfig.update(ticks=levels[::step])
                if cbarConfig.get('orientation', None) == 'horizontal':
                    step = step * 2 - 1
                cbarConfig.update(ticks=levels[::step])
            if extend:
                cbarConfig.update(extend=extend, extendfrac=0.02)
            self.colorbar(colorplot, unit=unit, **cbarConfig)
        return colorplot

    def percentileplot(self, data, x = None, *args, 
                alpha: float = None, zorder: int = None, 
                label: str = None, format: str = None, **kwargs):
        q1 = np.zeros(len(data))
        q3 = np.zeros(len(data))
        percentiles = kwargs.pop('percentile', [25, 75])
        alpha = self.config_['line']['alpha']['polygon'] if alpha is None else alpha
        kwargs.update(alpha=alpha)
        zorder = self.config_['line']['zorder']['polygon'] if zorder is None else alpha
        kwargs.update(zorder=zorder)
        
        if label is not None:
            if format is not None:
                format = '{:' + format + '}'
                label = format.format(label)
            kwargs.update(label=label)
        
        for i in range(len(data)):
            q1[i], q3[i] = np.percentile(data[i], percentiles)

        self.currentAx_.fill_between(x, q1, q3, *args, **kwargs)

    def barbs(self, 
            u, v, x = None, y = None, color: str = 'white',
            linewidth: float = None, linestyle: str = None,
            alpha: float = None, zorder: int = None, length: float = 4.5,
            thinning: int = 12, **kwargs):
        linewidth = self.config_['line']['width']['barbs'] if linewidth is None else linewidth
        linestyle = self.config_['line']['style']['barbs'] if linestyle is None else linestyle
        alpha = self.config_['line']['alpha']['barbs'] if alpha is None else alpha
        zorder = self.config_['line']['zorder']['barbs'] if zorder is None else zorder

        # flagNorth = y >= 0

        x, y, u = self.transformData(x=x, y=y, data=u, checkCyclicPoint=True)
        _, _, v = self.transformData(data=v, checkCyclicPoint=True)
        kwargs.update(
            barbcolor=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha,
            zorder=zorder, length=length, regrid_shape=thinning,
            transform=self.currentAx_.projection)
        notNanIdx = ~np.isnan(u) & ~np.isnan(v)
        barbs = self.currentAx_.barbs(
            x[notNanIdx], y[notNanIdx], u[notNanIdx], v[notNanIdx], **kwargs
        )
        
        # if np.any(flagNorth & notNanIdx):
        #     flagNorth = flagNorth & notNanIdx
        #     barbsNorth = self.currentAx_.barbs(
        #         x[flagNorth], y[flagNorth], u[flagNorth], v[flagNorth], **kwargs
        #     )
        # else:
        #     barbsNorth = None
        # flagSouth = ~flagNorth
        # if np.any(flagSouth & notNanIdx) and (flagSouth & notNanIdx).sum() > 4:
        #     flagSouth = flagSouth & notNanIdx
        #     barbsSouth = self.currentAx_.barbs(
        #         x[flagSouth], y[flagSouth], u[flagSouth], v[flagSouth], 
        #         flip_barb=True, **kwargs
        #     )
        # else:
        #     barbsSouth = None
        return barbs
    
    def quiver(self, 
            u, v, x = None, y = None, label: bool = False,
            labelConfig: dict = None, color: str = 'white', 
            linewidth: float = None, linestyle: str = None,
            alpha: float = None, zorder: int = None,
            width: float = 0.0012, thinning: int = 40, 
            scale: float = 500, **kwargs):
        labelConfig = labelConfig or dict()
        linewidth = self.config_['line']['width']['barbs'] if linewidth is None else linewidth
        linestyle = self.config_['line']['style']['barbs'] if linestyle is None else linestyle
        alpha = self.config_['line']['alpha']['barbs'] if alpha is None else alpha
        zorder = self.config_['line']['zorder']['barbs'] if zorder is None else zorder

        x, y, u = self.transformData(x=x, y=y, data=u, checkCyclicPoint=True)
        _, _, v = self.transformData(data=v, checkCyclicPoint=True)
        kwargs.update(
            color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha,
            zorder=zorder, width=width, regrid_shape=thinning, scale=scale,
            transform=self.currentAx_.projection)
        notNanIdx = ~np.isnan(u) & ~np.isnan(v)
        quiver = self.currentAx_.quiver(
            x[notNanIdx], y[notNanIdx], u[notNanIdx], v[notNanIdx], **kwargs
        )
        if label:
            labelX = labelConfig.get('x', 0.75)
            labelY = labelConfig.get('y', 1.05)
            labelScale = labelConfig.get('scale', 1)
            unit = labelConfig.get('unit', None)
            fontFamily = self.config_['font']['family']
            fontsize = labelConfig.get('fontsize', self.config_['font']['size']['legend'])
            if unit:
                keyText = f'{labelScale} {unit}'
            else:
                keyText = f'{labelScale}'
            self.currentAx_.quiverkey(
                quiver, labelX, labelY, labelScale, keyText, labelcolor=color,
                labelpos='W', fontproperties=dict(family=fontFamily, size=fontsize)
            )
        return quiver
        
    def boxplot(self, data,
                zorder: int = None, 
                label: str = None, format: str = None, 
                xTicksRotation: float = None, **kwargs):
        zorder = self.config_['line']['zorder']['polygon'] if zorder is None else zorder
        kwargs.update(zorder=zorder)
        
        if label is not None:
            if format is not None:
                format = '{:' + format + '}'
                label = format.format(label)
            kwargs.update(label=label)

        self.currentAx_.boxplot(data, **kwargs)
        if xTicksRotation:
            for l in self.currentAx_.get_xticklabels():
                l.set_rotation(xTicksRotation)

    def bar(self, data, offset: float = 0, width: float = 0.25,
            xlabel: np.ndarray | list = None,
            yLimit: float | str = None,
            transform: bool = False,
            zorder: int = None, alpha: float = None,
            showValue: bool = False,
            valueConfig: dict = {},
            addAxes = None,
            *args, **kwargs):

        x = np.arange(1, len(data) + 1) * width
        zorder = self.config_['line']['zorder']['polygon'] if zorder is None else zorder
        alpha = self.config_['line']['alpha']['polygon'] if alpha is None else alpha
        kwargs.update(zorder=zorder, alpha=alpha)

        if transform:
            kwargs.update(transform=self.currentAx_.transAxes)
        
        label = kwargs.get('label')
        if 'format' in kwargs:
            fmt = kwargs.pop('format')
            fmt = '{:' + fmt + '}'
            label = fmt.format(label) if label is not None else None
            kwargs.update(label=label)
        
        if yLimit == 'auto':
            yLimit = (np.nanmin(data), np.nanmax(data) + 1)
        
        axHandle = None
        if addAxes:
            axHandle = plt.axes(addAxes)
        else:
            axHandle = self.currentAx_
        axHandle.set_ylim(yLimit)
        if xlabel is not None:
            axHandle.set_xticks(x+offset, xlabel)
        ret = axHandle.bar(x+offset, data, width=width, *args, **kwargs)

        if showValue:
            fmt = valueConfig.get('format', 'd')
            color = valueConfig.get('color', self.config_['font']['color']['tick'])
            fontsize = valueConfig.get('fontsize', self.config_['font']['size']['annot'])
            xLoc = valueConfig.get('xloc', 0)
            yLoc = valueConfig.get('yloc', 0.25)
            verticalAlignment = valueConfig.get('verticalalignment', 'center')
            showNonZero = valueConfig.get('showNonZero', False)
            fmt = '{:' + fmt + '}'
            value = valueConfig.get('value', data)
            for ix, iy, iv in zip(x, data, value):
                if showNonZero and iy == 0:
                    continue
                else:
                    if iy >= 0:
                        axHandle.text(ix+offset + xLoc, iy + yLoc, fmt.format(iv), 
                                    horizontalalignment='center', 
                                    verticalalignment=verticalAlignment,
                                    color=color,
                                    zorder=zorder+1,
                                    fontsize=fontsize)
                    else:
                        axHandle.text(ix+offset + xLoc, iy - yLoc, fmt.format(iv), 
                                    horizontalalignment='center', color=color,
                                    zorder=zorder+1,
                                    fontsize=fontsize)
        return ret
    
    def withinMapRange(self, x, y):
        mapRange = self.currentAxInfo_.get('map').get('mapRange')
        yMin, yMax, xMin, xMax = mapRange
        xMin = (xMin + 360) % 360
        xMax = (xMax + 360) % 360
        x = (x + 360) % 360
        flag = True
        if x < xMin or x > xMax or y < yMin or y > yMax:
            flag = False
        return flag

    def text(self, x, y, s, transform: bool = False, **kwargs):
        flagWithin = True
        if 'map' in self.currentAxInfo_.keys() and transform:
            # flagWithin = self.withinMapRange(x, y)
            kwargs.update(transform=ccrs.PlateCarree())
        # elif transform:
        #     kwargs.update(transform=self.currentAx_.transAxes)
        
        zorder = kwargs.get('zorder', None)
        zorder = self.config_['font']['zorder']['text'] if zorder is None else zorder
        kwargs.update(zorder=zorder)
        if flagWithin:
            return self.currentAx_.text(x, y, s, **kwargs)
    
    def legend(self, *args, axis: Optional[str] = 'current', 
               unique: Optional[bool] = True, sort: Optional[bool] = False,
               superLegend: Optional[bool] = False, **kwargs):
        if axis == 'all':
            handles = []
            labels = []
            if self.currentAxIdx_ is not None:
                self.axes_[self.currentAxIdx_] = self.currentAx_
            for idx, ax in self.axes_.items():
                axHandle, axLabel = ax.get_legend_handles_labels()
                handles.extend(axHandle)
                labels.extend(axLabel)
            if unique:
                uniqueHandles = []
                uniqueLabels = []
                for i, (h, l) in enumerate(zip(handles, labels)):
                    if l not in labels[:i]:
                        uniqueHandles.append(h)
                        uniqueLabels.append(l)
                handles = uniqueHandles
                labels = uniqueLabels
                if sort:
                    sortIdx = np.argsort(uniqueLabels)
                    uniqueHandles = [uniqueHandles[i] for i in sortIdx]
                    uniqueLabels = [uniqueLabels[i] for i in sortIdx]

            ret = self.figure_.legend(handles, labels, *args, **kwargs)
        else:
            handles, labels = self.currentAx_.get_legend_handles_labels()
            if unique:
                uniqueHandles = []
                uniqueLabels = []
                for i, (h, l) in enumerate(zip(handles, labels)):
                    if l not in labels[:i]:
                        uniqueHandles.append(h)
                        uniqueLabels.append(l)
                if sort:
                    sortIdx = np.argsort(uniqueLabels).tolist()
                    uniqueHandles = [uniqueHandles[i] for i in sortIdx]
                    uniqueLabels = [uniqueLabels[i] for i in sortIdx]
                handles = uniqueHandles
                labels = uniqueLabels

            if superLegend:
                ret = self.figure_.legend(handles, labels, *args, **kwargs)
            else:
                ret = self.currentAx_.legend(handles, labels, *args, **kwargs)
        return ret

    def title(self, 
            title: Optional[str] = '', signature: Optional[str] = None,
            fontsize: Optional[int] = None,
            fontFamily: Optional[str] = None,
            location: Optional[Union[list, tuple]] = [0, 1.035],
            color: Optional[str] = None, 
            horizontalAlignment: Optional[str] = 'left',
            verticalAlignment: Optional[str] = 'bottom',
            superTitle: Optional[bool] = False,
            **kwargs):
        if signature is not None:
            title = title + '@' + signature
        fontsize = self.config_['font']['size']['title'] if fontsize is None else fontsize
        fontFamily = self.config_['font']['family'] if fontFamily is None else fontFamily
        color = self.config_['font']['color']['title'] if color is None else color

        if superTitle and len(self.axes_.keys()) > 1:
            for idx in self.axes_.keys():
                shape, loc, rowspan, colspan = idx
                if loc == (0, 0):
                    transform = self.axes_[idx].transAxes
        else:
            transform = self.currentAx_.transAxes
        
        ret = self.figure_.text(
            *location, title, transform=transform, color=color,
            fontsize=fontsize, family=fontFamily, 
            horizontalalignment=horizontalAlignment,
            verticalalignment=verticalAlignment)
        return ret
    
    def timestamp(self, 
            init: Union[str, datetime.datetime],
            fcst: Union[int, str],
            duration: Optional[int] = None,
            end: Optional[datetime.datetime] = None,
            location: Optional[Union[list, tuple]] = [0, 1.005],
            fontsize: Optional[int] = None,
            fontFamily: Optional[str] = None,
            color: Optional[str] = None,
            zorder: Optional[int] = None,
            horizontalAlignment: Optional[str] = 'left',
            verticalAlignment: Optional[str] = 'bottom',
            superTitle: Optional[bool] = False,
            **kwargs):
        fontsize = self.config_['font']['size']['timestamp'] if fontsize is None else fontsize
        fontFamily = self.config_['font']['family'] if fontFamily is None else fontFamily
        if isinstance(init, str):
            init = datetime.datetime.strptime('%Y%m%d_%H')
        if duration is not None:
            if duration > 0:
                fcst = fcst, fcst + duration
            else:
                fcst = fcst + duration, fcst
        try:
            if isinstance(fcst, str):
                fcstTry = fcst.strip('[()]').split(',')
                if len(fcstTry) == 2:
                    for i in range(2):
                        fcstTry[i] = int(fcstTry[i])
                    fcst = tuple(fcstTry)
        except:
            pass

        fmt = kwargs.pop('format', None)
        f = ''
        if fmt is not None:
            try:
                f = '{:' + fmt + '}'
                initStr = f.format(init)
            except:
                f = '{:%Y/%m/%d %HZ}'
                initStr = f.format(init)
        else:
            f = '{:%Y/%m/%d %HZ}'
            initStr = f.format(init)
        
        if isinstance(fcst, int):
            validTime = init + datetime.timedelta(hours=fcst)
            validTimeStr = f.format(validTime)
            subtitle = f'{initStr} [+{fcst:d}h] valid at {validTimeStr}'
        elif isinstance(fcst, str):
            if fcst == 'an':
                subtitle = f'{initStr} [analysis]'
            elif fcst == 'bg':
                subtitle = f'{initStr} [background]'
            elif fcst == 'period' and end is not None:
                if fmt is not None:
                    try:
                        f = '{:' + fmt + '}'
                        endStr = f.format(end)
                    except:
                        f = '{:%Y/%m/%d %HZ}'
                        endStr = f.format(end)
                else:
                    f = '{:%Y/%m/%d %HZ}'
                    endStr = f.format(end)
                subtitle = f'From {initStr} to {endStr}'
            else:
                subtitle = f'{initStr} {fcst}'
        else:
            fcst = tuple(fcst)
            fromHour, toHour = fcst
            begin = init + datetime.timedelta(hours=fromHour)
            end = init + datetime.timedelta(hours=toHour)
            beginStr = f.format(begin)
            endStr = f.format(end)
            subtitle = (f'{initStr} [+{fromHour:d}~{toHour}h] valid from '
                        f'{beginStr} to {endStr}')
        color = self.config_['font']['color']['timestamp'] if color is None else color
        zorder = self.config_['font']['zorder']['timestamp'] if zorder is None else zorder
        
        if superTitle and len(self.axes_.keys()) > 1:
            for idx in self.axes_.keys():
                shape, loc, rowspan, colspan = idx
                if loc == (0, 0):
                    transform = self.axes_[idx].transAxes
        else:
            transform = self.currentAx_.transAxes
        
        ret = self.figure_.text(
            *location, subtitle, transform=transform, color=color,
            fontsize=fontsize, family=fontFamily, 
            horizontalalignment= horizontalAlignment,
            verticalalignment=verticalAlignment)
        return ret
    
    def panelNumber(self, 
            number: int = None, style: str = 'a',
            fontsize: Optional[int] = None,
            fontFamily: Optional[str] = None,
            location: Optional[Union[list, tuple]] = [0.5, -0.05],
            color: Optional[str] = None, 
            horizontalAlignment: Optional[str] = 'center',
            verticalAlignment: Optional[str] = 'top',
            **kwargs):
        if style == 'a':
            panelNumber = f'({chr(97+number)})'
        elif style == 'A':
            panelNumber = f'({chr(65+number)})'
        else:
            panelNumber = f'({number})'
        fontsize = self.config_['font']['size']['title'] if fontsize is None else fontsize
        fontFamily = self.config_['font']['family'] if fontFamily is None else fontFamily
        color = self.config_['font']['color']['title'] if color is None else color

        transform = self.currentAx_.transAxes
        
        ret = self.figure_.text(
            *location, panelNumber, transform=transform, color=color,
            fontsize=fontsize, family=fontFamily, 
            horizontalalignment=horizontalAlignment,
            verticalalignment=verticalAlignment)
        return ret
    
    def save(self, path: Union[str, Sequence], **kwargs):
        full = self.config_['figure']['full']
        dpi = self.config_['figure']['dpi']
        if isinstance(path, str):
            path = [path]
        if self.currentAxIdx_ is not None:
            self.axes_[self.currentAxIdx_] = self.currentAx_
        if len(self.axes_) > 0:
            self.figure_.tight_layout()
        
        for i in path:
            if full:
                self.figure_.savefig(
                    i, dpi=dpi, bbox_inches='tight', edgecolor='none',
                    pad_inches=0.0, **kwargs)
            else:
                self.figure_.savefig(
                    i, dpi=dpi, bbox_inches='tight', edgecolor='none',
                    pad_inches=0.04, **kwargs)
    
    def clear(self):
        plt.close('all')