import logging
import os, sys
import time

import numpy as np
import xarray as xr

from Base import npDatetimeToDatetime
from Preprocess.Model import destagger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

rttovRoot = os.environ.get('RTTOV_ROOT', None)
hasRttov = False
if rttovRoot is not None:
    sys.path.insert(0, os.path.join(os.path.abspath(rttovRoot), 'wrapper'))
    try:
        import pyrttov
        from pyrttov.rttype import gasUnitType
        from pyrttov.option import THERMAL_SOLVER_CHOU, CLOUD_OVERLAP_MAX_RANDOM
        hasRttov = True
    except:
        logger.error('Failed to import pyrttov')

constStefanBoltzmann = 5.67 * 1.0e-8
coefA = 1.228
coefB = -1.106 * 1.0e-3

def olr2BrightnessTemperature(olr: np.ndarray):
    fluxT = np.float_power(olr / constStefanBoltzmann, 0.25)
    brightnessT = (-coefA + np.sqrt(np.square(coefA) + 4.0 * coefB * fluxT)) / (
        2.0 * coefB)
    return brightnessT
    
def grid2Profile(data: xr.DataArray):
    dims = data.dims
    if 'Time' in dims:
        data = data.stack(profiles=('Time', 'y', 'x'))
    else:
        data = data.stack(profiles=('y', 'x'))
    data = data.transpose('profiles', ...)
    if 'z' in dims:
        data = data.isel(z=slice(None, None, -1))
    return data

def profile2Grid(data: xr.DataArray):
    data = data.unstack('profiles')
    return data

def expand2Nprofiles(data: np.ndarray, nprofiles: int):
    return np.ones((nprofiles, 1)) * data

def setupRttovIR(
        platform: str = 'goes_16', instrument: str = 'abi', 
        irChannel: list = [13, 14], variableGas: str = 'o3co2',
        nproc: int = 1):
    config = dict()
    rttovObject = pyrttov.Rttov()

    instrumentFullName = f'{platform}_{instrument}'
    coefFilename = f'{rttovRoot}/rtcoef_rttov14/rttov13pred54L/rtcoef_{instrumentFullName}_{variableGas}.dat'
    hydrotableFilename = f'{rttovRoot}/rtcoef_rttov14/hydrotable_visir/rttov_hydrotable_{instrumentFullName}.dat'
    
    rttovObject.FileCoef = coefFilename
    # Solar radiation only impact channels with wavelengths below 5µm
    rttovObject.Options.Solar = True
    rttovObject.Options.VerboseWrapper = False
    rttovObject.Options.Hydrometeors = True
    rttovObject.Options.Nthreads = nproc

    rttovObject.FileHydrotable = hydrotableFilename

    try:
        rttovObject.loadInst(irChannel)
    except pyrttov.RttovError as e:
        logger.error(f'Failed to load instrument(s): {e}')
        raise Exception('Error in RTTOV simulation.')
    
    irAtlas = pyrttov.Atlas()
    irAtlas.AtlasPath = f'{rttovRoot}/emis_data'

    config.update(
        rttov = rttovObject,
        irAtlas = irAtlas,
    )
    return config

# Refer 
def rttovBrightnessTemperature(
        lon: xr.DataArray,
        lat: xr.DataArray,
        hgt: xr.DataArray,
        landMask: xr.DataArray,
        t: xr.DataArray, 
        p: xr.DataArray,
        qVapor: xr.DataArray,
        qIce: xr.DataArray,
        qSnow: xr.DataArray,
        qCloud: xr.DataArray,
        cloudFrac: xr.DataArray,
        tSkin: xr.DataArray,
        dataSnowCover: xr.DataArray,
        t2: xr.DataArray,
        q2: xr.DataArray,
        u10: xr.DataArray,
        v10: xr.DataArray,
        albedo: xr.DataArray,
        emissivity: xr.DataArray,
        kappa: float = 1.0,
        platform: str = 'goes_16', instrument: str = 'abi', 
        irChannel: int | list = [13, 14], variableGas: str = 'o3co2',
        nproc: int = 2,
        w: xr.DataArray = None
    ):

    if not isinstance(irChannel, list):
        irChannel = [irChannel]

    nChannel = len(irChannel)
    config = setupRttovIR(platform, instrument, irChannel, variableGas, nproc=nproc)

    # Convert pressure from Pa to hPa
    pInHPa = p / 100

    qIceTotal = qIce + kappa * qSnow
    qIceTotal = qIceTotal.where(qIceTotal > 0.0, 0.0)
    qCloud = qCloud.where(qCloud > 0.0, 0.0)

    hgtInKm = hgt / 1000.0

    if w is not None:
        maxW = w.max(dim='z')

    lon = grid2Profile(lon)
    lat = grid2Profile(lat)
    hgtInKm = grid2Profile(hgtInKm)
    # 1 for land, 0 for water
    landMask = grid2Profile(landMask)
    pInHPa = grid2Profile(pInHPa)
    t = grid2Profile(t)
    qVapor = grid2Profile(qVapor)
    qIceTotal = grid2Profile(qIceTotal)
    qCloud = grid2Profile(qCloud)
    cloudFrac = grid2Profile(cloudFrac)
    if w is not None:
        maxW = grid2Profile(maxW)
        # Cloud classification follows NCAR/DART's way
        isCumulus = maxW > 0.5
    else:
        isCumulus = xr.ones_like(landMask, dtype=bool)

    nlevels = t.sizes.get('z')
    nProfiles = t.sizes.get('profiles')

    isCumulus = isCumulus.expand_dims(dim={'z': nlevels}).transpose('profiles', ...).to_numpy()
    isLand = landMask.expand_dims(dim={'z': nlevels}).transpose('profiles', ...).to_numpy()
    
    profileTime = t['XTIME'].to_numpy()
    profileTime = [npDatetimeToDatetime(i) for i in profileTime]
    profileTime = np.array(
        [np.array([i.year, i.month, i.day, i.hour, i.minute, i.second], 
                  dtype=np.int32) for i in profileTime], dtype=np.int32)

    tSkin = grid2Profile(tSkin)
    dataSnowCover = grid2Profile(dataSnowCover)
    t2 = grid2Profile(t2)
    q2 = grid2Profile(q2)
    u10 = grid2Profile(u10)
    v10 = grid2Profile(v10)
    albedo = grid2Profile(albedo)
    emissivity = grid2Profile(emissivity)

    qVapor = qVapor.astype(np.float64).where(qVapor > 1.0e-11, 1.0e-11)
    q2 = q2.astype(np.float64).where(q2> 1.0e-11, 1.0e-11)

    profiles = pyrttov.Profiles(nProfiles, nlevels+1, 1)
    profiles.GasUnits = gasUnitType('kg_per_kg')

    pHalf = destagger(pInHPa, staggerDim=1)
    pHalfTop = tSkin.copy(data = 2 * pInHPa.isel(z=0) - pHalf.isel(z=0))
    pHalfBottom = tSkin.copy(data = 2 * pInHPa.isel(z=-1) - pHalf.isel(z=-1))
    pHalf = xr.concat([pHalfTop, pHalf, pHalfBottom], 'z', coords='minimal')
    profiles.PHalf = pHalf.to_numpy()
    profiles.T = t.to_numpy()
    profiles.Q = qVapor.to_numpy()

    profiles.HydroFrac = cloudFrac.to_numpy()

    # Depending on the vertical velocity (>0.5 m/s or not) and land type
    zeros = 0.0 * qCloud
    profiles.Stco = xr.where((~isCumulus) & (isLand==1), qCloud.values, zeros.values)
    profiles.Stma = xr.where((~isCumulus) & (isLand==0), qCloud.values, zeros.values)
    profiles.Cucc = xr.where(isCumulus & (isLand==1), qCloud.values, zeros.values)
    profiles.Cuma = xr.where(isCumulus & (isLand==0), qCloud.values, zeros.values)
    profiles.Baum = qIceTotal.to_numpy()

    # zenangle, sunzenangle: used by BRDF, IR atlas if angular correction applied
    # azangle, sunazangle: used by BRDF
    angles = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    profiles.Angles = expand2Nprofiles(angles, nProfiles)

    windFetch = 1.0e5 * np.ones(nProfiles, dtype=np.float64)
    nearSurface = np.stack(
        [[t2.to_numpy(), q2.to_numpy(), u10.to_numpy(), v10.to_numpy(), windFetch]], axis=1).astype(np.float64).T
    profiles.NearSurface = nearSurface

    # skin T, salinity, snow_frac, foam_frac, fastem_coefsx5
    skin = np.array([270.0, 35.0, 0.0, 0.0, 3.0, 5.0, 15.0, 0.1, 0.3], dtype=np.float64)
    skin = expand2Nprofiles(skin, nProfiles)
    skin[:,0] = tSkin
    skin = np.expand_dims(skin, axis=1)
    profiles.Skin = skin
 
    surfaceType = np.array([0, 0], dtype=np.int32)
    surfaceType = expand2Nprofiles(surfaceType, nProfiles)
    surfaceType[:,0][landMask == 1] = 0
    surfaceType[:,0][landMask == 0] = 1
    surfaceType = np.expand_dims(surfaceType, axis=1)
    profiles.SurfType = surfaceType

    surfaceGeom = np.stack(
        [lat.to_numpy(), lon.to_numpy(), hgtInKm.to_numpy()], axis=1, dtype=np.float64)
    profiles.SurfGeom = surfaceGeom

    profiles.DateTimes = profileTime

    rttov = config['rttov']
    rttov.Profiles = profiles
    irAtlas = config['irAtlas']
    irAtlas.loadIrEmisAtlas(profileTime[0][1])

    surfaceEmisRefl = np.zeros((5, nProfiles, 1, len(irChannel)), dtype=np.float64)
    surfaceEmisRefl[0,:,:,:] = -1 # Emissivity
    surfaceEmisRefl[1,:,:,:] = -1 # Reflectance: albedo/np.pi
    surfaceEmisRefl[2,:,:,:] = 0  # Diffuse reflectance
    surfaceEmisRefl[3,:,:,:] = 0  # Specularity
    surfaceEmisRefl[4,:,:,:] = -1 # Effective tSurface
    # surfaceEmisRefl[:,:,:,:2] = irAtlas.getEmisBrdf(rttov)
    rttov.SurfEmisRefl = surfaceEmisRefl

    tt = time.time()
    rttov.runDirect()
    tt = time.time() - tt
    print(f'Take {tt} s')

    da = None
    for i, chn in enumerate(irChannel):
        data = tSkin.copy(data=rttov.BtRefl[:, i])
        data = profile2Grid(data)
        if da is None:
            da = data
        else:
            da = xr.concat([da, data], 'z', coords='minimal')
    
    zeros.close()
    data.close()
    lon.close()
    lat.close()
    hgt.close()
    landMask.close()
    t.close()
    qVapor.close()
    qIce.close()
    qSnow.close()
    qCloud.close()
    cloudFrac.close()
    tSkin.close()
    dataSnowCover.close()
    t2.close()
    q2.close()
    u10.close()
    v10.close()
    albedo.close()
    emissivity.close()
    if w is not None:
        maxW.close()
        w.close()
    return da

