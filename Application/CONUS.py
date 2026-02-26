import argparse
import datetime
import os
import sys

import xarray as xr

sys.path.insert(0, os.path.abspath('.'))

from Base import getInnerBoxMask
from Utilities import downloadRdaCONUS404
from Preprocess.Model import PreprocessWRF, GaussianSmoother, Watershed
from TempestExtremes import (
    CONUS404InputName, TempestExtremes, SyCLoPSClassifier, MCSClassifier, 
    BlobStatsParallelization)
from System import defineScratch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('USTORM (Unified Storm Tracking for Observations and '
        'multi-Resolution Models) low pressure system tracking and classification '
        'workflow script for CONUS404.'))
    parser.add_argument(
        '--begin', action='store', metavar='%YYYY%mm%dd_%H',
        required=True,
        help='The beginning of the calculation period.'
    )
    parser.add_argument(
        '--end', action='store', metavar='%YYYY%mm%dd_%H',
        required=True,
        help='The end of the calculation period.'
    )
    parser.add_argument(
        '--step', action='store', metavar='Hour',
        type=int, default=3,
        help='Time step of the data.'
    )
    parser.add_argument(
        '--te_path', '-te', action='store', metavar='TempestExtremesPath',
        default='',
        help='Path to tempestextremes bin directory.'
    )
    parser.add_argument(
        '--mpirun', '-mpi', action='store', metavar='MPIPath',
        default=None,
        help='Path to MPI launch program (e.g., mpirun or mpiexec).'
    )
    parser.add_argument(
        '--mpi_arg', action='store', metavar='MPIArg',
        default=None,
        help=('Arguments of the MPI launch program. '
              'For example, use \"-np N\" to run tempestextremes with N cores')
    )
    parser.add_argument(
        '--work_dir', action='store', metavar='WorkDir',
        default=None,
        help=('The work directory used to store results. '
              'Default: directory TempestExtremes under your scratch '
              'or home directory')
    )
    parser.add_argument(
        '--data_root', '--data_dir', action='store', metavar='DataRoot',
        default=None,
        help=('The root directory that is used to download and store data. '
              'Default: WorkDir/Data.')
    )
    parser.add_argument(
        '--nproc', action='store', metavar='N',
        type=int, default=8,
        help='Number of processes.'
    )
    parser.add_argument(
        '--assign_blob_index', action='store', metavar='AssignBlobIndex',
        type=bool, default=False,
        help='Assign indices for LPS blobs in masked files.'
    )
    parser.add_argument(
        '--full_result', '--full', action='store', metavar='StoreFullResult',
        type=bool, default=False,
        help='Whether to save verbose info.'
    )
    parser.add_argument(
        '--result_format', '--format', action='store', metavar='Format',
        default='parquet',
        help=('The format of result to be stored. Supported: '
              'parquet (default), xlsx, all (i.e., parquet & xlsx)')
    )
    parserArgs = parser.parse_args()
    tempestExtremesPath = vars(parserArgs).get('te_path')
    mpirunPath = vars(parserArgs).get('mpirun')
    mpiArgStr = vars(parserArgs).get('mpi_arg')
    mpiArg = mpiArgStr.replace('\"', '').replace('\'', '').split(' ') if (
        mpiArgStr is not None) else []
    
    scratchDir = defineScratch()
    workDir = vars(parserArgs).get('work_dir')
    workDir = os.path.join(scratchDir, 'TempestExtremes') if (
        workDir is None) else workDir
    dataRoot = vars(parserArgs).get('data_root')
    dataRoot = os.path.join(workDir, 'Data') if (dataRoot is None) else dataRoot
    beginTimeStr = vars(parserArgs).get('begin')
    endTimeStr = vars(parserArgs).get('end')
    assignBlobIndex = vars(parserArgs).get('assign_blob_index')
    fullResult = vars(parserArgs).get('full_result')
    reusltFormat = vars(parserArgs).get('result_format')
    step = vars(parserArgs).get('step')
    nproc = vars(parserArgs).get('nproc')

    # Download CONUS404 from RDA
    downloadRdaCONUS404(dataRoot, beginTimeStr, endTimeStr, step=step, override=False)

    preprocessDir = f'{workDir}/preprocess'

    tempestExtremes = TempestExtremes(tempestExtremesPath, mpirunPath, mpiArg)

    beginTime = datetime.datetime.strptime(beginTimeStr, "%Y%m%d_%H")
    endTime = datetime.datetime.strptime(endTimeStr, "%Y%m%d_%H")
    stitchNodesOutput = os.path.join(workDir, 'StitchNodesResult.txt')
    lpsMaskBlobStatsOutput = os.path.join(workDir, 'LPSMaskBlobStatsOutput.txt')
    blobStatsOutput = os.path.join(workDir, 'BlobStatsOutput.txt')
    frontalBlobStatsOutput = os.path.join(workDir, 'FrontalBlobStatsOutput.txt')
    classifierPreprocessFile = os.path.join(workDir, f'Preprocess_{beginTime:%Y%m%d%H}_{endTime:%Y%m%d%H}.parquet')
    frontClassifierPreprocessFile = os.path.join(workDir, f'FrontPreprocess_{beginTime:%Y%m%d%H}_{endTime:%Y%m%d%H}.parquet')
    classifierFinalFile = os.path.join(workDir, f'Result_{beginTime:%Y%m%d%H}_{endTime:%Y%m%d%H}')
    inputName = CONUS404InputName(root=dataRoot)
    inputName.setDateTime(begin=beginTime, end=endTime)
    inputName.generateInput(types=['wrf3d'])
    fileList = inputName.getInputAsList()
    inputName.generateInput(types=['wrf2d'])
    wrf2dList = inputName.getInputAsList()
    # Pre-process CONUS 404
    originalInvariantPath = inputName.getInvariantPaths().get('invariant')
    preprocessor = PreprocessWRF(fileList, wrf2dFiles=wrf2dList)
    preprocessor.setInvariantPath(originalInvariantPath)
    varRaname = dict(
        lev = 'ZNU',
        ilev = 'ZNW',
    )
    varUnitChange = dict(
        Z = 'm'
    )
    varProcess = dict(
        Z = dict(
            zCoord = 'P',
            var = 'Z',
            level = [92500, 85000, 70000, 50000, 30000],
            suffix = [925, 850, 700, 500, 300],
            method = 'linear',
        ),
        Vo = dict(
            derivative = 'vo',
            zCoord = 'P',
            var = ['U', 'V'],
            level = [50000],
            suffix = [500],
            mapFacX = 'MAPFAC_MX',
            mapFacY = 'MAPFAC_MY',
            method = 'linear',
        ),
        SLP = dict(
            derivative = 'slp',
            zCoord = 'P',
            var = ['Z', 'TK', 'P', 'QVAPOR'],
        ),
        U = dict(
            zCoord = 'P',
            var = 'U',
            level = [85000, 70000, 20000],
            suffix = [850, 700, 200],
            method = 'linear',
        ),
        V = dict(
            zCoord = 'P',
            var = 'V',
            level = [85000, 20000],
            suffix = [850, 200],
            method = 'linear',
        ),
        ULev = dict(
            var = 'U',
            level = [11],
            suffix = [12],
            method = 'linear',
        ),
        VLev = dict(
            var = 'V',
            level = [11],
            suffix = [12],
            method = 'linear',
        ),
        T = dict(
            zCoord = 'P',
            var = 'TK',
            level = [85000],
            suffix = [850],
            method = 'linear',
        ),
        RH = dict(
            derivative = 'relative_humidity_from_mixing_ratio',
            zCoord = 'P',
            var = ['P', 'TK', 'QVAPOR'],
            level = [85000, 70000, 10000],
            suffix = [850, 700, 100],
            method = 'linear',
        ),
        RHLev = dict(
            derivative = 'relative_humidity_from_mixing_ratio',
            var = ['P', 'TK', 'QVAPOR'],
            level = [42],
            suffix = [43], # ~100hPa
            method = 'linear',
        ),
        ThetaLev = dict(
            derivative = 'potential_temperature',
            level = [11],
            suffix = [12],
            var = ['TK', 'P'],
            method = 'linear',
        ),
    )
    preprocessedList = preprocessor.process(varProcess, preprocessDir, requireMapFactor=False, 
                         varRename=varRaname, varUnitChange=varUnitChange,
                         nproc=nproc, override=False)

    # Detect low pressure systems
    inputFileName = os.path.join(workDir, 'DetectNodesInputFilenames.txt')
    outputFileName = os.path.join(workDir, 'DetectNodesOutputFilenames.txt')
    logRoot = os.path.join(workDir, 'log')
    if not os.path.exists(logRoot):
        os.makedirs(logRoot)
    invariantBasename = os.path.basename(originalInvariantPath)
    tmpInvariantPath = os.path.join(workDir, invariantBasename)
    if not os.path.exists(tmpInvariantPath):
        invariantDataset = xr.open_dataset(originalInvariantPath)
        invariantDataset = invariantDataset[dict(Time=0)]
        invariantDataset = invariantDataset.drop_vars('Time')
        if 'HGT' in invariantDataset.variables.keys():
            invariantDataset = invariantDataset.rename_vars({
                                                            'HGT': 'Z0'})
        innerBoxMask = getInnerBoxMask(
            invariantDataset['XLONG'].to_numpy(), 
            invariantDataset['XLAT'].to_numpy(), offset=2.0)
        invariantDataset['InnerBoxMask'] = invariantDataset['LANDMASK'].copy(
            data=innerBoxMask)
        invariantDataset.to_netcdf(tmpInvariantPath)

    with open(inputFileName, 'w') as f:
        for i,j in zip(preprocessedList, wrf2dList):
            f.write(f'{i};{j};{tmpInvariantPath}\n')
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'DetectNodesOutput_{currentTime:%Y%m%d_%H%M%S}.txt')
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=step)

    logDir = os.path.join(logRoot, 'DetectNodes')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    detectNodesArg = [
        '--in_data_list', inputFileName,
        '--out_file_list', outputFileName,
        # Search minimum mean sea level pressure
        '--searchbymin', 'SLP',
        # Mean sea level pressure must increase 10 Pa within a 5.5 GCD
        '--closedcontourcmd', 'SLP,10,5.5,0',
        # Merge candidates within a 6.0 GCD, with the lower mean sea level pressure 
        # node taking precedence.
        '--mergedist', '6.0',
        '--thresholdcmd', 'InnerBoxMask,=,1,0',
        '--outputcmd',
        (
            # Mean sea level pressure (MSLP)
            'SLP,min,0;'
            # Maximum model bottem wind speed within 2.0 GCD (WS10)
            '_VECMAG(U10,V10),max,2.0;'
            # Greatest positive closed contour delta of MSLP over a 2.0 GCD (MSLPCC20)
            'SLP,posclosedcontour,2.0,0;'
            # Greatest positive closed contour delta of MSLP over a 5.5 GCD (MSLPCC55)
            'SLP,posclosedcontour,5.5,0;'
            # Average environmental deep-layer (200 - 850 hPa) wind shear 
            # over a 10.0 GCD (DeepShear)
            '_DIFF(_VECMAG(U200,V200),_VECMAG(U850,V850)),avg,10.0;'
            # Greatest decline of the upper-level (300 - 500 hPa) geopotential 
            # thickness within a 6.5 GCD of the maximum thickness node within 
            # a 1.0 GCD of the current node (UppThkCC)
            '_DIFF(Z300,Z500),negclosedcontour,6.5,1.0;'
            # Greatest decline of the mid-level (500 - 700 hPa) geopotential 
            # thickness within a 3.5 GCD of the maximum thickness node within 
            # a 1.0 GCD of the current node (MidThkCC)
            '_DIFF(Z500,Z700),negclosedcontour,3.5,1.0;'
            # Greatest decline of the lower-level (700 - 925 hPa) geopotential 
            # thickness within a 3.5 GCD of the maximum thickness node within 
            # a 1.0 GCD of the current node (LowerThkCC)
            '_DIFF(Z700,Z925),negclosedcontour,3.5,1.0;'
            # Greatest increase of the 500 hPa geopotential within a 3.5 GCD 
            # of the minimum geopotential node within a 1.0 GCD of the current node 
            # (Z500CC)
            'Z500,posclosedcontour,3.5,1.0;'
            # Avarage relative vorticity over a 2.5 GCD (Vo500Avg)
            'Vo500,avg,2.5;'
            # Maximum 100 hPa relative humidity within a 2.5 GCD (RH100Max)
            'RH100,max,2.5;'
            # Avarage 850 hPa relative humidity within a 2.5 GCD (RH850Avg)
            'RH850,avg,2.5;'
            # Avarage 700 hPa relative humidity within a 2.5 GCD (RH700Avg)
            'RH700,avg,2.5;'
            # 850 hPa air temperature at the node (T850)
            'T850,max,0.0;'
            # 850 hPa geopotential at the node (Z850)
            'Z850,min,0.0;'
            # 700 hPa geopotential at the node (Z700)
            'Z700,min,0.0;'
            # Surface geopotential at the node (Z0)
            'Z0,min,0;'
            # Difference between the weighted area mean of positive and negative 
            # values of 850 hPa eastward wind over a 5.5 GCD (U850Diff)
            'U850,posminusnegwtarea,5.5;'
            # Maximun poleward 200 hPa wind speed within a 1.0 GCD (WS200PMax)
            '_VECMAG(U200,V200),maxpoleward,1.0'
        ),
        '--timefilter', '3hr',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--regional',
        '--logdir', logDir
    ]
    tempestExtremes.run('DetectNodes', detectNodesArg)

    inputFileName = outputFileName
    outputFileName = stitchNodesOutput
    stitchNodesArg = [
        '--in_list', inputFileName,
        '--out', outputFileName,
        '--in_fmt',
        (
            'lon,lat,MSLP,WS,MSLPCC20,MSLPCC55,DeepShear,UppThkCC,MidThkCC,LowThkCC,'
            'Z500CC,Vo500Avg,RH100Max,RH850Avg,RH700Avg,T850,Z850,Z700,Z0,U850Diff,'
            'WS200PMax'
        ),
        '--range', '4.0',
        '--mintime', '18h',
        '--maxgap', '12h',
        '--threshold', 'MSLPCC55,>=,100.0,5'
    ]
    tempestExtremes.run('StitchNodes', stitchNodesArg)

    # Calculate smoothed relative vorticity using 850 hPa wind field.
    inputFileName = os.path.join(workDir, 'SmoothedVo850InputFilenames.txt')
    outputFileName = os.path.join(workDir, 'SmoothedVo850_S0.txt')
    with open(inputFileName, 'w') as f:
        f.writelines(line + '\n' for line in preprocessedList)
    logDir = os.path.join(logRoot, 'SmoothedVo850_S0')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    windFieldFileList = preprocessedList
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'SmoothedVo850_{currentTime:%Y%m%d_%H%M%S}_S0.nc')
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=step)
    variableProcessorArg = [
        '--in_data_list', inputFileName,
        '--out_data_list',  outputFileName,
        '--var', '_CURL{8,3}(ULev12,VLev12)',
        '--varout', 'Vorticity',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--regional',
        '--timefilter', '3hr',
        '--logdir', logDir
    ]
    # tempestExtremes.run('VariableProcessor', variableProcessorArg)

    # Flip the sign of vorticity in the Southern Hemisphere
    inputFileName = outputFileName
    outputFileName = os.path.join(workDir, 'SmoothedVo850_S1.txt')
    logDir = os.path.join(logRoot, 'SmoothedVo850_S1')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'SmoothedVo850_{currentTime:%Y%m%d_%H%M%S}_S1.nc')
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=step)
    variableProcessorArg = [
        '--in_data_list', inputFileName,
        '--out_data_list',  outputFileName,
        '--var', '_COND(_LAT(),Vorticity,_PROD(Vorticity,-1))',
        '--varout', 'CyclonicVorticity',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--regional',
        '--logdir', logDir
    ]
    # tempestExtremes.run('VariableProcessor', variableProcessorArg)
    
    # Search for cyclonic regions
    inputFileName = os.path.join(workDir, 'SmoothedVo850_S1.txt')
    outputFileName = os.path.join(workDir, 'DetectBlobsOutputFilenames.txt')
    logDir = os.path.join(logRoot, 'DetectBlobs')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    vorticityFileList = []
    vorticityBlobFileList = []
    with open(inputFileName, 'r') as f:
        vorticityFileList = f.read().splitlines()
    inputFileName = os.path.join(workDir, 'DetectBlobsInputFilenames.txt')
    with open(inputFileName, 'w') as f:
        for i, j in zip(vorticityFileList, windFieldFileList):
            f.write(f'{i};{j}\n')
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'WatershedOutput_{currentTime:%Y%m%d_%H%M%S}.nc')
            vorticityBlobFileList.append(currentOutputFile)
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=step)

    # Smooth CyclonicVorticity again for vorticity gradient calculation
    outputFileName = os.path.join(workDir, 'SmoothedVo850_S2.txt')
    gaussianSmoothedVorticityFileList = []
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'SmoothedVo850_{currentTime:%Y%m%d_%H%M%S}_S2.nc')
            gaussianSmoothedVorticityFileList.append(currentOutputFile)
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=step)

    smoother = GaussianSmoother(vorticityFileList, gaussianSmoothedVorticityFileList)
    smoother.process(vars='CyclonicVorticity', sigma=8.0, nproc=nproc)

    # Size blob segmentation with watershed algorithm
    watershedProcessor = Watershed(
        gaussianSmoothedVorticityFileList, vorticityBlobFileList, nproc=nproc)
    watershedProcessor.process(
        varname='CyclonicVorticity', newVarname='GradientMask', 
        filterMin=2.0e-5, inverse=True)

    # Assign indices to the marked blobs
    lpsBlobOutput = None
    if assignBlobIndex:
        inputFileName = os.path.join(workDir, 'AssignBlobIndexIutputFilenames.txt')
        with open(inputFileName, 'w') as f:
            for i in vorticityBlobFileList:
                f.write(f'{i}\n')
        outputFileName = os.path.join(workDir, 'AssignBlobIndexOutputFilenames.txt')
        vorticityBlobFileList = []
        with open(outputFileName, 'w') as f:
            currentTime = beginTime
            while currentTime <= endTime:
                currentOutputFile = os.path.join(workDir,
                                                f'AssignBlobIndexOutput_{currentTime:%Y%m%d_%H%M%S}.nc')
                vorticityBlobFileList.append(currentOutputFile)
                f.write(f'{currentOutputFile}\n')
                currentTime += datetime.timedelta(hours=step)
        stitchBlobsArg = [
            '--in_list', inputFileName,
            '--out_list', outputFileName,
            '--var', 'GradientMask',
            '--outvar', 'GradientMask',
            '--tagonly',
            '--latname', 'XLAT',
            '--lonname', 'XLONG',
            '--regional'
        ]
        tempestExtremes.run('StitchBlobs', stitchBlobsArg)
        lpsBlobOutput = outputFileName

    # Summary the information of each cyclonic regions, which will be used 
    # in the SyCLoPS classifier by pairing with each low pressure system node
    inputFileName = os.path.join(workDir, 'BlobStatsInputFilenames.txt')
    with open(inputFileName, 'w') as f:
        for i, j in zip(vorticityBlobFileList, windFieldFileList):
            f.write(f'{i};{j}\n')
    outputFileName = blobStatsOutput
    mpiNpIndex = mpiArg.index('-np')
    mpiNProc = int(mpiArg[mpiNpIndex+1])
    blobStatsArg = [
        '--var', 'GradientMask',
        '--out', 'centlon,centlat,minlat,maxlat,minlon,maxlon,area',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--out_fulltime',
        '--regional',
    ]
    blobStatsParallelization = BlobStatsParallelization(
        tempestExtremes, inputFileName, blobStatsArg)
    blobStatsParallelization.run(outputFileName, nproc=nproc)

    inputFileName = os.path.join(workDir, 'LPSMaskInputFilenames_S0.txt')
    with open(inputFileName, 'w') as f:
        for i in vorticityBlobFileList:
            f.write(f'{i}\n')
    outputFileName = os.path.join(workDir, 'LPSMaskOutputFilenames_S0.txt')
    logDir = os.path.join(logRoot, 'LPSMask')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    lpsMaskFileList = []
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'LPSMaskOutput_{currentTime:%Y%m%d_%H%M%S}_S0.nc')
            f.write(f'{currentOutputFile}\n')
            lpsMaskFileList.append(currentOutputFile)
            currentTime += datetime.timedelta(hours=step)
    nodeFileFilterArg = [
        '--in_nodefile', stitchNodesOutput,
        '--in_nodefile_type', 'SN',
        '--in_data_list', inputFileName,
        '--in_fmt',
        (
            'lon,lat,MSLP,WS,MSLPCC20,MSLPCC55,DeepShear,UppThkCC,MidThkCC,LowThkCC,'
            'Z500CC,Vo500Avg,RH100Max,RH850Avg,RH700Avg,T850,Z850,Z700,Z0,U850Diff,'
            'WS200PMax'
        ),
        '--out_data_list', outputFileName,
        '--regional',
        '--bydist', '1.0',
        '--maskvar', 'LPSMask',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--logdir', logDir
    ]
    tempestExtremes.run('NodeFileFilter', nodeFileFilterArg)

    # Solution for no longitude/latitude information in NodeFileFilter outputs.
    dsInvariant = xr.open_dataset(tmpInvariantPath)
    for i in lpsMaskFileList:
        ds = xr.open_dataset(i)
        ds['XLONG'] = dsInvariant['XLONG']
        ds['XLAT'] = dsInvariant['XLAT']
        ds.to_netcdf(f'{i}_tmp')
        ds.close()
        os.rename(f'{i}_tmp', i)
    dsInvariant.close()

    inputFileName = outputFileName
    outputFileName = os.path.join(workDir, 'LPSMaskOutputFilenames_S1.txt')
    lpsMaskFileList = []
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'LPSMaskOutput_{currentTime:%Y%m%d_%H%M%S}_S1.nc')
            lpsMaskFileList.append(currentOutputFile)
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=step)
    stitchBlobsArg = [
        '--in_list', inputFileName,
        '--out_list', outputFileName,
        '--var', 'LPSMask',
        '--outvar', 'LPSMask',
        '--tagonly',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--regional'
    ]
    tempestExtremes.run('StitchBlobs', stitchBlobsArg)
    lpsMaskOutput = outputFileName

    inputFileName = os.path.join(workDir, 'LPSMaskBlobStatsInputFilenames.txt')
    with open(inputFileName, 'w') as f:
        for i,j in zip(lpsMaskFileList, vorticityBlobFileList):
            f.write(f'{i};{j}\n')
    outputFileName = lpsMaskBlobStatsOutput
    blobStatsArg = [
        '--var', 'LPSMask',
        '--out', 'centlon,centlat,minlat,maxlat,minlon,maxlon,area',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--out_fulltime',
        '--overlapvar', 'GradientMask',
        '--regional',
    ]
    blobStatsParallelization = BlobStatsParallelization(
        tempestExtremes, inputFileName, blobStatsArg)
    blobStatsParallelization.run(outputFileName, nproc=nproc)

    # Calculate the F diagnostic
    # Step 0: only to store intermediate variables, can be removed later
    inputFileName = os.path.join(workDir, 'FrontalDiagnostic850InputFilenames.txt')
    outputFileName = os.path.join(workDir, 'FrontalDiagnostic850_S0.txt')
    logDir = os.path.join(logRoot, 'FrontalDiagnostic850_S0')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    with open(inputFileName, 'w') as f:
        for i in preprocessedList:
            f.write(f'{i}\n')
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'FrontalDiagnostic850_{currentTime:%Y%m%d_%H%M%S}_S0.nc')
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=step)
    variableProcessorArg = [
        '--in_data_list', inputFileName,
        '--out_data_list',  outputFileName,
        '--var', ('_PROD(_CURL{4,1}(ULev12,VLev12),_GRADMAG{4,1}(ThetaLev12));'
            '_PROD(_VECDOTGRAD{4,1}(ULev12,VLev12,ThetaLev12),-1);'
            '_GRADMAG{4,1}(ThetaLev12);'
            '_CURL{4,1}(ULev12,VLev12);'
            '_PROD(_F(),0.0000045)'),
        '--varout', 'FrontalDiagnostic850Var1;ThetaAdvection850;ThetaGradMagnitude850;FrontalVo850;FrontalDiagnostic850Ref',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--regional',
        '--logdir', logDir
    ]
    # tempestExtremes.run('VariableProcessor', variableProcessorArg)

    # Step 1: calculate F diagnostic
    inputFileName = outputFileName
    outputFileName = os.path.join(workDir, 'FrontalDiagnostic850_S1.txt')
    frontalDiagnostic850FileList = []
    logDir = os.path.join(logRoot, 'FrontalDiagnostic850_S1')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'FrontalDiagnostic850_{currentTime:%Y%m%d_%H%M%S}_S1.nc')
            frontalDiagnostic850FileList.append(currentOutputFile)
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=step)
    variableProcessorArg = [
        '--in_data_list', inputFileName,
        '--out_data_list',  outputFileName,
        '--var', ('_PROD(_DIV(FrontalDiagnostic850Var1,FrontalDiagnostic850Ref),_SIGN(_F()))'),
        '--varout', 'FrontalDiagnostic850',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--regional',
        '--logdir', logDir
    ]
    # tempestExtremes.run('VariableProcessor', variableProcessorArg)

    # Search for frontal regions
    inputFileName = os.path.join(workDir, 'FrontalDiagnostic850_S1.txt')
    outputFileName = os.path.join(workDir, 'DetectFrontalBlobsOutputFilenames.txt')
    logDir = os.path.join(logRoot, 'DetectFrontalBlobs')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    frontalBlobFileList = []
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'DetectFrontalBlobsOutput_{currentTime:%Y%m%d_%H%M%S}.nc')
            frontalBlobFileList.append(currentOutputFile)
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=step)
    detectBlobsArg = [
        '--in_data_list', inputFileName,
        '--out_list', outputFileName,
        '--thresholdcmd',
        '(FrontalDiagnostic850,>=,1.0,0)',
        '--geofiltercmd', 'area,>=,1e4km2',
        '--tagvar', 'FrontalMask',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--regional',
        '--timefilter', '3hr',
        '--logdir', logDir
    ]
    tempestExtremes.run('DetectBlobs', detectBlobsArg)

    # Assign indices to the marked blobs
    frontalBlobOutput = None
    if assignBlobIndex:
        inputFileName = os.path.join(workDir, 'AssignFrontalBlobIndexIutputFilenames.txt')
        with open(inputFileName, 'w') as f:
            for i in frontalBlobFileList:
                f.write(f'{i}\n')
        outputFileName = os.path.join(workDir, 'AssignFrontalBlobIndexOutputFilenames.txt')
        frontalBlobFileList = []
        with open(outputFileName, 'w') as f:
            currentTime = beginTime
            while currentTime <= endTime:
                currentOutputFile = os.path.join(workDir,
                                                f'AssignFrontalBlobIndexOutput_{currentTime:%Y%m%d_%H%M%S}.nc')
                frontalBlobFileList.append(currentOutputFile)
                f.write(f'{currentOutputFile}\n')
                currentTime += datetime.timedelta(hours=step)
        stitchBlobsArg = [
            '--in_list', inputFileName,
            '--out_list', outputFileName,
            '--var', 'FrontalMask',
            '--outvar', 'FrontalMask',
            '--tagonly',
            '--latname', 'XLAT',
            '--lonname', 'XLONG',
            '--regional'
        ]
        tempestExtremes.run('StitchBlobs', stitchBlobsArg)
        frontalBlobOutput = outputFileName

    # Summary the information of each frontal regions, which will be used 
    # in the SyCLoPS classifier by pairing with each low pressure system node
    inputFileName = os.path.join(workDir, 'FrontalBlobStatsInputFilenames.txt')
    with open(inputFileName, 'w') as f:
        for iFrontalBlob, iVortBlob, iPreprocessed in zip(
            frontalBlobFileList, vorticityBlobFileList, preprocessedList):
            f.write(f'{iFrontalBlob};{iVortBlob};{iPreprocessed}\n')
    outputFileName = frontalBlobStatsOutput
    blobStatsArg = [
        '--var', 'FrontalMask',
        '--out', 'centlon,centlat,minlat,maxlat,minlon,maxlon,area',
        '--latname', 'XLAT',
        '--lonname', 'XLONG',
        '--timefilter', '3hr',
        '--out_fulltime',
        '--regional',
        '--overlapvar', 'GradientMask',
        '--sumvar', 'RHLev43',
    ]
    blobStatsParallelization = BlobStatsParallelization(
        tempestExtremes, inputFileName, blobStatsArg)
    blobStatsParallelization.run(
        outputFileName, nproc=nproc)

    lpsClassifier = SyCLoPSClassifier(stitchNodesOutput, blobStatsOutput, tmpInvariantPath, 
                                      lpsMaskBlobStatsOutput=lpsMaskBlobStatsOutput,
                                      nproc=nproc, flagGeopotentialHeight=True, 
                                      latName='XLAT', lonName='XLONG',
                                      rhTropicalThreshold=15)
    lpsClassifier.preprocess(classifierPreprocessFile, override=True)
    lpsClassifier.classify(classifierFinalFile, full=fullResult, resultFormat=reusltFormat)

    if assignBlobIndex:
        classifierFinalFile = f'{classifierFinalFile}.parquet'
        frontClassifier = MCSClassifier(
            frontalBlobStatsOutput, classifierFinalFile,
            tmpInvariantPath, classifierType='front',
            lonName='XLONG', latName='XLAT', regional=True, nproc=nproc)
        frontClassifier.preprocess(frontClassifierPreprocessFile, override=True)