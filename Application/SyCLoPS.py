'''
Details of SyCLoPS algorithms can be found at 
https://doi.org/10.1029/2024JD041287
'''
import argparse
import datetime
import os
import sys

import xarray as xr

sys.path.insert(0, os.path.abspath('.'))

from Utilities import downloadFromRda
from TempestExtremes import ERA5InputName, TempestExtremes, SyCLoPSClassifier
from System import defineScratch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('SyCLoPS (The System for Classification of Low-Pressure Systems): '
                     'Details of SyCLoPS algorithms can be found at '
                     'https://doi.org/10.1029/2024JD041287'))
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
        '--full_result', '--full', action='store', metavar='StoreFullResult',
        default=False,
        help='Whether to save verbose info.'
    )
    parser.add_argument(
        '--result_format', '--format', action='store', metavar='Format',
        default='parquet',
        help=('The format of result to be stored. Supported: '
              'parquet (default), csv, all (i.e., parquet & csv)')
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
    fullResult = vars(parserArgs).get('full_result')
    reusltFormat = vars(parserArgs).get('result_format')

    # Download ERA5 from RDA
    downloadFromRda(dataRoot, beginTimeStr, endTimeStr, override=False)

    stitchNodesOutput = os.path.join(workDir, 'StitchNodesResult.txt')
    blobStatsOutput = os.path.join(workDir, 'BlobStatsOutput.txt')
    tmpInvariantPath = ''

    tempestExtremes = TempestExtremes(tempestExtremesPath, mpirunPath, mpiArg)

    beginTime = datetime.datetime.strptime(beginTimeStr, "%Y%m%d_%H")
    endTime = datetime.datetime.strptime(endTimeStr, "%Y%m%d_%H")
    classifierPreprocessFile = os.path.join(workDir, f'Preprocess_{beginTime:%Y%m%d%H}_{endTime:%Y%m%d%H}.parquet')
    classifierFinalFile = os.path.join(workDir, f'Result_{beginTime:%Y%m%d%H}_{endTime:%Y%m%d%H}')
    inputName = ERA5InputName(root=dataRoot)
    inputName.setDateTime(begin=beginTime, end=endTime)
    inputName.generateInput()

    # Detect low pressure systems
    inputFileName = os.path.join(workDir, 'DetectNodesInputFilenames.txt')
    outputFileName = os.path.join(workDir, 'DetectNodesOutputFilenames.txt')
    logRoot = os.path.join(workDir, 'log')
    if not os.path.exists(logRoot):
        os.makedirs(logRoot)
    invariantPaths = inputName.getInvariantPaths()
    if invariantPaths:
        for var in invariantPaths.keys():
            currentInvariantPath = invariantPaths[var]
            currentInvariantBasename = os.path.basename(currentInvariantPath)
            tmpInvariantPath = os.path.join(workDir, currentInvariantBasename)
            if not os.path.exists(tmpInvariantPath):
                invariantDataset = xr.open_dataset(currentInvariantPath)
                invariantDataset = invariantDataset[dict(time=0)]
                invariantDataset = invariantDataset.drop_vars('time')
                if 'Z' in invariantDataset.variables.keys():
                    invariantDataset = invariantDataset.rename_vars({
                                                                    'Z': 'Z0'})
                invariantDataset.to_netcdf(tmpInvariantPath)
            inputName.replace(currentInvariantPath, tmpInvariantPath)
    inputName.dump(inputFileName)
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'DetectNodesOutput_{currentTime:%Y%m%d_%H%M%S}.txt')
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(days=1)

    logDir = os.path.join(logRoot, 'DetectNodes')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    detectNodesArg = [
        '--in_data_list', inputFileName,
        '--out_file_list', outputFileName,
        # Search minimum mean sea level pressure
        '--searchbymin', 'MSL',
        # Mean sea level pressure must increase 10 Pa within a 5.5 GCD
        '--closedcontourcmd', 'MSL,10,5.5,0',
        # Merge candidates within a 6.0 GCD, with the lower mean sea level pressure 
        # node taking precedence.
        '--mergedist', '6.0',
        '--outputcmd',
        (
            # Mean sea level pressure (MSLP)
            'MSL,min,0;'
            # Maximum 10m wind speed within 2.0 GCD (WS10)
            '_VECMAG(VAR_10U,VAR_10V),max,2.0;'
            # Greatest positive closed contour delta of MSLP over a 2.0 GCD (MSLPCC20)
            'MSL,posclosedcontour,2.0,0;'
            # Greatest positive closed contour delta of MSLP over a 5.0 GCD (MSLPCC55)
            'MSL,posclosedcontour,5.5,0;'
            # Average environmental deep-layer (200 - 850 hPa) wind shear 
            # over a 10.0 GCD (DeepShear)
            '_DIFF(_VECMAG(U(200hPa),V(200hPa)),_VECMAG(U(850hPa),V(850hPa))),avg,10.0;'
            # Greatest decline of the upper-level (300 - 500 hPa) geopotential 
            # thickness within a 6.5 GCD of the maximum thickness node within 
            # a 1.0 GCD of the current node (UppThkCC)
            '_DIFF(Z(300hPa),Z(500hPa)),negclosedcontour,6.5,1.0;'
            # Greatest decline of the mid-level (500 - 700 hPa) geopotential 
            # thickness within a 3.5 GCD of the maximum thickness node within 
            # a 1.0 GCD of the current node (MidThkCC)
            '_DIFF(Z(500hPa),Z(700hPa)),negclosedcontour,3.5,1.0;'
            # Greatest decline of the lower-level (700 - 925 hPa) geopotential 
            # thickness within a 3.5 GCD of the maximum thickness node within 
            # a 1.0 GCD of the current node (LowerThkCC)
            '_DIFF(Z(700hPa),Z(925hPa)),negclosedcontour,3.5,1.0;'
            # Greatest increase of the 500 hPa geopotential within a 3.5 GCD 
            # of the minimum geopotential node within a 1.0 GCD of the current node 
            # (Z500CC)
            'Z(500hPa),posclosedcontour,3.5,1.0;'
            # Avarage relative vorticity over a 2.5 GCD (Vo500Avg)
            'VO(500hPa),avg,2.5;'
            # Maximum 100 hPa relative humidity within a 2.5 GCD (RH100Max)
            'R(100hPa),max,2.5;'
            # Avarage 850 hPa relative humidity within a 2.5 GCD (RH850Avg)
            'R(850hPa),avg,2.5;'
            # 850 hPa air temperature at the node (T850)
            'T(850hPa),max,0.0;'
            # 850 hPa geopotential at the node (Z850)
            'Z(850hPa),min,0.0;'
            # Surface geopotential at the node (Z0)
            'Z0,min,0;'
            # Difference between the weighted area mean of positive and negative 
            # values of 850 hPa eastward wind over a 5.5 GCD (U850Diff)
            'U(850hPa),posminusnegwtarea,5.5;'
            # Maximun poleward 200 hPa wind speed within a 1.0 GCD (WS200PMax)
            '_VECMAG(U(200hPa),V(200hPa)),maxpoleward,1.0'
        ),
        '--timefilter', '3hr',
        '--latname', 'latitude',
        '--lonname', 'longitude',
        '--logdir', logDir
    ]
    tempestExtremes.run('DetectNodes', detectNodesArg, 'Done', slice(-65, -61))

    inputFileName = outputFileName
    outputFileName = stitchNodesOutput
    stitchNodesArg = [
        '--in_list', inputFileName,
        '--out', outputFileName,
        '--in_fmt',
        (
            'lon,lat,MSLP,WS,MSLPCC20,MSLPCC55,DeePShear,UppThkCC,MidThkCC,LowThkCC,'
            'Z500CC,Vo500Avg,RH100Max,RH850Avg,T850,Z850,Z0,U850Diff,WS200PMax'
        ),
        '--range', '4.0',
        '--mintime', '18h',
        '--maxgap', '12h',
        '--threshold', 'MSLPCC55,>=,100.0,5'
    ]
    tempestExtremes.run('StitchNodes', stitchNodesArg, 'Done', slice(-65, -61))

    # Calculate smoothed relative vorticity using 850 hPa wind field.
    inputFileName = os.path.join(workDir, 'SmoothedVo850InputFilenames.txt')
    outputFileName = os.path.join(workDir, 'SmoothedVo850_S0.txt')
    logDir = os.path.join(logRoot, 'SmoothedVo850_S0')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    windFieldFileList = inputName.generateInput(
        ['128_131_u.ll025uv', '128_132_v.ll025uv'], True).splitlines()
    inputName.dump(inputFileName)
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'SmoothedVo850_{currentTime:%Y%m%d_%H%M%S}_S0.nc')
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(days=1)
    variableProcessorArg = [
        '--in_data_list', inputFileName,
        '--out_data_list',  outputFileName,
        '--var', '_CURL{8,3}(U(850hPa),V(850hPa))',
        '--varout', 'Vorticity',
        '--latname', 'latitude',
        '--lonname', 'longitude',
        '--timefilter', '3hr',
        '--logdir', logDir
    ]
    tempestExtremes.run('VariableProcessor', variableProcessorArg)

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
            currentTime += datetime.timedelta(days=1)
    variableProcessorArg = [
        '--in_data_list', inputFileName,
        '--out_data_list',  outputFileName,
        '--var', '_COND(_LAT(),Vorticity,_PROD(Vorticity,-1))',
        '--varout', 'CyclonicVorticity',
        '--latname', 'latitude',
        '--lonname', 'longitude',
        '--logdir', logDir
    ]
    tempestExtremes.run('VariableProcessor', variableProcessorArg)

    # Search for cyclonic regions
    inputFileName = outputFileName
    outputFileName = os.path.join(workDir, 'DetectBlobsOutputFilenames.txt')
    logDir = os.path.join(logRoot, 'DetectBlobs')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    vorticityFileList = []
    with open(inputFileName, 'r') as f:
        vorticityFileList = f.read().splitlines()
    with open(inputFileName, 'w') as f:
        for i, j in zip(vorticityFileList, windFieldFileList):
            f.write(f'{i};{j}\n')
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'DetectBlobsOutput_{currentTime:%Y%m%d_%H%M%S}.nc')
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(days=1)
    detectBlobsArg = [
        '--in_data_list', inputFileName,
        '--out_list', outputFileName,
        '--thresholdcmd',
        ('((CyclonicVorticity,>=,2e-5,0) & (_VECMAG(U(925hPa),V(925hPa)),>=,12.0,0))'
         ' | (CyclonicVorticity,>=,4e-5,0)'),
        '--geofiltercmd', 'area,>=,1e4km2',
        '--tagvar', 'blobmask',
        '--latname', 'latitude',
        '--lonname', 'longitude',
        '--timefilter', '3hr',
        '--logdir', logDir
    ]
    tempestExtremes.run('DetectBlobs', detectBlobsArg)

    # Summary the information of each cyclonic regions, which will be used 
    # in the SyCLoPS classifier by pairing with each low pressure system node
    inputFileName = outputFileName
    outputFileName = blobStatsOutput
    detectBlobsOutputList = []
    with open(inputFileName, 'r') as f:
        detectBlobsOutputList = f.read().splitlines()
    with open(inputFileName, 'w') as f:
        for i, j in zip(detectBlobsOutputList, windFieldFileList):
            f.write(f'{i};{j}\n')
    blobStatsArg = [
        '--in_list', inputFileName,
        '--out_file', outputFileName,
        '--findblobs',
        '--var', 'blobmask',
        '--out', 'centlon,centlat,minlat,maxlat,minlon,maxlon,area',
        '--latname', 'latitude',
        '--lonname', 'longitude',
        '--out_fulltime',
        '--out_isotime',
        # Integrated kinetic energy (IKE)
        '--sumvar', '_PROD(_SUM(_POW(U(925hPa),2),_POW(V(925hPa),2)),0.5)'
    ]
    tempestExtremes.run('BlobStats', blobStatsArg)

    lpsClassifier = SyCLoPSClassifier(stitchNodesOutput, blobStatsOutput, tmpInvariantPath)
    lpsClassifier.preprocess(classifierPreprocessFile, override=True)
    lpsClassifier.classify(classifierFinalFile, full=fullResult, resultFormat=reusltFormat)
