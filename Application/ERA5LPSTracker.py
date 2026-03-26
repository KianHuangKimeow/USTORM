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

from Base import getBoxMask
from Preprocess.Model import GaussianSmoother, PreprocessERA5, Watershed
from System import defineScratch
from TempestExtremes import (
  ERA5InputName, TempestExtremes, SyCLoPSClassifier, MCSClassifier, BlobStatsParallelization)
from Utilities import downloadRdaERA5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('USTORM (Unified Storm Tracking for Observations and '
        'multi-Resolution Models) low pressure system tracking and classification '
        'workflow script for ERA5.'))
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
    parser.add_argument(
        '--regional_mask_file', action='store', metavar='RegionalMask',
        default=None,
        help=('The regional mask file.')
    )
    parser.add_argument(
        '--regional_mask_variable', action='store', metavar='RegionalMaskVar',
        default='RegionalMask',
        help=('The regional mask variable name.')
    )
    parser.add_argument(
        '--inner_mask_variable', action='store', metavar='InnerMaskVar',
        default='InnerBoxMask',
        help=('The regional mask variable name.')
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
    regionalMaskFilename = vars(parserArgs).get('regional_mask_file')
    regionalMaskVar = vars(parserArgs).get('regional_mask_variable')
    innerMaskVar = vars(parserArgs).get('inner_mask_variable')

    beginTime = datetime.datetime.strptime(beginTimeStr, "%Y%m%d_%H")
    endTime = datetime.datetime.strptime(endTimeStr, "%Y%m%d_%H")

    stitchNodesOutput = os.path.join(workDir, 'StitchNodesResult.txt')
    lpsMaskBlobStatsOutput = os.path.join(workDir, 'LPSMaskBlobStatsOutput.txt')
    blobStatsOutput = os.path.join(workDir, 'BlobStatsOutput.txt')
    frontalBlobStatsOutput = os.path.join(workDir, 'FrontalBlobStatsOutput.txt')
    frontClassifierPreprocessFile = os.path.join(workDir, f'FrontPreprocess_{beginTime:%Y%m%d%H}_{endTime:%Y%m%d%H}.parquet')
    tmpInvariantPath = ''

    # Download ERA5 from RDA
    downloadRdaERA5(dataRoot, beginTime, endTime, override=False)

    preprocessDir = f'{workDir}/preprocess'

    tempestExtremes = TempestExtremes(tempestExtremesPath, mpirunPath, mpiArg)

    classifierPreprocessFile = os.path.join(workDir, f'Preprocess_{beginTime:%Y%m%d%H}_{endTime:%Y%m%d%H}.parquet')
    classifierFinalFile = os.path.join(workDir, f'Result_{beginTime:%Y%m%d%H}_{endTime:%Y%m%d%H}')
    inputName = ERA5InputName(root=dataRoot)
    inputName.setDateTime(begin=beginTime, end=endTime)
    mlSpFiles = inputName.generateInput(
        ['128_134_sp.regn320sc'], output=True, step=6).splitlines()
    mlTFiles = inputName.generateInput(
        ['0_5_0_0_0_t.regn320sc'], output=True, step=6).splitlines()
    mlQFiles = inputName.generateInput(
        ['0_5_0_1_0_q.regn320sc'], output=True, step=6).splitlines()
    mlUFiles = inputName.generateInput(
        ['0_5_0_2_2_u.regn320uv'], output=True, step=6).splitlines()
    mlVFiles = inputName.generateInput(
        ['0_5_0_2_3_v.regn320uv'], output=True, step=6).splitlines()

    # Detect low pressure systems
    inputFileName = os.path.join(workDir, 'DetectNodesInputFilenames.txt')
    outputFileName = os.path.join(workDir, 'DetectNodesOutputFilenames.txt')
    logRoot = os.path.join(workDir, 'log')
    if not os.path.exists(logRoot):
        os.makedirs(logRoot)
    invariantPaths = inputName.getInvariantPaths()

    preprocessor = PreprocessERA5(
        mlSpFiles=mlSpFiles, mlTFiles=mlTFiles, mlQFiles=mlQFiles,
        mlUFiles=mlUFiles, mlVFiles=mlVFiles)
    varProcess = dict(
        RLev = dict(
            derivative = 'relative_humidity_from_specific_humidity',
            level = [61],
            suffix = [62],
            method = 'linear',
        ),
        ThetaLev = dict(
            derivative = 'potential_temperature',
            level = [114],
            suffix = [115],
            method = 'linear',
        ),
    )
    
    inputName.generateInput()
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
    
    maskFileMl = None
    if regionalMaskFilename is not None and invariantPaths:
        currentInvariantPathPl = list(invariantPaths.values())
        currentInvariantPathPl = currentInvariantPathPl[0]
        maskInvariantDs = xr.open_dataset(regionalMaskFilename)
        daDomainMask = maskInvariantDs[innerMaskVar]
        maskInvariantDs[regionalMaskVar] = daDomainMask.copy(data=xr.zeros_like(daDomainMask))
        daDomainMask.close()
        if 'Z0' in maskInvariantDs.variables.keys():
            maskInvariantDs = maskInvariantDs.drop_vars(['Z0'])
        maskInvariantPath = preprocessor.regrid(
            workDir, [maskInvariantDs], currentInvariantPathPl, 'nearest_s2d', 
            fileSuffix='ll025sc', varRename={'XLONG': 'lon', 'XLAT': 'lat'},
            regridderSuffix='CONUS404.ll025sc', nproc=nproc)
        inputName.addInvariantPath('mask.pl', maskInvariantPath[0])
        tmpInvariantDs = xr.open_dataset(maskInvariantPath[0])
        tmpInvariantDs[regionalMaskVar] = tmpInvariantDs[regionalMaskVar].copy(
            data=getBoxMask(
                tmpInvariantDs['longitude'].to_numpy(), tmpInvariantDs['latitude'].to_numpy(), 
                maskInvariantDs['XLONG'].to_numpy(), maskInvariantDs['XLAT'].to_numpy(), 
                offset=0.04))
        tmpInvariantDs.to_netcdf(f'{maskInvariantPath[0]}_tmp')
        tmpInvariantDs.close()
        os.rename(f'{maskInvariantPath[0]}_tmp', maskInvariantPath[0])

        currentInvariantPathMl = mlTFiles[0]
        maskInvariantPath = preprocessor.regrid(
            workDir, [maskInvariantDs], currentInvariantPathMl, 'nearest_s2d', 
            fileSuffix='regn320sc', varRename={'XLONG': 'lon', 'XLAT': 'lat'},
            regridderSuffix='CONUS404.regn320sc', nproc=nproc)
        inputName.addInvariantPath('mask.ml', maskInvariantPath[0])
        tmpInvariantDs = xr.open_dataset(maskInvariantPath[0])
        tmpInvariantDs[regionalMaskVar] = tmpInvariantDs[regionalMaskVar].copy(
            data=getBoxMask(
                tmpInvariantDs['longitude'].to_numpy(), tmpInvariantDs['latitude'].to_numpy(), 
                maskInvariantDs['XLONG'].to_numpy(), maskInvariantDs['XLAT'].to_numpy(), 
                offset=0.04))
        tmpInvariantDs.to_netcdf(f'{maskInvariantPath[0]}_tmp')
        tmpInvariantDs.close()
        os.rename(f'{maskInvariantPath[0]}_tmp', maskInvariantPath[0])
        maskFileMl = maskInvariantPath[0]
        maskInvariantDs.close()
            
    inputName.generateInput(exclusiveType=['e5.oper.an.ml'], addInvariant=['mask.pl'])
    if invariantPaths:
        for var in invariantPaths.keys():
            currentInvariantPath = invariantPaths[var]
            currentInvariantBasename = os.path.basename(currentInvariantPath)
            tmpInvariantPath = os.path.join(workDir, currentInvariantBasename)
            inputName.replace(currentInvariantPath, tmpInvariantPath)
    currentInvariantPath = invariantPaths['128_129_z.ll025sc']
    currentInvariantBasename = os.path.basename(currentInvariantPath)
    tmpInvariantPath = os.path.join(workDir, currentInvariantBasename)

    preprocessedList = preprocessor.process(varProcess, preprocessDir, 
                         nproc=nproc, override=False)
    
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
            # Greatest positive closed contour delta of MSLP over a 5.5 GCD (MSLPCC55)
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
            # Avarage 700 hPa relative humidity within a 2.5 GCD (RH700Avg)
            'R(700hPa),avg,2.5;'
            # 850 hPa air temperature at the node (T850)
            'T(850hPa),max,0.0;'
            # 850 hPa geopotential at the node (Z850)
            'Z(850hPa),min,0.0;'
            # 700 hPa geopotential at the node (Z700)
            'Z(700hPa),min,0.0;'
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
    if regionalMaskFilename:
        detectNodesArg.extend(['--thresholdcmd', f'{innerMaskVar},=,1,0'])

    tempestExtremes.run('DetectNodes', detectNodesArg)

    inputFileName = outputFileName
    outputFileName = stitchNodesOutput
    stitchNodesArg = [
        '--in_list', inputFileName,
        '--out', outputFileName,
        '--in_fmt',
        (
            'lon,lat,MSLP,WS,MSLPCC20,MSLPCC55,DeepShear,UppThkCC,MidThkCC,LowThkCC,'
            'Z500CC,Vo500Avg,RH100Max,RH850Avg,RH700Avg,T850,Z850,Z700,Z0,U850Diff,WS200PMax'
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
    logDir = os.path.join(logRoot, 'SmoothedVo850_S0')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    windFieldFileList = inputName.generateInput(
        ['0_5_0_2_2_u.regn320uv', '0_5_0_2_3_v.regn320uv'], output=True, step=6).splitlines()
    inputName.dump(inputFileName)
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'SmoothedVo850_{currentTime:%Y%m%d_%H%M%S}_S0.nc')
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=6)
    variableProcessorArg = [
        '--in_data_list', inputFileName,
        '--out_data_list',  outputFileName,
        '--var', '_CURL{8,3}(U(114),V(114))',
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
            currentTime += datetime.timedelta(hours=6)
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
            currentTime += datetime.timedelta(hours=6)

    # Size blob segmentation with watershed algorithm
    watershedProcessor = Watershed(
        vorticityFileList, vorticityBlobFileList, nproc=nproc)
    watershedProcessor.process(
        varname='CyclonicVorticity', newVarname='GradientMask', 
        filterMin=2.0e-5, inverse=True, maskFile=maskFileMl, maskVar=regionalMaskVar)

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
                currentTime += datetime.timedelta(hours=6)
        stitchBlobsArg = [
            '--in_list', inputFileName,
            '--out_list', outputFileName,
            '--var', 'GradientMask',
            '--outvar', 'GradientMask',
            '--tagonly',
            '--latname', 'latitude',
            '--lonname', 'longitude'
        ]
        tempestExtremes.run('StitchBlobs', stitchBlobsArg)
        lpsBlobOutput = outputFileName

    # Summary the information of each cyclonic regions, which will be used 
    # in the SyCLoPS classifier by pairing with each low pressure system node
    windFieldFileList = inputName.generateInput(
        ['128_131_u.ll025uv', '128_132_v.ll025uv'], output=True, step=6).splitlines()
    inputFileName = os.path.join(workDir, 'BlobStatsInputFilenames.txt')
    with open(inputFileName, 'w') as f:
        for i, j in zip(vorticityBlobFileList, windFieldFileList):
            f.write(f'{i};{j}\n')
    outputFileName = blobStatsOutput
    blobStatsArg = [
        '--var', 'GradientMask',
        '--out', 'centlon,centlat,minlat,maxlat,minlon,maxlon,area',
        '--latname', 'latitude',
        '--lonname', 'longitude',
        '--out_fulltime',
    ]
    blobStatsParallelization = BlobStatsParallelization(
        tempestExtremes, inputFileName, blobStatsArg)
    blobStatsParallelization.run(outputFileName, nproc=nproc)

    inputFileName = os.path.join(workDir, 'LPSMaskInputFilenames_S0.txt')
    with open(inputFileName, 'w') as f:
        for i in vorticityBlobFileList:
            f.write(f'{i}\n')
    outputFileName = os.path.join(workDir, 'LPSMaskOutputFilenames_S0.txt')
    lpsMaskFileList = []
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'LPSMaskOutput_{currentTime:%Y%m%d_%H%M%S}_S0.nc')
            f.write(f'{currentOutputFile}\n')
            lpsMaskFileList.append(currentOutputFile)
            currentTime += datetime.timedelta(hours=6)
    nodeFileFilterArg = [
        '--in_nodefile', stitchNodesOutput,
        '--in_nodefile_type', 'SN',
        '--in_data_list', inputFileName,
        '--in_fmt',
        (
            'lon,lat,MSLP,WS,MSLPCC20,MSLPCC55,DeepShear,UppThkCC,MidThkCC,LowThkCC,'
            'Z500CC,Vo500Avg,RH100Max,RH850Avg,RH700Avg,T850,Z850,Z700,Z0,U850Diff,WS200PMax'
        ),
        '--out_data_list', outputFileName,
        '--regional',
        '--bydist', '1.0',
        '--maskvar', 'LPSMask',
        '--latname', 'latitude',
        '--lonname', 'longitude',
    ]
    tempestExtremes.run('NodeFileFilter', nodeFileFilterArg)

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
            currentTime += datetime.timedelta(hours=6)
    stitchBlobsArg = [
        '--in_list', inputFileName,
        '--out_list', outputFileName,
        '--var', 'LPSMask',
        '--outvar', 'LPSMask',
        '--tagonly',
        '--latname', 'latitude',
        '--lonname', 'longitude',
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
        '--latname', 'latitude',
        '--lonname', 'longitude',
        '--out_fulltime',
        '--overlapvar', 'GradientMask',
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
        # for i in preprocessedList:
        #     f.write(f'{i}\n')
        for i, j, k in zip(mlUFiles, mlVFiles, preprocessedList):
            f.write(f'{i};{j};{k}\n')
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'FrontalDiagnostic850_{currentTime:%Y%m%d_%H%M%S}_S0.nc')
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=6)
    variableProcessorArg = [
        '--in_data_list', inputFileName,
        '--out_data_list',  outputFileName,
        '--var', ('_PROD(_CURL{4,1}(U(114),V(114)),_GRADMAG{4,1}(ThetaLev115));'
            '_PROD(_VECDOTGRAD{4,1}(U(114),V(114),ThetaLev115),-1);'
            '_GRADMAG{4,1}(ThetaLev115);'
            '_CURL{4,1}(U(114),V(114));'
            '_PROD(_F(),0.0000045)'),
        '--varout', 'FrontalDiagnostic850Var1;ThetaAdvection850;ThetaGradMagnitude850;FrontalVo850;FrontalDiagnostic850Ref',
        '--latname', 'latitude',
        '--lonname', 'longitude',
        '--logdir', logDir
    ]
    tempestExtremes.run('VariableProcessor', variableProcessorArg)

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
            currentTime += datetime.timedelta(hours=6)
    variableProcessorArg = [
        '--in_data_list', inputFileName,
        '--out_data_list',  outputFileName,
        '--var', ('_PROD(_DIV(FrontalDiagnostic850Var1,FrontalDiagnostic850Ref),_SIGN(_F()))'),
        '--varout', 'FrontalDiagnostic850',
        '--latname', 'latitude',
        '--lonname', 'longitude',
        '--logdir', logDir
    ]
    tempestExtremes.run('VariableProcessor', variableProcessorArg)

    # Search for frontal regions
    inputFileName = os.path.join(workDir, 'DetectFrontalBlobsInputFilenames.txt')
    outputFileName = os.path.join(workDir, 'DetectFrontalBlobsOutputFilenames.txt')
    logDir = os.path.join(logRoot, 'DetectFrontalBlobs')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    frontalBlobFileList = []
    with open(inputFileName, 'w') as f:
        for i in frontalDiagnostic850FileList:
            f.write(f'{i}')
            if maskFileMl is not None:
                f.write(f';{maskFileMl}')
            f.write('\n')
    with open(outputFileName, 'w') as f:
        currentTime = beginTime
        while currentTime <= endTime:
            currentOutputFile = os.path.join(workDir,
                                             f'DetectFrontalBlobsOutput_{currentTime:%Y%m%d_%H%M%S}.nc')
            frontalBlobFileList.append(currentOutputFile)
            f.write(f'{currentOutputFile}\n')
            currentTime += datetime.timedelta(hours=6)
    detectBlobsThresholdCmd = 'FrontalDiagnostic850,>=,1.0,0'
    if regionalMaskFilename is not None and regionalMaskVar is not None:
        detectBlobsThresholdCmd += f';{regionalMaskVar},>=,1.0,0'
    detectBlobsArg = [
        '--in_data_list', inputFileName,
        '--out_list', outputFileName,
        '--thresholdcmd',
        f'{detectBlobsThresholdCmd}',
        '--geofiltercmd', 'area,>=,1e4km2',
        '--tagvar', 'FrontalMask',
        '--latname', 'latitude',
        '--lonname', 'longitude',
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
                currentTime += datetime.timedelta(hours=6)
        stitchBlobsArg = [
            '--in_list', inputFileName,
            '--out_list', outputFileName,
            '--var', 'FrontalMask',
            '--outvar', 'FrontalMask',
            '--tagonly',
            '--latname', 'latitude',
            '--lonname', 'longitude',
        ]
        tempestExtremes.run('StitchBlobs', stitchBlobsArg)
        frontalBlobOutput = outputFileName

    # Summary the information of each frontal regions, which will be used 
    # in the SyCLoPS classifier by pairing with each low pressure system node
    inputFileName = os.path.join(workDir, 'FrontalBlobStatsInputFilenames.txt')
    with open(inputFileName, 'w') as f:
        for iFrontalBlob, iVortBlob, iPreprocessedFile in zip(
            frontalBlobFileList, vorticityBlobFileList, preprocessedList):
            f.write(f'{iFrontalBlob};{iVortBlob};{iPreprocessedFile}\n')
    outputFileName = frontalBlobStatsOutput
    blobStatsArg = [
        '--var', 'FrontalMask',
        '--out', 'centlon,centlat,minlat,maxlat,minlon,maxlon,area',
        '--latname', 'latitude',
        '--lonname', 'longitude',
        '--timefilter', '3hr',
        '--out_fulltime',
        '--overlapvar', 'GradientMask',
        '--sumvar', 'RLev62',
    ]
    blobStatsParallelization = BlobStatsParallelization(
        tempestExtremes, inputFileName, blobStatsArg)
    blobStatsParallelization.run(outputFileName, nproc=nproc)

    lpsClassifier = SyCLoPSClassifier(stitchNodesOutput, blobStatsOutput, tmpInvariantPath, 
                                      lpsMaskBlobStatsOutput=lpsMaskBlobStatsOutput,
                                      nproc=nproc, flagGeopotentialHeight=False, 
                                      latName='latitude', lonName='longitude')
    lpsClassifier.preprocess(classifierPreprocessFile, override=True)
    lpsClassifier.classify(classifierFinalFile, full=fullResult, resultFormat=reusltFormat)

    if assignBlobIndex:
        classifierFinalFile = f'{classifierFinalFile}.parquet'
        frontClassifier = MCSClassifier(
            frontalBlobStatsOutput, classifierFinalFile, 
            tmpInvariantPath, classifierType='front',
            regional=False, nproc=nproc)
        frontClassifier.preprocess(frontClassifierPreprocessFile, override=True)