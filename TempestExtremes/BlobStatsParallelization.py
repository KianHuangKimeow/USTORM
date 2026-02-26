from itertools import repeat
import logging
import math
from multiprocessing import Pool
import os
import platform
import stat
import subprocess

import polars as pl

from TempestExtremes import TempestExtremes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BlobStatsParallelization:
    def __init__(
            self, te: TempestExtremes, inputFileName: str = None, args: list = None):
        self.te_ = te
        self.inputFileName_ = inputFileName
        self.args_ = args
        self.outputFileNames_ = []

    def runSingle(self, inputFilName: str, outputFilenName: str, args: list):
        args.extend(
            ['--in_list', inputFilName, '--out_file', outputFilenName])
        self.te_.run('BlobStats', args)
        return outputFilenName
    
    def run(self, outputFileName: str, overlap: int = 1, 
            mpiPath: str = 'mpirun', mpiNProc: int = None,
            nproc: int = 1, filePrefix: str = 'BlobStats'):
        if mpiNProc is not None:
            nproc = mpiNProc
        inputFileNames = []
        inputFileChunks = []
        outputFileList = []
        with open(self.inputFileName_, 'r') as f:
            inputFileNames = f.readlines()
        
        nFile = len(inputFileNames)
        size = math.ceil(nFile / nproc)

        # One day overlap for stitching
        inputFileNameParts = self.inputFileName_.rsplit('.', 1)
        outputFileNameParts = outputFileName.rsplit('.', 1)
        for i in range(nproc):
            iInputFilename = f'{inputFileNameParts[0]}.{i}'
            if len(inputFileNameParts) > 1:
                iInputFilename = f'{iInputFilename}.{inputFileNameParts[1]}'
            startIndex = max(0, i*size-overlap)
            endIndex = min((i+1)*size+overlap, nFile)
            iInputContent = inputFileNames[startIndex:endIndex]
            with open(iInputFilename, 'w') as f:
                f.writelines(iInputContent)
            inputFileChunks.append(iInputFilename)
            iOutputFileName = f'{outputFileNameParts[0]}.{i}'
            if len(outputFileNameParts) > 1:
                iOutputFileName = f'{iOutputFileName}.{outputFileNameParts[1]}'
            outputFileList.append(iOutputFileName)

        if mpiNProc is not None:
            dirName = os.path.dirname(inputFileNameParts[0])
            wrapperName = os.path.join(dirName, f'{filePrefix}Wrapper.sh')
            currentOS = platform.system()
            executable = 'BlobStats'
            executable = f'{executable}.exe' if currentOS == 'Windows' else executable
            blobStatsPath = self.te_.findExecutablePath(executable)
            rankInputName = f'{inputFileNameParts[0]}.${{rank}}'
            if len(inputFileNameParts) > 1:
                rankInputName = f'{rankInputName}.{inputFileNameParts[1]}'
            rankOutputName = f'{outputFileNameParts[0]}.${{rank}}'
            if len(outputFileNameParts) > 1:
                rankOutputName = f'{rankOutputName}.{outputFileNameParts[1]}'
            argsStr = ''
            for i in self.args_:
                argsStr = f'{argsStr} {i}'
            argsStr = (f'--in_list "{rankInputName}" --out_file "{rankOutputName}" {argsStr}'
                        f' > "{dirName}/{filePrefix}.${{rank}}.log"')
            with open(wrapperName, 'w') as f:
                f.write(
                    ('#!/bin/bash -f\n'
                     'rank=$PMI_RANK\n'
                     f'{blobStatsPath} {argsStr}'))
            wrapperMode = os.stat(wrapperName).st_mode
            wrapperMode = wrapperMode | stat.S_IXUSR
            os.chmod(wrapperName, wrapperMode)
            cmd = [mpiPath, '-np', str(mpiNProc), wrapperName]
            logger.warning(cmd)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            while True:
                stdout = proc.stdout.readline()
                if stdout == '' and proc.poll() is not None:
                    break
                if stdout:
                    logger.info(stdout)
            if proc.wait() == 0:
                logger.warning(f'Job {filePrefix}Wrapper.sh succeed!')
            else:
                raise Exception(f'Job {filePrefix}Wrapper.sh failed!')
            self.outputFileNames_ = outputFileList
        else:
            with Pool(processes=nproc) as pool:
                self.outputFileNames_ = pool.starmap(
                    self.runSingle, zip(inputFileChunks, outputFileList, repeat(self.args_)))
            
        self.stitchResult(outputFileName, self.outputFileNames_, overlap=overlap)
            
    def stitchResult(self, outputFileName: str, inputFileNames: list, overlap: int = 1, 
                     renumber: bool = True):
        dfList = []
        df = pl.DataFrame(
            {'bid': [], 'iBlobOutputTime': [], 'content':[]}, 
            schema={
                'bid': pl.Int64,
                'iBlobOutputTime': pl.Int64,
                'content': pl.String
            })
        for iInputFileName in inputFileNames:
            try:
                iDf = pl.read_csv(
                    iInputFileName, has_header=False, infer_schema=False,
                    separator='\t', quote_char=None)
                iDf = iDf.with_columns(
                    pl.col('column_1').cast(pl.Int64).alias('column_1'),
                    pl.col('column_2').cast(pl.Int64).alias('column_2'))
                iDf = iDf.with_columns(
                    pl.concat_str(pl.all().exclude(['column_1', 'column_2']), separator='\t').alias(
                        'content')).drop(pl.all().exclude(['column_1', 'column_2', 'content'])).rename({
                            'column_1': 'bid',
                            'column_2': 'iBlobOutputTime'})
                dfList.append(iDf)
            except:
                if iInputFileName != inputFileNames[-1]:
                    raise Exception(f'File {iInputFileName} is either empty or broken.')
                else:
                    pass

        if overlap > 0:
            for i in range(1, len(dfList)):
                head = dfList[i].row(index=0)
                firstOverlap = dfList[i-1].with_row_index().filter(
                    pl.col('content')==head[2])
                removeLast = True
                if (firstOverlap.shape[0] > 1):
                    Exception('Cannot stitch BlobStats results.')
                elif (firstOverlap.shape[0] == 1):
                    lastBid = firstOverlap['bid'][0]
                    lastIndex = firstOverlap['index'][0]
                else:
                    lastBid = dfList[i-1].tail(1)['bid'][0]
                    lastIndex = dfList[i-1].with_row_index().tail(1)['index'][0]
                    removeLast = False
                
                if removeLast:
                    dfList[i-1] = dfList[i-1].with_row_index().filter(pl.col('index')<lastIndex).drop('index')
                if head[0] != lastBid and head[0] == 1:
                    dfList[i] = dfList[i].with_columns((pl.col('bid')+lastBid-1).alias('bid'))
                elif head[0] == lastBid + 1:
                    pass
                elif head[0] != 1:
                    Exception(
                        (f'File {inputFileNames[i]} has a non-zero bid {head[0]} but '
                        f'not match with the one in the overlaping record: {lastBid}'))
                logger.warning(f'Connecting point at bid: {lastBid}')
        df = pl.concat(dfList).with_columns(
            pl.col('bid').cast(pl.String), pl.col('iBlobOutputTime').cast(pl.String))
        df = df.with_columns(pl.concat_str(pl.all(), separator='\t').alias('content')).drop(['bid', 'iBlobOutputTime'])
        df.write_csv(outputFileName, include_header=False, quote_style='never')
