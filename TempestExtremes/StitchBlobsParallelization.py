import logging
import math
import os
import platform
import shutil
import stat
import subprocess

from TempestExtremes import TempestExtremes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class StitchBlobsParallelization:
    def __init__(
            self, te: TempestExtremes, inputFileName: str = None, args: list = None):
        self.te_ = te
        self.inputFileName_ = inputFileName
        self.args_ = args
        self.outputFileNameRanges_ = []
        self.outputFileNameLists_ = []
        self.copyFilesNames_ = []
    
    def run(self, outputFileName: str, overlap: int = 1, mpiPath: str = 'mpirun', 
            mpiNProc: int = 1, filePrefix: str = 'StitchBlobs', 
            listOnly: bool = False):
        inputFileNames = []
        outputFileNames = []
        inputFileChunks = []
        self.outputFileNameLists_ = []
        with open(self.inputFileName_, 'r') as f:
            inputFileNames = f.readlines()
        with open(outputFileName, 'r') as f:
            outputFileNames = f.readlines()
        
        nFile = len(inputFileNames)
        size = math.ceil(nFile / mpiNProc)

        inputFileNameParts = self.inputFileName_.rsplit('.', 1)
        outputFileNameParts = outputFileName.rsplit('.', 1)
        for i in range(mpiNProc):
            iInputFilename = f'{inputFileNameParts[0]}.{i}'
            if len(inputFileNameParts) > 1:
                iInputFilename = f'{iInputFilename}.{inputFileNameParts[1]}'
            startIndex = max(0, i*size-overlap)
            endIndex = min((i+1)*size+overlap, nFile)
            iInputContent = inputFileNames[startIndex:endIndex]
            startIndexFinal = overlap if i > 0 else 0
            endIndexFinal = len(iInputContent)-overlap if i < (mpiNProc - 1) else len(iInputContent)
            if overlap > 0:
                copyIndex = startIndexFinal + overlap if i > 0 else 0
                for j in range(0, copyIndex):
                    iContentParts = iInputContent[j].rsplit('.', 1)
                    iContent = f'{iContentParts[0]}.{i}'
                    if len(iContentParts) > 0:
                        iContent = f'{iContent}.{iContentParts[1]}'
                    logger.warning(f'Making a copy for {iInputContent[j]}')
                    shutil.copy(iInputContent[j].strip(), iContent.strip())
                    iInputContent[j] = iContent
                    self.copyFilesNames_.append(iContent)
                
            with open(iInputFilename, 'w') as f:
                f.writelines(iInputContent)
            inputFileChunks.append(iInputFilename)
            iOutputFileName = f'{outputFileNameParts[0]}.{i}'
            if len(outputFileNameParts) > 1:
                iOutputFileName = f'{iOutputFileName}.{outputFileNameParts[1]}'
            iOutputContent = outputFileNames[startIndex:endIndex]
            if overlap > 0 and not listOnly:
                for j in range(len(iOutputContent)):
                    iContentParts = iOutputContent[j].rsplit('.', 1)
                    iContent = f'{iContentParts[0]}.{i}'
                    if len(iContentParts) > 0:
                        iContent = f'{iContent}.{iContentParts[1]}'
                    iOutputContent[j] = iContent
            self.outputFileNameRanges_.append(
                [startIndexFinal, endIndexFinal])
            with open(iOutputFileName, 'w') as f:
                f.writelines(iOutputContent)
            self.outputFileNameLists_.append(iOutputFileName)
        
        if listOnly:
            return self.outputFileNameLists_

        dirName = os.path.dirname(inputFileNameParts[0])
        wrapperName = os.path.join(dirName, f'{filePrefix}Wrapper.sh')
        currentOS = platform.system()
        executable = 'StitchBlobs'
        executable = f'{executable}.exe' if currentOS == 'Windows' else executable
        executablePath =  self.te_.findExecutablePath(executable)
        rankInputName = f'{inputFileNameParts[0]}.${{rank}}'
        if len(inputFileNameParts) > 1:
            rankInputName = f'{rankInputName}.{inputFileNameParts[1]}'
        rankOutputName = f'{outputFileNameParts[0]}.${{rank}}'
        if len(outputFileNameParts) > 1:
            rankOutputName = f'{rankOutputName}.{outputFileNameParts[1]}'
        argsStr = ''
        for i in self.args_:
            argsStr = f'{argsStr} {i}'
        argsStr = (f'--in_list "{rankInputName}" --out_list "{rankOutputName}" {argsStr}'
                   f' > "{dirName}/{filePrefix}.${{rank}}.log"')
        
        with open(wrapperName, 'w') as f:
            f.write(('#!/bin/bash   \n'
                     'rank=$PMI_RANK\n'
                     'echo $rank\n'
                     f'{mpiPath} -np 1 {executablePath} {argsStr}'))
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
        for i in self.copyFilesNames_:
            if os.path.exists(i):
                os.remove(i)
        if proc.wait() == 0:
            logger.warning(f'Job {filePrefix}Wrapper.sh succeed!')
        else:
            raise Exception(f'Job {filePrefix}Wrapper.sh failed!')
        
        return self.outputFileNameLists_
        
    def rename(self, outputFileList: list):
        for i in range(len(outputFileList)):
            content = []
            startIndex, endIndexFinal = self.outputFileNameRanges_[i]
            with open(outputFileList[i], 'r') as f:
                content = f.readlines()
            for j in range(startIndex, endIndexFinal):
                filename = content[j].strip()
                fileNameParts = filename.rsplit(f'.{i}', 1)
                filenameNew = fileNameParts[0]
                if len(fileNameParts) > 1:
                    filenameNew = f'{filenameNew}{fileNameParts[1]}'
                logger.warning(f'Renaming {filename} to {filenameNew}')
                os.rename(filename, filenameNew)
            
        
