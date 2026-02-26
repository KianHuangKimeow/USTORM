import logging
import os
import platform
import shutil
import subprocess
from typing import Optional, Sequence

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Program:
    def __init__(self, name: str, path: str, mpiPath: Optional[str] = '',
                 mpiArg: Optional[Sequence] = []):
        self.name_ = name
        if os.path.exists(path):
            self.path_ = path
        else:
            self.path_ = None
        self.setMPI(mpiPath,
                    mpiArg)
        self.exitMark_ = None

    def setExitMark(self, mark: str, loc: int) -> None:
        self.exitMark_ = mark
        self.exitMarkLoc_ = loc

    def setMPI(self, mpiPath: str,
               mpiArg: Optional[Sequence] = []) -> None:
        if mpiPath is None:
            mpiPath = ''
        if len(mpiPath) > 0:
            if os.path.exists(mpiPath):
                self.mpiPath_ = mpiPath
            else:
                logging.error(f'{mpiPath} does not exist!')
        self.mpiPath_ = mpiPath
        self.mpiArg_ = mpiArg

    def findExecutablePath(self, executable):
        if self.path_ is not None:
            return os.path.join(self.path_, executable)
        else:
            return shutil.which(executable)

    def run(self, executable, arg: Optional[Sequence] = [],
            flagMpi: bool = True) -> None:
        currentOS = platform.system()
        executable = f'{executable}.exe' if currentOS == 'Windows' else executable
        executablePath =  self.findExecutablePath(executable)
        if not os.path.exists(executablePath):
            raise Exception(f'{executablePath} does not exist!')
        if len(self.mpiPath_) > 0 and flagMpi:
            cmd = [self.mpiPath_, *self.mpiArg_, executablePath, *arg]
        else:
            cmd = [executablePath, *arg]
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
            logger.warning(f'Job {self.name_}/{executable} succeed!')
        else:
            raise Exception(f'Job {self.name_}/{executable} failed!')
