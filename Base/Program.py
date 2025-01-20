import logging
import os
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
            logging.error(f'{path} does not exist!')
        self.setMPI(mpiPath,
                    mpiArg)
        self.exitMark_ = None

    def setExitMark(self, mark: str, loc: int) -> None:
        self.exitMark_ = mark
        self.exitMarkLoc_ = loc

    def setMPI(self, mpiPath: str,
               mpiArg: Optional[Sequence] = []) -> None:
        if len(mpiPath) > 0:
            if os.path.exists(mpiPath):
                self.mpiPath_ = mpiPath
        else:
            logging.error(f'{mpiPath} does not exist!')
        self.mpiPath_ = mpiPath
        self.mpiArg_ = mpiArg

    def run(self, executable, arg: Optional[Sequence] = [],
            exitMark: Optional[str] = None, exitMarkLoc: Optional[int] = None) -> None:
        executablePath = os.path.join(self.path_, executable)
        if not os.path.exists(executablePath):
            raise Exception(f'{executablePath} does not exist!')
        cmd = [self.mpiPath_, *self.mpiArg_, executablePath, *arg]
        logger.warning(cmd)
        ret = subprocess.run(cmd, capture_output=True, text=True)
        stdout = ret.stdout.strip('\n')
        stderr = ret.stderr.strip('\n')
        if (exitMark is not None) and (exitMarkLoc is not None):
            if stdout[exitMarkLoc] == exitMark:
                logger.warning(f'Job {self.name_}/{executable} succeed!')
            else:
                logger.error(stdout)
                logger.error(stderr)
                raise Exception(f'Job {self.name_}/{executable} failed!')
        else:
            if ret.returncode == 0:
                logger.warning(f'Job {self.name_}/{executable} succeed!')
            else:
                logger.error(stdout)
                logger.error(stderr)
                raise Exception(f'Job {self.name_}/{executable} failed!')
