import logging
import os
import subprocess
from typing import Any, Optional, Sequence

from Base import Program

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TempestExtremes(Program):
    def __init__(self, path: str, mpiPath: Optional[str] = '',
                 mpiArg: Optional[Sequence] = []):
        super().__init__(self.__class__.__name__, path, mpiPath, mpiArg)
