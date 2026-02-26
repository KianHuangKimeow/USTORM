import datetime
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class InputNameBase:
    def __init__(self, root, step: int = 24):
        if isinstance(root, dict):
            for k, v in root.items():
                if not os.path.exists(v):
                    logger.error(f'{k}: {v} does not exist!')
            self.root_ = root
        else:
            if os.path.exists(root):
                self.root_ = root
            else:
                logger.error(f'{root} does not exist!')
        
        self.inputNameList_ = ''
        self.invariantPaths_ = {}
        self.step_ = datetime.timedelta(hours=step)

    def setDateTime(self, begin: datetime.datetime, end: datetime.datetime):
        assert (begin <= end)
        self.begin_ = begin
        self.end_ = end

    def addInvariantPath(self, var, filename):
        self.invariantPaths_[var] = filename

    def generateInput(self, variables: list = [], output: bool = False):
        return self.inputNameList_
    
    def getInputAsList(self):
        return self.inputNameList_.splitlines()

    def dump(self, filename):
        basename = os.path.basename(filename)
        if not os.path.exists(filename):
            logger.warning(f'{basename} does not exist! Creating a new one...')
        with open(filename, "w") as f:
            f.write(self.inputNameList_)

    def getInvariantPaths(self):
        return self.invariantPaths_

    def replace(self, old, new):
        self.inputNameList_ = self.inputNameList_.replace(old, new)