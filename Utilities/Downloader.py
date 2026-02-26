import sys
import os
from urllib.request import build_opener

from Base import Program

class Wget(Program):
    def __init__(self, path: str):
        super().__init__(self.__class__.__name__, path)

def download(url: str, dist: str, override: bool = False):
    ofile = os.path.basename(url)
    dir = os.path.dirname(dist)
    if not os.path.isdir(dir):
        sys.stdout.write(
            dir + ' does not exist, making one ... ')
        sys.stdout.flush()
        os.makedirs(dir)
        sys.stdout.write("done\n")
    if os.path.exists(dist) and not override:
        sys.stdout.write(ofile + ' exists. Skip.\n')
    else:
        sys.stdout.write('downloading ' + ofile + ' ... ')
        sys.stdout.flush()
        opener = build_opener()
        infile = opener.open(url)
        outfile = open(dist, 'wb')
        outfile.write(infile.read())
        outfile.close()
        sys.stdout.write('done\n')


def downloadWget(url: str, dist: str, override: bool = False):
    ofile = os.path.basename(url)
    dir = os.path.dirname(dist)
    wget = Wget('')
    if not os.path.isdir(dir):
        sys.stdout.write(
            dir + ' does not exist, making one ... ')
        sys.stdout.flush()
        os.makedirs(dir)
        sys.stdout.write("done\n")
    if os.path.exists(dist) and not override:
        sys.stdout.write(ofile + ' exists. Skip.\n')
    else:
        sys.stdout.write('downloading ' + ofile + ' ...\n')
        sys.stdout.flush()
        args = ['-O', dist, url]
        wget.run('wget', arg=args, flagMpi=False)
        sys.stdout.write('done\n')