import getpass
import logging
import os
import platform
import socket

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def detectMachine() -> str:
    host = socket.getfqdn().lower()
    logger.warning(f'The host is {host}')
    machineID = ''
    if 'derecho.hpc.ucar.edu' in host:
        machineID = 'derecho'
    elif 'wasatch.peaks' in host or 'chpc.utah.edu' in host:
        machineID = 'utah'
    else:
        machineID = 'unknown'
    return machineID


def defineScratch() -> str:
    machineID = detectMachine()
    username = getpass.getuser()
    currentOS = platform.system()
    scratchDir = ''
    if machineID == 'derecho':
        scratchDir = os.path.join('/glade/derecho/scratch', username)
    elif machineID == 'utah':
        scratchDir = os.path.join('/scratch/general/nfs1', username)
    else:
        homeEnvVar = ''
        if currentOS == 'Windows':
            homeEnvVar = 'USERPROFILE'
        else:
            homeEnvVar = 'HOME'
        scratchDir = os.environ[homeEnvVar]
    return scratchDir


def findHomeDir() -> str:
    currentOS = platform.system()
    if currentOS == 'Windows':
        homeEnvVar = 'USERPROFILE'
    else:
        homeEnvVar = 'HOME'
    homeDir = os.environ[homeEnvVar]
    return homeDir
