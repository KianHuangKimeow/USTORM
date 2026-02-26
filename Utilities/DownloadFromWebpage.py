import datetime
import hashlib
import os
import re
import sys
from typing import Optional, Sequence
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from .Downloader import download

def findFilesFromUrl(url: str, reStr, cacheDir: Optional[str] = None):
    foundCache = ''
    expiredCache = ''
    if cacheDir is not None:
        currentTime = datetime.datetime.now()
        urlHash = hashlib.md5(url.encode('utf-8')).hexdigest()
        if not os.path.isdir(cacheDir):
            os.makedirs(cacheDir)

        for filename in os.listdir(cacheDir):
            match = re.search(urlHash, filename)
            if match:
                idx = match.end() + 1
                try:
                    cacheTimeStr = filename[idx::].split('.', 1)[0]
                    cacheTime = datetime.datetime.strptime(cacheTimeStr, '%Y%m%d_%H%M%S')
                    if currentTime - cacheTime < datetime.timedelta(hours=3):
                        foundCache = os.path.join(cacheDir, filename)
                    else:
                        expiredCache = os.path.join(cacheDir, filename)
                        sys.stdout.write(
                            'Cache file ' + expiredCache + ' has expired, deleting ...\n')
                        sys.stdout.flush()
                        os.remove(expiredCache)
                except:
                    pass

    if os.path.exists(foundCache):
        with open(foundCache) as f:
            bs = BeautifulSoup(f, 'html.parser')
    else:
        url = url.replace(' ', '%20')
        request = Request(url)
        response = urlopen(request).read()
        bs = BeautifulSoup(response, 'html.parser')
        
    results = bs.find_all(href=re.compile(reStr))
    urlList = []
    fileList = []
    for i in results:
        iUrl = f'{url}/{i.get('href')}'
        urlList.append(iUrl)
        fileList.append(i.text)

    if cacheDir is not None:
        if len(foundCache) < 1:
            cacheFile = os.path.join(cacheDir, f'{urlHash}_{currentTime:%Y%m%d_%H%M%S}.html')
            with open(cacheFile, 'w') as f:
                f.write(str(bs))

    return urlList, fileList


def downloadFromWebpage(url: str, distDir: str, reStr: str, 
                        override: Optional[bool] = False,
                        cacheDir: Optional[str] = None) -> Sequence:
    urlList, fileList = findFilesFromUrl(url, reStr, cacheDir)
    finalList = []
    for i, j in zip(urlList, fileList):
        dist = os.path.join(distDir, j)
        download(i, dist, override)
        finalList.append(j)
    return finalList
