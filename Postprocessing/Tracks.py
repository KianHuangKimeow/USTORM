import datetime

import polars as pl

def extractCurrentTrack(
    df: pl.DataFrame, currentTime: datetime.datetime,
    tidName: str = 'TID'):
    currentTid = df.filter(pl.col('UTC') == currentTime)[tidName]
    subDf = df.filter(
        (pl.col('UTC') <= currentTime) & 
        (pl.col(tidName).is_in(currentTid))
    )
    return subDf