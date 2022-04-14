from datetime import datetime, timezone
from pytest import Instance
import pytz

import pandas as pd
import matplotlib.pyplot as plt

import finplot_lib as fplt

from data_pipeline.market_profile_reader import MarketProfileReader


def create_observation_matrix(mp_slice):
    pass

if __name__ == '__main__':
    inst = 'btcusdt'
    # input in HKT
    start = datetime(2021, 12, 10, 0, 0, 0)
    end = datetime(2021, 12, 12, 0, 0, 0)

    profile = MarketProfileReader()
    profile.load_data_from_influx(inst=inst, start=start, end=end, env='local')
    
    # slice_dt = pytz.timezone('Asia/Hong_Kong').localize(datetime(2022,3,17,17,23,0)) # input in HKT
    slice_start = pytz.timezone('Asia/Hong_Kong').localize(datetime(2021,12,10,0,0,0)) # input in HKT
    slice_end = pytz.timezone('Asia/Hong_Kong').localize(datetime(2021,12,12,0,0,0)) # input in HKT
    
    # mp_slice = profile[slice_dt]
    mp_slice = profile[slice_start:slice_end]

    ochl = pd.DataFrame(
        {
            'o': [mp.open for mp in mp_slice],
            'c': [mp.close for mp in mp_slice],
            'h': [mp.high for mp in mp_slice],
            'l': [mp.low for mp in mp_slice],
            # 'v': [mp.volume_qty for mp in mp_slice]
        },
        index=[mp.timepoint for mp in mp_slice]
    )
    
    # ohlcv.plot(y='o')
    # plt.show()
    fplt.candlestick_ochl(datasrc=ochl, candle_width=0.2)
    fplt.show()