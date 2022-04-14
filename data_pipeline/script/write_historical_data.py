import argparse
from datetime import datetime
import os
import json

import pandas as pd

from data_pipeline.historical_volume_data_recorder import HistoricalVolumeDataRecorder

if __name__ == '__main__':
    '''
    Read raw aggTrade data and transform it into candlestick and volume data by price levels.
    Output to csv in the same directory as raw data.
    csv of orderflow data with suffix -volume_df and candlestick with suffix -ohlc_df respectively.
    '''
    # read argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", dest="env", help="system environment", metavar="environment", default="local")
    args = parser.parse_args()
    if args.env == 'local':
        env = 'local'
    elif args.env == 'server':
        env = 'prod'
    
    # read config
    with open(f'{os.path.dirname(__file__)}/../../config/volume_data_recorder_config.json') as f:
        configs = json.load(f)['Config'] # list of configs for each thread
    config = configs[0] # btc
    
    base_path = '/Users/tyson/dev/orichal/aggtrades_data/btcusdt'
    year_months = pd.date_range(start=datetime(2020,1,1), end=datetime(2022,2,28), freq='1M')
    fnames = [f'BTCUSDT-aggTrades-{dt.year}-{str(dt.month).zfill(2)}.csv' for dt in year_months]
    
    recorder = HistoricalVolumeDataRecorder(env, config)
    for fname in fnames:
        print(fname)
        fpath = f'{base_path}/{fname}'
        recorder.read_and_write(fpath=fpath, write_to_csv=True)
        print(f'{datetime.now()} - Done writing {fpath}')
