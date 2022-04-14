from datetime import datetime
import argparse
import os
import json
import csv

from pprint import pprint
import pandas as pd
from tqdm import tqdm

from data_pipeline.historical_volume_data_recorder import HistoricalVolumeDataRecorder


if __name__ == '__main__':

    # read argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", dest="env", help="system environment", metavar="environment", default="local")
    args = parser.parse_args()
    if args.env == 'local':
        env = 'local'
    elif args.env == 'server':
        env = 'prod'
    
    # read config
    with open(f'{os.path.dirname(__file__)}/../../../config/volume_data_recorder_config.json') as f:
        configs = json.load(f)['Config'] # list of configs for each thread
    config = configs[0] # btc
    
    base_path = '/Users/tyson/dev/orichal/aggtrades_data/influx_feed'
    year_months = pd.date_range(start=datetime(2020,1,1), end=datetime(2022,2,28), freq='1M')
    ohlc_fnames = [f'BTCUSDT-aggTrades-{dt.year}-{str(dt.month).zfill(2)}-ohlc_df.csv' for dt in year_months]
    volume_fnames = [f'BTCUSDT-aggTrades-{dt.year}-{str(dt.month).zfill(2)}-volume_df.csv' for dt in year_months]
    
    # recorder = HistoricalVolumeDataRecorder(env, config)
    for ohlc_f, volume_f in zip(ohlc_fnames, volume_fnames):
        print(ohlc_f, volume_f)
        ohlc_fpath = f'{base_path}/{ohlc_f}'
        volume_fpath = f'{base_path}/{volume_f}'
        ohlc_df = pd.read_csv(ohlc_fpath)
        volume_df = pd.read_csv(volume_fpath)

        ohlc_df['T'] = pd.to_datetime(ohlc_df['T'], format='%Y-%m-%d %H:%M:%S', utc=True)
        volume_df['T'] = pd.to_datetime(volume_df['T'], format='%Y-%m-%d %H:%M:%S', utc=True)

        # write to db
        # recorder._write_to_db(volume_df, ohlc_df, verbose=True)

        # check for missing dates
        print('ohlc:' + str(len(ohlc_df)))
        maxdf, mindf = max(ohlc_df['T']), min(ohlc_df['T'])
        
        check_dts = list(ohlc_df['T'])
        all_dts = list(pd.date_range(start=mindf, end=maxdf, freq='1min'))
        pprint([dt for dt in all_dts if dt not in check_dts]) # slow tho; grab a coffee
        print('==================')
        # print('orderflow:' + str(len(volume_df)))
