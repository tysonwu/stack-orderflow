'''
Backend for updating GUI data.
Uses websocket server to receive message for real time updating the GUI.
'''

import argparse
import yaml
import os
from threading import Thread
from datetime import datetime
import time

import pandas as pd
import talib as ta

import finplot_lib as fplt

from data_pipeline.volume_data_recorder import VolumeDataRecorder
from data_pipeline.market_profile_reader import MarketProfileSlice, MarketProfileReader
from plotting_orderflow import OrderflowPlotter

class RealTimeDataStream(VolumeDataRecorder):

    def __init__(self, env, config):
        super().__init__(env, config, turn_on_ws=True)

        # data storages
        self.ohlcv = pd.DataFrame()
        self.mp_slice = []
        
    
    def connect(self):
        self.thread_connect = Thread(target=self.run)
        self.thread_connect.daemon = True
        self.thread_connect.start()        


    def _announce_minute_data(self, dt: datetime, ohlcv_df: pd.DataFrame, volume_df: pd.DataFrame, no_update: bool = False):
        if not no_update:
            ohlcv_df['T'] = pd.to_datetime(ohlcv_df['T'])
            ohlcv_df = ohlcv_df.set_index('T')
            ohlcv_df = ohlcv_df['o h l c v pot_ask pot_bid pot'.split()]

            self.ohlcv = pd.concat([self.ohlcv, ohlcv_df])
            ohlcv_data = ohlcv_df.loc[dt].to_dict()
            orderflow_row = volume_df['p q d b a'.split()].values
            self.mp_slice.append(MarketProfileSlice(self.inst, dt, ohlcv_data, orderflow_row))


    def _load_recent_candle_from_db(self):
        '''
        TODO: functionality to load from influxdb
        '''
        profile = MarketProfileReader()
                    

def realtime_update_plot():
    '''Called at regular intervals by a timer.'''

    # print(plots.keys())
    plotter.update_datasrc(ds.ohlcv, ds.mp_slice)
    plotter.calculate_plot_features()

    datasrc = {}
    for k in plotter.plots:
        if k in ['candlestick']:
            datasrc[k] = plotter.ohlcv[['o', 'c', 'h', 'l']]
        elif k in ['volume']:
            datasrc[k] = plotter.ohlcv[['o', 'c', 'v']]
        elif k in ['poc']:
            datasrc[k] = plotter.ohlcv['poc']
        elif k.startswith('ema'):
            datasrc[k] = plotter.ohlcv[k]
        elif k in ['delta_heatmap']:
            datasrc[k] = plotter.delta_heatmap
        elif k in ['bid_labels']:
            datasrc[k] = plotter.price_level_texts[['t', 'p', 'b']]
        elif k in ['ask_labels']:
            datasrc[k] = plotter.price_level_texts[['t', 'p', 'a']]
        elif k in ['pot_heatmap']:
            datasrc[k] = plotter.pot_heatmap
        elif k in ['pot_ask']:
            datasrc[k] = plotter.pot_df[['ts', 'ask_label_height', 'ask_label']]
        elif k in ['pot_bid']:
            datasrc[k] = plotter.pot_df[['ts', 'bid_label_height', 'bid_label']]
        elif k in ['macd_diff']:
            datasrc[k] = plotter.ohlcv[['o', 'c', 'macd_diff']]
        elif k in ['macd']:
            datasrc[k] = plotter.ohlcv['macd']
        elif k in ['macd_signal']:
            datasrc[k] = plotter.ohlcv['macd_signal']
        elif k in ['cvd']:
            datasrc[k] = plotter.cvd
        elif k in ['stochrsi_fastk']:
            datasrc[k] = plotter.ohlcv['fastk']
        elif k in ['stochrsi_fastd']:
            datasrc[k] = plotter.ohlcv['fastd']
        else:
            print(f'Warning: some plot objects not updated - {k}')

    for k in plotter.plots:
        plotter.plots[k].update_data(datasrc[k], gfx=False)

    for k in plotter.plots:
        plotter.plots[k].update_gfx()
    

if __name__ == '__main__':
    '''
    Real time streaming of aggTrade data. Data are transformed and written into influx directly.
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
    with open(f'{os.path.dirname(__file__)}config/volume_data_recorder_config.yaml') as f:
        configs = yaml.full_load(f)['Config'] # list of configs for each thread

    # process_list = []
    # for config in configs:
    #     volume_recorder_p = Process(target=main_ws, args=(env, config,), daemon=True)
    #     process_list.append(volume_recorder_p)
    #     volume_recorder_p.start()

    # for p in process_list:
    #     p.join()

    # single asset
    config = configs[0] # single asset
    ds = RealTimeDataStream(env, config)
    '''
    TODO: if need to load data from influxdb, load here
    '''
    ds.connect()

    plotter = OrderflowPlotter(
        token=ds.token, 
        interval=ds.resample_period, 
        increment=ds.round_px_to_nearest, 
        ohlcv=ds.ohlcv, 
        mp_slice=ds.mp_slice
    )

    while not plotter.plots: # plot objects that has to be updated every timer callback trigger
        print('Awaiting data...')
        plotter.update_datasrc(ds.ohlcv, ds.mp_slice)
        plotter.orderflow_plot()
        time.sleep(1)

    print('Received first data point. Plot initiated.')

    '''Also refer to example/complicated.py'''    
    fplt.timer_callback(realtime_update_plot, 1) # update every second
    fplt.show()