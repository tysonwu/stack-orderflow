from datetime import datetime

import numpy as np
import pandas as pd
from unicorn_binance_websocket_api import BinanceWebSocketApiManager
from unicorn_fy import UnicornFy

from influx_db_handler import InfluxDbHandler

class VolumeDataRecorder:

    def __init__(self, env, config, turn_on_ws=True):
        if turn_on_ws:
            self._spot_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.com")
        self.data = []
        self.env = env
        self.influxdb_handler = InfluxDbHandler(env=self.env, logger=None)

        # set config attribute
        self.inst = config['inst']
        self.channel = config['channel'] 
        self.token = config['token']
        self.resample_period = config['resample_period']

        # currently only support 1 min resample!
        assert self.resample_period == '1min', 'Reample period has to be `1min`'

        self.round_px_to_nearest = float(config['round_px_to_nearest'])


    def run(self):
        '''
        See reference from `unicorn_binance_websocket_api.manager`
        https://github.com/LUCIT-Systems-and-Development/unicorn-binance-websocket-api/blob/master/unicorn_binance_websocket_api/manager.py
        '''
        # self._time_clock_handler.timeClockEvent += self._transform_feed
        self._spot_websocket_api_manager.create_stream([self.channel], [self.inst])

        # check if it is an incomplete candle (because not ran from 0th second)
        first_incomplete_candle = True if datetime.now().second != 0 else False

        current_minute = datetime.now().minute
        while True:
            # with self._lock:
            oldest_data_from_stream_buffer = self._spot_websocket_api_manager.pop_stream_data_from_stream_buffer()

            # callback
            if oldest_data_from_stream_buffer:
                data = UnicornFy.binance_com_websocket(oldest_data_from_stream_buffer)
                if 'result' not in data: # first line from websocket is a trivial dict = {'result': None, 'id': 1, 'unicorn_fied': ['binance.com', '0.11.0']}
                    t = datetime.utcfromtimestamp(int(data['event_time']) // 1000)
                    p =  float(data['price'])
                    if data['is_market_maker']:
                        q = -float(data['quantity'])
                    else:
                        q = float(data['quantity'])

                    # check if we it's time to call influx to write and clean up self.data
                    if t.minute != current_minute:
                        ohlc_df, volume_df = self._transform_feed_and_write_and_flush(no_write=first_incomplete_candle)
                        candle_time = max(ohlc_df['T'])
                        self._announce_minute_data(candle_time, ohlc_df, volume_df, no_update=first_incomplete_candle)
                        current_minute = t.minute
                        first_incomplete_candle = False

                    self.data.append([t, p, q])
                    # print(f'{t} {p=} {q=}, {len(self.data)=}')


    def _announce_minute_data(self, dt, ohlcv_df, volume_df, no_update):
        pass


    def _write_to_db(self, volume_df, ohlc_df, verbose):
        if verbose:
            print(volume_df)
            print(ohlc_df)
            print(f'{datetime.utcnow()} - Writing ohlcv data, orderflow data into influx...')

        # then write data via influx API
        self.influxdb_handler.write_from_dataframe(
            bucket='orderflow-data',
            df=volume_df,
            time_col='T', 
            measurement='orderflow', 
            tags=['s', 'p'], 
            fields=['q', 'd', 'a', 'b'], 
            precision='s',
            )

        self.influxdb_handler.write_from_dataframe(
            bucket='ohlcv-data',
            df=ohlc_df,
            time_col='T', 
            measurement='ohlcv', 
            tags=['s'], 
            fields=['o', 'h', 'l', 'c', 'v', 'pot_ask', 'pot_bid', 'pot'], 
            precision='s',
            )


    def _transform_feed_and_write_and_flush(self, no_write=False, verbose=True):
        # with self._lock:
        if self.data:
            # when the current minute finishes, do the following:

            # to conform into influx format
            df = pd.DataFrame(self.data, columns=['T', 'p', 'q'])

            # calculate tape speed
            pot_ask = len(df[df['q'] >= 0])
            pot_bid = len(df[df['q'] < 0])
            pot = pot_ask + pot_bid

            # HH:MM:SS means the data between HH:MM:00 to HH:MM:59; aka. the datetime is the open time
            ohlc_df = df.copy()
            ohlc_df['q'] = ohlc_df['q'].abs()
            ohlc_df = ohlc_df.resample(self.resample_period, on='T', origin='epoch', label='left').agg({'p': ['first', 'max', 'min', 'last'], 'q': 'sum'}).sort_index()
            ohlc_df.columns = ohlc_df.columns.get_level_values(1)
            ohlc_df = ohlc_df.rename(columns={'first': 'o', 'max': 'h', 'min': 'l', 'last': 'c', 'sum': 'v'})
            ohlc_df['v'] = np.round(ohlc_df['v'], 8)
            ohlc_df['s'] = self.inst

            # PoT is Pace of Tape - trade intensity
            ohlc_df['pot_ask'] = pot_ask
            ohlc_df['pot_bid'] = pot_bid
            ohlc_df['pot'] = pot
            ohlc_df = ohlc_df.reset_index(drop=False)

            df['p'] = self.round_to(df['p'], nearest=self.round_px_to_nearest)
            delta_df = df.groupby(['p']).resample(self.resample_period, on='T', origin='epoch', label='left').agg({'q': 'sum'}).reset_index().sort_values(['T','p'])#.pivot(columns=['T'], values='q', index='p')
            volume_df = df.copy()
            volume_df['q'] = volume_df['q'].abs()
            volume_df = volume_df.groupby(['p']).resample(self.resample_period, on='T', origin='epoch', label='left').agg({'q': 'sum'}).reset_index().sort_values(['T','p'])#.pivot(columns=['T'], values='q', index='p')
            volume_df = volume_df[volume_df['q'] != 0.0]                    

            '''
            q: quantity(volume): ask + bid
            d: delta: ask - bid
            b: (q - d) / 2 = bid
            a: (q + d) / 2 = ask
            '''
            volume_df['p'] = np.round(volume_df['p'], 8)
            volume_df['q'] = np.round(volume_df['q'], 8)
            volume_df['d'] = np.round(delta_df['q'], 8)
            volume_df['a'] = np.round((volume_df['q'] + volume_df['d']) / 2, 8)
            volume_df['b'] = np.round((volume_df['q'] - volume_df['d']) / 2, 8)
            volume_df['s'] = self.inst

            if not no_write:
                self._write_to_db(volume_df=volume_df, ohlc_df=ohlc_df, verbose=verbose)
                if verbose:
                    print(f'{datetime.utcnow()} - Finished writing, flushed data queue')
            else:
                if verbose:
                    print(f'{datetime.utcnow()} - Discarded current candle as {no_write=}, flushed data queue')
            
            # reset data array
            self.data = []
            return ohlc_df, volume_df
        else:
            return
            

    @staticmethod
    def round_to(pxs: float, nearest: int):
        return nearest * np.round(pxs / nearest)
