
from threading import Thread, Lock
from datetime import datetime, timedelta
import pytz

import pandas as pd
import numpy as np
from unicorn_binance_websocket_api import BinanceWebSocketApiManager
from unicorn_fy import UnicornFy


from market_profile import MarketProfileSlice


'''
Backend for updating GUI data.
Stream and update aggTrades data from Binance given an instrument name.
Uses websocket server to receive message for real time updating the GUI.
'''

class RealTimeDataStream:

    def __init__(self, config):
        self._spot_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.com")
        self.data = []

        # set config attribute
        self.inst = config['inst']
        self.channel = config['channel'] 
        self.token = config['token']
        self.resample_period = config['resample_period']
        # currently only support 1 min resample!
        assert self.resample_period == '1min', 'Reample period has to be `1min`'
        self.round_px_to_nearest = float(config['round_px_to_nearest'])

        # data storages
        self.ohlcv = pd.DataFrame()
        self.mp_slice = []
        self._lock = Lock()
    

    def connect(self):
        self.thread_connect = Thread(target=self.run, args=(True,))
        self.thread_connect.daemon = True
        self.thread_connect.start()        


    def run(self):
        '''
        See reference from `unicorn_binance_websocket_api.manager`
        https://github.com/LUCIT-Systems-and-Development/unicorn-binance-websocket-api/blob/master/unicorn_binance_websocket_api/manager.py
        '''
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
                    t = datetime.utcfromtimestamp(int(data['event_time']) // 1000).replace(tzinfo=pytz.utc)
                    p =  float(data['price'])
                    if data['is_market_maker']:
                        q = -float(data['quantity'])
                    else:
                        q = float(data['quantity'])

                    # check if we it's time to call influx to write and clean up self.data
                    if t.minute != current_minute:
                        ohlc_df, volume_df = self._transform_feed_and_flush()
                        candle_time = max(ohlc_df['T'])
                        self._announce_minute_data(candle_time, ohlc_df, volume_df, no_update=first_incomplete_candle)
                        current_minute = t.minute
                        first_incomplete_candle = False

                    self.data.append([t, p, q])
                    # print(f'{t} {p=} {q=}, {len(self.data)=}')


    def _announce_minute_data(self, dt: datetime, ohlcv_df: pd.DataFrame, volume_df: pd.DataFrame, no_update: bool = False):
        with self._lock:
            if not no_update:
                print(dt)
                ohlcv_df['T'] = pd.to_datetime(ohlcv_df['T'])
                ohlcv_df = ohlcv_df.set_index('T')
                ohlcv_df = ohlcv_df['o h l c v pot_ask pot_bid pot'.split()]

                self.ohlcv = pd.concat([self.ohlcv, ohlcv_df])
                ohlcv_data = ohlcv_df.loc[dt].to_dict()
                orderflow_row = volume_df['p q d b a'.split()].values
                self.mp_slice.append(MarketProfileSlice(self.inst, dt, ohlcv_data, orderflow_row))

                # always only store recent n timepoints in datastream
                self._flush_old_data(keep=10_000)


    def _flush_old_data(self, keep: int):
        if len(self.ohlcv) > keep:
            self.ohlcv = self.ohlcv.iloc[-keep:]
            self.mp_slice = self.mp_slice[-keep:]


    def _load_recent_candle_from_db(self, load_recent=30):
        '''
        functionality to load from influxdb
        load recent n minutes of candles
        '''
        profile = MarketProfileReader()
        end = datetime.now()
        start = end - timedelta(minutes=load_recent) # in local time
        profile.load_data_from_influx(inst=self.inst, start=start, end=end, env=self.env)
        slice_start = pytz.timezone('Asia/Hong_Kong').localize(start) # input in local time
        slice_end = pytz.timezone('Asia/Hong_Kong').localize(end) # input in local time
        mp_slice = profile[slice_start:slice_end]
        # override ohlcv df and mp_slice list, which are suppsoed to be empty before conencting to ws
        self.ohlcv = pd.DataFrame({
            'o': [mp.open for mp in mp_slice],
            'h': [mp.high for mp in mp_slice],
            'l': [mp.low for mp in mp_slice],
            'c': [mp.close for mp in mp_slice],
            'v': [mp.volume_qty for mp in mp_slice],
            'pot': [mp.pot for mp in mp_slice],
            'pot_ask': [mp.pot_ask for mp in mp_slice],
            'pot_bid': [mp.pot_bid for mp in mp_slice],
        }, index=[mp.timepoint for mp in mp_slice])
        self.mp_slice = profile[slice_start:slice_end]
        print(f'Loaded recent {len(self.ohlcv)} timepoint data from InfluxDB.')


    def _transform_feed_and_flush(self, verbose=True):
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

            # if not no_write:
            #     self._write_to_db(volume_df=volume_df, ohlc_df=ohlc_df, verbose=verbose)
            #     if verbose:
            #         print(f'{datetime.utcnow()} - Finished writing, flushed data queue')

            if verbose:
                print(f'{datetime.utcnow()} - Flushed data queue')
            
            # reset data array
            self.data = []
            return ohlc_df, volume_df
        else:
            return
            

    @staticmethod
    def round_to(pxs: float, nearest: int):
        return nearest * np.round(pxs / nearest)
