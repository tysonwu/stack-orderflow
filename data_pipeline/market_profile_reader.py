from datetime import datetime
import pytz

import numpy as np
import pandas as pd

from pprint import pprint

from influx_db_handler import InfluxDbHandler
from data_pipeline.market_profiler import find_poc

class MarketProfileSlice:

    def __init__(self, inst: str, timepoint: datetime, ohlcv: dict, orderflow: np.ndarray):
        '''
        ohlcv
        o: open
        h: high
        l: low
        c: close
        v: volume
        pot_ask: pace of tape of ask
        pot_bid: pace of tape of bid
        pot: pace of tape

        orderflow
        p: price level
        q: total volume
        d: volume delta
        b: bid volume
        a: ask volume
        '''
        self.inst = inst
        self.timepoint = timepoint

        # read ohlcv data
        self.open = ohlcv['o']
        self.high = ohlcv['h']
        self.low = ohlcv['l']
        self.close = ohlcv['c']
        self.volume_qty = ohlcv['v']
        # pace of tape
        self.pot = ohlcv['pot']
        self.pot_ask = ohlcv['pot_ask']
        self.pot_bid = ohlcv['pot_bid']

        self.open_range = (self.open, self.close)
        self.profile_range = (self.high, self.low)
        self.mid_price = round((self.high + self.low) / 2, 8)

        # read orderflow data
        self.n_levels = len(orderflow)
        self.price_levels = np.array([row[0] for row in orderflow])
        self.price_levels_range = (max(self.price_levels), min(self.price_levels))
        self.delta_qty = round(sum([row[2] for row in orderflow]), 8)
        self.total_bid_qty = round(sum([row[3] for row in orderflow]), 8)
        self.total_ask_qty = round(sum([row[4] for row in orderflow]), 8)

        self.profile = pd.DataFrame({'q': [row[1] for row in orderflow]}, index=self.price_levels)
        self.delta_profile = pd.DataFrame({timepoint: [row[2] for row in orderflow]}, index=self.price_levels)
        self.bidask_profile = pd.DataFrame({'b': [row[3] for row in orderflow], 'a': [row[4] for row in orderflow]}, index=self.price_levels)
        # calc POC
        self.poc_volume, self.poc_price_level = find_poc(self.profile)

        # print all local attributes
        # pprint(vars(self))


class MarketProfileReader:
    
    def __init__(self):
        self.inst = None
        self.timepoint_range = [] # soon be a list
        self.ohlcv_data = None # soon be a pd.DataFrame
        self.orderflow_data = {} # soon be a dict


    def __getitem__(self, date):

        if isinstance(date, datetime):
            if date not in self.timepoint_range:
                raise KeyError(f'{date} not found in market profile.')
            ohlcv_data = self.ohlcv_data.loc[date].to_dict()
            orderflow_data = self.orderflow_data[date]
            return MarketProfileSlice(self.inst, date, ohlcv_data, orderflow_data)
        elif isinstance(date, slice):
            '''
            return slice between start and end inclusive
            '''
            start = date.start
            end = date.stop
            if not isinstance(start, datetime) or not isinstance(end, datetime):
                raise TypeError("Slice must start and end with datetime.")
            result = []
            for date in self.timepoint_range:
                if start <= date <= end:
                    ohlcv_data = self.ohlcv_data.loc[date].to_dict()
                    orderflow_data = self.orderflow_data[date]                    
                    result.append(MarketProfileSlice(self.inst, date, ohlcv_data, orderflow_data))
            return result
        else:
            raise TypeError("Key must be datetime object.")


    def _convert_records_to_slice(self, ohlcv_generator, orderflow_generator): # make market slice generator from these two generators
        print(f'{datetime.now()} Start iterating for ohlcv data')
        ohlcv_data = []
        for record in ohlcv_generator:
            self.timepoint_range.append(record['_time'])
            ohlcv_data.append(
                [
                    float(record['o']), 
                    float(record['h']), 
                    float(record['l']), 
                    float(record['c']),
                    float(record['v']),
                    float(record['pot_ask']),
                    float(record['pot_bid']),
                    float(record['pot']),
                ]
            )

        self.ohlcv_data = pd.DataFrame(ohlcv_data, columns=['o', 'h', 'l', 'c', 'v', 'pot_ask', 'pot_bid', 'pot'], index=self.timepoint_range)
        
        print(f'{datetime.now()} Start iterating for orderflow data')
        for record in orderflow_generator:
            if (t := record['_time']) in self.timepoint_range:
                if t in self.orderflow_data:
                    self.orderflow_data[t] = np.append(self.orderflow_data[t], np.array([
                        [
                            round(float(record['p']), 8), # don't want precision error
                            float(record['q']),
                            float(record['d']),
                            float(record['b']),
                            float(record['a']),
                        ]
                    ]), axis=0)
                else:
                    self.orderflow_data[t] = np.array([
                        [
                            round(float(record['p']), 8), # don't want precision error
                            float(record['q']),
                            float(record['d']),
                            float(record['b']),
                            float(record['a']),
                        ]
                    ])
                
            else:
                # if this happen there is mis-recorded data
                print('Warning: time of orderflow point not in `timepoint_range`!')
        print(f'{datetime.now()} Finish iterating for orderflow data')


    def load_data_from_influx(self, inst: str, start: datetime, end: datetime, env: str):

        self.inst = inst
        handler = InfluxDbHandler(env=env, logger=None)

        start_ts = int(datetime.timestamp(start))
        end_ts = int(datetime.timestamp(end))

        ohlcv_query = f'''
        from(bucket: "ohlcv-data")
        |> range(start: {start_ts}, stop: {end_ts})
        |> filter(fn: (r) => r["_measurement"] == "ohlcv")
        |> filter(fn: (r) => r["s"] == "{inst}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns:["_time"])
        '''

        orderflow_query = f'''
        from(bucket: "orderflow-data")
        |> range(start: {start_ts}, stop: {end_ts})
        |> filter(fn: (r) => r["_measurement"] == "orderflow")
        |> filter(fn: (r) => r["s"] == "{inst}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> filter(fn: (r) => r["q"] > 0)
        |> sort(columns:["_time", "p"])
        '''

        ohlcv_generator = handler.query_stream_iter(query=ohlcv_query)
        orderflow_generator = handler.query_stream_iter(query=orderflow_query)
        self._convert_records_to_slice(ohlcv_generator, orderflow_generator)
