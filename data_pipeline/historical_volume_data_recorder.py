from datetime import datetime
import argparse
import os
import json
import csv

import pandas as pd
from tqdm import tqdm

from data_pipeline.volume_data_recorder import VolumeDataRecorder

class HistoricalVolumeDataRecorder(VolumeDataRecorder):

    def __init__(self, env, config):
        super().__init__(env, config, turn_on_ws=False)


    def _write_to_csv(self, volume_df, ohlc_df, verbose, output_path_prefix=None):
        if verbose:
            print(volume_df)
            print(ohlc_df)
            print(f'{datetime.utcnow()} - Writing ohlcv data, orderflow data into csv...')

        volume_df.to_csv(f'{output_path_prefix}volume_df.csv', index=False)
        ohlc_df.to_csv(f'{output_path_prefix}ohlc_df.csv', index=False)

        print(f'Saved csv with path and prefix: {output_path_prefix}')


    def read_and_write(self, fpath, write_to_csv=False):
        # read the csv line by line
        total_ohlc_df = None
        total_volume_df = None

        with open(fpath, 'r') as obj:
            csv_reader = csv.reader(obj)

            current_hour = None
            current_minute = None
            for data in tqdm(csv_reader):
                if data:
                    t = datetime.utcfromtimestamp(float(data[5]) / 1000)
                    p =  float(data[1])
                    if data[6] == 'True':
                        q = -float(data[2])
                    else:
                        q = float(data[2])

                    # check if we it's time to call influx to write and clean up self.data
                    if current_minute is not None and current_hour is not None:
                        '''
                        TODO: An even safer check of feed condition with datetime.year, month, day, hour, minute, 0 to be exact!
                        '''
                        if (t.minute != current_minute) or (t.hour != current_hour):
                            # not writing into influxdb at this moment. Write it after all iterations to make writing more efficient.
                            ohlc_df, volume_df = self._transform_feed_and_write_and_flush(no_write=True, verbose=False)
                            if total_ohlc_df is None:
                                total_ohlc_df = ohlc_df
                            else:
                                total_ohlc_df = pd.concat([total_ohlc_df, ohlc_df])
                            if total_volume_df is None:
                                total_volume_df = volume_df
                            else:
                                total_volume_df = pd.concat([total_volume_df, volume_df])
                            current_minute = t.minute
                            current_hour = t.hour

                        self.data.append([t, p, q])

                    current_minute = t.minute
                    current_hour = t.hour
                # print(f'{t} {p=} {q=}, {len(self.data)=}')
            else:
                # when the for loop finishes, also transform in the very last chunk of data
                ohlc_df, volume_df = self._transform_feed_and_write_and_flush(no_write=True, verbose=False)
                if total_ohlc_df is None:
                    total_ohlc_df = ohlc_df
                else:
                    total_ohlc_df = pd.concat([total_ohlc_df, ohlc_df])
                if total_volume_df is None:
                    total_volume_df = volume_df
                else:
                    total_volume_df = pd.concat([total_volume_df, volume_df])

        # finally write to db
        if write_to_csv:
            output_path_prefix = f'{fpath.split(".")[0]}-'
            self._write_to_csv(volume_df=total_volume_df, ohlc_df=total_ohlc_df, verbose=True, output_path_prefix=output_path_prefix)
        else:
            self._write_to_db(volume_df=total_volume_df, ohlc_df=total_ohlc_df, verbose=True)
