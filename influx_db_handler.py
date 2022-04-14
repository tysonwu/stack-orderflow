'''
Handling read and write to InfluxDB
Measurement is the meaning of the value eg. volume, price, cvd, etc.
Tag is the asset name
Field is the actual numeric value of the measurement, ie. the measured value of the tag at that timestamp
Fields has to be float!
'''

import os
from collections import OrderedDict
from csv import DictReader

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS, WriteOptions, WriteType


class InfluxDbHandler:

    def __init__(self, env, logger):
        self.logger = logger
        if env == 'local':
            self._client = InfluxDBClient.from_config_file(os.path.dirname(__file__) + '/config/influx_config_local.ini')
        elif env == 'prod':
            self._client = InfluxDBClient.from_config_file(os.path.dirname(__file__) + '/config/influx_config_prod.ini')
        else:
            raise NotImplementedError('env is neither local nor prod!')

        self.test_connection()

    def test_connection(self):
        health = self._client.health()
        if health.status == "pass":
            print(f"{self._client} - Connection success.")
        else:
            print(f"{self._client} - Connection failure: {health.message}!")

    @staticmethod
    def _parse_row(row: OrderedDict, time_col: str, measurement: str, tags: list, fields: list):
        point = Point(measurement)
        for tag in tags:
            point = point.tag(tag, row[tag])
        for field in fields:
            point = point.field(field, float(row[field]))
        point = point.time(row[time_col])
        return point

    def write_from_file(self, bucket, f, csv_headers, time_col, measurement, tags, fields, precision='s', skip_header=False):
        '''
        Write csv file to InfluxDB
        '''
        # data = rx \
        #     .from_iterable(DictReader(open(f, 'r'), fieldnames=[time_col]+fields)) \
        #     .pipe(ops.map(lambda row: self._parse_row(row, time_col, measurement, tags, fields)))
        # self._write_client.write(bucket=self.bucket, org=self._client.org, record=data)
        reader = DictReader(open(f, 'r'), fieldnames=csv_headers)
        if skip_header:
            next(reader)
        data = [self._parse_row(row, time_col, measurement, tags, fields) for row in reader]
        
        with self._client.write_api(write_options=WriteOptions(batch_size=1000)) as _write_client:
            _write_client.write(
                bucket=bucket,
                org=self._client.org,
                record=data,
                write_precision=precision
            )

    def write_from_dataframe(self, bucket, df, time_col, measurement, tags, fields, precision='s'):
        '''
        Write pd.DataFrame to InfluxDB
        index has to be time
        '''
        # for field in tags:
        #     df[field] = field
        for field in fields:
            df[field] = df[field].astype('float')
        data = df[[time_col] + tags + fields]
        data = data.set_index(time_col)

        with self._client.write_api(write_options=WriteOptions(batch_size=1000)) as _write_client:
            # _write_client.write(self.bucket, self._client.org, data)
            _write_client.write(
                bucket=bucket,
                org=self._client.org,
                record=data,
                data_frame_measurement_name=measurement,
                data_frame_tag_columns=tags,
                write_precision=precision,
            )

    def query_data(self, query):
        '''
        Returns a list of Flux objects
        The Flux object provides the following methods for accessing your data:
        https://docs.influxdata.com/influxdb/v2.0/api-guide/client-libraries/python/
        get_measurement(): Returns the measurement name of the record.
        get_field(): Returns the field name.
        get_value(): Returns the actual field value.
        values: Returns a map of column values.
        values.get("<your tag>"): Returns a value from the record for given column.
        get_time(): Returns the time of the record.
        get_start(): Returns the inclusive lower time bound of all records in the current table.
        get_stop(): Returns the exclusive upper time bound of all records in the current table.
        '''
        res = self._client.query_api().query(org=self._client.org, query=query)
        results = []
        for table in res:
            for record in table.records:
                # results.append(record)
                results.append(record.values)
        return results

    def query_data_iter(self, query):
        res = self._client.query_api().query(org=self._client.org, query=query)
        return res

    def query_stream_iter(self, query):
        res = self._client.query_api().query_stream(org=self._client.org, query=query)
        return res

    '''
    Below methods are deprecated, not maintained:
    '''

    def read_market_data_df(self, exchange, resolution, symbol, data_length='10000d'):
        query = f'from(bucket:"{self.bucket}") |> range(start: -{data_length}) |> filter(fn: (r) => r._measurement == "{exchange}_{resolution}")' \
                f'|> filter(fn: (r) => r.symbol == "{symbol}") ' \
                f'|> pivot(rowKey:["_time"],columnKey: ["_field"], valueColumn: "_value")'
        result = self._client.query_api().query_data_frame(org=self._client.org, query=query)

        result.drop(columns=['_start', '_stop', 'result', 'table', '_measurement', 'symbol'], inplace=True)
        result.rename(columns={'_time': 'datetime'}, inplace=True)
        result.set_index('datetime', inplace=True)

        return result

    def read_market_data_daily_close_df(self, exchange, symbol, data_length='10000d'):
        if isinstance(symbol, list):
            query_list = [f'r.symbol == "{i}"' for i in symbol]
            query_sym_string = f'({" or ".join(query_list)})'
        else:
            query_sym_string = f'r.symbol == "{symbol}"'

        query = f'from(bucket:"{self.bucket}") |> range(start: -{data_length}) |> filter(fn: (r) => r._measurement == "{exchange}_1h")' \
                f'|> filter(fn: (r) => {query_sym_string} and r._field == "close" ) ' \
                f'|> aggregateWindow(every: 1d, fn: last) ' \
                f'|> pivot(rowKey:["_time"],columnKey: ["_field"], valueColumn: "_value")'
        result = self._client.query_api().query_data_frame(org=self._client.org, query=query)

        result.drop(columns=['_start', '_stop', 'result', 'table', '_measurement'], inplace=True)
        result.rename(columns={'_time': 'datetime'}, inplace=True)
        result.set_index('datetime', inplace=True)

        return result