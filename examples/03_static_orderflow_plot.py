import pandas as pd

from market_profile import MarketProfileSlice
from orderflow_plotter import OrderflowPlotter

'''
Plotting of orderflow chart
- Candlestick
- Orderflow data by price level
- Volume bar
- Classic MACD
- CVD
- StochRSI
'''

if __name__ == '__main__':

    inst = 'btcusdt'
    increment = 10
    token = inst.upper()
    interval = '1m'

    ohlcv = pd.read_csv('examples/data/sample_data_2.csv', index_col=0)
    ohlcv.index = pd.to_datetime(ohlcv.index)

    profile = pd.read_csv('examples/data/orderflow_data.csv')
    profile['t'] = pd.to_datetime(profile['t'])

    mp_slices=[]
    dts = list(ohlcv.index)
    for dt in dts:
        ohlcv_data = ohlcv.loc[dt].to_dict()
        orderflow_data = profile[profile['t'] == dt][['p', 'q', 'd', 'b', 'a']].to_numpy()
        mp_slices.append(MarketProfileSlice(inst, dt, ohlcv_data, orderflow_data))

    plotter = OrderflowPlotter(token, interval, increment, ohlcv, mp_slices)
    plotter.orderflow_plot()
    plotter.show()
