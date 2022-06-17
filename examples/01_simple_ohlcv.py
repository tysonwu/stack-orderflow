import pandas as pd
import talib as ta

from finplotter.finplotter import FinPlotter

'''
Simple candlestick plot with a basic EMA line
'''

if __name__ == '__main__':

    inst = 'btcusdt'.upper()
    interval = '1m'

    metadata = dict(inst=inst, interval=interval)

    df = pd.read_csv('examples/data/sample_data.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df['ema20'] = ta.EMA(df['c'], timeperiod=10)

    plotter = FinPlotter(metadata=metadata, figsize=(1800, 1000))
    plotter.plot_candlestick(data=df[['o', 'h', 'l', 'c', 'v']], with_volume=True) # has to be this name and this order
    plotter.add_line(data=df['ema20'], color='#F4D03F', width=3, legend='EMA')
    plotter.show()