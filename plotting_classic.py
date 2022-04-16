from datetime import datetime, timezone
import pytz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import talib as ta

import finplot_lib as fplt

from data_pipeline.market_profile_reader import MarketProfileReader

'''
Plotting of orderflow chart
- Candlestick
- Orderflow data by price level
- Volume bar
- Classic MACD
- CVD
- StochRSI

Pyqtgraph ref:
https://pyqtgraph.readthedocs.io/en/latest/index.html
https://doc.qt.io/qtforpython-5/contents.html
'''

if __name__ == '__main__':
    inst = 'btcusdt'
    # input in HKT
    start = datetime(2021, 12, 10, 0, 0, 0)
    end = datetime(2021, 12, 10, 6, 0, 0)

    profile = MarketProfileReader()
    profile.load_data_from_influx(inst=inst, start=start, end=end, env='local')
    
    # slice_dt = pytz.timezone('Asia/Hong_Kong').localize(datetime(2022,3,17,17,23,0)) # input in HKT
    slice_start = pytz.timezone('Asia/Hong_Kong').localize(start) # input in HKT
    slice_end = pytz.timezone('Asia/Hong_Kong').localize(end) # input in HKT
    
    # mp_slice = profile[slice_dt]
    mp_slice = profile[slice_start:slice_end]

    ohlcv = pd.DataFrame(
        {
            'o': [mp.open for mp in mp_slice],
            'h': [mp.high for mp in mp_slice],
            'l': [mp.low for mp in mp_slice],
            'c': [mp.close for mp in mp_slice],
            'v': [mp.volume_qty for mp in mp_slice],
            'd': [mp.delta_qty for mp in mp_slice],
        },
        index=[mp.timepoint for mp in mp_slice]
    )

    ema_n = [20, 50, 200]
    ema_colors = ['#33DCCD50', '#ADE03670', '#F4D03F80']
    for n in ema_n:
        ohlcv[f'ema{n}'] = ta.EMA(ohlcv['c'], timeperiod=n)

    macd = [12, 26, 9]
    ohlcv['macd'], ohlcv['macd_signal'], ohlcv['macd_diff'] = ta.MACD(ohlcv['c'], fastperiod=macd[0], slowperiod=macd[1], signalperiod=macd[2])

    stoch_rsi = [21, 14, 14]
    ohlcv['fastk'], ohlcv['fastd'] = ta.STOCHRSI(ohlcv['c'], timeperiod=stoch_rsi[0], fastk_period=stoch_rsi[1], fastd_period=stoch_rsi[2], fastd_matype=0)

    print('Drawing plot...')
    
    # plotting
    fplt.foreground = '#D6DBDF'
    fplt.background = '#151E26'
    fplt.legend_border_color = '#ffffff30' # make transparent
    fplt.legend_fill_color = '#ffffff10' # make transparent
    fplt.legend_text_color = fplt.foreground
    fplt.top_graph_scale = 2 # top graph and bottom graph has ratio of r:1
    fplt.winx, fplt.winy, fplt.winw, fplt.winh = 0,0,1600,1600
    ax, ax3, ax2, ax4 = fplt.create_plot(
        title='Chart',
        rows=4, # main candlestick = ax / MACD = ax3 / CVD = ax2 / StochRSI = ax4
        maximize=False,
        init_zoom_periods=18,
        row_stretch_factors=[3, 1, 1, 1]
    )

    # placeholder for tick info; updated with fplt.set_time_inspector(func)
    hover_label = fplt.add_legend('', ax=ax)

    # set max zoom: for orderflow data, allow more zoom (default was 20)
    fplt.max_zoom_points = 20

    # add candlestick
    # this is the version of candlestick without orderflow data
    candlestick_plot = fplt.candlestick_ochl(datasrc=ohlcv[['o', 'c', 'h', 'l']], candle_width=0.7, ax=ax)
    
    # add volume
    volume_plot = fplt.volume_ocv(ohlcv[['o', 'c', 'v']], candle_width=0.7, ax=ax.overlay())

    # plot EMAs
    for n, color in zip(ema_n, ema_colors):
        fplt.plot(ohlcv[f'ema{n}'], ax=ax.overlay(), legend=f'EMA {n}', color=color)

    # plot stoch RSI
    stoch_rsi_plot = fplt.plot(ohlcv['fastk'], ax=ax4, legend=f'StochRSI Fast k: {stoch_rsi[1]} Timeperiod: {stoch_rsi[0]}')
    fplt.plot(ohlcv['fastd'], ax=ax4, legend=f'StochRSI Fast d: {stoch_rsi[2]} Timeperiod: {stoch_rsi[0]}')

    thresholds = [20, 80]
    for th in thresholds:
        rsi_threshold_line = pg.InfiniteLine(pos=th, angle=0, pen=fplt._makepen(color='#ffffff50', style='- - '))
        ax4.addItem(rsi_threshold_line, ignoreBounds=True)
    fplt.add_band(*thresholds, color='#2980B920', ax=ax4)

    vb = stoch_rsi_plot.getViewBox()
    vb.setBackgroundColor('#00000000')

    # plot MACD
    macd_plot = fplt.volume_ocv(ohlcv[['o', 'c', 'macd_diff']], ax=ax3, candle_width=0.7, colorfunc=fplt.strength_colorfilter)
    fplt.plot(ohlcv['macd'], ax=ax3, legend=f'MACD ({macd[0]}, {macd[1]}, {macd[2]})')
    fplt.plot(ohlcv['macd_signal'], ax=ax3, legend='Signal')

    vb = macd_plot.getViewBox()
    vb.setBackgroundColor('#00000000')

    '''
    Ref: examples/snp500.py
    '''
    # plot cvd
    line_color = '#F4D03F'
    cvd_plot = fplt.plot(np.cumsum(ohlcv['d']), ax=ax2, legend='CVD', color=line_color, fillLevel=0, brush=line_color+'10')
    # and set background
    vb = cvd_plot.getViewBox()
    vb.setBackgroundColor('#00000000')

    '''
    Ref: examples/complicated.py
    '''
    # set bull body to same color as bull frame; otherwise it is default background color (transparent)
    bull = '#1ABC9C'
    bear = '#E74C3C'
    fplt.candle_bull_color = bull
    fplt.candle_bull_body_color = bull
    fplt.candle_bear_color = bear
    candlestick_plot.colors.update({
        'bull_body': fplt.candle_bull_color
    })

    transparency = '45'
    volume_plot.colors.update({
        'bull_frame': fplt.candle_bull_color + transparency, 
        'bull_body': fplt.candle_bull_body_color + transparency,
        'bear_frame': fplt.candle_bear_color + transparency, 
        'bear_body': fplt.candle_bear_color + transparency,
    })
    
    # set gridlines
    ax.showGrid(x=True, y=True, alpha=0.2)
    ax2.showGrid(x=True, y=True, alpha=0.2)
    ax3.showGrid(x=True, y=True, alpha=0.2)
    ax4.showGrid(x=True, y=True, alpha=0.2)
    
    # add YAxis item at the right
    # ax.axes['right'] = {'item': fplt.YAxisItem(vb=ax.vb, orientation='right')}
    # ax2.axes['right'] = {'item': fplt.YAxisItem(vb=ax2.vb, orientation='right')}

    # add legend of ohlcv data
    '''
    Ref: examples/snp500.py
    '''
    def update_legend_text(x, y):
        dt = datetime.fromtimestamp(x // 1000000000)
        utcdt = dt.astimezone(pytz.utc)
        # dt = dt.replace(tzinfo=timezone.utc)
        row = ohlcv.loc[utcdt]
        # format html with the candle and set legend
        fmt = '<span style="color:%s; margin: 16px;">%%s</span>' % (bull if (row['o'] < row['c']).all() else bear)
        rawtxt = '<span style="font-size:14px">%%s %%s</span> &nbsp; O: %s H: %s L: %s C: %s Delta: %s' % (fmt, fmt, fmt, fmt, fmt)
        hover_label.setText(rawtxt % ('TOKEN', 'INTERVAL', row['o'], row['h'], row['l'], row['c'], row['d']))
    fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
    
    # additional crosshair info
    def enrich_info(x, y, xtext, ytext):
        o = ohlcv.iloc[x]['o']
        h = ohlcv.iloc[x]['h']
        l = ohlcv.iloc[x]['l']
        c = ohlcv.iloc[x]['c']
        add_xt = f'\t{xtext}'
        add_yt = f'\tLevel: {ytext}\n\n\tOpen: {o}\n\tHigh: {h}\n\tLow: {l}\n\tClose: {c}'
        return add_xt, add_yt
    
    fplt.add_crosshair_info(enrich_info, ax=ax)

    '''
    Ref: examples/complicated.py
    '''
    # set dark themes ====================
    pg.setConfigOptions(foreground=fplt.foreground, background=fplt.background)

    # window background
    for win in fplt.windows:
        win.setBackground(fplt.background)

    # axis, crosshair, candlesticks, volumes
    axs = [ax for win in fplt.windows for ax in win.axs]
    vbs = set([ax.vb for ax in axs])
    axs += fplt.overlay_axs
    axis_pen = fplt._makepen(color=fplt.foreground)
    for ax in axs:
        ax.axes['left']['item'].setPen(axis_pen)
        ax.axes['left']['item'].setTextPen(axis_pen)
        ax.axes['bottom']['item'].setPen(axis_pen)
        ax.axes['bottom']['item'].setTextPen(axis_pen)
        if ax.crosshair is not None:
            ax.crosshair.vline.pen.setColor(pg.mkColor(fplt.foreground))
            ax.crosshair.hline.pen.setColor(pg.mkColor(fplt.foreground))
            ax.crosshair.xtext.setColor(fplt.foreground)
            ax.crosshair.ytext.setColor(fplt.foreground)
    # ====================================

    fplt.show()