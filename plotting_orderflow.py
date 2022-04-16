from datetime import datetime, timezone, timedelta
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

def plot():

    # =======widnow setup=========================================================================
    # plotting
    fplt.foreground = '#D6DBDF'
    fplt.background = '#151E26'
    fplt.legend_border_color = '#ffffff30' # make transparent
    fplt.legend_fill_color = '#ffffff10' # make transparent
    fplt.legend_text_color = fplt.foreground
    # this is overriden if rows_stretch_factor is set in fplt.create_plot()
    # fplt.top_graph_scale = 3 # top graph and bottom graph has ratio of r:1
    fplt.winx, fplt.winy, fplt.winw, fplt.winh = 100,100,1600,1600

    ax, ax5, ax3, ax2, ax4 = fplt.create_plot(
        title='StackOrderflow',
        rows=5, # main candlestick = ax / pace of tape = ax5 / MACD = ax3 / CVD = ax2 / StochRSI = ax4
        maximize=False,
        init_zoom_periods=18,
        row_stretch_factors=[3, 0.3, 1, 1, 1]
    )

    # placeholder for tick info; updated with fplt.set_time_inspector(func)
    hover_label = fplt.add_legend('', ax=ax)

    # set max zoom: for orderflow data, allow more zoom (default was 20)
    fplt.max_zoom_points = 5
    fplt.lod_labels = 700 # when number of labels exceed this number, stop generating them

    # =======start plotting=========================================================================

    print('Drawing plot...')
    # add candlestick
    # this is the version of candlestick without orderflow data
    # candlestick_plot = fplt.candlestick_ochl(datasrc=ohlcv[['o', 'c', 'h', 'l']], candle_width=0.7, ax=ax)

    # and this is the version with orderflow data; thinner candle and put aside
    candlestick_plot = fplt.candlestick_ochl_orderflow(datasrc=ohlcv[['o', 'c', 'h', 'l']], candle_width=.075, ax=ax)
    
    # add volume
    volume_plot = fplt.volume_ocv(ohlcv[['o', 'c', 'v']], candle_width=0.2, ax=ax.overlay(scale=0.18))

    # poc plot
    # can choose below either: plot a connected curve of POCs, or explicitly mark the POC level for each point
    # fplt.plot(ohlcv['poc'], ax=ax, legend='POC', color='#008FFF')
    for t, poc in ohlcv['poc'].to_dict().items():
        fplt.add_line(p0=(t-timedelta(seconds=24), poc), p1=(t+timedelta(seconds=24), poc), color='#008FFF90', width=2, ax=ax)

    # plot EMAs
    for n, color in zip(ema_n, ema_colors):
        fplt.plot(ohlcv[f'ema{n}'], ax=ax, legend=f'EMA {n}', color=color)

    # add heatmap
    fplt.delta_heatmap(delta_heatmap, filter_limit=0.7, whiteout=0.0, rect_size=1.0)

    # add price level bid ask text
    def human_format(num, sigfig=2):
        '''
        Ref: https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
        '''
        # 1g means 1 sigfig
        num = float(f'{num:.{sigfig}g}')
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

    small_font = pg.Qt.QtGui.QFont()
    small_font.setPixelSize(9)
    price_levels_text_rows = [None] * len(ohlcv)
    
    for idx, mp in enumerate(mp_slice):
        bidask_profile = mp.bidask_profile.reset_index().rename(columns={'index': 'p'})
        bidask_profile['t'] = int(mp.timepoint.timestamp())
        price_levels_text_rows[idx] = bidask_profile
    price_level_texts = pd.concat(price_levels_text_rows, axis=0, sort=True).reset_index(drop=True)

    fplt.labels(
        price_level_texts['t'], price_level_texts['p'], 
        [human_format(txt) for txt in price_level_texts['b']], ax=ax, anchor=(1, 0.5), color=fplt.foreground, qfont=small_font
    )
    fplt.labels(
        price_level_texts['t'], price_level_texts['p'], 
        [human_format(txt) for txt in price_level_texts['a']], ax=ax, anchor=(0, 0.5), color=fplt.foreground, qfont=small_font
    )

    # ==============================================================================================
    # add PoT table
    '''
    Ref: examples/bubble-table.py
    '''
    def skip_y_crosshair_info(x, y, xt, yt): # we don't want any Y crosshair info on the table
        return xt, ''

    ax5.set_visible(yaxis=False)
    fplt.add_crosshair_info(skip_y_crosshair_info, ax=ax5)
    fplt.set_y_range(0, 2, ax5)
    
    pot_colmap = fplt.ColorMap([0.0, 0.3, 1.0], [[255, 255, 255, 10], [255, 155, 0, 200], [210, 48, 9, 230]]) # traffic light colors
    ts = [int(t.timestamp()) for t in ohlcv.index]

    # background color
    pot_df = ohlcv[['pot_ask', 'pot_bid']].copy().rename(columns={'pot_ask': 1, 'pot_bid': 0})
    fplt.heatmap(pot_df[[1, 0]], colmap=pot_colmap, colcurve=lambda x: x, ax=ax5)
    medium_font = pg.Qt.QtGui.QFont()
    medium_font.setPixelSize(11)
    pot_plot = fplt.labels(ts, [1.5] * len(ohlcv), [human_format(txt) for txt in ohlcv['pot_ask']], ax=ax5, anchor=(0.5, 0.5), color=fplt.foreground, qfont=medium_font)
    pot_plot = fplt.labels(ts, [0.5] * len(ohlcv), [human_format(txt) for txt in ohlcv['pot_bid']], ax=ax5, anchor=(0.5, 0.5), color=fplt.foreground, qfont=medium_font)
    
    # maybe find a better way to deal with this
    fplt.legend_border_color = '#151E26'
    fplt.legend_fill_color = '#ffffff80'
    fplt.legend_text_color = '#151E26'
    fplt.add_legend('ASK (upper) BID (lower)', ax= ax5)
    # revert back after adding this legend
    fplt.legend_border_color = '#ffffff30'
    fplt.legend_fill_color = '#ffffff10'
    fplt.legend_text_color = fplt.foreground    

    # and set background
    vb = pot_plot.getViewBox()
    vb.setBackgroundColor('#00000000')

    # ==============================================================================================
    # plot MACD
    macd_plot = fplt.volume_ocv(ohlcv[['o', 'c', 'macd_diff']], ax=ax3, candle_width=0.2, colorfunc=fplt.strength_colorfilter)
    fplt.plot(ohlcv['macd'], ax=ax3, legend=f'MACD ({macd[0]}, {macd[1]}, {macd[2]})')
    fplt.plot(ohlcv['macd_signal'], ax=ax3, legend='Signal')

    vb = macd_plot.getViewBox()
    vb.setBackgroundColor('#00000020')

    # ==============================================================================================
    '''
    Ref: examples/snp500.py
    '''
    # plot cvd
    line_color = '#F4D03F'
    cvd_plot = fplt.plot(np.cumsum(ohlcv['d']), ax=ax2, legend='CVD', color=line_color, fillLevel=0, brush=line_color+'10')
    # and set background
    vb = cvd_plot.getViewBox()
    vb.setBackgroundColor('#00000000')

    # ==============================================================================================
    # plot stoch RSI
    stoch_rsi_plot = fplt.plot(ohlcv['fastk'], ax=ax4, legend=f'StochRSI Fast k: {stoch_rsi[1]} Timeperiod: {stoch_rsi[0]}')
    fplt.plot(ohlcv['fastd'], ax=ax4, legend=f'StochRSI Fast d: {stoch_rsi[2]} Timeperiod: {stoch_rsi[0]}')

    thresholds = [20, 80]
    for th in thresholds:
        rsi_threshold_line = pg.InfiniteLine(pos=th, angle=0, pen=fplt._makepen(color='#ffffff50', style='- - '))
        ax4.addItem(rsi_threshold_line, ignoreBounds=True)
    fplt.add_band(*thresholds, color='#2980B920', ax=ax4)

    vb = stoch_rsi_plot.getViewBox()
    vb.setBackgroundColor('#00000020')

    # ==============================================================================================
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
    ax2.showGrid(x=True, y=True, alpha=0.1)
    ax3.showGrid(x=True, y=True, alpha=0.1)
    ax4.showGrid(x=True, y=True, alpha=0.1)
    ax5.showGrid(x=True, y=False, alpha=0.1)
    
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
        # o = ohlcv.iloc[x]['o']
        # h = ohlcv.iloc[x]['h']
        # l = ohlcv.iloc[x]['l']
        # c = ohlcv.iloc[x]['c']
        mp = mp_slice[x]
        bapr = mp.bidask_profile.copy()
        # infer the increment value between price levels
        try:
            increment = (bapr.index[1] - bapr.index[0]) / 2
        except IndexError:
            increment = 0
        if y > mp.price_levels_range[0] + increment  or y < mp.price_levels_range[1] - increment:
            add_yt = f'\tLevel: {ytext}' # not showing orderflow info if cursor is outside range
        else:
            bapr.index = bapr.index - increment # 
            plr = bapr[bapr.index <= y].iloc[-1] # the price level row
            pl = round(plr.name + increment, 8) # the original price level
            dpr = mp.delta_profile
            a, b, d = plr['a'], plr['b'], dpr.loc[pl].values[0]
            add_yt = f'\tLevel: {ytext}\n\n\tAsk: {a}\n\tBid: {b}\n\tDelta: {d}' # not showing ask and bid value if cursor is outside range
        add_xt = f'\t{xtext}'
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

if __name__ == '__main__':

    inst = 'btcusdt'
    # input in HKT
    start = datetime(2021, 12, 10, 0, 0, 0)
    end = datetime(2021, 12, 10, 12, 0, 0)

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
            'poc': [mp.poc_price_level for mp in mp_slice],
            'pot': [mp.pot for mp in mp_slice],
            'pot_ask': [mp.pot_ask for mp in mp_slice],
            'pot_bid': [mp.pot_bid for mp in mp_slice],
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

    delta_heatmap_rows = [None] * len(ohlcv)
    # make heatmap df
    for idx in range(len(ohlcv)):
        delta_heatmap_rows[idx] = mp_slice[idx].delta_profile.T
    # concat rows
    delta_heatmap = pd.concat(delta_heatmap_rows, axis=0, sort=True)

    plot()
