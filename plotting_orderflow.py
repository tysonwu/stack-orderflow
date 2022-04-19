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

class OrderflowPlotter:

    ema_n = [20, 50, 200]
    macd = [12, 26, 9]
    stoch_rsi = [14, 3, 3]

    def __init__(self, token: str, interval: str, increment: float, ohlcv: pd.DataFrame, mp_slice: list):
        self.token = token
        self.interval = interval
        self.increment = increment
        self.ohlcv = ohlcv
        self.mp_slice = mp_slice
        self.plots = {} # pyqtgraph plot objects


    def update_datasrc(self, ohlcv, mp_slice):
        self.ohlcv = ohlcv
        self.mp_slice = mp_slice


    def calculate_plot_features(self):

        self.ohlcv['d'] = [mp.delta_qty for mp in self.mp_slice]
        self.ohlcv['poc'] = [mp.poc_price_level for mp in self.mp_slice]

        for n in self.ema_n:
            self.ohlcv[f'ema{n}'] = ta.EMA(self.ohlcv['c'], timeperiod=n)

        self.ohlcv['macd'], self.ohlcv['macd_signal'], self.ohlcv['macd_diff'] = ta.MACD(self.ohlcv['c'], fastperiod=self.macd[0], slowperiod=self.macd[1], signalperiod=self.macd[2])

        self.ohlcv['fastk'], self.ohlcv['fastd'] = ta.STOCHRSI(self.ohlcv['c'], timeperiod=self.stoch_rsi[0], fastk_period=self.stoch_rsi[1], fastd_period=self.stoch_rsi[2], fastd_matype=0)

        delta_heatmap_rows = [None] * len(self.ohlcv)
        # make heatmap df
        for idx in range(len(self.ohlcv)):
            delta_heatmap_rows[idx] = self.mp_slice[idx].delta_profile.T
        # concat rows
        self.delta_heatmap = pd.concat(delta_heatmap_rows, axis=0, sort=True)

        price_levels_text_rows = [None] * len(self.ohlcv)
        
        for idx, mp in enumerate(self.mp_slice):
            bidask_profile = mp.bidask_profile.reset_index().rename(columns={'index': 'p'})
            bidask_profile['t'] = int(mp.timepoint.timestamp())
            price_levels_text_rows[idx] = bidask_profile
        self.price_level_texts = pd.concat(price_levels_text_rows, axis=0, sort=True).reset_index(drop=True)
        self.price_level_texts['a'] = self.price_level_texts['a'].apply(lambda x: self.human_format(x))
        self.price_level_texts['b'] = self.price_level_texts['b'].apply(lambda x: self.human_format(x))

        ts = [int(t.timestamp()) for t in self.ohlcv.index]
        self.pot_heatmap = self.ohlcv[['pot_ask', 'pot_bid']].copy().rename(columns={'pot_ask': 1, 'pot_bid': 0})

        self.pot_df = self.ohlcv[['pot_ask', 'pot_bid']].copy().reset_index(drop=True)
        self.pot_df['ts'] = ts
        self.pot_df['ask_label_height'] = 1.5
        self.pot_df['bid_label_height'] = 0.5
        self.pot_df['ask_label'] = self.pot_df['pot_ask'].apply(lambda x: self.human_format(x))
        self.pot_df['bid_label'] = self.pot_df['pot_bid'].apply(lambda x: self.human_format(x))

        self.cvd = np.cumsum(self.ohlcv['d'])


    def orderflow_plot(self):

        if self.ohlcv.empty:
            return None

        self.calculate_plot_features()

        # =======widnow setup=========================================================================
        # plotting
        ema_colors = ['#33DCCD50', '#ADE03670', '#F4D03F80']
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
        # candlestick_plot = fplt.candlestick_ochl(datasrc=self.ohlcv[['o', 'c', 'h', 'l']], candle_width=0.7, ax=ax)

        # and this is the version with orderflow data; thinner candle and put aside
        self.plots['candlestick'] = fplt.candlestick_ochl_orderflow(datasrc=self.ohlcv[['o', 'c', 'h', 'l']], candle_width=.075, ax=ax)
        
        # add volume
        self.plots['volume'] = fplt.volume_ocv(self.ohlcv[['o', 'c', 'v']], candle_width=0.2, ax=ax.overlay(scale=0.18))

        # poc plot
        # can choose below either: plot a connected curve of POCs, or explicitly mark the POC level for each point
        self.plots['poc'] = fplt.plot(self.ohlcv['poc'], ax=ax, legend='POC', color='#008FFF')

        # for t, poc in self.ohlcv['poc'].to_dict().items():
            # fplt.add_line(p0=(t-timedelta(seconds=24), poc), p1=(t+timedelta(seconds=24), poc), color='#008FFF90', width=2, ax=ax)

        # plot EMAs
        for n, color in zip(self.ema_n, ema_colors):
            self.plots[f'ema{n}'] = fplt.plot(self.ohlcv[f'ema{n}'], ax=ax, legend=f'EMA {n}', color=color)

        # add heatmap
        self.plots['delta_heatmap'] = fplt.delta_heatmap(self.delta_heatmap, filter_limit=0.7, whiteout=0.0, rect_size=1.0)


        small_font = pg.Qt.QtGui.QFont()
        small_font.setPixelSize(9)

        self.plots['bid_labels'] = fplt.labels(
            self.price_level_texts[['t', 'p', 'b']], ax=ax, anchor=(1, 0.5), color=fplt.foreground, qfont=small_font
        )
        self.plots['ask_labels'] = fplt.labels(
            self.price_level_texts[['t', 'p', 'a']], ax=ax, anchor=(0, 0.5), color=fplt.foreground, qfont=small_font
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

        # background color
        medium_font = pg.Qt.QtGui.QFont()
        medium_font.setPixelSize(11)

        self.plots['pot_heatmap'] = fplt.heatmap(self.pot_heatmap, colmap=pot_colmap, colcurve=lambda x: x, ax=ax5)

        self.plots['pot_ask'] = fplt.labels(self.pot_df[['ts', 'ask_label_height', 'ask_label']], ax=ax5, anchor=(0.5, 0.5), color=fplt.foreground, qfont=medium_font)
        self.plots['pot_bid'] = fplt.labels(self.pot_df[['ts', 'bid_label_height', 'bid_label']], ax=ax5, anchor=(0.5, 0.5), color=fplt.foreground, qfont=medium_font)
        
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
        vb = self.plots['pot_heatmap'].getViewBox()
        vb.setBackgroundColor('#00000000')

        # ==============================================================================================
        # plot MACD
        self.plots['macd_diff'] = fplt.volume_ocv(self.ohlcv[['o', 'c', 'macd_diff']], ax=ax3, candle_width=0.2, colorfunc=fplt.strength_colorfilter)
        self.plots['macd'] = fplt.plot(self.ohlcv['macd'], ax=ax3, legend=f'MACD ({self.macd[0]}, {self.macd[1]}, {self.macd[2]})')
        self.plots['macd_signal'] = fplt.plot(self.ohlcv['macd_signal'], ax=ax3, legend='Signal')

        vb = self.plots['macd'].getViewBox()
        vb.setBackgroundColor('#00000020')

        # ==============================================================================================
        '''
        Ref: examples/snp500.py
        '''
        # plot cvd
        line_color = '#F4D03F'
        self.plots['cvd'] = fplt.plot(self.cvd, ax=ax2, legend='CVD', color=line_color, fillLevel=0, brush=line_color+'10')
        # and set background
        vb = self.plots['cvd'].getViewBox()
        vb.setBackgroundColor('#00000000')

        # ==============================================================================================
        # plot stoch RSI
        self.plots['stochrsi_fastk'] = fplt.plot(self.ohlcv['fastk'], ax=ax4, legend=f'StochRSI Fast k: {self.stoch_rsi[1]} Timeperiod: {self.stoch_rsi[0]}')
        self.plots['stochrsi_fastd'] = fplt.plot(self.ohlcv['fastd'], ax=ax4, legend=f'StochRSI Fast d: {self.stoch_rsi[2]} Timeperiod: {self.stoch_rsi[0]}')

        thresholds = [20, 80]
        for th in thresholds:
            rsi_threshold_line = pg.InfiniteLine(pos=th, angle=0, pen=fplt._makepen(color='#ffffff50', style='- - '))
            ax4.addItem(rsi_threshold_line, ignoreBounds=True)
        fplt.add_band(*thresholds, color='#2980B920', ax=ax4)

        vb = self.plots['stochrsi_fastk'].getViewBox()
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
        self.plots['candlestick'].colors.update({
            'bull_body': fplt.candle_bull_color
        })

        transparency = '45'
        self.plots['volume'].colors.update({
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
            utcdt = dt.astimezone(pytz.utc).replace(tzinfo=None)
            # dt = dt.replace(tzinfo=timezone.utc)
            row = self.ohlcv.loc[utcdt]
            # format html with the candle and set legend
            fmt = '<span style="color:%s; margin: 16px;">%%s</span>' % (bull if (row['o'] < row['c']).all() else bear)
            rawtxt = '<span style="font-size:14px">%%s %%s</span> &nbsp; O: %s H: %s L: %s C: %s Delta: %s' % (fmt, fmt, fmt, fmt, fmt)
            hover_label.setText(rawtxt % (self.token, self.interval, row['o'], row['h'], row['l'], row['c'], row['d']))
        fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
        
        # additional crosshair info
        def enrich_info(x, y, xtext, ytext):
            # o = self.ohlcv.iloc[x]['o']
            # h = self.ohlcv.iloc[x]['h']
            # l = self.ohlcv.iloc[x]['l']
            # c = self.ohlcv.iloc[x]['c']
            mp = self.mp_slice[x]
            bapr = mp.bidask_profile.copy()
            y_increment = self.increment / 2
            if y > mp.price_levels_range[0] + y_increment  or y < mp.price_levels_range[1] - y_increment:
                add_yt = f'\tLevel: {ytext}' # not showing orderflow info if cursor is outside range
            else:
                bapr.index = bapr.index - y_increment # 
                plr = bapr[bapr.index <= y].iloc[-1] # the price level row
                pl = round(plr.name + y_increment, 8) # the original price level
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


    def show(self):
        return fplt.show()


    @staticmethod
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



if __name__ == '__main__':

    inst = 'btcusdt'
    token = inst.upper()
    interval = '1m'
    increment = 10
    # input in HKT
    start = datetime(2022, 4, 18, 23, 0, 0)
    end = datetime(2022, 4, 19, 6, 0, 0)

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
            'pot': [mp.pot for mp in mp_slice],
            'pot_ask': [mp.pot_ask for mp in mp_slice],
            'pot_bid': [mp.pot_bid for mp in mp_slice],
        },
        index=[mp.timepoint for mp in mp_slice]
    )

    plotter = OrderflowPlotter(token, interval, increment, ohlcv, mp_slice)
    plotter.orderflow_plot()
    plotter.show()
