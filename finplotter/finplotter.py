from datetime import datetime, timezone
import pytz

import pyqtgraph as pg

import finplotter.finplot_library as fplt


class FinPlotter:

    BULL = '#1ABC9C'
    BEAR = '#E74C3C'

    def __init__(self, metadata=None, figsize=(1800, 1000)):
        self.ax = None # the canvas
        self._load_metadata(metadata)
        self._style_setup(figsize)


    def _load_metadata(self, metadata):
        if metadata is not None:
            self.inst = metadata.get('inst', '')
            self.interval = metadata.get('interval', '')


    def _style_setup(self, figsize):

        fplt.foreground = '#D6DBDF'
        fplt.background = '#151E26'
        fplt.legend_border_color = '#ffffff30' # make transparent
        fplt.legend_fill_color = '#ffffff10' # make transparent
        fplt.legend_text_color = fplt.foreground
        fplt.top_graph_scale = 2 # top graph and bottom graph has ratio of r:1
        fplt.winx, fplt.winy, fplt.winw, fplt.winh = 0, 0, figsize[0], figsize[1]

        # set max zoom: for orderflow data, allow more zoom (default was 20)
        fplt.max_zoom_points = 20

        # set bull body to same color as bull frame; otherwise it is default background color (transparent)
        # candlestick color
        fplt.candle_bull_color = self.BULL
        fplt.candle_bull_body_color = self.BULL
        fplt.candle_bear_color = self.BEAR

        # creat plot 
        self.ax = fplt.create_plot(
            title=self.inst,
            rows=1,
            maximize=False,
            init_zoom_periods=18,
        )

        axis_pen = fplt._makepen(color=fplt.foreground)
        self.ax.crosshair.vline.pen.setColor(pg.mkColor(fplt.foreground))
        self.ax.crosshair.hline.pen.setColor(pg.mkColor(fplt.foreground))
        self.ax.crosshair.xtext.setColor(fplt.foreground)
        self.ax.crosshair.ytext.setColor(fplt.foreground)

        self.ax.axes['left']['item'].setPen(axis_pen)
        self.ax.axes['left']['item'].setTextPen(axis_pen)
        self.ax.axes['bottom']['item'].setPen(axis_pen)
        self.ax.axes['bottom']['item'].setTextPen(axis_pen)


    def plot_candlestick(self, data, with_volume=False):

        if with_volume:
            assert len(data.columns) == 5, f'There should be exactly five columns named o, h, l, c, v under {with_volume=} Please check with your passed data.'
        else:
            assert len(data.columns) == 4, f'There should be exactly four columns named o, h, l, c under {with_volume=}. Please check with your passed data.'

        ohlc = data.copy()
        ax = self.ax

        # placeholder for tick info; updated with fplt.set_time_inspector(func)
        hover_label = fplt.add_legend('', ax=self.ax)

        # add candlestick
        # this is the version of candlestick without orderflow data
        candlestick_plot = fplt.candlestick_ochl(datasrc=ohlc[['o', 'c', 'h', 'l']], candle_width=0.7, ax=ax)
        
        if with_volume:
            # add volume
            transparency = '45'
            volume_plot = fplt.volume_ocv(ohlc[['o', 'c', 'v']], candle_width=0.7, ax=ax.overlay())
            volume_plot.colors.update({
                'bull_frame': fplt.candle_bull_color + transparency, 
                'bull_body': fplt.candle_bull_body_color + transparency,
                'bear_frame': fplt.candle_bear_color + transparency, 
                'bear_body': fplt.candle_bear_color + transparency,
            })
        
        # set gridlines
        ax.showGrid(x=True, y=True, alpha=0.2)

        # add legend of ohlc data
        def update_legend_text(x, y):
            dt = datetime.fromtimestamp(x // 1000000000)
            utcdt = dt.astimezone(pytz.utc).replace(tzinfo=timezone.utc)
            # dt = dt.replace(tzinfo=timezone.utc)
            row = ohlc.loc[utcdt]
            # format html with the candle and set legend
            fmt = '<span style="color:%s; margin: 16px;">%%s</span>' % (self.BULL if (row['o'] < row['c']).all() else self.BEAR)
            rawtxt = '<span style="font-size:14px">%%s %%s</span> &nbsp; O: %s H: %s L: %s C: %s' % (fmt, fmt, fmt, fmt)
            hover_label.setText(rawtxt % (self.inst, self.interval, row['o'], row['h'], row['l'], row['c']))
        fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
        
        # additional crosshair info
        def enrich_info(x, y, xtext, ytext):
            o = ohlc.iloc[x]['o']
            h = ohlc.iloc[x]['h']
            l = ohlc.iloc[x]['l']
            c = ohlc.iloc[x]['c']
            add_xt = f'\t{xtext}'
            add_yt = f'\tLevel: {ytext}\n\n\tOpen: {o}\n\tHigh: {h}\n\tLow: {l}\n\tClose: {c}'
            return add_xt, add_yt
        
        fplt.add_crosshair_info(enrich_info, ax=ax)


    def add_line(self, data, color='#F4D03F', width=1, legend=None):
        fplt.plot(data, ax=self.ax, color=color, width=width, legend=legend)


    def show(self):
        fplt.show()