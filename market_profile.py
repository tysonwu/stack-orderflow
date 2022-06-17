from datetime import datetime

import numpy as np
import pandas as pd

def _midmax_idx(array):
    if len(array) == 0:
        return None

    # Find candidate maxima
    maxima_idxs = np.argwhere(array == np.amax(array))[:,0]
    if len(maxima_idxs) == 1:
        return maxima_idxs[0]
    elif len(maxima_idxs) <= 1:
        return None

    # Find the distances from the midpoint to find
    # the maxima with the least distance
    midpoint = len(array) / 2
    v_norm = np.vectorize(np.linalg.norm)
    maximum_idx = np.argmin(v_norm(maxima_idxs - midpoint))

    return maxima_idxs[maximum_idx]


def find_poc(profile) -> float:
    """Calculates metrics based on market profiles

    Args:
        profile (pd.DataFrame): a dataframe with price levels as index and one column named 'q' for quantity
    """
    poc_idx = _midmax_idx(profile['q'].tolist())
    if poc_idx is not None:
        poc_volume = profile['q'].iloc[poc_idx]
        poc_price_level = profile.index[poc_idx]
    else:
        poc_volume = None
        poc_price_level = None

    return poc_volume, poc_price_level


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
        # delta: defined as ask volume minus bid volume
        self.delta_profile = pd.DataFrame({timepoint: [row[2] for row in orderflow]}, index=self.price_levels)
        self.bidask_profile = pd.DataFrame({'b': [row[3] for row in orderflow], 'a': [row[4] for row in orderflow]}, index=self.price_levels)
        # calc POC
        self.poc_volume, self.poc_price_level = find_poc(self.profile)
