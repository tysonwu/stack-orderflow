import numpy as np


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
