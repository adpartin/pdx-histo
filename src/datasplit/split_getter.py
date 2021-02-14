import os
import numpy as np
import pandas as pd


def get_unq_split_ids(all_split_files):
    """ List containing the full path of each split. """
    unq = [all_split_files[i].split(os.sep)[-1].split('_')[1] for i, p in enumerate(all_split_files)]
    # unq = []
    # for i, p in enumerate(all_split_files): 
    #     sp = all_split_files[i].split(os.sep)[-1].split('_')[1]
    # unq.append(sp)
    unq = np.unique(unq)
    return unq


def get_data_by_id(idx, X, Y, meta=None):
    """ Returns a tuple of (features (x), target (y), metadata (m))
    for an input array of indices (idx). """
    x_data = X.iloc[idx, :].reset_index(drop=True)
    y_data = np.squeeze(Y.iloc[idx, :]).reset_index(drop=True)
    if meta is not None:
        m_data = meta.iloc[idx, :].reset_index(drop=True)
    else:
        meta = None
    return x_data, y_data, m_data