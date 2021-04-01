import numpy as np
import pandas as pd
from typing import List, Optional


class TidyData():

    def __init__(self, df,
                 ge_prfx: str="ge_",
                 dd_prfx: str="dd_",
                 split_ids: Optional[List]=None):
        """ ... """
        self.df = df
        self.ge_prfx = ge_prfx
        self.dd_prfx = dd_prfx
        self.splits_id = splits_id

        if self.ge_prfx is not None:
            self.ge_cols = [c for c in self.df.columns if c.startswith(self.ge_prfx)]
        if self.dd_prfx is not None:
            self.dd_cols = [c for c in self.df.columns if c.startswith(self.dd_prfx)]

    def split(self):
        # TODO
        pass

    def create_scaler(self, use: str="all"):
        # TODO
        if use == "all":
            print("scale using all data")
        elif use == "train":
            print("scale using only training data")

    def is_ge_scaled(self):
        return True if self.scaled_flag is True else False

    def scale_ge(self):
        if self.ge is not None:
            self.ge = pd.DataFrame(ge_scaler.transform(self.ge), columns=self.ge_cols, dtype=cfg.GE_DTYPE)
            self.scaled_flag = True

    def get_meta(self, of: str="all"):
        if of == "all":
            pass
        elif of == "train":
            pass
        elif of == "val":
            pass
        elif of == "test":
            pass
        else:
            pass


def split_data_and_extract_fea(data, ids, split_on,
                               ge_cols, dd1_cols, dd2_cols,
                               ge_scaler=None, dd2_scaler=None, dd1_scaler=None,
                               index_col_name: Optional[str]=None,
                               ge_dtype=np.float32, dd_dtype=np.float32):
    """ Split data into T/V/E using the provided ids and extract the separate
    features.

    Args:
        index_col_name : column that is used to index the dataset

    TODO:
    create class TidyData
    """
    # Obtain the relevant ids
    if index_col_name in data.columns:
        df = data[data[index_col_name].isin(ids)]
    else:
        df = data.iloc[ids, :]

    # df = df.sort_values(split_on, ascending=True)  # TODO: this line makes the model worse! why??
    df = df.reset_index(drop=True)

    # Extract features
    ge, dd1, dd2 = df[ge_cols], df[dd1_cols], df[dd2_cols]

    # Extract meta
    meta = df.drop(columns=ge_cols + dd1_cols + dd2_cols)

    # Scale
    if dd1_scaler is not None:
        dd1 = pd.DataFrame(dd1_scaler.transform(dd1), columns=dd1_cols, dtype=dd_dtype)

    if dd2_scaler is not None:
        dd2 = pd.DataFrame(dd2_scaler.transform(dd2), columns=dd2_cols, dtype=dd_dtype)

    if ge_scaler is not None:
        ge = pd.DataFrame(ge_scaler.transform(ge), columns=ge_cols, dtype=ge_dtype)

    return ge, dd1, dd2, meta


def extract_fea(data, 
                ge_cols, dd1_cols, dd2_cols,
                ge_scaler=None, dd2_scaler=None, dd1_scaler=None,
                ge_dtype=np.float32, dd_dtype=np.float32):
    """ Split data into T/V/E using the provided ids and extract the separate
    features.

    TODO:
    create class TidyData
    """
    # Extract features
    ge, dd1, dd2 = data[ge_cols], data[dd1_cols], data[dd2_cols]

    # Scale
    if dd1_scaler is not None:
        dd1 = pd.DataFrame(dd1_scaler.transform(dd1), columns=dd1_cols, dtype=dd_dtype)

    if dd2_scaler is not None:
        dd2 = pd.DataFrame(dd2_scaler.transform(dd2), columns=dd2_cols, dtype=dd_dtype)

    if ge_scaler is not None:
        ge = pd.DataFrame(ge_scaler.transform(ge), columns=ge_cols, dtype=ge_dtype)

    return ge, dd1, dd2
