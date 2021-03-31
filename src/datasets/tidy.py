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


def split_data_and_extract_fea(data, ids,
                               ge_cols, dd_cols,
                               ge_scaler, dd_scaler,
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
        df = df.reset_index(drop=True)
    else:
        df = data.iloc[ids, :].reset_index(drop=True)

    # Extract features
    ge, dd = df[ge_cols], df[dd_cols]

    # Scale
    dd = pd.DataFrame(dd_scaler.transform(dd), columns=dd_cols, dtype=dd_dtype)
    ge = pd.DataFrame(ge_scaler.transform(ge), columns=ge_cols, dtype=ge_dtype)
    return ge, dd
