import numpy as np
import pandas as pd
import sklearn
from typing import List, Dict, Optional
from src.config import cfg


class TidyData():
    """ This class manages tidy data for drug response prediction. """

    def __init__(self, data,
                 ge_prfx: str=None,
                 dd1_prfx: str=None,
                 dd2_prfx: str=None,
                 index_col_name: Optional[str]=None,
                 # split_ids: Optional[List]=None
                 split_ids: Optional[Dict]={"tr_id": [], "vl_id": [], "te_id": []}
                 ):
        """ 
        Args:
            df : tidy dataframe that features and metadata (including the response)
            index_col_name : col name in data by which we should split the data using split_ids

        Example:
        TidyData(data, ge_prfx="ge_", dd1_prfx="dd1_", dd2_prfx="dd2_", index_col_name="index",
                 split_ids={"tr_id": tr_id, "vl_id": vl_id, "te_id": te_id})
        """
        self.data = data
        self.ge_prfx = ge_prfx
        self.dd1_prfx = dd1_prfx
        self.dd2_prfx = dd2_prfx
        self.index_col_name = index_col_name
        self.split_ids = split_ids

        # self.is_ge_scaled = False
        # self.is_dd1_scaled = False
        # self.is_dd2_scaled = False

        if self.ge_prfx is not None:
            self.ge_cols = [c for c in self.data.columns if c.startswith(self.ge_prfx)]

        if self.dd1_prfx is not None:
            self.dd1_cols = [c for c in self.data.columns if c.startswith(self.dd1_prfx)]

        if self.dd2_prfx is not None:
            self.dd2_cols = [c for c in self.data.columns if c.startswith(self.dd2_prfx)]

        fea_cols = self.ge_cols + self.dd1_cols + self.dd2_cols
        self.meta_cols = [c for c in self.data.columns if c not in fea_cols]
        # self.meta = self.data[self.meta_cols]

        # import ipdb; ipdb.set_trace()
        # Train data
        tr_data = self.split(df=self.data, ids=split_ids["tr_id"])
        tr_ge  = tr_data[self.ge_cols]
        tr_dd1 = tr_data[self.dd1_cols]
        tr_dd2 = tr_data[self.dd2_cols]
        self.tr_meta = tr_data[self.meta_cols]
        self.tr_ge = self.scale_ge(tr_ge)
        self.tr_dd2 =self.scale_dd1(tr_dd1)
        self.tr_dd2 =self.scale_dd2(tr_dd2)
        del tr_data, tr_ge, tr_dd1, tr_dd2

        # Validation data
        vl_data = self.split(df=self.data, ids=split_ids["vl_id"])
        vl_ge  = vl_data[self.ge_cols]
        vl_dd1 = vl_data[self.dd1_cols]
        vl_dd2 = vl_data[self.dd2_cols]
        self.vl_meta = vl_data[self.meta_cols]
        self.vl_ge  = pd.DataFrame(self.ge_scaler.transform(vl_ge), columns=self.ge_cols, dtype=cfg.GE_DTYPE)
        self.vl_dd1 = pd.DataFrame(self.dd1_scaler.transform(vl_dd1), columns=self.dd1_cols, dtype=cfg.DD_DTYPE)
        self.vl_dd2 = pd.DataFrame(self.dd2_scaler.transform(vl_dd2), columns=self.dd2_cols, dtype=cfg.DD_DTYPE)
        del vl_data, vl_ge, vl_dd1, vl_dd2

        # Test data
        te_data = self.split(df=self.data, ids=split_ids["te_id"])
        te_ge  = te_data[self.ge_cols]
        te_dd1 = te_data[self.dd1_cols]
        te_dd2 = te_data[self.dd2_cols]
        self.te_meta = te_data[self.meta_cols]
        self.te_ge  = pd.DataFrame(self.ge_scaler.transform(te_ge), columns=self.ge_cols, dtype=cfg.GE_DTYPE)
        self.te_dd1 = pd.DataFrame(self.dd1_scaler.transform(te_dd1), columns=self.dd1_cols, dtype=cfg.DD_DTYPE)
        self.te_dd2 = pd.DataFrame(self.dd2_scaler.transform(te_dd2), columns=self.dd2_cols, dtype=cfg.DD_DTYPE)
        del te_data, te_ge, te_dd1, te_dd2

        # self.tr_ge = self.split(df=self.ge, ids=split_ids["tr_id"])
        # self.vl_ge = self.split(df=self.ge, ids=split_ids["vl_id"])
        # self.te_ge = self.split(df=self.ge, ids=split_ids["te_id"])

        # self.tr_dd1 = self.split(df=self.dd1, ids=split_ids["tr_id"])
        # self.vl_dd1 = self.split(df=self.dd1, ids=split_ids["vl_id"])
        # self.te_dd1 = self.split(df=self.dd1, ids=split_ids["te_id"])

        # self.tr_dd2 = self.split(df=self.dd2, ids=split_ids["tr_id"])
        # self.vl_dd2 = self.split(df=self.dd2, ids=split_ids["vl_id"])
        # self.te_dd2 = self.split(df=self.dd2, ids=split_ids["te_id"])

        # self.tr_meta = self.split(df=self.meta, ids=split_ids["tr_id"])
        # self.vl_meta = self.split(df=self.meta, ids=split_ids["vl_id"])
        # self.te_meta = self.split(df=self.meta, ids=split_ids["te_id"])

    def get_scaler(self, fea_df,
                   scaler_name: str="standard",
                   use_set: str="all",
                   print_fn=print):
        """ Returns a sklearn scaler object. """
        fea_df = fea_df.drop_duplicates().reset_index(drop=True)
        if fea_df.shape[0] == 0:
            # TODO: add warning!
            return None

        # if use == "all":
        #     print("scale using all data")
        # elif use == "train":
        #     print("scale using only training data")

        if scaler_name == "standard":
            scaler = sklearn.preprocessing.StandardScaler()
        elif scaler_name == "minmax":
            scaler = sklearn.preprocessing.MinMaxScaler()
        elif scaler_name == "robust":
            scaler = sklearn.preprocessing.RobustScaler()
        else:
            print_fn(f"The specified scaler {scaler_name} is not supported (not scaling).")
            return None

        scaler.fit(fea_df)
        return scaler

    def scale_ge(self, df):
        if self.ge_cols is not None:
            dtype = cfg.GE_DTYPE
            cols = self.ge_cols
            # fea = self.data[cols].reset_index(drop=True)
            fea = df[cols].reset_index(drop=True)
            scaler = self.get_scaler(fea)

            self.ge_scaler = scaler
            # self.is_ge_scaled = True
            # self.ge = pd.DataFrame(scaler.transform(fea), columns=cols, dtype=dtype)
            return pd.DataFrame(scaler.transform(fea), columns=cols, dtype=dtype)

    def scale_dd1(self, df):
        if self.dd1_cols is not None:
            dtype = cfg.DD_DTYPE
            cols = self.dd1_cols
            # fea = self.data[cols].reset_index(drop=True)
            fea = df[cols].reset_index(drop=True)
            scaler = self.get_scaler(fea)

            self.dd1_scaler = scaler
            # self.is_dd1_scaled = True
            # self.dd1 = pd.DataFrame(scaler.transform(fea), columns=cols, dtype=dtype)
            return pd.DataFrame(scaler.transform(fea), columns=cols, dtype=dtype)

    def scale_dd2(self, df):
        if self.dd2_cols is not None:
            dtype = cfg.DD_DTYPE
            cols = self.dd2_cols
            # fea = self.data[cols].reset_index(drop=True)
            fea = df[cols].reset_index(drop=True)
            scaler = self.get_scaler(fea)

            self.dd2_scaler = scaler
            # self.is_dd2_scaled = True
            # self.dd2 = pd.DataFrame(scaler.transform(fea), columns=cols, dtype=dtype)
            return pd.DataFrame(scaler.transform(fea), columns=cols, dtype=dtype)

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

    def split(self, df, ids):
        """ ... """
        # Obtain the relevant ids
        if self.index_col_name in self.data:
            df = self.data[self.data[self.index_col_name].isin(ids)].reset_index(drop=True)
        else:
            df = self.data.iloc[ids, :].reset_index(drop=True)
        return df

    # def create_scaler(self, use: str="all"):
    #     # TODO
    #     if use == "all":
    #         print("scale using all data")
    #     elif use == "train":
    #         print("scale using only training data")


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
