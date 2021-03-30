"""
!!!!!!!!!!!!!!!!!!!!  Experimetnal script !!!!!!!!!!!!!!!!!!!!
"""

from pathlib import Path
import numpy as np
import pandas as pd
from pandas_streaming.df import train_test_apart_stratify

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


fdir = Path(__file__).resolve().parent

dpath = Path("/vol/ml/apartin/projects/pdx-histo/data/processed/tidy_partially_balanced")
data = pd.read_csv(dpath/"annotations.csv")

# split_on = "Group"
split_on = "slide"
target_name = "Response"
# te_size = 0.1
te_size = 0.2
seeds = np.array(range(5))
seed = seeds[0]

# import ipdb; ipdb.set_trace()

df = data.reset_index()[["index", split_on, target_name]]
# df = df.loc[idx_vec]

# tt = {}
# vv = {}
# ee = {}
# for i, seed in enumerate(seeds):
#     df_new, test = train_test_apart_stratify(df, group=split_on, stratify=target_name, force=True, test_size=te_size, random_state=seed)
#     # df_new, test = train_test_apart_stratify(df, group=split_on, stratify=target_name, force=True, test_size=te_size, random_state=None)
#     print(set(df_new[split_on]).intersection(set(test[split_on])))
#     te_id = sorted(test["index"].values)
#     ee[i] = te_id

#     train, val = train_test_apart_stratify(df_new, group=split_on, stratify=target_name, force=False, test_size=te_size, random_state=seed)
#     # train, val = train_test_apart_stratify(df_new, group=split_on, stratify=target_name, force=False, test_size=te_size, random_state=None)
#     print(set(train[split_on]).intersection(set(val[split_on])))
#     tr_id = sorted(train["index"].values)
#     vl_id = sorted(val["index"].values)
#     vv[i] = vl_id
#     tt[i] = tr_id

# import ipdb; ipdb.set_trace()
# print(ee[0])
# print(ee[1])
# print(ee[0] == ee[1])

# np.random.seed(1)
cv_folds = 5
shuffle = True
random_state = 1
# cv = GroupKFold(n_splits=cv_folds)
cv = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)

import ipdb; ipdb.set_trace()
tt = {}
ee = {}
for i, (tr_id, te_id) in enumerate(cv.split(X=df, y=df[target_name], groups=df[split_on])):
    # ee[i] = sorted(te_id)
    # tt[i] = sorted(tr_id)
    ee[i] = te_id
    tt[i] = tr_id
    tr_df = df.loc[tr_id]
    te_df = df.loc[te_id]
    print(set(tr_df[split_on]).intersection(set(te_df[split_on])))
    print(tr_df[target_name].value_counts())
    print(te_df[target_name].value_counts())

import ipdb; ipdb.set_trace()
print(ee[0])
print(ee[1])
print(ee[0] == ee[1])
tt1 = tt
ee1 = ee
print("\nDone.")


# np.random.seed(2)
cv_folds = 5
shuffle = True
random_state = 2
# cv = GroupKFold(n_splits=cv_folds)
cv = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)

import ipdb; ipdb.set_trace()
tt = {}
ee = {}
for i, (tr_id, te_id) in enumerate(cv.split(X=df, y=df[target_name], groups=df[split_on])):
    # ee[i] = sorted(te_id)
    # tt[i] = sorted(tr_id)
    ee[i] = te_id
    tt[i] = tr_id
    tr_df = df.loc[tr_id]
    te_df = df.loc[te_id]
    print(set(tr_df[split_on]).intersection(set(te_df[split_on])))

import ipdb; ipdb.set_trace()
print(ee[0])
print(ee[1])
print(ee[0] == ee[1])
tt2 = tt
ee2 = ee
print("\nDone.")

# print(ee1)
# print(ee2)
print(ee1[0])
print(ee2[0])
