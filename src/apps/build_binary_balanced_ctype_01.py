""" 
Build a dataset for binary classification.
The dataset is balanced in terms of ctype.
Specifically, we extracted all response samples and matched the same
number of non-response samples of the same ctype.
"""
import os
import sys
assert sys.version_info >= (3, 5)

from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np

# Seed
np.random.seed(42)


fdir = Path(__file__).resolve().parent

DATADIR = fdir/'../../data'
MAIN_APPDIR = fdir/'../../apps'
APPNAME = 'bin_ctype_balanced_01'


outdir = MAIN_APPDIR/APPNAME
os.makedirs(outdir, exist_ok=True)

# import ipdb; ipdb.set_trace()

datapath = DATADIR/'data_merged.csv'
data = pd.read_csv(datapath)
print('Master dataframe', data.shape)


# -----------------------------------------------------------
# Subsample data to create balanced dataset in terms of ctype
# -----------------------------------------------------------

# Get the top_n most prevalent ctypes in terms of the number wsi slides
top_n = 2
ctypes_df = data.groupby('ctype').agg({'smp': 'nunique', 'sample_id': 'nunique', 'image_id': 'nunique'}).reset_index()
ctypes_df = ctypes_df.sort_values('image_id', ascending=False)
# ctypes_df = ctypes_df.rename(columns={'sample_id': 'sample_ids', 'image_id': 'image_ids'})
pprint(ctypes_df)

ctypes_df_top = ctypes_df[:top_n].reset_index(drop=True)
pprint(ctypes_df_top)

ctypes = ctypes_df_top.ctype.values
min_count = ctypes_df_top.image_id.min()

# Aggregate non-responders to balance the responders
print('\nCreate balanced dataset.')
dfs = []
for ctype in ctypes:
    # print(ctype)
    aa = data[data.ctype == ctype]
    aa = aa.drop_duplicates(subset=['image_id'])
    aa = aa.sample(n=min_count)
    dfs.append(aa)

df = pd.concat(dfs, axis=0).reset_index(drop=True)
df = df.sort_values('ctype', ascending=True).reset_index(drop=True)

# make labels only 0 and 1
df['ctype_label'] = df['ctype_label'].map(lambda x: 0 if x == 10 else x)
pprint(df['ctype_label'].value_counts())

print('Final dataframe', df.shape)
df.to_csv(outdir/'annotations.csv', index=False)
print('\nDone.')
