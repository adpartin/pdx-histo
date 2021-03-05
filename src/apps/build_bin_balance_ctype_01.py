""" 
Build a dataset for binary classification.
The dataset is balanced in terms of ctype.
To generate the dataset, we used the data_merged.csv file
which was constructed by merging meta from histology slides,
crossref file (which came with the slides) and drug response df.

We got samples from the top n most prevalent ctypes in terms of
the number slides.
"""
import os
import sys
assert sys.version_info >= (3, 5)

import argparse
from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np

# from config import cfg

fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/'..'))
from config import cfg

# parser = argparse.ArgumentParser()
# parser.add_argument('-sf', '--slideflow',
#                     action='store_true',
#                     help='Add slideflow columns.')
# args = parser.parse_args()

APPNAME = 'bin_ctype_balance_01'
outdir = cfg.MAIN_APPDIR/APPNAME
os.makedirs(outdir, exist_ok=True)

import ipdb; ipdb.set_trace()
datapath = cfg.DATADIR/'data_merged.csv'
data = pd.read_csv(datapath)
print('\nMaster dataframe', data.shape)

df = data[~data.image_id.isin(cfg.BAD_SLIDES)].reset_index(drop=True)

# -----------------------------------------------------------
# Subsample data to create balanced dataset in terms of ctype
# -----------------------------------------------------------

# Get the top_n most prevalent ctypes in terms of the number wsi slides
top_n = 2
ctypes_df = data.groupby('ctype').agg({'smp': 'nunique', 'sample_id': 'nunique', 'image_id': 'nunique'}).reset_index()
ctypes_df = ctypes_df.sort_values('image_id', ascending=False)
# ctypes_df = ctypes_df.rename(columns={'sample_id': 'sample_ids', 'image_id': 'image_ids'})
pprint(ctypes_df)

print(f'\nGet samples of the {top_n} most prevalent ctypes in terms of number of histology slides.')
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

# save annotations file
df.to_csv(outdir/cfg.ANNOTATIONS_FILENAME, index=False)
print('\nFinal dataframe', df.shape)

# add slideflow columns and save annotations file
print('\nCreate and save annotations file for slideflow.')
df_sf = df.reset_index(drop=True)
df_sf.insert(loc=1, column='submitter_id', value=df_sf['image_id'].values, allow_duplicates=False)
df_sf.insert(loc=2, column='slide', value=df_sf['image_id'].values, allow_duplicates=False)
df_sf.to_csv(outdir/cfg.SF_ANNOTATIONS_FILENAME, index=False)

print('\nDone.')
