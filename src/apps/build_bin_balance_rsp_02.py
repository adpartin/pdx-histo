""" 
Build a dataset for binary classification with multimodal features.
The dataset is balanced in terms of drug response and ctype.
Specifically, we extracted all response samples and matched the same
number of non-response samples of the same ctype.

Here we use the full set of features including, slides, rna, and meta.
"""
import os
import sys
assert sys.version_info >= (3, 5)

from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np

fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/'..'))
from config import cfg

import load_data
from load_data import PDX_SAMPLE_COLS

APPNAME = 'bin_rsp_balance_02'
outdir = cfg.MAIN_APPDIR/APPNAME
os.makedirs(outdir, exist_ok=True)

target = 'Response'

# Load data
rsp = load_data.load_rsp()
rna = load_data.load_rna()
dd = load_data.load_dd()
cref = load_data.load_crossref()
pdx = load_data.load_pdx_meta2()

# Merge rsp with rna
print('\nMerge rsp and rna')
print(rsp.shape)
print(rna.shape)
rsp_rna = rsp.merge(rna, on='Sample', how='inner')
print(rsp_rna.shape)

# Merge with dd
print('Merge with descriptors')
print(rsp_rna.shape)
print(dd.shape)
rsp_rna_dd = rsp_rna.merge(dd, left_on='Drug1', right_on='ID', how='inner').reset_index(drop=True)
print(rsp_rna_dd.shape)

# Merge with pdx meta
print('Merge with pdx meta')
print(pdx.shape)
print(rsp_rna_dd.shape)
rsp_rna_dd_pdx = pdx.merge(rsp_rna_dd, on=['patient_id', 'specimen_id'], how='inner')
print(rsp_rna_dd_pdx.shape)

# Merge cref
print('Merge with cref')
# (we loose some samples because we filter the bad slides)
print(cref.shape)
print(rsp_rna_dd_pdx.shape)
data = cref.merge(rsp_rna_dd_pdx, on=PDX_SAMPLE_COLS, how='inner')
print(data.shape)

# Add 'slide' column
data.insert(loc=5, column='slide', value=data['image_id'], allow_duplicates=True)

# Remove duplicates for drug prediction with images only
# Note!
# this is required for predicting with images only
# df = data.copy()
# df = df.drop_duplicates(subset=[target, 'image_id'])
# df = df.sort_values(target, ascending=False)
# df = df.drop_duplicates(subset=['image_id'], keep='first')
# df = df.reset_index(drop=True)

ge_cols = [c for c in rna.columns if c.startswith('ge_')]
rna = rna.sort_values('Sample', ascending=True)
dup_vec = rna.duplicated(subset=ge_cols, keep=False)
print('Dups', sum(dup_vec))

# -----------------------------------------------------------------------------
# Subsample data to create balanced dataset in terms of drug response and ctype
# -----------------------------------------------------------------------------
print('\nSubsample master dataframe to create balanced dataset.')
r0 = data[data[target] == 0]  # non-responders
r1 = data[data[target] == 1]  # responders

# Aggregate non-responders to balance the responders
dfs = []
for ctype, count in r1['ctype'].value_counts().items():
    # print(ctype, count)
    aa = r0[r0.ctype == ctype]
    if aa.shape[0] > count:
        aa = aa.sample(n=count)
    dfs.append(aa)

# Concat non-responders and responders
aa = pd.concat(dfs, axis=0)
df = pd.concat([aa, r1], axis=0)
df = df.sort_values('smp', ascending=True).reset_index(drop=True)
del dfs, aa, ctype, count, r0, r1

# --------------------------------------------------------------
# import ipdb; ipdb.set_trace()

bb = data[~data['smp'].isin(df['smp'].values)]
bb = bb[bb['ctype'].isin(df['ctype'].unique())]

dfs = []
for ctype, count in df['ctype'].value_counts().items():
    # print(ctype, count)
    aa = bb[bb.ctype == ctype]
    if aa.shape[0] > count:
        aa = aa.sample(n=count)
    dfs.append(aa)

aa = pd.concat(dfs, axis=0)
df = pd.concat([aa, df], axis=0)
df = df.sort_values('smp', ascending=True).reset_index(drop=True)
del dfs, aa, ctype, count
# --------------------------------------------------------------

df = df.reset_index()
pprint(df.groupby(['ctype', target]).agg({'index': 'nunique'}).reset_index().rename(columns={'index': 'samples'}))

pprint(df[target].value_counts())

# Save annotations file
df.to_csv(outdir/cfg.ANNOTATIONS_FILENAME, index=False)
print('\nFinal dataframe', df.shape)

# add slideflow columns and save annotations file
print('\nCreate and save annotations file for slideflow.')
df_sf = df.reset_index(drop=True)
df_sf.insert(loc=1, column='submitter_id', value=df_sf['image_id'].values, allow_duplicates=False)
if 'slide' not in df_sf.columns:
    df_sf.insert(loc=2, column='slide', value=df_sf['image_id'].values, allow_duplicates=False)
df_sf.to_csv(outdir/cfg.SF_ANNOTATIONS_FILENAME, index=False)

print('\nDone.')
