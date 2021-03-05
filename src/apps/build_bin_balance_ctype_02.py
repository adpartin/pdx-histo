""" 
Build a dataset for binary classification.
The dataset is balanced in terms of ctype.
Specifically, we extracted the 2 most prevalent ctypes
and created a balanced dataset.

It generates a dataset with samples that have both histology rna and data.
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

from build_df import load_rna
from merge_meta_files import load_crossref, load_pdx_meta

# parser = argparse.ArgumentParser()
# parser.add_argument('-sf', '--slideflow',
#                     action='store_true',
#                     help='Add slideflow columns.')
# args = parser.parse_args()

APPNAME = 'bin_ctype_balance_02'
outdir = cfg.MAIN_APPDIR/APPNAME
os.makedirs(outdir, exist_ok=True)

import ipdb; ipdb.set_trace()
#datapath = cfg.DATADIR/'data_merged.csv'
#data = pd.read_csv(datapath)
#print('\nMaster dataframe', data.shape)

# Load data
rna = load_rna()
cref = load_crossref()
pdx = load_pdx_meta()

mrg_cols = ['model', 'patient_id', 'specimen_id', 'sample_id']

# Add columns to rna by parsing the Sample col
patient_id = rna['Sample'].map(lambda x: x.split('~')[0])
specimen_id = rna['Sample'].map(lambda x: x.split('~')[1])
sample_id = rna['Sample'].map(lambda x: x.split('~')[2])
model = [a + '~' + b for a, b in zip(patient_id, specimen_id)]
rna.insert(loc=1, column='model', value=model, allow_duplicates=True)
rna.insert(loc=2, column='patient_id', value=patient_id, allow_duplicates=True)
rna.insert(loc=3, column='specimen_id', value=specimen_id, allow_duplicates=True)
rna.insert(loc=4, column='sample_id', value=sample_id, allow_duplicates=True)
rna = rna.sort_values(mrg_cols)

# Remove bad samples with bad slides
cref = cref[~cref.image_id.isin(cfg.BAD_SLIDES)].reset_index(drop=True)


# -----------------------------------------------------------
# For some samples, we have histology slides but not rna-seq.
# Specifically, we miss rna-seq for 37 samples.
# -----------------------------------------------------------
# Subset the columns
df1 = cref[mrg_cols + ['image_id']]
df2 = rna

# Merge meta files
mrg = df1.merge(df2, on=mrg_cols, how='inner').reset_index(drop=True)
print('cref', df1.shape)
print('rna ', df2.shape)
print('mrg ', mrg.shape)

# Explore (merge and identify from which df the items are coming from)
# https://kanoki.org/2019/07/04/pandas-difference-between-two-dataframes/
# Find which items are missing in Yitan's file
mrg_outer = df1.merge(df2, on=mrg_cols, how='outer', indicator=True)
print('Outer merge', mrg_outer.shape)
print(mrg_outer['_merge'].value_counts())

miss = mrg_outer.loc[lambda x: x['_merge']=='left_only']
miss = miss.sort_values(mrg_cols, ascending=True)
print('\nMissing items', miss.shape)

# Consider filling the rnaseq from sample_ids of the sample model
# rna[rna.model.isin(miss.model)]

del df1, df2, mrg, mrg_outer, miss
# -----------------------------------------------------------

# Merge cref and rna
cref_rna = cref[mrg_cols + ['image_id']].merge(rna, on=mrg_cols, how='inner').reset_index(drop=True)
# Note that we also loose some samples when we merge with pdx metadata
data = pdx.merge(cref_rna, on=['patient_id', 'specimen_id'], how='inner').reset_index(drop=True)
# Re-org cols
cols = ['Sample', 'model', 'patient_id', 'specimen_id', 'sample_id', 'image_id', 
        'csite_src', 'ctype_src', 'csite', 'ctype', 'stage_or_grade']
ge_cols = [c for c in data.columns if str(c).startswith('ge_')]
data = data[cols + ge_cols]


# -----------------------------------------------------------
# Subsample data to create balanced dataset in terms of ctype
# -----------------------------------------------------------

# Get the top_n most prevalent ctypes in terms of the number wsi slides
top_n = 2
#ctypes_df = data.groupby('ctype').agg({'smp': 'nunique', 'sample_id': 'nunique', 'image_id': 'nunique'}).reset_index()
ctypes_df = data.groupby('ctype').agg({'sample_id': 'nunique', 'image_id': 'nunique'}).reset_index()
ctypes_df = ctypes_df.sort_values('image_id', ascending=False)
# ctypes_df = ctypes_df.rename(columns={'sample_id': 'sample_ids', 'image_id': 'image_ids'})
pprint(ctypes_df)

print(f'\nGet samples of the {top_n} most prevalent ctypes.')
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
#df['ctype_label'] = df['ctype_label'].map(lambda x: 0 if x == 10 else x)
#pprint(df['ctype_label'].value_counts())
pprint(df['ctype'].value_counts())

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
