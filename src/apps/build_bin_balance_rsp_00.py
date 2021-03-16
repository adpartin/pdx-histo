""" 
Build a dataset for binary classification with multimodal features.
The dataset is balanced in terms of drug response and ctype.
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

fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/'..'))
from config import cfg

APPNAME = 'bin_rsp_balance_01'
outdir = cfg.MAIN_APPDIR/APPNAME
os.makedirs(outdir, exist_ok=True)

# import ipdb; ipdb.set_trace()
datapath = cfg.DATADIR/'data_merged.csv'
data = pd.read_csv(datapath)
print('\nMaster dataframe', data.shape)

df = data[~data.image_id.isin(cfg.BAD_SLIDES)].reset_index(drop=True)


# Remove duplicates for drug prediction with images only
# Note!
# this is required for predicting with images only
df = df.drop_duplicates(subset=['Response', 'image_id'])
df = df.sort_values('Response', ascending=False)
# df[df.duplicated(subset=['image_id'], keep=False)].sort_values('image_id')[['image_id', 'Response']]
df = df.drop_duplicates(subset=['image_id'], keep='first')


# -----------------------------------------------------------------------------
# Subsample data to create balanced dataset in terms of drug response and ctype
# -----------------------------------------------------------------------------
print('\nSubsample master dataframe to create balanced dataset.')
r0 = df[df.Response == 0]  # non-responders
r1 = df[df.Response == 1]  # responders

# Aggregate non-responders to balance the responders
dfs = []
for ctype, count in r1.ctype.value_counts().items():
    # print(ctype, count)
    aa = r0[r0.ctype == ctype]
    if aa.shape[0] > count:
        aa = aa.sample(n=count)
    dfs.append(aa)

# Concat non-responders and responders
aa = pd.concat(dfs, axis=0)
df = pd.concat([aa, r1], axis=0).reset_index(drop=True)
df = df.sort_values('smp', ascending=True)
del dfs, aa, ctype, count

df = df.reset_index()
pprint(df.groupby(['ctype', 'Response']).agg({'index': 'nunique'}).reset_index().rename(columns={'index': 'samples'}))

pprint(df['Response'].value_counts())

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


# -------------------------------------
# Copy slides to training_slides folder
# -------------------------------------
# print('\nCopy slides to training_slides folder.')
# src_img_path = DATADIR/'doe-globus-pdx-data'
# dst_img_path = DATADIR/'training_slides'
# os.makedirs(dst_img_path, exist_ok=True)

# exist = []
# copied = []
# for fname in data.image_id.unique():
#     if (dst_img_path/f'{fname}.svs').exists():
#         exist.append(fname)
#     else:
#         _ = shutil.copyfile(str(src_img_path/f'{fname}.svs'), str(dst_img_path/f'{fname}.svs'))
#         copied.append(fname)

# print(f'Copied slides:   {len(copied)}')
# print(f'Existing slides: {len(exist)}')
