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

# Seed
np.random.seed(42)


fdir = Path(__file__).resolve().parent

DATADIR = fdir/'../../data'
MAIN_APPDIR = fdir/'../../apps'
APPNAME = 'bin_rsp_balanced_01'


outdir = MAIN_APPDIR/APPNAME
os.makedirs(outdir, exist_ok=True)

# import ipdb; ipdb.set_trace()

datapath = DATADIR/'data_merged.csv'
data = pd.read_csv(datapath)
print('Master dataframe', data.shape)


# -----------------------------------------------------------------------------
# Subsample data to create balanced dataset in terms of drug response and ctype
# -----------------------------------------------------------------------------
print('\nSubsample master dataframe to create balanced dataset.')
r0 = data[data.Response == 0]  # non-responders
r1 = data[data.Response == 1]  # responders

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
print('Final dataframe', df.shape)

aa = df.reset_index()
pprint(aa.groupby(['ctype', 'Response']).agg({'index': 'nunique'}).reset_index().rename(columns={'index': 'samples'}))

data = df
del dfs, df, aa, ctype, count

data.to_csv(outdir/'annotations.csv', index=False)
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
