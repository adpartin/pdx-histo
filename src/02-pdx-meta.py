"""
Merge three metadata files to create ../data/meta/meta_merged.csv:
1. ImageID_PDMRID_CrossRef.csv Meta that came with PDX images
2. PDX_Meta_Information.csv Meta from Yitan
3. meta_from_wsi_images.csv Meta that I (ap) extracted from WSI (SVS) images using histolab

Note! Before you run this, you need to generate meta_from_wsi_images.csv with 01-get-meta-from-images.ipynb.

Note! Yitan's file has some missing samples for which we do have the images.<br>
The missing samples either don't have response data or expression data.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from glob import glob
from pprint import pprint

dirpath = Path(__file__).resolve().parent


# =================================================
#   Part 1 - Merge meta files 1 and 2
# =================================================

# --------------------
# Loads two meta files
# --------------------
# Data path
datapath = dirpath/'../data'
metapath = datapath/'meta'

# PDX image meta (from NCI/Globus)
df_img = pd.read_csv(metapath/'ImageID_PDMRID_CrossRef.csv')
df_img = df_img.rename(columns={'Model': 'model'})
df_img = df_img[df_img['model'].notna()]
df_img = df_img.rename(columns={'Sample ID': 'sample_id', 
                                'Image ID': 'image_id',
                                'Capture Date': 'capture_date',
                                'Date Loaded to BW_Transfers': 'date_loaded_to_bw_transfers'})
df_img.insert(loc=1, column='patient_id',  value=df_img['model'].map(lambda x: x.split('~')[0]), allow_duplicates=True)
df_img.insert(loc=2, column='specimen_id', value=df_img['model'].map(lambda x: x.split('~')[1]), allow_duplicates=True)

# PDX Meta (from Yitan)
# Yitan doesn't have sample_id (??)
df_typ = pd.read_csv(metapath/'PDX_Meta_Information.csv')
print('Image meta', df_img.shape)
print('Yitan meta', df_typ.shape)

pprint(df_img[:2])
pprint(df_typ[:2])

# Drop items w/o images
df_img = df_img.dropna(subset=['image_id'])
df_img = df_img.astype({'image_id': int})
print(df_img.shape)
pprint(df_img[:2])

c = 'tumor_site_from_data_src'
print(df_typ[c].nunique())
tt = df_typ.groupby([c]).agg({'patient_id': 'nunique', 'specimen_id': 'nunique'}).reset_index()
pprint(tt)


# -------------------------------
# Explore data (but don't modify)
# -------------------------------
# Subset the columns
df1 = df_img[['model', 'patient_id', 'specimen_id', 'sample_id', 'image_id']]
df2 = df_typ[['patient_id', 'specimen_id', 'stage_or_grade']]
pprint(df1[:2])
pprint(df2[:2])

# Merge meta files
df = df1.merge(df2, on=['patient_id', 'specimen_id'], how='inner').reset_index(drop=True)

print(df1.shape)
print(df2.shape)
print(df.shape)

# Note that some items are missing in Yitan's file.
# The missing samples either don't have response or expression data.

# Explore (merge and identify from which df the items are coming from)
# https://kanoki.org/2019/07/04/pandas-difference-between-two-dataframes/
df = df1.merge(df2, on=['patient_id', 'specimen_id'], how='outer', indicator=True)
print('Inner merge', df.shape)
pprint(df[:2])

print('In both         ', df[df['_merge']=='both'].shape)
print('In left or right', df[df['_merge']!='both'].shape)

# Find which items are missing in Yitan's file
df_miss = df1.merge(df2, on=['patient_id', 'specimen_id'], how='outer', indicator=True).loc[lambda x : x['_merge']=='left_only']
df_miss = df_miss.sort_values(['patient_id', 'specimen_id'], ascending=True)
print('\nMissing items', df_miss.shape)
pprint(df_miss)


# -------------------
# Merge the two files
# -------------------
# Merge meta files
df_mrg = df_img.merge(df_typ, on=['patient_id', 'specimen_id'], how='inner').reset_index(drop=True)

# Drop cols
df_mrg = df_mrg.drop(columns=['capture_date', 'date_loaded_to_bw_transfers'])

# Sort
df_mrg = df_mrg.sort_values(['patient_id', 'specimen_id', 'sample_id'], ascending=True).reset_index(drop=True)

print(df1.shape)
print(df2.shape)
print(df_mrg.shape)
pprint(df_mrg[:2])


# ----------------------------
#   Stats
# ----------------------------
aa = df_mrg.groupby(['patient_id', 'specimen_id']).agg({'image_id': 'nunique'}).reset_index()
aa = aa.sort_values(by=['patient_id', 'specimen_id'])
print('Total images', aa['image_id'].sum())
pprint(aa)

c = 'tumor_site_from_data_src'
print(df_mrg[c].nunique())
tt = df_mrg.groupby([c]).agg({'patient_id': 'nunique', 
                              'specimen_id': 'nunique', 
                              'image_id': 'nunique'}).reset_index()
pprint(tt)

# print(df_mrg['simplified_tumor_site'].nunique())
# df_mrg['simplified_tumor_site'].value_counts()

c = 'tumor_type_from_data_src'
print(df_mrg[c].nunique())
tt = df_mrg.groupby(['tumor_type_from_data_src']).agg({'patient_id': 'nunique', 
                                                       'specimen_id': 'nunique', 
                                                       'image_id': 'nunique'}).reset_index()
pprint(tt)

# print(df_mrg['simplified_tumor_type'].nunique())
# df_mrg['simplified_tumor_type'].value_counts()

pprint(df_mrg['stage_or_grade'].value_counts())


# ========================================================
#   Part 2 - Merge with meta that we extracted from images
# ========================================================
meta_img = pd.read_csv(metapath/'meta_from_wsi_images.csv')
col_rename = {'aperio.ImageID': 'image_id',
              'aperio.MPP': 'MPP',
              'openslide.level[0].height': 'height',
              'openslide.level[0].width': 'width',
              'openslide.objective-power': 'power'
             }
meta_img = meta_img.rename(columns=col_rename)
print(meta_img.shape)

cols = set(meta_img.columns).intersection(set(list(col_rename.values())))
cols = ['image_id'] + [c for c in cols if c != 'image_id']
meta_img = meta_img[cols]
pprint(meta_img[:3])

# Finally, merge
df_final = df_mrg.merge(meta_img, how='inner', on='image_id')

print(df_mrg.shape)
print(meta_img.shape)
print(df_final.shape)
pprint(df_final[:3])

df_final.to_csv(metapath/'meta_merged.csv', index=False)