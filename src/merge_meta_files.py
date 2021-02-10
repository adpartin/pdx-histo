"""
Merge 3 metadata files to create ../data/meta/meta_merged.csv:

1. _ImageID_PDMRID_CrossRef.xlsx:  meta comes with PDX slides; crossref
2. PDX_Meta_Information.csv:       meta from Yitan; pdx_meta
3. meta_from_wsi_slides.csv:       meta extracted from SVS slides using openslide; slides_meta

Note! Before running this code, generate meta_from_wsi_slides.csv with get_meta_from_slides.py.

Note! Yitan's file has some missing samples for which we do have the slides.
The missing samples either don't have response data or expression data.
"""

import os
import sys
from pathlib import Path
import glob
from pprint import pprint
import pandas as pd
import numpy as np


dirpath = Path(__file__).resolve().parent


def get_dups(df):
    """ Return df of duplicates. """
    df = df[df.duplicated(keep=False) == True]
    return df


def _explore(cref, pdx):
    """ Local func to exolore intermediate data. """

    c = 'tumor_site_from_data_src'
    print(pdx[c].nunique())
    tt = pdx.groupby([c]).agg({'patient_id': 'nunique', 'specimen_id': 'nunique'}).reset_index()
    pprint(tt)

    # Subset the columns
    df1 = cref[['model', 'patient_id', 'specimen_id', 'sample_id', 'image_id']]
    df2 = pdx[['patient_id', 'specimen_id', 'stage_or_grade']]
    pprint(df1[:2])
    pprint(df2[:2])

    # Merge meta files
    df = df1.merge(df2, on=['patient_id', 'specimen_id'], how='inner').reset_index(drop=True)
    print(df1.shape)
    print(df2.shape)
    print(df.shape)

    # Explore (merge and identify from which df the items are coming from)
    # https://kanoki.org/2019/07/04/pandas-difference-between-two-dataframes/
    df = df1.merge(df2, on=['patient_id', 'specimen_id'], how='outer', indicator=True)
    print('Inner merge', df.shape)
    pprint(df[:2])

    print('In both         ', df[df['_merge']=='both'].shape)
    print('In left or right', df[df['_merge']!='both'].shape)

    # Find which items are missing in Yitan's file
    # df = df1.merge(df2, on=['patient_id', 'specimen_id'], how='outer' ,indicator=True).loc[lambda x : x['_merge']=='right_only']
    df_miss = df1.merge(df2, on=['patient_id', 'specimen_id'], how='outer', indicator=True).loc[lambda x : x['_merge']=='left_only']
    df_miss = df_miss.sort_values(['patient_id', 'specimen_id'], ascending=True)
    print('\nMissing items', df_miss.shape)
    pprint(df_miss)


def stats(df_mrg):
    """ Local func to exolore intermediate data. """

    aa = df_mrg.groupby(['patient_id', 'specimen_id']).agg({'image_id': 'nunique'}).reset_index().sort_values(by=['patient_id', 'specimen_id'])
    print('Total images', aa['image_id'].sum())
    pprint(aa)

    c = 'tumor_site_from_data_src'
    print(df_mrg[c].nunique())
    tt = df_mrg.groupby([c]).agg({'patient_id': 'nunique', 'specimen_id': 'nunique', 'image_id': 'nunique'}).reset_index()
    tt

    c = 'tumor_type_from_data_src'
    print(df_mrg[c].nunique())
    # tt = df_mrg[c].value_counts().reset_index().sort_values('index')
    tt = df_mrg.groupby(['tumor_type_from_data_src']).agg({'patient_id': 'nunique', 'specimen_id': 'nunique', 'image_id': 'nunique'}).reset_index()
    tt

    df_mrg['stage_or_grade'].value_counts()


def load_crossref(metapath, crossref_fname):
    # PDX slide meta (from NCI/Globus)
    cref = pd.read_excel(metapath/crossref_fname, engine='openpyxl', header=2)
    cref = cref.rename(columns={'Capture Date': 'capture_date',
                                'Image ID': 'image_id',
                                'Model': 'model',
                                'Sample ID': 'sample_id',
                                'Date Loaded to BW_Transfers': 'date_loaded_to_bw_transfers'})
    cref = cref.dropna(subset=['model', 'image_id']).reset_index(drop=True)
    cref.insert(loc=1, column='patient_id',  value=cref['model'].map(lambda x: x.split('~')[0]), allow_duplicates=True)
    cref.insert(loc=2, column='specimen_id', value=cref['model'].map(lambda x: x.split('~')[1]), allow_duplicates=True)
    # cref = cref.astype({'image_id': int})
    cref['image_id'] = [int(x) if ~np.isnan(x) else x for x in cref['image_id'].values]
    return cref


def load_pdx_meta(metapath, pdx_meta_fname):
    # PDX meta (from Yitan)
    # Yitan doesn't have sample_id (??)
    pdx = pd.read_csv(metapath/'PDX_Meta_Information.csv')
    # pdx = pd.read_excel(metapath/pdx_meta_fname, engine='openpyxl')
    # pdx = pdx.dropna(subset=['patient_id']).reset_index(drop=True)
    return pdx


def load_slides_meta(metapath, slides_meta_fname):
    slides_meta = pd.read_csv(metapath/slides_meta_fname)
    col_rename = {'aperio.ImageID': 'image_id',
                  'aperio.MPP': 'MPP',
                  'openslide.level[0].height': 'height',
                  'openslide.level[0].width': 'width',
                  'openslide.objective-power': 'power'
                 }
    slides_meta = slides_meta.rename(columns=col_rename)

    cols = list(set(slides_meta.columns).intersection(set(list(col_rename.values()))))
    # cols = ['image_id'] + [c for c in cols if c != 'image_id']
    cols.remove('image_id')
    cols = ['image_id'] + cols 
    slides_meta = slides_meta[cols]
    return slides_meta


if __name__ == "__main__":

    # Path
    datapath = dirpath/'../data'
    slidespath = datapath/'doe-globus-pdx-data'  # path to raw WSI slides
    metapath = datapath/'meta'

    # Meta file names
    crossref_fname = '_ImageID_PDMRID_CrossRef.xlsx'
    slides_meta_fname = 'meta_from_wsi_slides.csv'
    pdx_meta_fname = 'PDX_Meta_Information.xlsx'

    # import ipdb; ipdb.set_trace(context=11)
    cref = load_crossref(metapath, crossref_fname)
    pdx = load_pdx_meta(metapath, pdx_meta_fname)
    slides_meta = load_slides_meta(metapath, slides_meta_fname)

    # print('Crossref: {}'.format(cref.shape))
    # print('PDX meta: {}'.format(pdx.shape))
    # pprint(cref[:2])
    # pprint(pdx[:2])

    # import ipdb; ipdb.set_trace(context=11)
    # _explore(cref, pdx)

    #
    # Merge crossref and pdx_meta
    #
    df_mrg = cref.merge(pdx, on=['patient_id', 'specimen_id'], how='inner').reset_index(drop=True)
    df_mrg = df_mrg.drop(columns=['capture_date', 'date_loaded_to_bw_transfers'])
    df_mrg = df_mrg.sort_values(['patient_id', 'specimen_id', 'sample_id'], ascending=True).reset_index(drop=True)

    print('Crossref:  {}'.format(cref.shape))
    print('PDX meta:  {}'.format(pdx.shape))
    print('1st merge: {}'.format(df_mrg.shape))
    # pprint(df_mrg[:2])

    # import ipdb; ipdb.set_trace(context=11)
    # _stats(df_mrg)

    #
    # Merge with slides_meta
    #
    df_final = df_mrg.merge(slides_meta, how='inner', on='image_id')

    print('\n1st merge:   {}'.format(df_mrg.shape))
    print('slides_meta: {}'.format(slides_meta.shape))
    print('df_final:    {}\n'.format(df_final.shape))
    pprint(df_final[:3])

    # Save
    print('\nSave merged metadata in csv.')
    df_final.to_csv(metapath/'meta_merged.csv', index=False)
    print('Done.')
