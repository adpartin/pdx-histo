"""
Build df using the following files and save in ../data/data_merged.csv:
* pdx drug response data
* rna expression
* mordred drug descriptors
* metadata
"""

import os
import sys
from pathlib import Path
import glob
from pprint import pprint
import pandas as pd
import numpy as np


dirpath = Path(__file__).resolve().parent


def load_rsp(rsp_dpath, verbose=False):
    """ Load drug response data. """
    rsp = pd.read_csv(rsp_dpath, sep='\t')

    # Keep single drug samples
    rsp = rsp[rsp['Drug2'].isna()].reset_index(drop=True)
    rsp = rsp.sort_values(by='Sample', ascending=True)
    rsp = rsp.drop(columns=['Source', 'Model', 'Drug2'])
    rsp['Sample'] = rsp['Sample'].map(lambda x: x.split('NCIPDM.')[1])

    rsp.insert(loc=1, column='model', value=rsp['Sample'].map(lambda x: x.split('~')[0]), allow_duplicates=True)
    rsp.insert(loc=2, column='patient_id', value=rsp['Sample'].map(lambda x: x.split('~')[1]), allow_duplicates=True)
    rsp.insert(loc=3, column='sample_id', value=rsp['Sample'].map(lambda x: x.split('~')[2]), allow_duplicates=True)
    
    if verbose:
        print('\nUnique samples {}'.format(rsp.Sample.nunique()))
        print('Unique drug    {}'.format(rsp.Drug1.nunique()))
        print(rsp.Response.value_counts())
        print(rsp.shape)
        pprint(rsp[:2])
    return rsp


def load_rna(rna_dpath, verbose=False):
    """ Load RNA-Seq data. """
    rna = pd.read_csv(rna_dpath, sep='\t')
    rna = rna[ rna.Sample.map(lambda x: x.split('.')[0]) == 'NCIPDM' ].reset_index(drop=True)
    rna = rna.sort_values(by='Sample', ascending=True)
    rna['Sample'] = rna['Sample'].map(lambda x: x.split('NCIPDM.')[1])
    rna = rna.rename(columns={x: 'ge_'+x for x in rna.columns[1:]})  # add prefix to the genes

    if verbose:
        print(rna.shape)
        pprint(rna.iloc[:2, :7])
    return rna


def load_dd(dd_dpath, verbose=False):
    """ Load drug descriptors. """
    dd = pd.read_csv(dd_dpath, sep='\t')
    
    if verbose:
        print(dd.shape)
        pprint(dd.iloc[:2, :7])
    return dd


def load_meta(meta_dpath, verbose=False):
    """ Load the combined metadata. """
    meta = pd.read_csv(meta_dpath)

    # Add the 'Sample' column
    vv = list()
    for i, r in meta[['patient_id', 'specimen_id', 'sample_id']].iterrows():
        vv.append( str(r['patient_id']) + '~' + str(r['specimen_id'] + '~' + str(r['sample_id'])) )
    meta.insert(loc=1, column='Sample', value=vv, allow_duplicates=False)

    # Rename cols
    meta = meta.rename(columns={'tumor_site_from_data_src': 'csite_src',
                                'tumor_type_from_data_src': 'ctype_src',
                                'simplified_tumor_site': 'csite',
                                'simplified_tumor_type': 'ctype'})

    def encode_categorical(vv):
        return {value: label for label, value in enumerate(sorted(np.unique(vv)))}

    # Encode categorical cols
    ctype_enc = encode_categorical(meta['ctype'])
    csite_enc = encode_categorical(meta['csite'])
    meta['ctype_label'] = meta['ctype'].map(lambda x: ctype_enc[x])
    meta['csite_label'] = meta['csite'].map(lambda x: csite_enc[x])

    meta = meta.drop(columns=['model', 'Notes', 'MPP', 'width', 'height', 'power'])

    if verbose:
        print(meta.shape)
        pprint(meta.iloc[:2, :7])
        pprint(meta['csite'].value_counts())
        pprint(meta['csite_label'].value_counts())

    return meta


if __name__ == "__main__":

    rsp_dpath = dirpath/'../data/studies/pdm/ncipdm_drug_response'
    rna_dpath = dirpath/'../data/combined_rnaseq_data_lincs1000_combat'
    dd_dpath = dirpath/'../data/dd.mordred.with.nans'
    meta_dpath = dirpath/'../data/meta/meta_merged.csv'

    # Load
    rsp = load_rsp(rsp_dpath)
    rna = load_rna(rna_dpath)
    dd = load_dd(dd_dpath)
    meta = load_meta(meta_dpath)

    # Merge
    data = rsp.merge(rna, on='Sample', how='inner')
    data = data.merge(dd, left_on='Drug1', right_on='ID', how='inner').reset_index(drop=True)
    data = data.merge(meta.drop(columns=['patient_id', 'specimen_id', 'sample_id']), on='Sample', how='inner').reset_index(drop=True)
    data = data.sample(frac=1.0, random_state=0, axis=0).reset_index(drop=True)
    
    def check_prfx(x):
        return True if x.startswith('ge_') or x.startswith('dd_') else False
        
    # Re-org columns
    meta_df = data.drop(columns=[c for c in data.columns if check_prfx(c)])
    fea_df = data.drop(columns=meta_df.columns)
    data = pd.concat([meta_df, fea_df], axis=1)
    print('Final dataframe {}'.format(data.shape))

    # save
    print('\nSave merged dataframe in csv.')
    data.to_csv(dirpath/'../data/data_merged.csv', index=False)
    print('Done.')
