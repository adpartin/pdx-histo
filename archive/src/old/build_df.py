"""
Build df using the following files and save in ../data/data_merged.csv:
* pdx drug response data
* rna expression
* mordred drug descriptors
* metadata
"""

import os
import sys
import argparse
from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np

fdir = Path(__file__).resolve().parent
from config import cfg

# RSP_DPATH = cfg.DATADIR/'studies/pdm/ncipdm_drug_response'
# RNA_DPATH = cfg.DATADIR/'combined_rnaseq_data_lincs1000_combat'
# DD_DPATH = cfg.DATADIR/'dd.mordred.with.nans'
# META_DPATH = cfg.DATADIR/'meta/meta_merged.csv'
    

def parse_args(args):
    parser = argparse.ArgumentParser(description='Build master dataframe.')

    parser.add_argument('--bad',
                        action='store_true',
                        help='Include samples with bad quality slides.')

    args, other_args = parser.parse_known_args()
    return args


def get_dups(df):
    """ Return df of duplicates. """
    df = df[df.duplicated(keep=False) == True]
    return df


def drop_dups(df, verbose=True):
    """ Drop duplicates. """
    n = df.shape[0]
    df = df.drop_duplicates()
    if verbose:
        if df.shape[0] < n:
            print(f'\nDropped {n - df.shape[0]} duplicates.\n')
    return df


def load_rsp(rsp_dpath=cfg.RSP_DPATH, verbose=False):
    """ Load drug response data. """
    rsp = pd.read_csv(rsp_dpath, sep='\t')
    rsp = drop_dups(rsp)

    # Keep single drug samples
    rsp = rsp[rsp['Drug2'].isna()].reset_index(drop=True)
    rsp = rsp.sort_values(by='Sample', ascending=True)
    rsp = rsp.drop(columns=['Source', 'Model', 'Drug2'])
    rsp['Sample'] = rsp['Sample'].map(lambda x: x.split('NCIPDM.')[1])

    # rsp.insert(loc=1, column='model', value=rsp['Sample'].map(lambda x: x.split('~')[0]), allow_duplicates=True)
    # rsp.insert(loc=2, column='patient_id', value=rsp['Sample'].map(lambda x: x.split('~')[1]), allow_duplicates=True)
    # rsp.insert(loc=3, column='sample_id', value=rsp['Sample'].map(lambda x: x.split('~')[2]), allow_duplicates=True)

    # Parse Sample into model, patient_id, specimen_id, sample_id
    patient_id = rsp['Sample'].map(lambda x: x.split('~')[0])
    specimen_id = rsp['Sample'].map(lambda x: x.split('~')[1])
    sample_id = rsp['Sample'].map(lambda x: x.split('~')[2])
    model = [a + '~' + b for a, b in zip(patient_id, specimen_id)]
    rsp.insert(loc=1, column='model', value=model, allow_duplicates=True)
    rsp.insert(loc=2, column='patient_id', value=patient_id, allow_duplicates=True)
    rsp.insert(loc=3, column='specimen_id', value=specimen_id, allow_duplicates=True)
    rsp.insert(loc=4, column='sample_id', value=sample_id, allow_duplicates=True)
    
    # Create column of unique treatments
    col_name = 'smp'
    if col_name not in rsp.columns:
        jj = [str(s) + '_' + str(d) for s, d in zip(rsp.Sample, rsp.Drug1)]
        rsp.insert(loc=0, column=col_name, value=jj, allow_duplicates=False)
        
    if verbose:
        print('\nUnique samples {}'.format(rsp.Sample.nunique()))
        print('Unique drugs   {}'.format(rsp.Drug1.nunique()))
        print(rsp['Response'].value_counts())
        print(rsp.shape)
        pprint(rsp[:2])
    return rsp


def load_rna(rna_dpath=cfg.RNA_DPATH, add_prefix: bool=True, verbose: bool=False):
    """ Load RNA-Seq data. """
    rna = pd.read_csv(rna_dpath, sep='\t')
    rna = drop_dups(rna)

    rna = rna[ rna.Sample.map(lambda x: x.split('.')[0]) == 'NCIPDM'].reset_index(drop=True)
    rna = rna.sort_values(by='Sample', ascending=True)
    rna['Sample'] = rna['Sample'].map(lambda x: x.split('NCIPDM.')[1])
    if add_prefix:
        rna = rna.rename(columns={x: 'ge_'+x for x in rna.columns[1:]})  # add prefix to the genes

    if verbose:
        print(rna.shape)
        pprint(rna.iloc[:2, :7])
    return rna


def load_dd(dd_dpath=cfg.DD_DPATH, verbose=False):
    """ Load drug descriptors. """
    dd = pd.read_csv(dd_dpath, sep='\t')
    dd = drop_dups(dd)
    
    if verbose:
        print(dd.shape)
        pprint(dd.iloc[:2, :7])
    return dd


def load_meta(meta_dpath=cfg.META_DPATH, verbose=False):
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


def main(args):
    args = parse_args(args)

    # import ipdb; ipdb.set_trace() 
    
    # Load
    rsp = load_rsp()
    rna = load_rna()
    dd = load_dd()
    meta = load_meta()

    # Merge
    data = rsp.merge(rna, on='Sample', how='inner')
    data = data.merge(dd, left_on='Drug1', right_on='ID', how='inner').reset_index(drop=True)
    data = data.merge(meta.drop(columns=['patient_id', 'specimen_id', 'sample_id']), on='Sample', how='inner').reset_index(drop=True)
    data = data.sample(frac=1.0, random_state=0, axis=0).reset_index(drop=True)  # TODO is this necessary??
    
    def check_prfx(x):
        return True if x.startswith('ge_') or x.startswith('dd_') else False
        
    # Re-org columns
    meta_df = data.drop(columns=[c for c in data.columns if check_prfx(c)])
    fea_df = data.drop(columns=meta_df.columns)
    data = pd.concat([meta_df, fea_df], axis=1)

    # Add column with tile counts
    # tiles_dir = cfg.DATADIR/'tiles_png'
    # counter = [len(list(tiles_dir.glob(f'{image_id}-tile-*.png'))) for image_id in data.image_id.values]
    # data.insert(loc=11, value=counter, column='n_tiles', allow_duplicates=True)

    # Drop samples which have bad quality slides
    if args.bad is False:
        bad_slides_list = cfg.BAD_SLIDES
        # data[data.image_id.isin(bad_slides_list)]
        data = data[~data.image_id.isin(bad_slides_list)].reset_index(drop=True)

    # import ipdb; ipdb.set_trace()
    print('\nFinal dataframe {}'.format(data.shape))

    # save
    print('Save merged dataframe in csv.')
    data.to_csv(cfg.DATADIR/'data_merged.csv', index=False)
    print('Done.')


if __name__ == "__main__":
    main(sys.argv[1:])
