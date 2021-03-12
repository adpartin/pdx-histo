import os
import sys
from pathlib import Path
import glob
from pprint import pprint
import pandas as pd
import numpy as np

fdir = Path(__file__).resolve().parent
from config import cfg


PDX_SAMPLE_COLS = ['model', 'patient_id', 'specimen_id', 'sample_id']


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


def parse_Sample_col(df):
    """ Parse Sample into model, patient_id, specimen_id, sample_id.
    Cast the values in the created columns to str.
    """
    assert 'Sample' in df.columns, "'Sample' col was not found the dataframe."
    
    patient_id = df['Sample'].map(lambda x: x.split('~')[0])
    specimen_id = df['Sample'].map(lambda x: x.split('~')[1])
    sample_id = df['Sample'].map(lambda x: x.split('~')[2])
    model = [a + '~' + b for a, b in zip(patient_id, specimen_id)]

    df.insert(loc=1, column='model', value=model, allow_duplicates=True)
    df.insert(loc=2, column='patient_id', value=patient_id, allow_duplicates=True)
    df.insert(loc=3, column='specimen_id', value=specimen_id, allow_duplicates=True)
    df.insert(loc=4, column='sample_id', value=sample_id, allow_duplicates=True)    

    # Cast and sort
    df = df.astype({c: str for c in PDX_SAMPLE_COLS})
    df = df.sort_values(PDX_SAMPLE_COLS).reset_index(drop=True)

    return df


def load_rsp(rsp_dpath=cfg.RSP_DPATH, single_drug=True, verbose=False):
    """ Load drug response data. """
    rsp = pd.read_csv(rsp_dpath, sep='\t')
    rsp = drop_dups(rsp)
    rsp = rsp.drop(columns=['Source', 'Model'])

    # Single drug or drug pairs
    if single_drug:
        rsp = rsp[rsp.Drug2.isna()].reset_index(drop=True)
        rsp = rsp.drop(columns=['Drug2'])
        rsp = rsp.drop_duplicates()
    else:
        raise ValueError("Didn't test this...")
        # rsp.loc[rsp.Drug2.isna(), 'Drug2'] = 'Drug1'
        
    # Remove 'NCIPDM.' from sample name
    rsp['Sample'] = rsp['Sample'].map(lambda x: x.split('NCIPDM.')[1])

    # Parse Sample and add columns for model, patient_id, specimen_id, sample_id
    # rsp = parse_Sample_col(rsp)
    
    # Create column of unique treatments
    col_name = 'smp'
    if col_name not in rsp.columns:
        smp = [str(s) + '_' + str(d) for s, d in zip(rsp['Sample'], rsp['Drug1'])]
        rsp.insert(loc=0, column=col_name, value=smp, allow_duplicates=False)
        
    if verbose:
        print('\nUnique samples {}'.format(rsp['Sample'].nunique()))
        print('Unique drugs   {}'.format(rsp['Drug1'].nunique()))
        print(rsp['Response'].value_counts())
        print(rsp.shape)
        pprint(rsp[:2])
    return rsp


def load_rna(rna_dpath=cfg.RNA_DPATH, add_prefix: bool=True, fea_dtype=np.float32):
    """ Load RNA-Seq data.
    Args:
        add_prefix: prefix each gene column name with 'ge_'.    
    """
    rna = pd.read_csv(rna_dpath, sep='\t')
    rna = drop_dups(rna)

    fea_id0 = 1
    fea_pfx = 'ge_'

    # Extract NCIPDM samples from master dataframe
    rna = rna[rna.Sample.map(lambda x: x.split('.')[0]) == 'NCIPDM'].reset_index(drop=True)
    # Remove 'NCIPDM.' from sample name
    rna['Sample'] = rna['Sample'].map(lambda x: x.split('NCIPDM.')[1])

    # Add prefix to gene names
    if add_prefix:
        rna = rna.rename(columns={c: fea_pfx + c for c in rna.columns[fea_id0:]})

    fea_cast = {c: fea_dtype for c in rna.columns if c.startswith(fea_pfx)}
    rna = rna.astype(fea_cast)
        
    # Parse Sample and add columns for model, patient_id, specimen_id, sample_id
    rna = parse_Sample_col(rna)
    
    return rna


def load_dd(dd_dpath=cfg.DD_DPATH, treat_nan=True, fea_dtype=np.float32):
    """ Load drug descriptors. """
    dd = pd.read_csv(dd_dpath, sep='\t')
    dd = drop_dups(dd)

    fea_pfx = 'dd_'

    dd_fea_names = [c for c in dd.columns if c.startswith(fea_pfx)]
    dd_fea = dd[dd_fea_names]
    dd_meta = dd.drop(columns=dd_fea.columns)

    # Filter dd
    if treat_nan and sum(dd.isna().sum() > 0):

        print('Descriptors with all NaN: {}'.format( sum(dd_fea.isna().sum(axis=0) == dd_fea.shape[0])) )
        print('Descriptors with any NaN: {}'.format( sum(dd_fea.isna().sum(axis=0) > 0) ))

        # Drop cols with all NaN
        print('Drop descriptors with all NaN ...')
        dd_fea = dd_fea.dropna(axis=1, how='all')

        # There are descriptors with a single unique value excluding NA (drop those)
        print('Drop descriptors with a single unique value (excluding NaNs) ...')
        # print(dd_fea.nunique(dropna=True).sort_values())
        col_ids = dd_fea.nunique(dropna=True).values == 1  # takes too long for large dataset
        dd_fea = dd_fea.iloc[:, ~col_ids]

        # col_ids = dd_fea.std(axis=0, skipna=True, numeric_only=True).values == 0
        # dd_fea = dd_fea.iloc[:, ~col_ids]
        
        # print(dd_fea.nunique(dropna=True).sort_values())
        # ii = dd_fea.nunique(dropna=True) == 1
        # gg = dd_fea.iloc[:, ii.values]

        print('Impute NaN.')
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imputer = KNNImputer(missing_values=np.nan, n_neighbors=5,
        #                      weights='uniform', metric='nan_euclidean',
        #                      add_indicator=False)
        col_names = dd_fea.columns
        # dd_fea = pd.DataFrame(imputer.fit_transform(dd_fea), columns=col_names, dtype=fea_dtype)
        dd_fea = pd.DataFrame(imputer.fit_transform(dd_fea), columns=col_names)
        print('Descriptors with any NaN: {}'.format( sum(dd_fea.isna().sum(axis=0) > 0) ))

        dd = pd.concat([dd_meta, dd_fea], axis=1)

    fea_cast = {c: fea_dtype for c in dd.columns if c.startswith(fea_pfx)}
    dd = dd.astype(fea_cast)
        
    return dd


def load_crossref(path=cfg.METAPATH/cfg.CROSSREF_FNAME, drop_bad_slides=True):
    """ PDX slide meta that comes with the origianl slides (from NCI/Globus). """
    cref = pd.read_excel(path, engine='openpyxl', header=2)
    
    cref = cref.rename(columns={'Capture Date': 'capture_date',
                                'Image ID': 'image_id',
                                'Model': 'model',
                                'Sample ID': 'sample_id',
                                'Date Loaded to BW_Transfers': 'date_loaded_to_bw_transfers'})
      
    cref = cref.drop(columns=['capture_date', 'date_loaded_to_bw_transfers', 'Notes'])    
    
    # Drop nan
    cref = cref.dropna(subset=['model', 'image_id']).reset_index(drop=True)
    
    cref.insert(loc=1, column='patient_id',  value=cref['model'].map(lambda x: x.split('~')[0]), allow_duplicates=True)
    cref.insert(loc=2, column='specimen_id', value=cref['model'].map(lambda x: x.split('~')[1]), allow_duplicates=True)
    
    # Cast and sort
    cref = cref.astype({c: str for c in PDX_SAMPLE_COLS})
    cref = cref.sort_values(PDX_SAMPLE_COLS).reset_index(drop=True)
    
    # Cast image_id values to int
    cref['image_id'] = [str(int(x)) for x in cref['image_id'].values]
    # cref['image_id'] = [int(x) if ~np.isnan(x) else x for x in cref['image_id'].values]
    
    # Remove bad samples with bad slides
    if drop_bad_slides:
        bad_slides = [str(slide) for slide in cfg.BAD_SLIDES]
        print('\nDrop bad slides from image_id.')
        print(f'cref: {cref.shape}')
        print(f'Bad slides: {bad_slides}')
        cref = cref[~cref.image_id.isin(bad_slides)].reset_index(drop=True)
        print(f'cref: {cref.shape}')
    
    return cref


# def load_pdx_meta(path=cfg.METAPATH/cfg.PDX_META_FNAME):
def load_pdx_meta():
    """ PDX meta (from Yitan). """
    path = '../data/meta/PDX_Meta_Information.xlsx'
    file_type = str(path).split('.')[-1]
    if file_type == 'csv':
        yy = pd.read_csv(path)
    elif file_type == 'xlsx':
        yy = pd.read_excel(path, engine='openpyxl')
        yy = yy.dropna(subset=['patient_id']).reset_index(drop=True)
    else:
        raise f"File type ({file_type}) not supported."

    col_rename = {'tumor_site_from_data_src': 'csite_src',
                  'tumor_type_from_data_src': 'ctype_src',
                  'simplified_tumor_site': 'csite',
                  'simplified_tumor_type': 'ctype'}
    yy = yy.rename(columns=col_rename)
    
    yy = yy.sort_values(['patient_id', 'specimen_id'], ascending=True).reset_index(drop=True)
    yy = yy.astype(str)
    return yy


def load_pdx_meta2(path=cfg.METAPATH/cfg.PDX_META_FNAME):
    """ PDX meta (from Yitan updated). """
    file_type = str(path).split('.')[-1]
    if file_type == 'csv':
        yy = pd.read_csv(path)
    elif file_type == 'xlsx':
        yy = pd.read_excel(path, engine='openpyxl')
        yy = yy.dropna(subset=['patient_id']).reset_index(drop=True)
    else:
        raise f"File type ({file_type}) not supported."
    
    yy = yy.sort_values(['patient_id', 'specimen_id'], ascending=True).reset_index(drop=True)
    yy = yy.astype(str)
    return yy


def load_pdx_meta_jc(path=cfg.METAPATH/'combined_metadata_2018May.txt'):
    """ PDX meta (from Judith Cohn).
    TODO: this dataframe may some useful columns. 
    """
    jj = pd.read_csv(path, sep='\t')
                   
    col_rename = {'sample_name': 'Sample',
                  'tumor_site_from_data_src': 'csite_src',
                  'tumor_type_from_data_src': 'ctype_src',
                  'simplified_tumor_site': 'csite',
                  'simplified_tumor_type': 'ctype'}
    jj = jj.rename(columns=col_rename)

    # Extract NCIPDM samples from master dataframe
    jj = jj[jj.Sample.map(lambda x: x.split('.')[0]) == 'NCIPDM'].reset_index(drop=True)
    # Remove 'NCIPDM.' from the sample name
    jj['Sample'] = jj['Sample'].map(lambda x: x.split('NCIPDM.')[1])
                   
    jj = jj.drop(columns=['core_str',
                          'copy_flag', 'dataset',
                          'gdc_icdo_topo_code', 'gdc_icdo_morph_code'])

    jj = jj[['Sample', 'patient_id', 'specimen_id', 'sample_id',
             'csite_src', 'ctype_src', 'csite', 'ctype']]    
    
    jj = jj.sort_values(['patient_id', 'specimen_id'], ascending=True).reset_index(drop=True)
    jj = jj.astype(str)
    return jj


def load_slides_meta(path=cfg.METAPATH/cfg.SLIDES_META_FNAME):
    """ Meta that was extracted from the actual slides. """
    slides_meta = pd.read_csv(path)
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


# def load_meta(meta_dpath=cfg.META_DPATH, verbose=False):
#     """ Load the combined metadata. """
#     meta = pd.read_csv(meta_dpath)

#     # Add the 'Sample' column
#     vv = list()
#     for i, r in meta[['patient_id', 'specimen_id', 'sample_id']].iterrows():
#         vv.append( str(r['patient_id']) + '~' + str(r['specimen_id'] + '~' + str(r['sample_id'])) )
#     meta.insert(loc=1, column='Sample', value=vv, allow_duplicates=False)

#     # Rename cols
#     meta = meta.rename(columns={'tumor_site_from_data_src': 'csite_src',
#                                 'tumor_type_from_data_src': 'ctype_src',
#                                 'simplified_tumor_site': 'csite',
#                                 'simplified_tumor_type': 'ctype'})

#     def encode_categorical(vv):
#         return {value: label for label, value in enumerate(sorted(np.unique(vv)))}

#     # Encode categorical cols
#     ctype_enc = encode_categorical(meta['ctype'])
#     csite_enc = encode_categorical(meta['csite'])
#     meta['ctype_label'] = meta['ctype'].map(lambda x: ctype_enc[x])
#     meta['csite_label'] = meta['csite'].map(lambda x: csite_enc[x])

#     meta = meta.drop(columns=['model', 'Notes', 'MPP', 'width', 'height', 'power'])

#     if verbose:
#         print(meta.shape)
#         pprint(meta.iloc[:2, :7])
#         pprint(meta['csite'].value_counts())
#         pprint(meta['csite_label'].value_counts())

#     return meta
