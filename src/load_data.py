import os
import sys
from pathlib import Path
import glob
from pprint import pprint
import pandas as pd
import numpy as np

fdir = Path(__file__).resolve().parent
# from config import cfg
from src.config import cfg


PDX_SAMPLE_COLS = ['model', 'patient_id', 'specimen_id', 'sample_id']


def encode_categorical(vv):
    return {value: label for label, value in enumerate(sorted(np.unique(vv)))}


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


def get_model_from_Sample(Sample):
    """ ... """
    patient_id = Sample.map(lambda x: x.split('~')[0])
    specimen_id = Sample.map(lambda x: x.split('~')[1])
    # sample_id = Sample.map(lambda x: x.split('~')[2])
    model = [a + '~' + b for a, b in zip(patient_id, specimen_id)]
    return model


def load_rsp(rsp_dpath=cfg.RSP_DPATH, single_drug=True, verbose=False):
    """ Load drug response data. """
    # import ipdb; ipdb.set_trace()
    rsp = pd.read_csv(rsp_dpath, sep='\t')
    rsp = rsp.reset_index()
    rsp = drop_dups(rsp)
    # rsp = rsp.drop(columns=['Source', 'Model'])

    # Cast
    rsp["Drug1"] = rsp["Drug1"].map(lambda s: s.strip())
    rsp["Drug2"] = rsp["Drug2"].map(lambda s: s.strip())
    rsp = rsp.astype({"Sample": str, "Drug1": str, "Drug2": str})

    if "Image_ID" in rsp.columns:
        # rsp = rsp.astype({"Image_ID": str})
        rsp = rsp.drop(columns="Image_ID")

    # Remove PDX samples that were re-grown from cryo-preserved samples
    # https://pdmr.cancer.gov/database/default.htm
    rsp = rsp[rsp["Sample"].map(lambda s: True if "RG" not in s else False)].reset_index(drop=True)

    # Remove 'NCIPDM.' from sample name
    rsp['Sample'] = rsp['Sample'].map(lambda x: x.split('NCIPDM.')[1])

    # ------------------------------------
    # Augment Drug2 from Drug1
    # TODO: need to test the code below!
    # ------------------------------------
    # # Copy Drug1 to Drug2 in case of single drug treatments
    # drug2 = []
    # for i, (d1, d2) in enumerate(zip(rsp["Drug1"], rsp["Drug2"])):
    #     if isinstance(d2, str) and d2.startswith("NSC."):
    #         drug2.append(d2)  # drug pair
    #     else:
    #         drug2.append(d1)  # single drug; copy to drug1 to drug2
    # rsp["Drug2"] = drug2        

    # # Create drug treatment string ids
    # rsp["trt"] = ["_".join(sorted([d1, d2])) for d1, d2 in zip(rsp["Drug1"], rsp["Drug2"])]        

    # # Create treatment groups (specimen-treatment pairs)
    # gg = rsp[["model", "trt"]].drop_duplicates(subset=["model", "trt"]).reset_index(drop=True)
    # gg = gg.reset_index().rename(columns={"index": "grp"})
    # gg = gg[["model", "trt", "grp"]]

    # # Assign treatment groups into master df
    # rsp = rsp.merge(gg, on=["model", "trt"], how="inner")

    # # Augment drug-pair treatments
    # rsp = rsp.reset_index(drop=True)  # reset index just in case
    # rsp["aug"] = False

    # # Find ids of drug-pair treatments
    # aug_ids = [ii for ii, (d1, d2) in enumerate(zip(rsp["Drug1"], rsp["Drug2"])) if d1 != d2]
    # df_aug = rsp.loc[aug_ids]
    # df_aug = df_aug.rename(columns={"Drug1": "Drug2", "Drug2": "Drug1"})
    # df_aug["aug"] = True        

    # # Create and save the final drug response dataset
    # rsp = pd.concat([rsp, df_aug], axis=0)
    # rsp = rsp.sort_values(["grp", "aug", "Sample"]).reset_index(drop=True)
    # ------------------------------------

    # import ipdb; ipdb.set_trace()

    # Single drug or drug pairs
    if single_drug:
        rsp = rsp.reset_index(drop=True)
        ids = [True if d1 == d2 else False for d1, d2 in zip(rsp["Drug1"], rsp["Drug2"])]
        rsp = rsp[ids].reset_index(drop=True)
        rsp = rsp.drop(columns=['Drug2'])
        rsp = rsp.drop_duplicates()

        # Create drug treatment string ids
        rsp["trt"] = rsp["Drug1"]

        # rsp = rsp[rsp.Drug2.isna()].reset_index(drop=True)
        # rsp = rsp.drop(columns=['Drug2'])
        # rsp["trt"] = rsp["Drug1"]  # Create drug treatment string ids
        # rsp = rsp.drop_duplicates()
    else:
        # Create drug treatment string ids
        # rsp["trt"] = ["_".join(sorted([d1, d2])) for d1, d2 in zip(rsp["Drug1"], rsp["Drug2"])]
        rsp["trt"] = [str(d1) + "_" + str(d2) for d1, d2 in zip(rsp["Drug1"], rsp["Drug2"])]

        # Augment drug-pair treatments
        rsp["aug"] = [True if d1 != d2 else False for (d1, d2) in zip(rsp["Drug1"], rsp["Drug2"])]

    # Parse Sample and add columns for model, patient_id, specimen_id, sample_id
    # rsp = parse_Sample_col(rsp)

    # Create column of treatments
    col_name = "smp"
    if col_name not in rsp.columns:
        smp = [str(s) + "_" + str(d) for s, d in zip(rsp["Sample"], rsp["trt"])]
        # rsp.insert(loc=0, column=col_name, value=smp, allow_duplicates=False)
        rsp.insert(loc=1, column=col_name, value=smp, allow_duplicates=False)

    # Create grp_name
    col_name = "grp_name"
    if col_name not in rsp.columns:
        model = get_model_from_Sample(rsp["Sample"])
        grp_name_values = [str(a) + "_" + str(b) for a, b in zip(model, rsp["trt"])]
        rsp[col_name] = grp_name_values

    if single_drug:
        # rsp = rsp[["smp", "Sample", "Drug1", "trt", "Group", "grp_name", "Response"]]
        rsp = rsp[["index", "smp", "Sample", "Drug1", "trt", "Group", "grp_name", "Response"]]
    else:
        rsp = rsp[["index", "smp", "Sample", "Drug1", "Drug2", "trt", "aug", "Group", "grp_name", "Response"]]

    if verbose:
        print("\nUnique samples   {}".format(rsp["Sample"].nunique()))
        print("Unique treatments {}".format(rsp["trt"].nunique()))
        print(rsp["Response"].value_counts())
        print(rsp.shape)
        pprint(rsp[:2])
    return rsp


def load_rna(rna_dpath=cfg.RNA_DPATH, add_prefix: bool=True, fea_dtype=np.float32):
    """ Load RNA-Seq data.
    Args:
        add_prefix: prefix each gene column name with "ge_".    
    """
    rna = pd.read_csv(rna_dpath, sep="\t")
    rna = drop_dups(rna)

    # Yitan's file
    if rna.columns[0] != "Sample":
        rna = rna.rename(columns={rna.columns[0]: "Sample"})

    fea_id0 = 1
    fea_pfx = "ge_"

    # Extract NCIPDM samples from master dataframe
    rna = rna[rna["Sample"].map(lambda x: x.split(".")[0]) == "NCIPDM"].reset_index(drop=True)
    # Remove 'NCIPDM.' from sample name
    rna["Sample"] = rna["Sample"].map(lambda x: x.split("NCIPDM.")[1])

    # Add prefix to gene names
    if add_prefix:
        rna = rna.rename(columns={c: fea_pfx + c for c in rna.columns[fea_id0:]})

    fea_cast = {c: fea_dtype for c in rna.columns if c.startswith(fea_pfx)}
    rna = rna.astype(fea_cast)

    # Parse Sample and add columns for model, patient_id, specimen_id, sample_id
    rna = parse_Sample_col(rna)

    return rna


def load_dd(dd_dpath=cfg.DD_DPATH, treat_nan=True, add_prefix: bool=True, fea_dtype=np.float32):
    """ Load drug descriptors. """
    dd = pd.read_csv(dd_dpath, sep='\t')
    dd = drop_dups(dd)

    # Yitan's file
    if dd.columns[0] != "ID":
        dd = dd.rename(columns={dd.columns[0]: "ID"})

    fea_id0 = 1
    fea_pfx = "dd_"

    # Add prefix to drug features
    if add_prefix:
        dd = dd.rename(columns={c: fea_pfx + c for c in dd.columns[fea_id0:] if ~c.startswith(fea_pfx)})

    dd_fea_names = [c for c in dd.columns if c.startswith(fea_pfx)]
    dd_fea = dd[dd_fea_names]
    dd_meta = dd.drop(columns=dd_fea.columns)

    # Filter dd
    if treat_nan and sum(dd.isna().sum() > 0):

        print("Descriptors with all NaN: {}".format( sum(dd_fea.isna().sum(axis=0) == dd_fea.shape[0])) )
        print("Descriptors with any NaN: {}".format( sum(dd_fea.isna().sum(axis=0) > 0) ))

        # Drop cols with all NaN
        print("Drop descriptors with all NaN ...")
        dd_fea = dd_fea.dropna(axis=1, how="all")

        # There are descriptors with a single unique value excluding NA (drop those)
        print("Drop descriptors with a single unique value (excluding NaNs) ...")
        # print(dd_fea.nunique(dropna=True).sort_values())
        col_ids = dd_fea.nunique(dropna=True).values == 1  # takes too long for large dataset
        dd_fea = dd_fea.iloc[:, ~col_ids]

        # col_ids = dd_fea.std(axis=0, skipna=True, numeric_only=True).values == 0
        # dd_fea = dd_fea.iloc[:, ~col_ids]

        # print(dd_fea.nunique(dropna=True).sort_values())
        # ii = dd_fea.nunique(dropna=True) == 1
        # gg = dd_fea.iloc[:, ii.values]

        print("Impute NaN.")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imputer = KNNImputer(missing_values=np.nan, n_neighbors=5,
        #                      weights="uniform", metric="nan_euclidean",
        #                      add_indicator=False)
        col_names = dd_fea.columns
        # dd_fea = pd.DataFrame(imputer.fit_transform(dd_fea), columns=col_names, dtype=fea_dtype)
        dd_fea = pd.DataFrame(imputer.fit_transform(dd_fea), columns=col_names)
        print("Descriptors with any NaN: {}".format( sum(dd_fea.isna().sum(axis=0) > 0) ))

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
        print(f'cref before: {cref.shape}')
        print(f'Bad slides: {bad_slides}')
        cref = cref[~cref.image_id.isin(bad_slides)].reset_index(drop=True)
        print(f'cref after: {cref.shape}')

    # Add passage number and sample types
    # TODO

    return cref


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


def load_pdx_meta2(path=cfg.METAPATH/cfg.PDX_META_FNAME, add_type_labels: bool=False):
    """ PDX meta (from Yitan updated). """
    file_type = str(path).split('.')[-1]
    if file_type == 'csv':
        yy = pd.read_csv(path)
    elif file_type == 'xlsx':
        yy = pd.read_excel(path, engine='openpyxl')
        yy = yy.dropna(subset=['patient_id']).reset_index(drop=True)
    else:
        raise f"File type ({file_type}) not supported."

    # Encode categorical cols
    if add_type_labels:
        ctype_enc = encode_categorical(yy['ctype'])
        csite_enc = encode_categorical(yy['csite'])
        yy['ctype_label'] = yy['ctype'].map(lambda x: ctype_enc[x])
        yy['csite_label'] = yy['csite'].map(lambda x: csite_enc[x])

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


def load_tidy_dataset_rsp(single_drug: bool=False, add_type_labels: bool=True):
    """ Create a tidy dataframe that contains gene expression, drug descriptors
    for both drugs, metadata, and the drug response.

    Args:
        single_drug: if True, include only single drug treatments; if False, single drug and drug pairs are included
        add_type_labels: if True, add the columns "csite_label" and "ctype_label"
    """
    # Load data
    rsp = load_rsp(single_drug=single_drug)
    rna = load_rna()
    dd = load_dd()
    cref = load_crossref()
    pdx = load_pdx_meta2(add_type_labels=True)

    # Merge rsp with rna
    print("\nMerge rsp and rna")
    print(rsp.shape)
    print(rna.shape)
    rsp_rna = rsp.merge(rna, on="Sample", how="inner")
    print(rsp_rna.shape)

    # Merge with dd
    print("Merge with descriptors")
    print(rsp_rna.shape)
    print(dd.shape)

    dd1 = dd.copy()
    dd2 = dd.copy()
    dd1 = dd1.rename(columns={"ID": "Drug1"})
    dd2 = dd2.rename(columns={"ID": "Drug2"})
    fea_id0 = 1
    fea_pfx = "dd_"
    dd1 = dd1.rename(columns={c: "dd1_" + c.split(fea_pfx)[1] for c in dd1.columns[fea_id0:] if ~c.startswith(fea_pfx)})
    dd2 = dd2.rename(columns={c: "dd2_" + c.split(fea_pfx)[1] for c in dd2.columns[fea_id0:] if ~c.startswith(fea_pfx)})

    tmp = rsp_rna.merge(dd1, left_on="Drug1", right_on="Drug1", how="inner")
    rsp_rna_dd = tmp.merge(dd2, left_on="Drug2", right_on="Drug2", how="inner")
    # print(rsp_rna_dd[["dd1_Uc", "dd2_Uc", "aug", "grp_name"]])
    print(rsp_rna_dd.shape)
    del dd, dd1, dd2, tmp

    # Merge with pdx meta
    print("Merge with pdx meta")
    print(pdx.shape)
    print(rsp_rna_dd.shape)
    rsp_rna_dd_pdx = pdx.merge(rsp_rna_dd, on=["patient_id", "specimen_id"], how="inner")
    print(rsp_rna_dd_pdx.shape)

    # Merge with pdx meta
    print("Merge with pdx meta")
    print(pdx.shape)
    print(rsp_rna_dd.shape)
    rsp_rna_dd_pdx = pdx.merge(rsp_rna_dd, on=["patient_id", "specimen_id"], how="inner")
    print(rsp_rna_dd_pdx.shape)

    # Merge cref
    print("Merge with cref")
    # (we loose some samples because we filter the bad slides)
    print(cref.shape)
    print(rsp_rna_dd_pdx.shape)
    data = cref.merge(rsp_rna_dd_pdx, on=PDX_SAMPLE_COLS, how="inner").reset_index(drop=True)
    print(data.shape)

    # -------------------
    # Explore (merge and identify from which df the items are coming from)
    # https://kanoki.org/2019/07/04/pandas-difference-between-two-dataframes/
    # --------
    # mrg_outer = cref.merge(rsp_rna_dd_pdx, on=PDX_SAMPLE_COLS, how='outer', indicator=True)
    # print('Outer merge', mrg_outer.shape)
    # print(mrg_outer['_merge'].value_counts())

    # miss_r = mrg_outer.loc[lambda x: x['_merge']=='right_only']
    # miss_r = miss_r.sort_values(PDX_SAMPLE_COLS, ascending=True)
    # print('Missing right items', miss_r.shape)

    # miss_l = mrg_outer.loc[lambda x: x['_merge']=='left_only']
    # miss_l = miss_l.sort_values(PDX_SAMPLE_COLS, ascending=True)
    # print('Missing left items', miss_l.shape)

    # print(miss_r.patient_id.unique())
    # jj = load_data.load_pdx_meta_jc()
    # miss_found = jj[ jj.patient_id.isin(miss_r.patient_id.unique()) ]
    # print(miss_r.Response.value_counts())
    # print(miss_found)

    # print(miss_r[miss_r.Response==1])
    # -------------------

    # Add 'slide' column
    data.insert(loc=5, column="slide", value=data["image_id"], allow_duplicates=True)

    if "index" in data.columns:
        # Put "index" in first column
        cols = data.columns.tolist()
        cols.remove("index")
        data = data[["index"] + cols]

    # Re-org cols
    dim = data.shape[1]
    meta_cols = ["index", "smp", "Sample",
                 "model", "patient_id", "specimen_id", "sample_id", "image_id", "slide",
                 "csite_src", "ctype_src", "csite", "ctype", "csite_label", "ctype_label",
                 "stage_or_grade",
                 "Drug1", "Drug2", "trt", "aug", "Group", "grp_name", "Response"]
    ge_cols = [c for c in data.columns if str(c).startswith('ge_')]
    # dd_cols = [c for c in data.columns if str(c).startswith('dd_')]
    dd1_cols = [c for c in data.columns if str(c).startswith("dd1_")]
    dd2_cols = [c for c in data.columns if str(c).startswith("dd2_")]
    data = data[meta_cols + ge_cols + dd1_cols + dd2_cols]
    assert data.shape[1] == dim, "There are missing cols after re-organizing the cols."

    return data


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
