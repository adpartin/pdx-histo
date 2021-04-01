""" 
Takes the original tfrecords that we got from Alex Pearson and uses
them to generate new tfrecords with additional data.

update_tfrecords_with_rna()
    updates tfrecords with RNA-seq and metadata of PDX samples

update_tfrecords_for_drug_rsp()
    creates tfrecord for drug response sample that contains histo
    slide, RNA-seq, drug descriptors, and drug response
"""
import os
import sys
assert sys.version_info >= (3, 5)

from pathlib import Path
import numpy as np
from typing import Optional

import tensorflow as tf
assert tf.__version__ >= '2.0'

# import load_data
# from load_data import PDX_SAMPLE_COLS
# from tfrecords import FEA_SPEC, FEA_SPEC_RSP, FEA_SPEC_RNA_NEW, original_tfr_names
# from tf_utils import _float_feature, _bytes_feature, _int64_feature

fdir = Path(__file__).resolve().parent
# from config import cfg
sys.path.append(str(fdir/".."))
import src
from src.config import cfg
from src import load_data
from src.load_data import PDX_SAMPLE_COLS
from src.tfrecords import FEA_SPEC, FEA_SPEC_RSP, FEA_SPEC_RSP_DRUG_PAIR, FEA_SPEC_RNA_NEW, original_tfr_names
from src.tf_utils import _float_feature, _bytes_feature, _int64_feature

# Seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


LABEL = '299px_302um'
directory = cfg.SF_TFR_DIR/LABEL

GREEN = '\033[92m'
ENDC = '\033[0m'

def green(text):
    return GREEN + str(text) + ENDC


n_samples = None
# n_samples = 4
single_drug = False


def update_tfrecords_for_drug_rsp(n_samples: Optional[int] = None, single_drug: bool=False) -> None:
    """
    Takes original tfrecords that we got from A. Pearson and updates them
    by addting more data including metadata of PDX samples, RNA-Seq, drug
    descriptors, and drug response.

    Args:
        n_samples : generate tfrecords for n_samples drug response samples
            (primarily used for debugging)
    """
    # Create path for the updated tfrecords
    if single_drug:
        outpath = cfg.SF_TFR_DIR_RSP/LABEL
    else:
        outpath = cfg.SF_TFR_DIR_RSP_DRUG_PAIR/LABEL
    os.makedirs(outpath, exist_ok=True)

    # Load data
    rsp = load_data.load_rsp(single_drug=single_drug)
    rna = load_data.load_rna()
    dd = load_data.load_dd()
    cref = load_data.load_crossref()
    pdx = load_data.load_pdx_meta2()

    # import ipdb; ipdb.set_trace()

    # Merge rsp with rna
    print("\nMerge rsp and rna")
    print(rsp.shape)
    print(rna.shape)
    rsp_rna = rsp.merge(rna, on="Sample", how="inner")
    print(rsp_rna.shape)

    # # Merge with dd
    # print(rsp_rna.shape)
    # print(dd.shape)
    # rsp_rna_dd = rsp_rna.merge(dd, left_on='Drug1', right_on='ID', how='inner').reset_index(drop=True)
    # print(rsp_rna_dd.shape)

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

    if n_samples is not None:
        data = data.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    # Re-org cols
    dim = data.shape[1]
    # meta_cols = ['smp', 'Sample', 'Drug1', 'Response',
    #              'model', 'patient_id', 'specimen_id', 'sample_id', 'image_id', 
    #              'csite_src', 'ctype_src', 'csite', 'ctype', 'stage_or_grade',
    #              'NAME', 'CLEAN_NAME', 'SMILES', 'ID']
    meta_cols = ["index", "smp", "Sample",
                 "model", "patient_id", "specimen_id", "sample_id", "image_id", 
                 "csite_src", "ctype_src", "csite", "ctype", "stage_or_grade",
                 "Drug1", "Drug2", "trt", "aug", "Group", "grp_name", "Response"]
    ge_cols = [c for c in data.columns if str(c).startswith('ge_')]
    # dd_cols = [c for c in data.columns if str(c).startswith('dd_')]
    dd1_cols = [c for c in data.columns if str(c).startswith("dd1_")]
    dd2_cols = [c for c in data.columns if str(c).startswith("dd2_")]
    data = data[meta_cols + ge_cols + dd1_cols + dd2_cols]
    assert data.shape[1] == dim, "There are missing cols after re-organizing the cols."

    # Create dict of slide ids. Each slide (key) contains a dict with metadata.
    mm = {}  # dict to store all metadata
    id_name = "smp"  # col name that contains the IDs for the samples 

    # import ipdb; ipdb.set_trace()

    # Iterate over rows a collect data into dict
    for i, row_data in data.iterrows():
        # Dict to contain metadata for the current slide
        sample_dct = {}
        smp = str(row_data[id_name])

        # Meta cols
        # Create a (key, value) pair for each meta col
        for c in meta_cols:
            sample_dct[c] = str(row_data[c])

        # Features cols
        # ge_data = list(row_data[ge_cols].values.astype(cfg.GE_DTYPE))
        # dd_data = list(row_data[dd_cols].values.astype(cfg.DD_DTYPE))
        # sample_dct['ge_data'] = ge_data
        # sample_dct['dd_data'] = dd_data
        ge_data = row_data[ge_cols].values.astype(cfg.GE_DTYPE)
        dd1_data = row_data[dd1_cols].values.astype(cfg.DD_DTYPE)
        dd2_data = row_data[dd2_cols].values.astype(cfg.DD_DTYPE)
        sample_dct["ge_data"] = ge_data.tobytes()
        sample_dct["dd1_data"] = dd1_data.tobytes()
        sample_dct['dd2_data'] = dd2_data.tobytes()
        
        mm[smp] = sample_dct
        
    print(f"A total of {len(mm)} drug response samples with tabular features.")

    # Common slides (that have both image data, other features, and drug response)
    slides = data["image_id"].unique().tolist()
    all_slides = original_tfr_names(label=LABEL)
    c_slides = set(slides).intersection(set(all_slides))
    print(f"A total of {len(c_slides)} drug response samples with tfrecords and tabular features.")

    # import ipdb; ipdb.set_trace()

    # Create a tfrecord for each sample (iter over samples)
    for i, slide_name in enumerate(sorted(c_slides)):
        # Name of original tfrecord to load that contains tiles for a single
        # histo slide
        rel_tfr = str(slide_name) + ".tfrecords"
        tfr = str(directory/rel_tfr)
        
        print(f"\r\033[K Creating tfrecords using {green(rel_tfr)} ({i+1} out of {len(c_slides)} tfrecords) ...",
              end="") 
        
        raw_dataset = tf.data.TFRecordDataset(tfr)
            
        # Iter over drug response samples that use the current slide
        samples = data[data["image_id"] == slide_name][id_name].values.tolist()
        for smp in samples:

            # Name of the output tfrecord for the current drug response sample
            tfr_fname = str(outpath/(smp + ".tfrecords"))
            writer = tf.io.TFRecordWriter(tfr_fname)

            # Iter over tiles of the current slide
            for tile_cnt, rec in enumerate(raw_dataset):
                # Features of the current rec from old tfrecord
                features = tf.io.parse_single_example(rec, features=FEA_SPEC)
                # tf.print(features.keys())

                # Extract slide name from old tfrecord and get the new metadata
                # to be added to the new tfrecord
                slide = features["slide"].numpy().decode("utf-8")
                slide_meta = mm[smp]

                ex = tf.train.Example(features=tf.train.Features(
                    feature={
                        # old features
                        "slide":       _bytes_feature(features["slide"].numpy()),  # image_id
                        "image_raw":   _bytes_feature(features["image_raw"].numpy()),

                        # new features
                        "index":       _bytes_feature(bytes(slide_meta["index"], "utf-8")),
                        "smp":         _bytes_feature(bytes(slide_meta["smp"], "utf-8")),
                        "Group":       _bytes_feature(bytes(slide_meta["Group"], "utf-8")),
                        "grp_name":    _bytes_feature(bytes(slide_meta["grp_name"], "utf-8")),

                        "Sample":      _bytes_feature(bytes(slide_meta["Sample"], "utf-8")),
                        "model":       _bytes_feature(bytes(slide_meta["model"], "utf-8")),
                        "patient_id":  _bytes_feature(bytes(slide_meta["patient_id"], "utf-8")),
                        "specimen_id": _bytes_feature(bytes(slide_meta["specimen_id"], "utf-8")),
                        "sample_id":   _bytes_feature(bytes(slide_meta["sample_id"], "utf-8")),
                        "image_id":    _bytes_feature(bytes(slide_meta["image_id"], "utf-8")),

                        "ctype":       _bytes_feature(bytes(slide_meta["ctype"], "utf-8")),
                        "csite":       _bytes_feature(bytes(slide_meta["csite"], "utf-8")),
                        "ctype_src":   _bytes_feature(bytes(slide_meta["ctype_src"], "utf-8")),
                        "csite_src":   _bytes_feature(bytes(slide_meta["csite_src"], "utf-8")),

                        # "Drug1":       _bytes_feature(bytes(slide_meta["Drug1"], "utf-8")),
                        # "NAME":        _bytes_feature(bytes(slide_meta["NAME"], "utf-8")),
                        # "CLEAN_NAME":  _bytes_feature(bytes(slide_meta["CLEAN_NAME"], "utf-8")),
                        # "ID":          _bytes_feature(bytes(slide_meta["ID"], "utf-8")),
                        "Drug1":       _bytes_feature(bytes(slide_meta["Drug1"], "utf-8")),
                        "Drug2":       _bytes_feature(bytes(slide_meta["Drug2"], "utf-8")),
                        "trt":         _bytes_feature(bytes(slide_meta["trt"], "utf-8")),
                        "aug":         _bytes_feature(bytes(slide_meta["aug"], "utf-8")),

                        "Response":    _int64_feature(int(slide_meta["Response"])),

                        # 'ge_data':     _float_feature(slide_meta['ge_data']),
                        # 'dd_data':     _float_feature(slide_meta['dd_data']),
                        "ge_data":      _bytes_feature(slide_meta["ge_data"]),
                        "dd1_data":     _bytes_feature(slide_meta["dd1_data"]),
                        "dd2_data":     _bytes_feature(slide_meta["dd2_data"]),
                    }
                ))
                
                writer.write(ex.SerializeToString())

            print(f"Total tiles in the sample {tile_cnt+1}")
            writer.close()
        print()
        
        
    # ------------------
    # Inspect a TFRecord
    # ------------------
    import ipdb; ipdb.set_trace()

    smp = samples[0]
    tfr_path = str(outpath/(smp + ".tfrecords"))
    raw_dataset = tf.data.TFRecordDataset(tfr_path)
    rec = next(raw_dataset.__iter__())
    # features = tf.io.parse_single_example(rec, features=fea_spec_new)
    if single_drug:
        features = tf.io.parse_single_example(rec, features=FEA_SPEC_RSP)
    else:
        features = tf.io.parse_single_example(rec, features=FEA_SPEC_RSP_DRUG_PAIR)
    print(np.frombuffer(features["ge_data"].numpy(), dtype=cfg.GE_DTYPE))
    print(np.frombuffer(features["dd1_data"].numpy(), dtype=cfg.DD_DTYPE))
    print(np.frombuffer(features["dd2_data"].numpy(), dtype=cfg.DD_DTYPE))
    tf.print(features.keys())

    print("\nDone.")


def update_tfrecords_with_rna(n_samples: Optional[int] = None) -> None:
    """
    Takes original tfrecords that we got from A. Pearson and updates them
    by addting more data including PDX samples metadata and RNA-Seq data.

    We take RNA data and metadata csv files (crossref file that comes with the
    histology slides and PDX meta that Yitan prepared), and merge them to
    obtain df that contains samples that have RNA data and the corresponding
    tfrecords.

    Only for those slides we update the tfrecords and store them in a new
    directory.
    """
    # Create path for the updated tfrecords
    # outpath = cfg.SF_TFR_DIR_RNA/LABEL
    outpath = cfg.SF_TFR_DIR_RNA_NEW/LABEL
    os.makedirs(outpath, exist_ok=True)

    # Load data
    rna = load_data.load_rna()
    cref = load_data.load_crossref()
    pdx = load_data.load_pdx_meta2()

    # Merge cref and rna
    print(cref.shape)
    print(rna.shape)
    cref_rna = cref.merge(rna, on=PDX_SAMPLE_COLS, how='inner')
    print(cref_rna.shape)

    # Merge with PDX meta
    print(pdx.shape)
    print(cref_rna.shape)
    data = pdx.merge(cref_rna, on=['patient_id', 'specimen_id'], how='inner').reset_index(drop=True)
    print(data.shape)

    if n_samples is not None:
        data = data.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    # Re-org cols
    dim = data.shape[1]
    meta_cols = ['Sample',
                 'model', 'patient_id', 'specimen_id', 'sample_id', 'image_id', 
                 'csite_src', 'ctype_src', 'csite', 'ctype', 'stage_or_grade']
    ge_cols = [c for c in data.columns if str(c).startswith('ge_')]
    data = data[meta_cols + ge_cols]
    assert data.shape[1] == dim, "There are missing cols after re-organizing the cols."

    # Create dict of slide ids. Each slide (key) contains a dict with metadata.
    assert sum(data.duplicated('image_id', keep=False)) == 0, 'There are duplicates of image_id in the df'
    mm = {}  # dict to store all metadata
    gg = {}
    id_name = 'image_id'  # col name that contains the IDs for the samples 

    # Iterate over rows a collect data into dict
    for i, row_data in data.iterrows():
        # Dict to contain metadata for the current sample (slide)
        sample_dct = {}
        smp = str(row_data[id_name])

        # Meta cols
        # Create a (key, value) pair for each meta col
        # meta_cols = [c for c in row_data.index if not c.startswith('ge_')]
        for c in meta_cols:
            sample_dct[c] = str(row_data[c])

        # import ipdb; ipdb.set_trace()

        # Features cols
        #ge_data = list(row_data[ge_cols].values.astype(cfg.GE_DTYPE))
        #sample_id['ge_data'] = ge_data
        ge_data = row_data[ge_cols].values.astype(cfg.GE_DTYPE)
        sample_dct['ge_data'] = ge_data.tobytes()
        # check
        # jj = np.frombuffer(sample_dct['ge_data'], dtype=cfg.GE_DTYPE)
        # print(all(jj == ge_data))

        mm[smp] = sample_dct
        gg[smp] = ge_data
        
    print(f'A total of {len(mm)} samples with image and rna data.')

    # Common slides (that have both image and rna data)
    #c_slides = [s for s in all_slides if s in mm.keys()]
    slides = data['image_id'].unique().tolist()
    all_slides = original_tfr_names(label=LABEL)
    c_slides = set(slides).intersection(set(all_slides))
    print(f'A total of {len(c_slides)} samples with tfrecords and rna data.')

    # Load tfrecords and update with new data
    for i, slide_name in enumerate(sorted(c_slides)):
        rel_tfr = str(slide_name) + '.tfrecords'
        tfr = str(directory/rel_tfr)
        
        raw_dataset = tf.data.TFRecordDataset(tfr)
            
        tfr_fname = str(outpath/rel_tfr)
        writer = tf.io.TFRecordWriter(tfr_fname)
        
        for tile_cnt, rec in enumerate(raw_dataset):
            # Features of the current rec from old tfrecord
            features = tf.io.parse_single_example(rec, features=FEA_SPEC)
            # tf.print(features.keys())

            # Extract slide name from old tfrecord and get the new metadata to be added to the new tfrecord
            slide = features['slide'].numpy().decode('utf-8')
            slide_meta = mm[slide]

            # slide, image_raw = _read_and_return_features(record)
            ex = tf.train.Example(features=tf.train.Features(
                feature={
                    # old features
                    'slide':       _bytes_feature(features['slide'].numpy()),  # image_id
                    'image_raw':   _bytes_feature(features['image_raw'].numpy()),

                    # new features
                    'Sample':      _bytes_feature(bytes(slide_meta['Sample'], 'utf-8')),
                    'model':       _bytes_feature(bytes(slide_meta['model'], 'utf-8')),
                    'patient_id':  _bytes_feature(bytes(slide_meta['patient_id'], 'utf-8')),
                    'specimen_id': _bytes_feature(bytes(slide_meta['specimen_id'], 'utf-8')),
                    'sample_id':   _bytes_feature(bytes(slide_meta['sample_id'], 'utf-8')),
                    'image_id':    _bytes_feature(bytes(slide_meta['image_id'], 'utf-8')),

                    'ctype':       _bytes_feature(bytes(slide_meta['ctype'], 'utf-8')),
                    'csite':       _bytes_feature(bytes(slide_meta['csite'], 'utf-8')),
                    'ctype_src':   _bytes_feature(bytes(slide_meta['ctype_src'], 'utf-8')),
                    'csite_src':   _bytes_feature(bytes(slide_meta['csite_src'], 'utf-8')),

                    'ge_data':     _bytes_feature(slide_meta['ge_data']),
                }
            ))
            
            writer.write(ex.SerializeToString())
            
        print(f"\r\033[K Created tfrecord using {green(rel_tfr)} ({i+1} out of {len(c_slides)}; {tile_cnt+1} tiles) ...", end="") 
        
        writer.close()
    print()
        
        
    # ------------------
    # Inspect a TFRecord
    # ------------------

    # import ipdb; ipdb.set_trace()

    smp = list(c_slides)[0]
    tfr_path = str(outpath/(str(smp) + '.tfrecords'))
    raw_dataset = tf.data.TFRecordDataset(tfr_path)
    rec = next(raw_dataset.__iter__())
    # features = tf.io.parse_single_example(rec, features=FEA_SPEC_RNA)
    features = tf.io.parse_single_example(rec, features=FEA_SPEC_RNA_NEW)
    ge_data = np.frombuffer(features['ge_data'].numpy(), dtype=cfg.GE_DTYPE)
    print(all(ge_data == gg[smp]))
    tf.print(features.keys())

    print('\nDone.')


# update_tfrecords_with_rna(n_samples)
update_tfrecords_for_drug_rsp(n_samples, single_drug)
