""" 
Takes the original tfrecords that we got from Alex Pearson and uses
them to generate new tfrecords with additional data.

update_tfrecords_with_rna()
    updates tfrecords with RNA-seq and metadata of PDX samples

update_tfrecords_for_drug_rsp()
    creates tfrecord for drug response sample that contains histo
    slide, RNA-seq, drug descriptors, and drug response

Examples:
$ python scripts/update_tfrecords.py --frac_tiles 0.1
$ python scripts/update_tfrecords.py --n_samples 3 --frac_tiles 0.1
$ python scripts/update_tfrecords.py --n_samples 3 --frac_tiles 0.1 --single_drug
"""
import os
import sys
assert sys.version_info >= (3, 5)

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from pprint import pprint
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
from src.sf_utils import green
from src.tfrecords import FEA_SPEC, FEA_SPEC_RSP, FEA_SPEC_RSP_DRUG_PAIR, FEA_SPEC_RNA_NEW, original_tfr_names
from src.tf_utils import _float_feature, _bytes_feature, _int64_feature, calc_examples_in_tfrecord
from src.utils.utils import Timer

# Seed
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)


parser = argparse.ArgumentParser("Create TFRecords.")
parser.add_argument("--n_samples",
                    type=int,
                    default=-1,
                    help="Total samples to process.")
parser.add_argument("--frac_tiles",
                    type=float,
                    # default=1.0,
                    default=None,
                    help="Fraction of tiles to use from each slide for creating TFRecords.")
parser.add_argument("--single_drug",
                    action="store_true",
                    help="Use only single-drug drug responses.")
parser.add_argument("--random",
                    action="store_true",
                    help="Take random tiles.")
# parser.add_argument("--max_tiles",
#                     type=int,
#                     default=None,
#                     help="Max tiles (if None, use the global min across all slides).")
parser.add_argument("--take_glob_min",
                    action="store_true",
                    help="Take same number of tiles from each slide which is the global min across slides.")
args, other_args = parser.parse_known_args()
pprint(args)


LABEL = "299px_302um"
# directory = cfg.SF_TFR_DIR/LABEL
directory = cfg.PDX_FIXED/LABEL

# single_drug = True
# single_drug = False  # drug pairs

timer = Timer()


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.

    Args:
        np_img: The image represented as a NumPy array.

    Returns:
        The NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64" or np_img.dtype == "float32":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)


def update_tfrecords_for_drug_rsp(n_samples: int=-1,
                                  single_drug: bool=False,
                                  frac_tiles: Optional[float]=None,
                                  # min_tiles: Optional[int]=None,
                                  take_glob_min: bool=False,
                                  random: bool=False) -> None:
    """
    Takes original tfrecords that we got from A. Pearson and updates them
    by addting more data including metadata of PDX samples, RNA-Seq, drug
    descriptors, and drug response.

    Args:
        n_samples : generate tfrecords for n_samples drug response samples
            (primarily used for debugging)
    """
    # Create path for the updated tfrecords
    # if single_drug:
    #     outpath = cfg.SF_TFR_DIR_RSP/LABEL
    # else:
    #     outpath = cfg.SF_TFR_DIR_RSP_DRUG_PAIR/LABEL
    if single_drug:
        # base_outdir = cfg.SF_TFR_DIR_RSP
        pass
    else:
        # base_outdir = cfg.SF_TFR_DIR_RSP_DRUG_PAIR
        base_outdir = cfg.PDX_FIXED_RSP_DRUG_PAIR
    if frac_tiles and frac_tiles < 1.0:
        base_outdir = str(base_outdir) + "_" + str(frac_tiles) + "_of_tiles"
    if random:
        base_outdir = str(base_outdir) + "_rnd"
    if args.take_glob_min:
        base_outdir = str(base_outdir) + "_glob_min"
    outpath = Path(base_outdir)/LABEL
    os.makedirs(outpath, exist_ok=True)

    # Load data
    data = load_data.load_tidy_dataset_rsp(single_drug=single_drug, add_type_labels=True)

    if n_samples > 0:
        data = data.sample(n=n_samples, random_state=cfg.seed).reset_index(drop=True)

    ge_cols = [c for c in data.columns if str(c).startswith('ge_')]
    dd1_cols = [c for c in data.columns if str(c).startswith("dd1_")]
    dd2_cols = [c for c in data.columns if str(c).startswith("dd2_")]
    cols = data.columns.tolist()
    meta_cols = [c for c in cols if c not in (ge_cols + dd1_cols + dd2_cols)]

    # Create dict of sample ids. Each sample (key) contains a dict with metadata.
    mm = {}  # dict to store all metadata
    id_name = "smp"  # col name that contains the IDs for the samples 


    # Iterate over rows and collect data into dict
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
    # all_slides = original_tfr_names(label=LABEL)
    all_slides = original_tfr_names(tfr_basedir=cfg.PDX_FIXED, label=LABEL)
    c_slides = set(slides).intersection(set(all_slides))
    print(f"A total of {len(c_slides)} slides that are relevant for our drug response samples.")

    tile_cnts = []

    # Obtain tile count per slide
    dct = {}
    for i, slide_name in enumerate(sorted(c_slides)):
        rel_tfr = str(slide_name) + ".tfrecords"
        tfr = str(directory/rel_tfr)
        max_tiles = calc_examples_in_tfrecord(tfr)
        dct[slide_name] = max_tiles
    aa = pd.DataFrame([dct]).T.reset_index()
    aa.columns = ["slide_name", "max_tiles"]

    # Specify number of tiles to take from each slide
    min_tiles_global = aa["max_tiles"].min()

    # Create a tfrecord for each sample (iter over samples)
    for i, slide_name in enumerate(sorted(c_slides)):
        # Name of original tfrecord to load that contains tiles for a single
        # histo slide
        rel_tfr = str(slide_name) + ".tfrecords"
        tfr = str(directory/rel_tfr)

        max_tiles = calc_examples_in_tfrecord(tfr)
        # n_tiles = int(frac_tiles * max_tiles)  # num to use from the current tfrecord (slide)
        if frac_tiles and frac_tiles < 1.0:
            n_tiles = int(frac_tiles * max_tiles)  # num to use from the current tfrecord (slide)
        elif args.take_glob_min:
            n_tiles = min_tiles_global
        else:
            n_tiles = max_tiles

        print(f"\r\033[K Creating drug response tfrecords using {green(rel_tfr)} (slide {i+1} out of {len(c_slides)} slides) ...", end="") 
        
        raw_dataset = tf.data.TFRecordDataset(tfr)

        # Draw random tiles (and not sequentially)
        if random:
            rnd_tile_ids = sorted(np.random.choice(range(max_tiles), size=n_tiles, replace=False))
            
        # Create folder to write tile images (png)
        img_dir = outpath/f"slide_{slide_name}"
        os.makedirs(img_dir, exist_ok=True)

        # Iter over drug response samples that use the current slide
        samples = data[data["image_id"] == slide_name][id_name].values.tolist()
        for smp in samples:

            # Name of the output tfrecord for the current drug response sample
            tfr_fname = str(outpath/(smp + ".tfrecords"))
            writer = tf.io.TFRecordWriter(tfr_fname)

            # Iter over tiles of the current slide
            tile_counter = 0
            for tile_id, rec in enumerate(raw_dataset):

                # Check if the tile_id is one of those drawn at random
                if random and (tile_id not in rnd_tile_ids):
                    continue
                else:
                    tile_counter += 1

                if tile_counter > n_tiles:
                    break

                # Features of the current rec from old tfrecord
                features = tf.io.parse_single_example(rec, features=FEA_SPEC)
                # tf.print(features.keys())

                # Extract slide name from old tfrecord and get the new metadata
                # to be added to the new tfrecord
                slide = features["slide"].numpy().decode("utf-8")
                slide_meta = mm[smp]

                # Write image file (png) into folder
                np_img = tf.image.decode_jpeg(features['image_raw'], channels=3).numpy()
                pil_img = np_to_pil(np_img)
                pil_img.save(img_dir/f"tile_{tile_id}.png", "PNG")

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
                        
                        "tile_id":     _bytes_feature(bytes(str(tile_id), "utf-8")),
                        # "tile_id":     _int64_feature(int(tile_id)),

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
                        # "ctype_label": _bytes_feature(bytes(slide_meta["ctype_label"], "utf-8")),
                        # "csite_label": _bytes_feature(bytes(slide_meta["csite_label"], "utf-8")),
                        "ctype_label": _int64_feature(int(slide_meta["ctype_label"])),
                        "csite_label": _int64_feature(int(slide_meta["csite_label"])),

                        # "Drug1":       _bytes_feature(bytes(slide_meta["Drug1"], "utf-8")),
                        # "NAME":        _bytes_feature(bytes(slide_meta["NAME"], "utf-8")),
                        # "CLEAN_NAME":  _bytes_feature(bytes(slide_meta["CLEAN_NAME"], "utf-8")),
                        # "ID":          _bytes_feature(bytes(slide_meta["ID"], "utf-8")),
                        "Drug1":       _bytes_feature(bytes(slide_meta["Drug1"], "utf-8")),
                        "Drug2":       _bytes_feature(bytes(slide_meta["Drug2"], "utf-8")),
                        "trt":         _bytes_feature(bytes(slide_meta["trt"], "utf-8")),
                        "aug":         _bytes_feature(bytes(slide_meta["aug"], "utf-8")),

                        "Response":    _int64_feature(int(slide_meta["Response"])),
                        # "Response":    _bytes_feature(bytes(str(slide_meta["Response"]), "utf-8")),

                        # 'ge_data':     _float_feature(slide_meta['ge_data']),
                        # 'dd_data':     _float_feature(slide_meta['dd_data']),
                        "ge_data":     _bytes_feature(slide_meta["ge_data"]),
                        "dd1_data":    _bytes_feature(slide_meta["dd1_data"]),
                        "dd2_data":    _bytes_feature(slide_meta["dd2_data"]),
                    }
                ))
                
                writer.write(ex.SerializeToString())

            # print(f"Total tiles in the sample {tile_id+1}")
            tile_cnts.append({"tfr_fname": tfr_fname.split(os.sep)[-1],
                              "smp": smp,
                              "slide": slide_name,
                              "max_tiles": max_tiles,
                              "n_tiles": n_tiles})
            writer.close()
        print()
        
    tile_cnts = pd.DataFrame(tile_cnts)
    tile_cnts = tile_cnts.drop_duplicates().reset_index(drop=True)
    meta = data[["smp", "Group", "grp_name", "Response"]]
    tile_cnts = tile_cnts.merge(meta, on="smp", how="inner").reset_index(drop=True)
    tile_cnts.to_csv(outpath/"tile_counts_per_slide.csv", index=False)


    # ------------------
    # Inspect a TFRecord
    # ------------------

    smp = samples[0]
    tfr_path = str(outpath/(smp + ".tfrecords"))
    raw_dataset = tf.data.TFRecordDataset(tfr_path)
    # rec = next(raw_dataset.take(1).__iter__())
    rec = next(raw_dataset.__iter__())
    if single_drug:
        features = tf.io.parse_single_example(rec, features=FEA_SPEC_RSP)
    else:
        features = tf.io.parse_single_example(rec, features=FEA_SPEC_RSP_DRUG_PAIR)
    print(np.frombuffer(features["ge_data"].numpy(), dtype=cfg.GE_DTYPE))
    # print(np.frombuffer(features["dd1_data"].numpy(), dtype=cfg.DD_DTYPE))
    # print(np.frombuffer(features["dd2_data"].numpy(), dtype=cfg.DD_DTYPE))
    tf.print(features.keys())

    for rec in raw_dataset.take(4):
        features = tf.io.parse_single_example(rec, features=FEA_SPEC_RSP_DRUG_PAIR)
        print("tile_id:     ", features["tile_id"].numpy())
        print("Response:    ", features["Response"].numpy())
        print("ctype_label: ", features["ctype_label"].numpy())
        print("csite_label: ", features["csite_label"].numpy())
        # print("ctype_label: ", tf.cast(tf.strings.to_number(features["ctype_label"]), tf.int64))

    print("\nDone.")
    return None


def update_tfrecords_with_rna(n_samples: int=-1) -> None:
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

    if n_samples > 0:
        data = data.sample(n=n_samples, random_state=cfg.seed).reset_index(drop=True)

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


# update_tfrecords_with_rna(args.n_samples)
update_tfrecords_for_drug_rsp(args.n_samples, args.single_drug,
                              args.frac_tiles, args.take_glob_min, args.random)
timer.display_timer()
