""" 
Takes the original tfrecords that we got from Alex Pearson and updates
them with new data including PDX sample metadata and RNA-Seq data.

We take metadata csv files (crossref file and comes with the histology
slides and PDX meta that Yitan prepared) and RNA data, and merge them
to obtain df that contains samples that have RNA data and TFRecords.

Only for those slides we update the tfrecords and store them in a new
directory.
"""
import os
import sys
assert sys.version_info >= (3, 5)

from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np

import tensorflow as tf
assert tf.__version__ >= '2.0'

# from build_tfrec import show_img  # (show_images, encode_categorical, np_img_to_bytes)
# from train_nn import get_tfr_files
from tf_utils import (calc_records_in_tfr_folder, calc_examples_in_tfrecord,
                      _float_feature, _bytes_feature, _int64_feature)

from build_df import load_rna
from merge_meta_files import load_crossref, load_pdx_meta

from tfrecords import FEA_SPEC #, FEA_SPEC_NEW

fdir = Path(__file__).resolve().parent
from config import cfg


label = '299px_302um'

GREEN = '\033[92m'
ENDC = '\033[0m'
def green(text):
    return GREEN + str(text) + ENDC

import ipdb; ipdb.set_trace()


# -------------------------------------------
# Get the dataframe with the metadata and RNA
# -------------------------------------------
# Load data
rna = load_rna()
cref = load_crossref()
pdx = load_pdx_meta()

mrg_cols = ['model', 'patient_id', 'specimen_id', 'sample_id']

# Add columns to rna by parsing the Sample col
patient_id = rna['Sample'].map(lambda x: x.split('~')[0])
specimen_id = rna['Sample'].map(lambda x: x.split('~')[1])
sample_id = rna['Sample'].map(lambda x: x.split('~')[2])
model = [a + '~' + b for a, b in zip(patient_id, specimen_id)]
rna.insert(loc=1, column='model', value=model, allow_duplicates=True)
rna.insert(loc=2, column='patient_id', value=patient_id, allow_duplicates=True)
rna.insert(loc=3, column='specimen_id', value=specimen_id, allow_duplicates=True)
rna.insert(loc=4, column='sample_id', value=sample_id, allow_duplicates=True)
rna = rna.sort_values(['model', 'patient_id', 'specimen_id', 'sample_id'])

# Remove bad samples with bad slides
cref = cref[~cref.image_id.isin(cfg.BAD_SLIDES)].reset_index(drop=True)

print(rna.shape)
print(cref.shape)
print(pdx.shape)

# Merge cref and rna
cref_rna = cref[mrg_cols + ['image_id']].merge(rna, on=mrg_cols, how='inner').reset_index(drop=True)
# Note that we also loose some samples when we merge with pdx metadata
data = pdx.merge(cref_rna, on=['patient_id', 'specimen_id'], how='inner').reset_index(drop=True)
# Re-org cols
cols = ['Sample', 'model', 'patient_id', 'specimen_id', 'sample_id', 'image_id', 
        'csite_src', 'ctype_src', 'csite', 'ctype', 'stage_or_grade']
ge_cols = [c for c in data.columns if str(c).startswith('ge_')]
data = data[cols + ge_cols]

# ---------------
# Update TFRecods
# ---------------
# Destination for the updated tfrecords
# outpath = cfg.SF_TFR_DIR
# outpath = Path(str(outpath) + '_updated')/label
outpath = cfg.SF_TFR_DIR_RNA/label
os.makedirs(outpath, exist_ok=True)

# Create dict of slide ids. Each slide contain a dict with metadata.
assert sum(cref_rna.duplicated('image_id', keep=False)) == 0, 'There are duplicates of image_id in the df'

mt = {}  # dict to store all metadata
GE_TYPE = np.float32

# Note that we use cref_rna since the subequent merge with pdx further
# looses a few samples.
for i, row_data in cref_rna.iterrows():
    # Dict to contain metadata for the current slide
    slide_dct = {}

    # Meta cols
    meta_cols = [c for c in row_data.index if not c.startswith('ge_')]
    for c in meta_cols:
        slide_dct[c] = str(row_data[c])

    # RNA cols
    ge_cols = [c for c in row_data.index if c.startswith('ge_')]
    ge_data = list(row_data[ge_cols].values.astype(GE_TYPE))
    slide_dct['ge_data'] = ge_data
    
    slide = str(row_data['image_id'])
    mt[slide] = slide_dct
    
print(f'A total of {len(mt)} samples with image and rna data.')


# Obtain slide names for which we need to update the tfrecords
directory = cfg.SF_TFR_DIR/label
tfr_files = list(directory.glob('*.tfrec*'))
print(f'A total of {len(tfr_files)} original tfrecords.')

# Slide names from tfrecords
slides = [s.name.split('.tfrec')[0] for s in tfr_files]

# Common slides (that have both image and rna data)
c_slides = [s for s in slides if s in mt.keys()]
print(f'A total of {len(c_slides)} samples with tfrecords and rna data.')

print('Missing tfrecords for the following slides (bad quality of histology slides): ',
       sorted(set(mt.keys()).difference(set(c_slides))))
# print(sorted(cfg.BAD_SLIDES))


# Load tfrecords and update with new data
for i, s in enumerate(sorted(c_slides)):
    rel_tfr = str(s) + '.tfrecords'
    tfr = str(directory/rel_tfr)  #join(directory, rel_tfr)
    
    print(f"\r\033[K Updating {green(rel_tfr)} ({i+1} out of {len(c_slides)} tfrecords) ...", end="") 
    
    tfr_fname = str(outpath/rel_tfr)
    writer = tf.io.TFRecordWriter(tfr_fname)
    
    raw_dataset = tf.data.TFRecordDataset(tfr)
        
    for rec in raw_dataset:
        features = tf.io.parse_single_example(rec, features=FEA_SPEC)  # rec features from old tfrecord
        # tf.print(features.keys())

        # Extract slide name from old tfrecord and get the new metadata to be added to the new tfrecord
        slide = features['slide'].numpy().decode('utf-8')
        slide_meta = mt[slide]

        # slide, image_raw = _read_and_return_features(record)
        
        ex = tf.train.Example(features=tf.train.Features(
            feature={
                # old features
                'slide':       _bytes_feature(features['slide'].numpy()),     # image_id
                'image_raw':   _bytes_feature(features['image_raw'].numpy()),

                # new features
                'model':       _bytes_feature(bytes(slide_meta['model'], 'utf-8')),
                'patient_id':  _bytes_feature(bytes(slide_meta['patient_id'], 'utf-8')),
                'specimen_id': _bytes_feature(bytes(slide_meta['specimen_id'], 'utf-8')),
                'sample_id':   _bytes_feature(bytes(slide_meta['sample_id'], 'utf-8')),
                'image_id':    _bytes_feature(bytes(slide_meta['image_id'], 'utf-8')),
                'Sample':      _bytes_feature(bytes(slide_meta['Sample'], 'utf-8')),
                'ge_data':     _float_feature(slide_meta['ge_data']),
            }
        ))
        
        writer.write(ex.SerializeToString())
        
    writer.close()
    
    
# ----------------------
# Try to load a TFRecord
# ----------------------
GE_LEN = len(slide_meta['ge_data'])

fea_spec_new = {
    'slide': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),

    'model': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'patient_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'specimen_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'sample_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'image_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'Sample': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'ge_data': tf.io.FixedLenFeature(shape=(GE_LEN,), dtype=tf.float32, default_value=None),
}

s = c_slides[0]
rel_tfr = str(s) + '.tfrecords'
tfr_path = str(outpath/rel_tfr)
raw_dataset = tf.data.TFRecordDataset(tfr_path)
rec = next(raw_dataset.__iter__())
features = tf.io.parse_single_example(rec, features=fea_spec_new)  # rec features from old tfrecord
tf.print(features.keys())

print('\nDone.')
